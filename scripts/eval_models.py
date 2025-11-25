import os
import json
import pickle
from tqdm import tqdm

import torch
import timm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

MODEL_CATALOG_PATH = (
    "../efficbench/Benjamin/tensors_with_outputs/timm_model_catalog.json"
)
RESULTS_PICKLE = "all_model_results.pickle"
# IMAGENET_VAL_PATH = os.environ.get("IMAGENET_VAL_PATH", "./imagenet-validation")  # Directory containing the val set
IMAGENET_VAL_PATH = "/weka/datasets/ImageNet2012/val/"

BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_catalog():
    with open(MODEL_CATALOG_PATH, "r") as f:
        return json.load(f)


def load_results():
    if os.path.exists(RESULTS_PICKLE):
        with open(RESULTS_PICKLE, "rb") as f:
            return pickle.load(f)
    else:
        return {
            "model_names": [],
            "all_correctness": None,
            "all_confidence": None,
        }


def get_dataloader(resize_size=256, target_size=224):
    val_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    val_dataset = datasets.ImageFolder(
        IMAGENET_VAL_PATH, transform=val_transform
    )
    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return loader, val_dataset


def eval_model_on_imagenet(model_name, device, dataloader, dataset):
    # Handle timm models with pretrained weights
    with torch.no_grad():
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        model.to(device)
        results = []
        idx_offset = 0
        for batch_idx, (inputs, targets) in enumerate(
            tqdm(dataloader, desc=f"Eval {model_name}")
        ):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            batch_size = inputs.shape[0]
            # Record individual datapoint results
            for i in range(batch_size):
                img_fp, _ = dataset.samples[idx_offset + i]
                correct = int(preds[i].cpu().item() == targets[i].cpu().item())
                conf = float(confidences[i].cpu().item())
                # Instead of storing dicts for each datapoint, accumulate correctness and confidence in vectors.
                if batch_idx == 0 and i == 0:
                    # On first call, initialize storage outside the loop if not already in "results"

                    if isinstance(results, list) and len(results) == 0:
                        results = np.zeros(2 * len(dataset), dtype=float)

                results[idx_offset + i] = float(
                    correct
                )  # first 50k are correctness, second 50k are confidences
                results[
                    len(dataset) + idx_offset + i
                ] = conf  # second 50k are confidences
            idx_offset += batch_size
        return results


def main():
    model_catalog = load_model_catalog()
    results_dict = load_results()
    dataloader, dataset = get_dataloader()

    # Initialize storage for intermediate results
    # If we have existing results, reconstruct the list from them
    all_results_list = []
    if (
        results_dict.get("all_correctness") is not None
        and results_dict.get("all_confidence") is not None
    ):
        num_samples = len(dataset)
        all_correctness = results_dict["all_correctness"]
        all_confidence = results_dict["all_confidence"]
        # Reconstruct the 1D arrays for each model
        for i in range(all_correctness.shape[0]):
            combined = np.concatenate([all_correctness[i], all_confidence[i]])
            all_results_list.append(combined)

    for model_entry in tqdm(model_catalog, desc="Evaluating models"):
        non_standard_dataloader = None
        if isinstance(
            model_entry, dict
        ):  # If catalog is a list of dicts with 'name'
            model_name = model_entry.get("name", "")
        else:
            model_name = str(model_entry)
            if "196" in model_name:
                non_standard_dataloader, non_standard_dataset = get_dataloader(
                    resize_size=256, target_size=196
                )
            elif "flexivit_" in model_name or "240" in model_name:
                non_standard_dataloader, non_standard_dataset = get_dataloader(
                    resize_size=256, target_size=240
                )
            elif "sehalonet33ts" in model_name or "256" in model_name:
                non_standard_dataloader, non_standard_dataset = get_dataloader(
                    resize_size=256, target_size=256
                )
            elif "336" in model_name:
                non_standard_dataloader, non_standard_dataset = get_dataloader(
                    resize_size=512, target_size=336
                )
            elif "384" in model_name:
                non_standard_dataloader, non_standard_dataset = get_dataloader(
                    resize_size=512, target_size=384
                )
            elif "448" in model_name:
                non_standard_dataloader, non_standard_dataset = get_dataloader(
                    resize_size=512, target_size=448
                )
            elif "512" in model_name:
                non_standard_dataloader, non_standard_dataset = get_dataloader(
                    resize_size=512, target_size=512
                )

        if not model_name:
            continue
        model_names = results_dict.get("model_names", []) or []
        if model_name in model_names:
            print(f"Skipping {model_name}: already in results")
            continue
        try:
            print(f"Evaluating {model_name} ...")
            per_datapoint_results = eval_model_on_imagenet(
                model_name,
                DEVICE,
                dataloader
                if non_standard_dataloader is None
                else non_standard_dataloader,
                dataset
                if non_standard_dataloader is None
                else non_standard_dataset,
            )

            # Add to results
            if results_dict.get("model_names") is None:
                results_dict["model_names"] = []
            results_dict["model_names"].append(model_name)
            all_results_list.append(per_datapoint_results)

            # Now, stack all results after every new model is added.
            # Each per_datapoint_results is a 1D array of length 2 * #samples.
            # After stacking, shape will be (#models, 2 * #samples)
            all_results_np = np.stack(
                all_results_list, axis=0
            )  # (#models, 2 * #samples)
            num_samples = len(dataset)
            num_models = all_results_np.shape[0]
            # Split into correctness and confidences, keeping shape (#models, #samples)
            all_correctness = all_results_np[
                :, :num_samples
            ]  # (num_models, num_samples)
            all_confidence = all_results_np[
                :, num_samples:
            ]  # (num_models, num_samples)

            # Update results_dict and save
            results_dict["all_correctness"] = all_correctness
            results_dict["all_confidence"] = all_confidence

            with open(RESULTS_PICKLE, "wb") as f:
                pickle.dump(results_dict, f)
            print(f"Saved results for {model_name} to {RESULTS_PICKLE}.")
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            continue


if __name__ == "__main__":
    main()
