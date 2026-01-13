import sys
import os
import numpy as np
import torch
import argparse

ROOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, ROOT_PATH)
from experiments import (
    compute_embedding,
)
from utils import (
    load_pickle,
)

sys.path.pop(0)


def main():
    parser = argparse.ArgumentParser(
        description="Predict model performance from target model outputs"
    )
    parser.add_argument(
        "--prediction_path",
        type=str,
        required=True,
        help="Path to predictions file (identical to predictions_test_v2)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pickled model file (identical to fitted_weights, should also contain transform)",
    )
    parser.add_argument(
        "--transform_path",
        type=str,
        default=None,
        help="Optional path to pickled transform file. If provided, will be used instead of transform from model_path.",
    )
    parser.add_argument(
        "--pca",
        type=int,
        default=None,
        help="Optional PCA value. If provided and not None, --transform_path must be given.",
    )
    args = parser.parse_args()

    # Assert that if pca is not None, then --transform_path should be given
    if args.pca is not None:
        assert (
            args.transform_path is not None
        ), "If --pca is provided (not None), --transform_path must be given."

    # Load predictions (predictions_test_v2)
    # Shape: (n_models, n_anchor_points, n_classes)
    predictions_test_v2 = load_pickle(args.prediction_path)

    # Load model (fitted_weights + transform)
    # Expected structure: dict with keys:
    # - "fitted_weights": fitted_weights dict with structure fitted_weights[sampling_name][number_item][fitted_model_type]
    # - "transform": the PCA transform used for embeddings
    # - "sampling_name": sampling name (optional, will be inferred if not provided)
    # - "number_item": number of items (optional, will be inferred if not provided)
    # - "fitted_model_type": model type (optional, will be inferred if not provided)
    # - "pca": pca value (optional, defaults to 256)
    model_data = load_pickle(args.model_path)

    # Extract model components
    if not isinstance(model_data, dict):
        raise ValueError(
            f"model_path must contain a dict. Got {type(model_data)}"
        )

    # Get transform (from transform_path if provided, otherwise from model_data)
    if args.transform_path is not None:
        transform_v2 = load_pickle(args.transform_path)
    else:
        # Get transform from model_data
        if "transform" in model_data:
            transform_v2 = model_data["transform"]
        else:
            raise ValueError(
                "model_path must contain 'transform' key, or --transform_path must be provided. "
                "The transform (PCA) is required to compute embeddings from predictions."
            )

    # Get fitted_weights
    if "fitted_weights" in model_data:
        fitted_weights = model_data["fitted_weights"]
    else:
        # Assume the dict itself is fitted_weights (excluding transform)
        fitted_weights = {
            k: v for k, v in model_data.items() if k != "transform"
        }
        if not fitted_weights:
            raise ValueError(
                "Could not find fitted_weights in model_path. "
                "Expected dict with 'fitted_weights' key or fitted_weights structure at top level."
            )

    # Get metadata or infer from structure
    if "sampling_name" in model_data:
        sampling_name = model_data["sampling_name"]
    else:
        sampling_name = list(fitted_weights.keys())[0]

    if "number_item" in model_data:
        number_item = model_data["number_item"]
    else:
        number_item = list(fitted_weights[sampling_name].keys())[0]

    if "fitted_model_type" in model_data:
        fitted_model_type = model_data["fitted_model_type"]
    else:
        fitted_model_type = list(
            fitted_weights[sampling_name][number_item].keys()
        )[0]

    # Get pca value (from command line if provided, otherwise from model_data, otherwise default to 256)
    if args.pca is not None:
        pca = args.pca
    else:
        pca = model_data.get("pca", 256)

    # Compute embeddings from predictions
    target_embeddings_v2, _ = compute_embedding(
        predictions_test_v2,
        anchor_indices=None,
        pca=pca,
        transform=transform_v2,
        apply_softmax=True,
    )

    # Get the fitted model
    fitted_model = fitted_weights[sampling_name][number_item][fitted_model_type]

    # Predict accuracies for each model
    predicted_accs = {}
    for target_model_idx in range(target_embeddings_v2.shape[0]):
        test_model_embedding = target_embeddings_v2[target_model_idx]

        # Convert to numpy if it's a torch tensor
        if isinstance(test_model_embedding, torch.Tensor):
            test_model_embedding_np = test_model_embedding.numpy()
        else:
            test_model_embedding_np = test_model_embedding

        # Predict using fitted model
        fitted_acc = fitted_model.predict(
            test_model_embedding_np.reshape(1, -1)
        )[0]

        predicted_accs[target_model_idx] = fitted_acc

    # Print predicted accuracies
    print("Predicted accuracies:")
    for model_idx, acc in predicted_accs.items():
        print(f"Model {model_idx}: {acc:.6f}")


if __name__ == "__main__":
    main()
