"""
Helper functions for post-processing lm-evaluation-harness results.
Extracted from notebooks/analyze_presaved_outputs.ipynb
"""

import os
import json
import numpy as np
from pathlib import Path
import sys

# Add repo root to path to import utils
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from utils import load_pickle, dump_pickle
from utils_for_notebooks import pad_predictions

sys.path.pop(0)


def load_jsonl(filename):
    """Load JSONL file and return list of dictionaries."""
    results = []
    with open(filename, "r") as infile:
        for line in infile:
            results.append(json.loads(line))
    return results


def find_jsonl_file_in_directory(directory_path):
    """
    Find the JSONL file in a directory. Looks for files matching pattern:
    samples_*_prompts_*.jsonl

    Parameters:
    - directory_path: Path to directory containing the JSONL file

    Returns:
    - Path to JSONL file, or None if not found
    """
    if not os.path.isdir(directory_path):
        # If it's already a file, return it
        if os.path.isfile(directory_path) and directory_path.endswith(".jsonl"):
            return directory_path
        return None

    # Look for JSONL files matching the pattern
    # Collect all matching files and return the most recent one
    matching_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if (
                file.endswith(".jsonl")
                and "samples" in file
                and "prompts" in file
            ):
                file_path = os.path.join(root, file)
                matching_files.append((file_path, os.path.getmtime(file_path)))

    if matching_files:
        # Return the most recently modified file
        return max(matching_files, key=lambda x: x[1])[0]

    # If not found, try to find any JSONL file (also return most recent)
    all_jsonl_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                all_jsonl_files.append((file_path, os.path.getmtime(file_path)))

    if all_jsonl_files:
        return max(all_jsonl_files, key=lambda x: x[1])[0]

    return None


def convert_jsonl_results_to_arrays(jsonl_results, metric, anchor_points=None):
    """
    Convert JSONL results to predictions and correctness arrays.

    Parameters:
    - jsonl_results: List of dictionaries from load_jsonl
    - metric: Metric name to use for correctness (e.g., "acc", "acc_norm")
    - anchor_points: Optional list of anchor point indices. If provided, results will be
                     reordered to match the anchor_points order based on doc_id.

    Returns:
    - predictions_2d: numpy array of shape (n_questions, n_choices)
    - correctness_1d: numpy array of shape (n_questions,)
    - n_questions: number of questions
    """
    n_questions = len(jsonl_results)

    # Extract predictions from resps, preserving doc_id for ordering
    results_with_doc_id = []
    for result in jsonl_results:
        # Extract logits from resps
        # resps[i][0][0] gives the logit for choice i
        resps = result.get("resps", [])
        if not resps:
            raise ValueError(f"Result missing 'resps' field")

        # Extract logits for each choice
        choice_logits = []
        for resp in resps:
            if isinstance(resp, list) and len(resp) > 0:
                if isinstance(resp[0], list) and len(resp[0]) > 0:
                    logit_str = resp[0][0]
                    logit = float(logit_str)
                    choice_logits.append(logit)
                else:
                    raise ValueError(f"Unexpected resps structure: {resp}")
            else:
                raise ValueError(f"Unexpected resps structure: {resp}")

        # Extract correctness from metric field (1.0 if correct, 0.0 if incorrect)
        acc = result.get(metric, 0.0)
        doc_id = result.get("doc_id", None)

        results_with_doc_id.append(
            {
                "predictions": choice_logits,
                "correctness": float(acc),
                "doc_id": doc_id,
            }
        )

    # If anchor_points is provided, filter to only anchor points and reorder to match anchor_points order
    if anchor_points is not None:
        # Create mapping from doc_id to result
        doc_id_to_result = {
            r["doc_id"]: r
            for r in results_with_doc_id
            if r["doc_id"] is not None
        }

        # Check order of doc_ids in original results
        original_doc_ids = [
            r["doc_id"] for r in results_with_doc_id if r["doc_id"] is not None
        ]

        # Verify we have all anchor points
        missing_doc_ids = set(anchor_points) - set(doc_id_to_result.keys())
        if missing_doc_ids:
            raise ValueError(
                f"Missing doc_ids in JSONL results: {sorted(missing_doc_ids)[:10]}... "
                f"Available doc_ids: {sorted(doc_id_to_result.keys())[:10]}..."
            )

        # Reorder results to match anchor_points order exactly
        # This ensures predictions are in the same order as when we do predictions[:, anchor_points, :]
        reordered_results = []
        for anchor_idx in anchor_points:
            reordered_results.append(doc_id_to_result[anchor_idx])

        # Verify reordering if needed (only print if order changed)
        reordered_doc_ids = [r["doc_id"] for r in reordered_results]
        if original_doc_ids != reordered_doc_ids:
            print(
                f"  Reordered results from {original_doc_ids[:5]}... to {reordered_doc_ids[:5]}..."
            )

        results_with_doc_id = reordered_results

    # Extract predictions and correctness in final order
    predictions_list = [r["predictions"] for r in results_with_doc_id]
    correctness_list = [r["correctness"] for r in results_with_doc_id]

    # Stack predictions: (n_questions, n_choices)
    predictions_2d = np.array(predictions_list)

    # Extract correctness: (n_questions,)
    correctness_1d = np.array(correctness_list, dtype=float)

    return predictions_2d, correctness_1d, n_questions


def convert_model_paths_to_target_outputs(
    model_id_to_path_mapping,
    scenario,
    metric,
    output_path=None,
    pad_to_size=None,
    anchor_points=None,
    force_recompute=False,
):
    """
    Convert a mapping of model_id to local file paths to target_outputs format and save to pickle.

    Parameters:
    - model_id_to_path_mapping: Dictionary mapping model_id -> local file path (directory or JSONL file)
    - scenario: Scenario name (e.g., "harness_hellaswag_10")
    - metric: Metric name to use for correctness (e.g., "acc", "acc_norm")
    - output_path: Path to save target_outputs.pkl (if None, uses default)
    - pad_to_size: Optional size to pad predictions to on the last axis. If None, no padding is applied.
    - anchor_points: Optional list of anchor point indices for reordering results
    - force_recompute: If True, overwrite existing models even if they already exist in target_outputs

    Returns:
    - target_outputs: Dictionary with the expected structure
    """
    assert output_path is not None, "output_path must be provided"
    if output_path is None:
        # Extract scenario base name for default path
        if scenario.startswith("harness_"):
            scenario_base = scenario.replace("harness_", "")
            if "_" in scenario_base:
                scenario_base = scenario_base.split("_")[0]
        else:
            scenario_base = scenario
        output_path = f"/home/oh/arubinstein17/github/disco-public/data/model_outputs/{scenario_base}/target_outputs.pkl"

    # Check if file exists and load it to append
    existing_outputs = None
    if os.path.exists(output_path):
        print(f"Loading existing target_outputs from {output_path}")
        existing_outputs = load_pickle(output_path)
        # Extract existing model IDs
        existing_model_ids = set(existing_outputs["Models"].keys())
        print(
            f"Found {len(existing_model_ids)} existing models: {existing_model_ids}"
        )
    else:
        print(f"Creating new target_outputs file at {output_path}")

    # Collect data for all models
    all_predictions = []
    all_correctness = []
    models_map = {}
    models_to_overwrite = (
        {}
    )  # Track models being overwritten: model_id -> existing_index
    processed_model_ids = (
        []
    )  # Track order of processed models to match with predictions
    n_questions = None

    for model_idx, (model_id, path) in enumerate(
        model_id_to_path_mapping.items()
    ):
        print(
            f"\nProcessing model {model_idx + 1}/{len(model_id_to_path_mapping)}: {model_id}"
        )

        # Skip if model already exists (unless force_recompute is True)
        if (
            existing_outputs is not None
            and model_id in existing_outputs["Models"]
            and not force_recompute
        ):
            print(f"  Model {model_id} already exists, skipping...")
            continue
        elif (
            existing_outputs is not None
            and model_id in existing_outputs["Models"]
            and force_recompute
        ):
            print(
                f"  Model {model_id} already exists, but --force_recompute is set, will overwrite..."
            )
            # We'll replace it at the same index - store the index for later
            existing_model_idx = existing_outputs["Models"][model_id]
            models_to_overwrite[model_id] = existing_model_idx

        # Find the JSONL file
        jsonl_path = find_jsonl_file_in_directory(path)
        if jsonl_path is None:
            print(f"  Error: Could not find JSONL file in {path}")
            continue

        print(f"  Found JSONL file: {jsonl_path}")

        # Load JSONL results
        try:
            jsonl_results = load_jsonl(jsonl_path)
        except Exception as e:
            print(f"  Error loading JSONL file {jsonl_path}: {e}")
            continue

        # Convert to arrays
        try:
            (
                predictions_2d,
                correctness_1d,
                model_n_questions,
            ) = convert_jsonl_results_to_arrays(
                jsonl_results, metric, anchor_points=anchor_points
            )
        except Exception as e:
            print(f"  Error converting JSONL results: {e}")
            continue

        # Pad predictions if pad_to_size is specified
        if pad_to_size is not None:
            n_choices = predictions_2d.shape[1]
            if n_choices < pad_to_size:
                # Pad each row using pad_predictions function
                padded_predictions_list = []
                for row in predictions_2d:
                    row_list = row.tolist()
                    padded_row = pad_predictions(
                        row_list, max_num_answers=pad_to_size
                    )
                    padded_predictions_list.append(padded_row)
                predictions_2d = np.array(padded_predictions_list)
                print(
                    f"  Padded predictions from {n_choices} to {pad_to_size} choices"
                )

        # Check consistency of number of questions
        if n_questions is None:
            n_questions = model_n_questions
        elif n_questions != model_n_questions:
            raise ValueError(
                f"Model {model_id} has {model_n_questions} questions, "
                f"but expected {n_questions}"
            )

        # Add to lists
        all_predictions.append(predictions_2d)
        all_correctness.append(correctness_1d)
        processed_model_ids.append(model_id)

        # Determine model index (append to existing, use existing index if force_recompute, or use new index)
        if (
            existing_outputs is not None
            and model_id in existing_outputs["Models"]
            and force_recompute
        ):
            # Reuse the existing index when overwriting
            model_index = existing_outputs["Models"][model_id]
        elif existing_outputs is not None:
            # Find the next available index
            max_existing_idx = (
                max(existing_outputs["Models"].values())
                if existing_outputs["Models"]
                else -1
            )
            model_index = max_existing_idx + 1
        else:
            model_index = model_idx

        models_map[model_id] = model_index
        print(f"  Added model {model_id} at index {model_index}")

    # Stack all models: (n_models, n_questions, n_choices) and (n_models, n_questions, 1)
    if existing_outputs is not None and len(all_predictions) == 0:
        print("No new models to add, keeping existing target_outputs")
        return existing_outputs

    if len(all_predictions) > 0:
        new_predictions = np.stack(
            all_predictions
        )  # (n_new_models, n_questions, n_choices)
        new_correctness = np.stack(all_correctness)[
            :, :, np.newaxis
        ]  # (n_new_models, n_questions, 1)

        if existing_outputs is not None:
            # Handle overwriting existing models if force_recompute is True
            if models_to_overwrite:
                # Create copies of existing arrays
                predictions = existing_outputs["predictions"].copy()
                correctness = existing_outputs["correctness"].copy()

                # Overwrite existing models at their indices
                # processed_model_ids matches the order of new_predictions and new_correctness
                for new_idx, model_id in enumerate(processed_model_ids):
                    if model_id in models_to_overwrite:
                        old_idx = models_to_overwrite[model_id]
                        predictions[old_idx] = new_predictions[new_idx]
                        correctness[old_idx] = new_correctness[new_idx]

                # Append any new models (not being overwritten)
                new_model_indices = [
                    i
                    for i, model_id in enumerate(processed_model_ids)
                    if model_id not in models_to_overwrite
                ]
                if new_model_indices:
                    predictions = np.concatenate(
                        [predictions, new_predictions[new_model_indices]],
                        axis=0,
                    )
                    correctness = np.concatenate(
                        [correctness, new_correctness[new_model_indices]],
                        axis=0,
                    )
            else:
                # No overwrites, just concatenate
                predictions = np.concatenate(
                    [existing_outputs["predictions"], new_predictions], axis=0
                )
                correctness = np.concatenate(
                    [existing_outputs["correctness"], new_correctness], axis=0
                )

            # Merge models map
            models_map = {**existing_outputs["Models"], **models_map}
            # Use existing datapoints and scenarios (should be the same)
            datapoints_map = existing_outputs["Datapoints"]
            scenarios_map = existing_outputs["Scenarios"]
        else:
            # Create new
            predictions = new_predictions
            correctness = new_correctness
            # Create Datapoints mapping: idx -> idx
            datapoints_map = {i: i for i in range(n_questions)}

            # Extract scenario base name
            if scenario.startswith("harness_"):
                scenario_base = scenario.replace("harness_", "")
                if "_" in scenario_base:
                    scenario_base = scenario_base.split("_")[0]
            else:
                scenario_base = scenario

            # Create Scenarios mapping: scenario_name -> list of all datapoint indices
            scenarios_map = {scenario_base: list(range(n_questions))}
    else:
        # Only existing models, no new ones
        predictions = existing_outputs["predictions"]
        correctness = existing_outputs["correctness"]
        models_map = existing_outputs["Models"]
        datapoints_map = existing_outputs["Datapoints"]
        scenarios_map = existing_outputs["Scenarios"]

    target_outputs = {
        "predictions": predictions,
        "correctness": correctness,
        "Models": models_map,
        "Datapoints": datapoints_map,
        "Scenarios": scenarios_map,
    }

    # Save to pickle file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dump_pickle(target_outputs, output_path)

    print(f"\nSaved target_outputs to {output_path}")
    print(f"Shape of predictions: {target_outputs['predictions'].shape}")
    print(f"Shape of correctness: {target_outputs['correctness'].shape}")
    print(f"Number of models: {len(target_outputs['Models'])}")
    print(f"Number of datapoints: {len(target_outputs['Datapoints'])}")
    print(f"Scenarios: {list(target_outputs['Scenarios'].keys())}")
    print(f"Model IDs: {list(target_outputs['Models'].keys())}")

    return target_outputs
