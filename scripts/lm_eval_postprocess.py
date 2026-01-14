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
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if (
                file.endswith(".jsonl")
                and "samples" in file
                and "prompts" in file
            ):
                return os.path.join(root, file)

    # If not found, try to find any JSONL file
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".jsonl"):
                return os.path.join(root, file)

    return None


def convert_jsonl_results_to_arrays(jsonl_results, metric):
    """
    Convert JSONL results to predictions and correctness arrays.

    Parameters:
    - jsonl_results: List of dictionaries from load_jsonl
    - metric: Metric name to use for correctness (e.g., "acc", "acc_norm")

    Returns:
    - predictions_2d: numpy array of shape (n_questions, n_choices)
    - correctness_1d: numpy array of shape (n_questions,)
    - n_questions: number of questions
    """
    n_questions = len(jsonl_results)

    # Extract predictions from resps
    predictions_list = []
    correctness_list = []

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

        predictions_list.append(choice_logits)

        # Extract correctness from metric field (1.0 if correct, 0.0 if incorrect)
        acc = result.get(metric, 0.0)
        correctness_list.append(float(acc))

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
):
    """
    Convert a mapping of model_id to local file paths to target_outputs format and save to pickle.

    Parameters:
    - model_id_to_path_mapping: Dictionary mapping model_id -> local file path (directory or JSONL file)
    - scenario: Scenario name (e.g., "harness_hellaswag_10")
    - metric: Metric name to use for correctness (e.g., "acc", "acc_norm")
    - output_path: Path to save target_outputs.pkl (if None, uses default)
    - pad_to_size: Optional size to pad predictions to on the last axis. If None, no padding is applied.

    Returns:
    - target_outputs: Dictionary with the expected structure
    """
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
    n_questions = None

    for model_idx, (model_id, path) in enumerate(
        model_id_to_path_mapping.items()
    ):
        print(
            f"\nProcessing model {model_idx + 1}/{len(model_id_to_path_mapping)}: {model_id}"
        )

        # Skip if model already exists
        if (
            existing_outputs is not None
            and model_id in existing_outputs["Models"]
        ):
            print(f"  Model {model_id} already exists, skipping...")
            continue

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
            ) = convert_jsonl_results_to_arrays(jsonl_results, metric)
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

        # Determine model index (append to existing or use new index)
        if existing_outputs is not None:
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
            # Concatenate with existing
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
