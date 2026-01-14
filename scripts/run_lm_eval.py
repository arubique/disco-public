"""
Thin wrapper to run the lm-evaluation-harness CLI from the git submodule in external/.

Example usage (from the repository root):

    python scripts/run_lm_eval.py \\
        --batch_size=8 \\
        --device=cuda:0 \\
        --gen_kwargs=max_gen_toks=128,output_scores=True,return_dict_in_generate=True \\
        --model=hf \\
        --model_args=pretrained=abacusai/MetaMath-bagel-34b-v0.2-c1500,trust_remote_code=True \\
        --num_fewshot=10 \\
        --output_path=/weka/oh/arubinstein17/github/lm-evaluation-harness/output/hellaswag_050126/llama13b_r71 \\
        --tasks=hellaswag_prompts \\
        --log_samples \\
        --anchor_points_path=/path/to/anchor_points.pkl
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from utils import load_pickle
from scripts.lm_eval_postprocess import (
    convert_model_paths_to_target_outputs,
    find_jsonl_file_in_directory,
)

sys.path.pop(0)


def _ensure_external_on_path() -> None:
    """
    Prepend the `external/lm-evaluation-harness` directory (git submodule)
    to sys.path so that `import lm_eval` resolves to that copy.
    """

    repo_root = Path(__file__).resolve().parents[1]
    submodule_root = repo_root / "external" / "lm-evaluation-harness"

    submodule_str = str(submodule_root)
    if submodule_str not in sys.path:
        sys.path.insert(0, submodule_str)


def _extract_model_id_from_args(model_args):
    """
    Extract model_id (pretrained value) from model_args string or dict.

    Parameters:
    - model_args: Either a string like "pretrained=model/name,trust_remote_code=True"
                  or a dict with 'pretrained' key

    Returns:
    - model_id string or None if not found
    """
    _ensure_external_on_path()
    from lm_eval.utils import simple_parse_args_string

    if isinstance(model_args, dict):
        return model_args.get("pretrained", None)
    elif isinstance(model_args, str):
        parsed = simple_parse_args_string(model_args)
        return parsed.get("pretrained", None)
    return None


def _infer_scenario_from_args(tasks, num_fewshot):
    """
    Infer scenario name from tasks and num_fewshot.
    Example: tasks="hellaswag_prompts", num_fewshot=10 -> "harness_hellaswag_10"
    """
    if not tasks:
        return None

    # Get first task name
    task_name = tasks.split(",")[0].strip()

    # Remove _prompts suffix if present
    if task_name.endswith("_prompts"):
        task_base = task_name[:-8]  # Remove '_prompts'
    else:
        task_base = task_name

    # Create scenario name
    if num_fewshot is not None:
        scenario = f"harness_{task_base}_{num_fewshot}"
    else:
        scenario = f"harness_{task_base}"

    return scenario


def main() -> None:
    _ensure_external_on_path()

    # Import after sys.path modification so we pick up the submodule package
    from lm_eval.__main__ import setup_parser, parse_eval_args, cli_evaluate

    # Setup parser and add our custom arguments
    parser = setup_parser()
    parser.add_argument(
        "--anchor_points_path",
        type=str,
        default=None,
        help="Path to anchor points pickle file. If provided, will post-process results after evaluation.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="acc_norm",
        help="Metric name to use for correctness (e.g., 'acc', 'acc_norm'). Default: 'acc_norm'",
    )
    parser.add_argument(
        "--pad_to_size",
        type=int,
        default=None,
        help="Optional size to pad predictions to on the last axis. If None, no padding is applied.",
    )
    parser.add_argument(
        "--target_outputs_path",
        type=str,
        default=None,
        help="Path to save target_outputs.pkl. If None, uses default based on scenario.",
    )

    # Parse arguments (this will include our custom args)
    args = parse_eval_args(parser)

    # Store our custom args before passing to cli_evaluate
    anchor_points_path = args.anchor_points_path
    metric = args.metric
    pad_to_size = args.pad_to_size
    target_outputs_path = args.target_outputs_path

    # Create a copy of args without our custom attributes for cli_evaluate
    # We'll create a new namespace with only the standard lm_eval arguments
    import copy

    args_for_eval = copy.deepcopy(args)
    # Remove our custom attributes
    if hasattr(args_for_eval, "anchor_points_path"):
        delattr(args_for_eval, "anchor_points_path")
    if hasattr(args_for_eval, "metric"):
        delattr(args_for_eval, "metric")
    if hasattr(args_for_eval, "pad_to_size"):
        delattr(args_for_eval, "pad_to_size")
    if hasattr(args_for_eval, "target_outputs_path"):
        delattr(args_for_eval, "target_outputs_path")

    # Run the evaluation
    cli_evaluate(args_for_eval)

    # Post-process if anchor_points_path is provided
    if anchor_points_path is not None:
        print("\n" + "=" * 80)
        print("Post-processing evaluation results...")
        print("=" * 80)

        # Extract model_id from model_args (use args_for_eval which has the same model_args)
        model_id = _extract_model_id_from_args(args_for_eval.model_args)
        if model_id is None:
            raise ValueError(
                f"Could not extract model_id from model_args: {args_for_eval.model_args}. "
                "Expected 'pretrained=model/name' in model_args."
            )
        print(f"Extracted model_id: {model_id}")

        # Find JSONL file in output_path
        if not args_for_eval.output_path:
            raise ValueError(
                "--output_path must be specified when using --anchor_points_path"
            )

        jsonl_path = find_jsonl_file_in_directory(args_for_eval.output_path)
        if jsonl_path is None:
            raise ValueError(
                f"Could not find JSONL file in output_path: {args_for_eval.output_path}"
            )
        print(f"Found JSONL file: {jsonl_path}")

        # Infer scenario from tasks and num_fewshot
        scenario = _infer_scenario_from_args(
            args_for_eval.tasks, args_for_eval.num_fewshot
        )
        if scenario is None:
            raise ValueError(
                f"Could not infer scenario from tasks: {args_for_eval.tasks}"
            )
        print(f"Inferred scenario: {scenario}")

        # Create model_id_to_path_mapping
        model_id_to_path_mapping = {model_id: jsonl_path}

        # Convert to target_outputs
        target_outputs = convert_model_paths_to_target_outputs(
            model_id_to_path_mapping=model_id_to_path_mapping,
            scenario=scenario,
            metric=metric,
            output_path=target_outputs_path,
            pad_to_size=pad_to_size,
        )

        # Load anchor points
        anchor_points = load_pickle(anchor_points_path)
        print(f"\nLoaded anchor points: {len(anchor_points)} indices")

        # Compute predictions tensor: predictions[:, anchor_points, :]
        predictions = target_outputs["predictions"][:, anchor_points, :]
        print(f"Computed predictions tensor shape: {predictions.shape}")
        print(
            f"  (n_models={predictions.shape[0]}, n_anchor_points={predictions.shape[1]}, n_choices={predictions.shape[2]})"
        )

        print("\n" + "=" * 80)
        print("Post-processing complete!")
        print("=" * 80)


if __name__ == "__main__":
    main()
