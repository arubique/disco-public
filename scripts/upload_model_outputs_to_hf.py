#!/usr/bin/env python3
"""
Upload `data/model_outputs.pickle` to the Hugging Face Hub as a *single* dataset repo.

The Hub dataset uses one configuration (subset) per logical block (manifest, models,
hellaswag, mmlu_*, …), each with a `train` split, so the viewer can show different columns
per subset. If an earlier upload mixed layouts on the same repo, delete the repo (or use a
new name) before pushing again.

Requires: `huggingface-cli login` or HF_TOKEN in the environment.

Example:
  huggingface-cli login
  python scripts/upload_model_outputs_to_hf.py

  # Fork or custom repo:
  python scripts/upload_model_outputs_to_hf.py --hub-base my-org/disco-model-outputs

  # Quick test: manifest + models + hellaswag + MMLU abstract algebra (if present in pickle)
  python scripts/upload_model_outputs_to_hf.py --debug
"""

import argparse
import os
import pickle
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_PATH)

from model_outputs_hf import (
    get_hub_base,
    push_model_outputs_to_hub,
)  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pickle-path",
        type=str,
        default=os.path.join(ROOT_PATH, "data", "model_outputs.pickle"),
        help="Path to model_outputs.pickle",
    )
    parser.add_argument(
        "--hub-base",
        type=str,
        default=None,
        help=(
            "Full Hub dataset repo id org/name (default: arubique/disco-model-outputs). "
            "Overrides DISCO_MODEL_OUTPUTS_HF_BASE if set."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private dataset repo",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (default: HF_TOKEN env or cached login)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Upload manifest + models + only harness_hellaswag_10 and "
            "harness_hendrycksTest_abstract_algebra_5 (if present); omit other tasks"
        ),
    )
    args = parser.parse_args()

    repo_id = get_hub_base(args.hub_base)

    with open(args.pickle_path, "rb") as f:
        data = pickle.load(f)

    token = args.token or os.environ.get("HF_TOKEN")
    if args.debug:
        print(
            f"Pushing debug dataset (manifest + models + hellaswag + mmlu abstract algebra "
            f"if present) to {repo_id} ..."
        )
    else:
        print(f"Pushing single dataset to {repo_id} ...")
    push_model_outputs_to_hub(
        repo_id, data, token=token, private=args.private, debug=args.debug
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
