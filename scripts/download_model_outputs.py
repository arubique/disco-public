#!/usr/bin/env python3
"""
Download `data/model_outputs.pickle` from the Hugging Face Hub (default) or Google Drive.

Hub layout is defined in `scripts/model_outputs_hf.py` (one dataset repo; splits per task).

Examples:
  export DISCO_MODEL_OUTPUTS_HF_BASE=my-org/disco-public-model-outputs
  python scripts/download_model_outputs.py

  python scripts/download_model_outputs.py --source gdrive
"""

import argparse
import os
import pickle
import sys
from typing import Optional

import gdown

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_PATH)

from model_outputs_hf import (
    download_model_outputs_from_hub,
    get_hub_base,
)  # noqa: E402

GDRIVE_FILE_ID = "1OFvbunu0MK3kZiM1U6NGi46O5iWE-Fm1"
DEFAULT_OUTPUT_PATH = os.path.join(ROOT_PATH, "data", "model_outputs.pickle")


def download_from_gdrive(output_path: str) -> None:
    print(
        f"Downloading pickled model outputs from Google Drive to {output_path}..."
    )
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
        output_path,
        quiet=False,
    )
    print("Download complete!")


def download_from_hf(
    output_path: str, hub_base: str, token: Optional[str]
) -> None:
    print(
        f"Downloading model outputs from Hugging Face Hub (base={hub_base!r})..."
    )
    data = download_model_outputs_from_hub(hub_base, token=token)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("hf", "gdrive"),
        default="hf",
        help="Where to download from (default: hf)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output pickle path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--hub-base",
        type=str,
        default=None,
        help="Full Hub dataset repo id org/name. Overrides DISCO_MODEL_OUTPUTS_HF_BASE.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (default: HF_TOKEN env or cached login)",
    )
    args = parser.parse_args()

    if args.source == "gdrive":
        download_from_gdrive(args.output_path)
        return

    try:
        hub_base = get_hub_base(args.hub_base)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    token = args.token or os.environ.get("HF_TOKEN")
    download_from_hf(args.output_path, hub_base, token)


if __name__ == "__main__":
    main()
