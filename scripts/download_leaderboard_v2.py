# The code is adapted from the tinyBenchmarks repo: https://github.com/felipemaiapolo/efficbench
import re
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import os
import argparse
import requests
import json
import sys


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
# from utils import dict_to_h5
from utils import dump_pickle

sys.path.pop(0)

CACHE_DIR = "./cache_dir"


# MMLU-Pro: open-llm-leaderboard v2 detail datasets use config
# "{creator}__{model}__leaderboard_mmlu_pro" (metric: acc)
MMLU_PRO_SCENARIO_SUFFIX = "__leaderboard_mmlu_pro"

DEFAULT_CSV_PATH = "benchmark_csvs/open-llm-leaderboard-v2.csv"
LB_SAVEPATH = "data/leaderboard_fields_raw_22042025.pickle"

EXTRA_KEYS = [
    # 'full_prompt',
    "example",
    "predictions",
]

# Pattern: href="https://huggingface.co/creator/model_name" (exclude datasets/ links)
_HF_MODEL_HREF = re.compile(
    r'href=["]*https?://huggingface\.co/(?!datasets/)([^"]+?)["]*\s',
    re.IGNORECASE,
)


def extract_model_id(value):
    """Extract provider/model_name from a cell that may be plain text or HTML.

    - If value is plain 'creator/model', return it.
    - If value contains href="https://huggingface.co/creator/model"...", return 'creator/model'.
    - Otherwise return None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    # Try href first (HTML export format)
    m = _HF_MODEL_HREF.search(s)
    if m:
        path = m.group(1)
        if "/" in path:
            return path
    # Plain "creator/model"
    if "/" in s:
        return s
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download MMLU-Pro model outputs from open-llm-leaderboard (v2)."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to leaderboard CSV with Model column (default: %(default)s)",
    )
    parser.add_argument(
        "--subsample_step",
        type=int,
        default=None,
        metavar="K",
        help="If set, use every K-th model only (e.g. 5 => 1st, 6th, 11th, ...)",
    )
    parser.add_argument("--lb_savepath", type=str, default=LB_SAVEPATH)
    parser.add_argument("--save_only_once", action="store_true")
    args = parser.parse_args()

    lb_savepath = args.lb_savepath

    df = pd.read_csv(args.csv_path)

    # Columns that may contain model IDs (plain "creator/model" or HTML with href=...)
    candidate_columns = ["Model", "fullname", "Base Model"]
    model_id_columns = [c for c in candidate_columns if c in df.columns]
    if not model_id_columns:
        raise ValueError(
            f"CSV must have at least one of {candidate_columns!r}. "
            f"Found columns: {list(df.columns)}"
        )

    # Extract provider/model_name from each column; concatenate and dedupe
    all_ids = []
    for col in model_id_columns:
        extracted = df[col].dropna().apply(extract_model_id)
        all_ids.extend(extracted[extracted.notna()].tolist())
    model_names = list(pd.unique(all_ids))

    if args.subsample_step is not None:
        model_names = [
            model_names[i]
            for i in range(0, len(model_names), args.subsample_step)
        ]
    print(f"Loaded {len(model_names)} models from {args.csv_path}")

    models = []
    for m in model_names:
        creator, model = tuple(m.split("/"))
        models.append(
            "open-llm-leaderboard/details_{:}__{:}".format(creator, model)
        )

    data = {}
    # MMLU-Pro uses metric "acc" and config name "{details_id}__leaderboard_mmlu_pro"
    metric = "acc"

    os.makedirs(CACHE_DIR, exist_ok=True)
    skipped = 0
    log = []
    for model in tqdm(models):
        # Config name for this model's MMLU-Pro run (open-llm-leaderboard v2)
        details_id = model.replace("open-llm-leaderboard/details_", "")
        s = details_id + MMLU_PRO_SCENARIO_SUFFIX

        try:
            if model not in data:
                data[model] = {}
            if s not in data[model]:
                data[model][s] = {}
            aux = load_dataset(model, s, cache_dir=CACHE_DIR)
            data[model][s]["dates"] = list(aux.keys())
            for extra_key in EXTRA_KEYS:
                data[model][s][extra_key] = aux["latest"][extra_key]
            try:
                data[model][s]["correctness"] = [
                    a[metric] for a in aux["latest"]["metrics"]
                ]
                print("\nOK {:} {:}\n".format(model, s))
                log.append("\nOK {:} {:}\n".format(model, s))
            except Exception as e:
                print(f"Error accessing dataset attribute: {e}")
                try:
                    data[model][s]["correctness"] = aux["latest"][metric]
                    print("\nOK {:} {:}\n".format(model, s))
                    log.append("\nOK {:} {:}\n".format(model, s))
                except Exception as e2:
                    print(f"Error loading dataset for {model} and {s}: {e2}")
                    skipped += 1
                    skip_model(data, model, s, 0, log)

        except Exception as e:
            print(f"Error loading dataset for {model} and {s}: {e}")
            skipped += 1
            skip_model(data, model, s, 0, log)

        if not args.save_only_once:
            dump_pickle(data, lb_savepath)

        print("\nModels skipped so far: {:}\n".format(skipped))

    # dict_to_h5(data, lb_savepath)
    if args.save_only_once:
        dump_pickle(data, lb_savepath)


def skip_model(data, model, scenario_name, skipped_aux, log):
    data[model][scenario_name] = None
    print("\nSKIP {:} {:}\n".format(model, scenario_name))
    skipped_aux += 1
    log.append("\nSKIP {:} {:}\n".format(model, scenario_name))
    return skipped_aux


def download_helm_lite(lb_savepath):
    def get_json_from_url(url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            json_data = response.json()
            return json_data
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    version_to_run = "v1.0.0"
    overwrite = False
    assert (
        lb_savepath is None or lb_savepath == "None"
    ), "lb_savepath must be None for helm_lite"
    df = pd.read_csv("./generating_data/download_helm/helm_lite.csv")
    tasks_list = list(df.Run)
    template_url = f"https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/{version_to_run}"
    # save_dir = f"/llmthonskdir/felipe/helm/lite/{version_to_run}"
    save_dir = f"./data/downloaded_helm_lite/{version_to_run}"
    for tasks in [tasks_list]:
        for task in tqdm(tasks):
            cur_save_dir = f"{save_dir}/{task}"
            os.makedirs(cur_save_dir, exist_ok=True)

            for file_type in [
                "run_spec",
                "stats",
                "per_instance_stats",
                "instances",
                "scenario_state",
                "display_predictions",
                "display_requests",
                "scenario",
            ]:
                save_path = f"{cur_save_dir}/{file_type}.json"
                if os.path.exists(save_path) and not overwrite:
                    continue

                # https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/babi_qa:task=15,model=AlephAlpha_luminous-base/scenario_state.json

                cur_url = f"{template_url}/{task}/{file_type}.json"
                json.dump(
                    get_json_from_url(cur_url), open(save_path, "w"), indent=2
                )


if __name__ == "__main__":
    main()
