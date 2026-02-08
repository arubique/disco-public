"""
Test generated source_outputs (from extract_model_outputs_from_raw_data_v2.py):

1. Subsample 5 random models.
2. For each: performance (sum of correctness / n_questions) should match MMLU-PRO Raw in the leaderboard CSV.
3. Optional (--check_correctness_consistency): correctness[model,q] == (gold[q] == pred_label[q]).
   Off by default because stored correctness is acc_norm (from leaderboard logprobs), which may not
   match (gold == parsed_resps) from our resps→pred conversion.

Usage:
  python tests/test_extract_model_outputs_v2.py --outputs_path data/model_outputs/mmlu/source_outputs.pkl --raw_path data/leaderboard_mmlu_pro_raw.pickle --csv_path benchmark_csvs/open-llm-leaderboard-v2.csv
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)
from utils import load_pickle
from scripts.download_leaderboard_v2 import sanitize_for_hf_repo

sys.path.pop(0)

MMLU_PRO_SCENARIO_SUFFIX = "__leaderboard_mmlu_pro"
CHOICE_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
# Tolerance for mean correctness vs CSV score (floating point + possible rounding in CSV)
PERF_TOLERANCE = 1e-4


def _source_model_to_csv_id(model_name):
    """Map source_outputs model key to CSV Model column value (creator/model)."""
    # e.g. open-llm-leaderboard/0-hero__Matter-0.2-7B-DPO-details -> 0-hero/Matter-0.2-7B-DPO
    if not model_name.startswith(
        "open-llm-leaderboard/"
    ) or not model_name.endswith("-details"):
        return None
    mid = model_name[len("open-llm-leaderboard/") : -len("-details")]
    return mid.replace("__", "/", 1)


def _get_scenario_key(entry):
    if not isinstance(entry, dict):
        return None
    for k in entry:
        if k.endswith(MMLU_PRO_SCENARIO_SUFFIX):
            return k
    return None


def _target_to_indices(target, n_questions):
    """Convert target (list of gold labels) to int indices 0..K-1."""
    out = np.full(n_questions, -1, dtype=np.int64)
    if target is None:
        return out
    raw = (
        list(target)
        if hasattr(target, "__iter__") and not isinstance(target, (str, bytes))
        else [target]
    )
    n = min(n_questions, len(raw))
    for i in range(n):
        t = raw[i]
        if isinstance(t, (int, float)):
            out[i] = int(t)
        elif isinstance(t, str):
            t = t.strip().upper()
            if t:
                out[i] = CHOICE_TO_INDEX.get(t[0], -1)
    return out


def load_csv_scores(csv_path):
    """Return dict: csv_model_id (sanitized creator/model) -> MMLU-PRO Raw (float, 0-1)."""
    df = pd.read_csv(csv_path)
    if "Model" not in df.columns or "MMLU-PRO Raw" not in df.columns:
        raise ValueError(
            f"CSV must have columns 'Model' and 'MMLU-PRO Raw'. Found: {list(df.columns)}"
        )
    df = df.dropna(subset=["Model", "MMLU-PRO Raw"])
    # Normalize model names with sanitize_for_hf_repo so keys match _source_model_to_csv_id(output)
    out = {}
    for _, row in df.iterrows():
        model_str = str(row["Model"]).strip()
        model_str = (
            sanitize_for_hf_repo(model_str)
            .split("huggingface.co_")[1]
            .split("__style")[0]
            .split("__sty")[0]
            .replace("_", "/")
        )
        if "/" in model_str:
            creator, model = model_str.split("/", 1)
            key = (
                f"{sanitize_for_hf_repo(creator)}/{sanitize_for_hf_repo(model)}"
            )
        else:
            key = sanitize_for_hf_repo(model_str)
        if key not in out:
            out[key] = float(row["MMLU-PRO Raw"])
    return out


def load_gold_from_raw(raw_path, n_questions):
    """Load gold labels (target) from v2 raw pickle; return (n_questions,) int array."""
    raw = load_pickle(raw_path)
    scenario_key = None
    target = None
    for _name, scenarios_dict in raw.items():
        if not isinstance(scenarios_dict, dict):
            continue
        sk = _get_scenario_key(scenarios_dict)
        if sk is None:
            continue
        rec = scenarios_dict.get(sk)
        if rec is None:
            continue
        t = rec.get("target")
        if t is not None:
            scenario_key = sk
            target = t
            break
    if target is None:
        raise ValueError("Raw pickle has no 'target' in any model record.")
    return _target_to_indices(target, n_questions)


def run_tests(
    outputs_path,
    raw_path,
    csv_path,
    seed=42,
    n_sample=5,
    skip_correctness_consistency=True,
):
    """Run tests. skip_correctness_consistency=True by default because stored
    correctness comes from leaderboard acc_norm (from logprobs), which may not
    match (gold == parsed_resps) from our resps→pred conversion."""
    source = load_pickle(outputs_path)
    csv_scores = load_csv_scores(csv_path)

    predictions = source["predictions"]
    correctness = source["correctness"]
    models_map = source["Models"]

    n_models, n_questions, pad_size = predictions.shape
    assert correctness.shape == (n_models, n_questions, 1)

    gold = (
        load_gold_from_raw(raw_path, n_questions)
        if not skip_correctness_consistency
        else None
    )

    # Subsample models that exist in CSV
    model_names = list(models_map.keys())
    available = [
        m for m in model_names if _source_model_to_csv_id(m) in csv_scores
    ]
    if len(available) < n_sample:
        raise ValueError(
            f"Only {len(available)} models from outputs appear in CSV; need at least {n_sample}. "
            "Check that outputs and CSV refer to the same leaderboard."
        )
    random.seed(seed)
    chosen = random.sample(available, n_sample)

    errors = []
    for model_name in chosen:
        idx = models_map[model_name]
        csv_id = _source_model_to_csv_id(model_name)
        expected_perf = csv_scores[csv_id]

        # 1) Performance: mean(correctness) vs CSV MMLU-PRO Raw
        perf = float(np.sum(correctness[idx, :, 0]) / n_questions)
        if not np.isclose(perf, expected_perf, atol=PERF_TOLERANCE, rtol=0):
            errors.append(
                f"Model {model_name}: computed performance {perf:.6f} != CSV MMLU-PRO Raw {expected_perf:.6f}"
            )

        # 2) Optional: correctness == (gold == predicted label) where we have valid parsed pred.
        #    Skip when using acc_norm (leaderboard may compute from logprobs, not resps).
        if not skip_correctness_consistency:
            pred_val = np.asarray(predictions[idx, :, 0], dtype=np.float64)
            valid_pred = np.isfinite(pred_val) & (pred_val > -np.inf)
            valid_gold = gold >= 0
            expected_correct = np.zeros(n_questions, dtype=np.float64)
            pred_int = np.full(n_questions, -1, dtype=np.int64)
            pred_int[valid_pred] = np.round(pred_val[valid_pred]).astype(
                np.int64
            )
            match = valid_gold & valid_pred & (pred_int == gold)
            expected_correct[match] = 1.0
            if not np.allclose(correctness[idx, :, 0], expected_correct):
                bad = np.where(
                    np.abs(correctness[idx, :, 0] - expected_correct) > 1e-6
                )[0]
                errors.append(
                    f"Model {model_name}: correctness disagrees with (gold == pred_label) at {len(bad)} indices (e.g. {bad[:5].tolist()})"
                )

    if errors:
        for e in errors:
            print("FAIL:", e)
        raise AssertionError(f"{len(errors)} check(s) failed.")
    print(
        f"PASS: All checks passed for {n_sample} sampled models (seed={seed})."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test generated source_outputs against CSV and gold."
    )
    parser.add_argument(
        "--outputs_path",
        type=str,
        default=os.path.join(
            ROOT_PATH, "data", "model_outputs", "mmlu", "source_outputs.pkl"
        ),
        help="Path to generated source_outputs pickle",
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        default=os.path.join(
            ROOT_PATH, "data", "leaderboard_mmlu_pro_raw.pickle"
        ),
        help="Path to v2 raw pickle (for gold/target)",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=os.path.join(
            ROOT_PATH, "benchmark_csvs", "open-llm-leaderboard-v2.csv"
        ),
        help="Path to leaderboard CSV (Model, MMLU-PRO Raw)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for subsampling"
    )
    parser.add_argument(
        "--n_sample", type=int, default=5, help="Number of models to subsample"
    )
    parser.add_argument(
        "--check_correctness_consistency",
        action="store_true",
        help="Also check correctness == (gold == pred_label). Default off because acc_norm is from logprobs, not resps.",
    )
    args = parser.parse_args()

    run_tests(
        outputs_path=args.outputs_path,
        raw_path=args.raw_path,
        csv_path=args.csv_path,
        seed=args.seed,
        n_sample=args.n_sample,
        skip_correctness_consistency=not args.check_correctness_consistency,
    )


if __name__ == "__main__":
    main()
