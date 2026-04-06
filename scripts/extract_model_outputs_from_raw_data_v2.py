"""
Extract model outputs from v2 leaderboard raw pickle (e.g. leaderboard_mmlu_pro_raw.pickle)
into the same format as source_outputs.pkl used by two_stages_v2 and downstream code.

Predictions match lm-eval ``acc_norm`` for multiple_choice (ConfigurableTask.process_results):
``pred_norm = argmax(lls / completion_len)`` where ``lls`` are per-choice loglikelihoods from
``filtered_resps``/``resps`` and ``completion_len[i] = len(choice_i)`` for choices from
``doc_to_choice(doc)`` (same as ``leaderboard_mmlu_pro``: one letter per option, A..).

If ``doc`` is missing for a row, falls back to plain ``argmax(lls)``.

Output dict has:
  - predictions: (n_models, n_questions, n_answers) float array
  - correctness: (n_models, n_questions, 1) float array
  - Models: dict model_name -> local_index
  - Datapoints: dict datapoint_idx -> datapoint_idx
  - Scenarios: dict scenario_name -> list of datapoint indices
"""

import argparse
import os
import string

import numpy as np

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.insert(0, ROOT_PATH)
from utils import dump_pickle, load_pickle

sys.path.pop(0)

MMLU_PRO_SCENARIO_SUFFIX = "__leaderboard_mmlu_pro"

# Gold / choice order: option index 0 == "A", 1 == "B", ... (MMLU-Pro uses up to J).
CHOICE_TO_INDEX = {chr(ord("A") + i): i for i in range(26)}


def _parse_logprob_from_choice_cell(cell):
    """Parse one choice cell from Hub resps/filtered_resps into a float logprob."""
    v = cell
    while isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    if isinstance(v, (list, tuple)) and len(v) >= 1:
        x = v[0]
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return np.nan
    return np.nan


def _legacy_resps_row_to_choice_index(resps_row):
    """Fallback when row is not a list of per-choice logprobs (older / text-answer format)."""
    if resps_row is None:
        return -np.inf
    if isinstance(resps_row, str):
        s = resps_row.strip().upper()
        if s:
            return float(CHOICE_TO_INDEX.get(s[0], -1))
        return -np.inf
    if isinstance(resps_row, (int, float, np.floating)):
        return float(resps_row)
    if isinstance(resps_row, (list, tuple)) and len(resps_row):
        r = resps_row[0]
        while isinstance(r, (list, tuple)) and len(r) == 1:
            r = r[0]
        if isinstance(r, str):
            s = r.strip().upper()
            if s:
                return float(CHOICE_TO_INDEX.get(s[0], -1))
        if isinstance(r, (int, float, np.floating)):
            return float(r)
    return -np.inf


def _doc_to_choice_mmlu_pro_leaderboard(doc):
    """
    Same strings as lm_eval.tasks.leaderboard.mmlu_pro.utils.doc_to_choice (used for completion_len).
    """
    if not isinstance(doc, dict) or "options" not in doc:
        return None
    n = len(doc["options"])
    if n <= 0 or n > len(string.ascii_uppercase):
        return None
    return [string.ascii_uppercase[i] for i in range(n)]


def _resps_row_to_mc_pred_norm_index(resps_row, doc):
    """
    Replicate lm_eval.api.task.ConfigurableTask.process_results (OUTPUT_TYPE multiple_choice):
        choices = doc_to_choice(doc)
        completion_len = np.array([float(len(i)) for i in choices])
        pred_norm = np.argmax(lls / completion_len)
    """
    if resps_row is None:
        return -np.inf
    choices = _doc_to_choice_mmlu_pro_leaderboard(doc)
    if choices is None:
        return _resps_row_to_argmax_choice_index(resps_row)
    if not isinstance(resps_row, (list, tuple)) or len(resps_row) == 0:
        return _legacy_resps_row_to_choice_index(resps_row)

    logprobs = [_parse_logprob_from_choice_cell(c) for c in resps_row]
    m = min(len(choices), len(logprobs))
    if m == 0:
        return -np.inf
    choices = choices[:m]
    logprobs = logprobs[:m]

    lls = np.asarray(logprobs, dtype=np.float64)
    finite = np.isfinite(lls) & ~np.isnan(lls)
    if not np.any(finite):
        return _legacy_resps_row_to_choice_index(resps_row)
    lls = np.where(finite, lls, -np.inf)

    completion_len = np.array(
        [float(len(c)) for c in choices], dtype=np.float64
    )
    # lm-eval divides as-is; avoid div-by-zero if a choice were ""
    completion_len = np.maximum(completion_len, 1.0)
    normed = lls / completion_len
    return float(int(np.argmax(normed)))


def _resps_row_to_argmax_choice_index(resps_row):
    """
    One row = one question: either a list of K choice cells (logprob + greedy flag),
    or a legacy text/single-value answer.
    Returns float index in [0, K-1] or -inf if unknown.
    """
    if resps_row is None:
        return -np.inf
    if not isinstance(resps_row, (list, tuple)):
        return _legacy_resps_row_to_choice_index(resps_row)
    if len(resps_row) == 0:
        return -np.inf

    logprobs = [_parse_logprob_from_choice_cell(c) for c in resps_row]
    finite_mask = np.isfinite(logprobs) & ~np.isnan(logprobs)
    if not np.any(finite_mask):
        return _legacy_resps_row_to_choice_index(resps_row)

    logprobs_arr = np.array(logprobs, dtype=np.float64)
    logprobs_arr[~finite_mask] = -np.inf
    j = int(np.nanargmax(logprobs_arr))
    return float(j)


def _pick_resps_record(rec):
    """Prefer filtered_resps (flat [logprob, greedy] per choice); else resps."""
    fr = rec.get("filtered_resps")
    if fr is not None and hasattr(fr, "__len__") and len(fr) > 0:
        return fr
    return rec.get("resps")


def _resps_to_prediction_indices(resps, n_questions, max_answers=1, docs=None):
    """
    Per-question predicted choice index matching lm-eval acc_norm when ``docs`` is set
    (pred_norm = argmax(lls / len(choice_i))). Same index order as options: 0=A, 1=B, ...
    """
    out = np.full((n_questions, max_answers), -np.inf, dtype=np.float64)
    if resps is None or len(resps) == 0:
        return out
    n = min(n_questions, len(resps))
    for i in range(n):
        doc_i = None
        if docs is not None and i < len(docs):
            doc_i = docs[i]
        if isinstance(doc_i, dict) and doc_i is not None:
            out[i, 0] = _resps_row_to_mc_pred_norm_index(resps[i], doc_i)
        else:
            out[i, 0] = _resps_row_to_argmax_choice_index(resps[i])
    return out


def _get_scenario_key(entry):
    """Return the key that ends with __leaderboard_mmlu_pro, or None."""
    if not isinstance(entry, dict):
        return None
    for k in entry:
        if k.endswith(MMLU_PRO_SCENARIO_SUFFIX):
            return k
    return None


def extract_source_outputs_from_v2_raw(raw_path, pad_to_size=31):
    """
    Load v2 leaderboard raw pickle and build source_outputs dict.

    Raw pickle format: {model_name: {scenario_name: {correctness, resps, ...}}}

    pad_to_size: size of the last dimension of predictions (pad with -inf).
    """
    raw = load_pickle(raw_path)

    models = []
    scenario_key = None
    n_questions = None

    for model_name, scenarios_dict in raw.items():
        if not isinstance(scenarios_dict, dict):
            continue
        sk = _get_scenario_key(scenarios_dict)
        if sk is None:
            continue
        rec = scenarios_dict[sk]
        if rec is None:
            continue
        correctness = rec.get("correctness")
        if correctness is None:
            continue
        try:
            nc = len(correctness)
        except TypeError:
            continue
        if scenario_key is None:
            scenario_key = sk
            n_questions = nc
        # elif sk != scenario_key or nc != n_questions:
        elif nc != n_questions:
            continue
        models.append((model_name, rec))

    if not models or scenario_key is None or n_questions is None:
        raise ValueError(
            "No valid model data found in raw pickle; need at least one model "
            "with non-None correctness for the MMLU-Pro scenario."
        )

    n_models = len(models)
    n_answers = 1

    correctness_arr = np.zeros((n_models, n_questions, 1), dtype=np.float64)
    predictions_arr = np.full(
        (n_models, n_questions, pad_to_size), -np.inf, dtype=np.float64
    )

    for i, (model_name, rec) in enumerate(models):
        corr = rec["correctness"]
        if hasattr(corr, "__iter__") and not isinstance(corr, (str, bytes)):
            corr = list(corr)
        else:
            corr = [float(corr)] * n_questions
        correctness_arr[i, :, 0] = np.array(
            corr[:n_questions], dtype=np.float64
        )

        resps = _pick_resps_record(rec)
        docs_raw = rec.get("doc")
        docs_list = None
        if docs_raw is not None and not isinstance(
            docs_raw, (str, bytes, dict)
        ):
            try:
                docs_list = list(docs_raw)
            except TypeError:
                docs_list = None
        pred = _resps_to_prediction_indices(
            resps, n_questions, max_answers=1, docs=docs_list
        )
        predictions_arr[i, :, :1] = pred[:, :1]

    # Same format as two_stages.build_outputs_dict / source_outputs.pkl
    scenarios_map = {scenario_key: list(range(n_questions))}
    models_map = {name: i for i, (name, _) in enumerate(models)}
    datapoints_map = {i: i for i in range(n_questions)}

    return {
        "predictions": predictions_arr,
        "correctness": correctness_arr,
        "Models": models_map,
        "Datapoints": datapoints_map,
        "Scenarios": scenarios_map,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract source_outputs from v2 leaderboard raw pickle (e.g. MMLU-Pro)."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.path.join(
            ROOT_PATH, "data", "leaderboard_mmlu_pro_raw.pickle"
        ),
        help="Path to v2 raw pickle (default: data/leaderboard_mmlu_pro_raw.pickle)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(
            ROOT_PATH, "data", "model_outputs", "mmlu", "source_outputs.pkl"
        ),
        help="Path to output pickle, same format as source_outputs.pkl (default: data/model_outputs/mmlu/source_outputs.pkl)",
    )
    parser.add_argument(
        "--pad_to_size",
        type=int,
        default=31,
        help="Size of the last dimension of predictions array (pad with -inf). Default: 31.",
    )
    args = parser.parse_args()

    print(f"Loading v2 raw data from {args.input_path}...")
    source_outputs = extract_source_outputs_from_v2_raw(
        args.input_path, pad_to_size=args.pad_to_size
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    dump_pickle(source_outputs, args.output_path)
    print(
        f"Saved source_outputs: predictions {source_outputs['predictions'].shape}, "
        f"correctness {source_outputs['correctness'].shape}, "
        f"models {len(source_outputs['Models'])} -> {args.output_path}"
    )


if __name__ == "__main__":
    main()
