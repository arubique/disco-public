"""
Hugging Face Hub layout for `model_outputs.pickle` as a *single* dataset repository.

Splits (manifest, models, hellaswag, mmlu_*, …) are stored as **pandas-friendly tables**
so the Hub dataset viewer shows real columns. Task subsets use long format: one row per
(sample, model) with `sample_idx`, `model_idx`, `correctness`, and `logit_0` … `logit_{k-1}`
for leading scores only. Trailing choice columns that are entirely non-finite padding (`-inf`
/ NaN in the pickle) are **dropped** on upload. The manifest stores `prediction_width`
(original `predictions.shape[2]`) per task; download **pads** with `-inf` to restore the
same arrays as the source pickle. Legacy Hub tables may lack `prediction_width` or use a
`predictions` list column; download supports both.

Each block is pushed as its own Hub **configuration** (`push_to_hub(repo_id, config_name, split="train")`)
so schemas may differ; the viewer subset dropdown lists config names (`manifest`, `models`, `hellaswag`, …).

Download supports:
  - **v3 (tabular)**: manifest rows with `original_data_key` column
  - **v2 (legacy)**: manifest row with `pickle` column containing a pickled dict
"""

import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    pd = None  # type: ignore
    _PANDAS_IMPORT_ERROR = e
else:
    _PANDAS_IMPORT_ERROR = None

PICKLE_COLUMN = "pickle"
SUMMARY_COLUMN = "summary"
MODELS_SHARD = "models"
MANIFEST_SPLIT = "manifest"
MODELS_SPLIT = "models"
DATA_SHARD_PREFIX = "data__"
# Hub: v2 = pickle blobs per split; v3 = tabular (pandas) splits
MANIFEST_FORMAT_VERSION_HUB_PICKLE = 2
MANIFEST_FORMAT_VERSION_HUB_TABULAR = 3
MANIFEST_FORMAT_VERSION_LOCAL = 1

RESERVED_SPLIT_NAMES: Set[str] = {MANIFEST_SPLIT, MODELS_SPLIT}

# Matches `pad_predictions` in utils_for_notebooks: unused answer slots in the pickle.
PREDICTION_PADDING_VALUE = float("-inf")

# With --debug upload: only these harness keys (if present in the pickle) plus manifest/models.
DEBUG_UPLOAD_DATA_KEYS: Tuple[str, ...] = (
    "harness_hellaswag_10",
    "harness_hendrycksTest_abstract_algebra_5",
)


def slice_model_outputs_to_debug_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep full ``models`` and only task blocks listed in `DEBUG_UPLOAD_DATA_KEYS` that exist.
    Use when comparing a full reference pickle to a Hub dataset uploaded with ``--debug``.
    """
    inner = {
        k: data["data"][k] for k in DEBUG_UPLOAD_DATA_KEYS if k in data["data"]
    }
    return {"models": data["models"], "data": inner}


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for tabular Hub upload/download. Install with: pip install pandas"
        ) from _PANDAS_IMPORT_ERROR
    return pd


DEFAULT_MODEL_OUTPUTS_HF_REPO = "arubique/disco-model-outputs"


def get_hub_base(cli_value: Optional[str]) -> str:
    """Resolve Hub dataset repo id: CLI ``--hub-base``, then ``DISCO_MODEL_OUTPUTS_HF_BASE``, then default."""
    if cli_value:
        return cli_value
    v = os.environ.get("DISCO_MODEL_OUTPUTS_HF_BASE")
    if v:
        return v
    return DEFAULT_MODEL_OUTPUTS_HF_REPO


def data_key_to_shard_name(data_key: str) -> str:
    return f"{DATA_SHARD_PREFIX}{data_key}"


def shard_name_to_data_key(shard_name: str) -> Optional[str]:
    if shard_name == MODELS_SHARD:
        return None
    if not shard_name.startswith(DATA_SHARD_PREFIX):
        return None
    return shard_name[len(DATA_SHARD_PREFIX) :]


def iter_shard_names(data: Dict[str, Any]) -> List[str]:
    if "models" not in data or "data" not in data:
        raise ValueError(
            "Expected pickle dict with top-level keys 'models' and 'data'."
        )
    names = [MODELS_SHARD]
    names.extend(data_key_to_shard_name(k) for k in sorted(data["data"].keys()))
    return names


def build_manifest(data: Dict[str, Any]) -> Dict[str, Any]:
    """Local / test manifest listing internal shard names (v1)."""
    return {
        "format_version": MANIFEST_FORMAT_VERSION_LOCAL,
        "shards": iter_shard_names(data),
    }


def split_model_outputs_to_shards(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map shard_name -> object (for local tests)."""
    shards: Dict[str, Any] = {MODELS_SHARD: data["models"]}
    for key in sorted(data["data"].keys()):
        shards[data_key_to_shard_name(key)] = data["data"][key]
    return shards


def merge_shards_to_model_outputs(
    shard_objects: Dict[str, Any]
) -> Dict[str, Any]:
    if MODELS_SHARD not in shard_objects:
        raise ValueError(f"Missing shard {MODELS_SHARD!r}")
    models = shard_objects[MODELS_SHARD]
    inner: Dict[str, Any] = {}
    for name, obj in shard_objects.items():
        if name == MODELS_SHARD:
            continue
        key = shard_name_to_data_key(name)
        if key is None:
            raise ValueError(f"Unexpected shard name: {name!r}")
        inner[key] = obj
    return {"models": models, "data": inner}


def model_outputs_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    try:
        assert_model_outputs_equal(a, b)
        return True
    except AssertionError:
        return False


def assert_model_outputs_equal(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    assert set(a.keys()) == {"models", "data"}
    assert set(b.keys()) == {"models", "data"}
    assert a["models"] == b["models"]
    assert set(a["data"].keys()) == set(b["data"].keys())
    for k in a["data"]:
        da, db = a["data"][k], b["data"][k]
        assert set(da.keys()) == set(db.keys())
        for kk in da:
            va, vb = da[kk], db[kk]
            if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
                np.testing.assert_array_equal(va, vb)
            else:
                assert va == vb


def _sanitize_split_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "split"
    if s[0].isdigit():
        s = "s_" + s
    return s[:200]


def default_split_name_for_data_key(data_key: str) -> str:
    if data_key == "harness_arc_challenge_25":
        base = "arc_challenge"
    elif data_key == "harness_hellaswag_10":
        base = "hellaswag"
    elif data_key == "harness_winogrande_5":
        base = "winogrande"
    elif data_key.startswith("harness_truthfulqa"):
        base = "truthfulqa_" + data_key[len("harness_truthfulqa_") :]
    elif data_key.startswith("harness_hendrycksTest_"):
        rest = data_key[len("harness_hendrycksTest_") :]
        if rest.endswith("_5"):
            rest = rest[:-2]
        base = "mmlu_" + rest
    elif data_key.startswith("harness_"):
        base = data_key[len("harness_") :]
    else:
        base = data_key
    return _sanitize_split_name(base)


def _prediction_full_width(block: Dict[str, Any]) -> int:
    return int(np.asarray(block["predictions"], dtype=np.float64).shape[2])


def _trimmed_logit_column_count(p: np.ndarray) -> int:
    """Drop trailing columns that are entirely non-finite (Hub shows them as nulls)."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 3:
        raise ValueError(f"predictions must be (S,M,A); got {p.shape}")
    _, _, a = p.shape
    k = a
    while k > 0 and np.all(~np.isfinite(p[:, :, k - 1])):
        k -= 1
    return max(k, 1)


def build_hub_split_layout(
    data: Dict[str, Any], *, debug: bool = False
) -> Dict[str, Any]:
    used: Set[str] = set(RESERVED_SPLIT_NAMES)
    data_splits: Dict[str, str] = {}
    prediction_widths: Dict[str, int] = {}

    def _add_task(dk: str) -> None:
        base = default_split_name_for_data_key(dk)
        sn = base
        n = 2
        while sn in used:
            sn = _sanitize_split_name(f"{base}__{n}")
            n += 1
        used.add(sn)
        data_splits[sn] = dk
        prediction_widths[sn] = _prediction_full_width(data["data"][dk])

    if debug:
        for dk in DEBUG_UPLOAD_DATA_KEYS:
            if dk in data["data"]:
                _add_task(dk)
    else:
        for dk in sorted(data["data"].keys()):
            _add_task(dk)

    return {
        "format_version": MANIFEST_FORMAT_VERSION_HUB_TABULAR,
        "model_split": MODELS_SPLIT,
        "data_splits": data_splits,
        "prediction_widths": prediction_widths,
    }


def _blob_to_bytes(blob: Any) -> bytes:
    if isinstance(blob, bytes):
        return blob
    if isinstance(blob, (memoryview, bytearray)):
        return bytes(blob)
    return bytes(blob)


def _task_block_to_dataframe(block: Dict[str, Any]) -> Any:
    """Long-format rows; only leading logit_* columns (trailing all-padding columns omitted)."""
    pd = _require_pandas()
    c = np.asarray(block["correctness"], dtype=np.float64)
    p = np.asarray(block["predictions"], dtype=np.float64)
    if c.ndim != 2 or p.ndim != 3:
        raise ValueError(
            f"Expected correctness (S,M) and predictions (S,M,A); got {c.shape}, {p.shape}"
        )
    S, M = c.shape
    a_full = p.shape[2]
    k = _trimmed_logit_column_count(p)
    if k > a_full:
        k = a_full
    sample_idx = np.repeat(np.arange(S, dtype=np.int64), M)
    model_idx = np.tile(np.arange(M, dtype=np.int64), S)
    correctness = c.reshape(-1, order="C")
    pred_flat = p.reshape(S * M, a_full)
    col_data: Dict[str, Any] = {
        "sample_idx": sample_idx,
        "model_idx": model_idx,
        "correctness": correctness.astype(np.float64),
    }
    for j in range(k):
        col_data[f"logit_{j}"] = pred_flat[:, j].astype(np.float64)
    return pd.DataFrame(col_data)


def _logit_column_names(df: Any) -> List[str]:
    names = [c for c in df.columns if c.startswith("logit_")]
    names.sort(key=lambda x: int(x.split("_", 1)[1]))
    return names


def _task_features_for_num_choices(num_choices: int) -> Any:
    from datasets import Features, Value

    fields: Dict[str, Any] = {
        "sample_idx": Value("int64"),
        "model_idx": Value("int64"),
        "correctness": Value("float64"),
    }
    for j in range(num_choices):
        fields[f"logit_{j}"] = Value("float64")
    return Features(fields)


def _dataframe_to_task_block(
    df: Any, *, full_answer_dim: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Rebuild correctness (S,M) and predictions (S,M,A).

    If full_answer_dim is set (from manifest `prediction_width`), pad trailing choice
    dimensions with PREDICTION_PADDING_VALUE so the array matches the original pickle.
    """
    _require_pandas()
    df = df.sort_values(
        ["sample_idx", "model_idx"], kind="mergesort"
    ).reset_index(drop=True)
    S = int(df["sample_idx"].max()) + 1
    M = int(df["model_idx"].max()) + 1
    correctness = df["correctness"].values.reshape(S, M, order="C")

    if "predictions" in df.columns:
        first = df["predictions"].iloc[0]
        if isinstance(first, np.ndarray):
            a = int(first.shape[0])
        else:
            a = len(first)
        pred_stack = np.stack(
            [np.asarray(x, dtype=np.float64) for x in df["predictions"].values]
        )
        predictions = pred_stack.reshape(S, M, a)
        return {"correctness": correctness, "predictions": predictions}

    logit_cols = _logit_column_names(df)
    if not logit_cols:
        raise ValueError(
            "Task table has no logit_* columns and no predictions column; cannot rebuild arrays."
        )
    k = len(logit_cols)
    pred_stack = np.column_stack(
        [df[c].astype(np.float64).values for c in logit_cols]
    )
    a_target = full_answer_dim if full_answer_dim is not None else k
    if a_target < k:
        raise ValueError(
            f"prediction_width {a_target} < number of logit columns {k}"
        )
    if a_target == k:
        predictions = pred_stack.reshape(S, M, k)
    else:
        predictions = np.full(
            (S, M, a_target), PREDICTION_PADDING_VALUE, dtype=np.float64
        )
        predictions[:, :, :k] = pred_stack.reshape(S, M, k)
    return {"correctness": correctness, "predictions": predictions}


def _build_manifest_dataframe(layout: Dict[str, Any]) -> Any:
    pd = _require_pandas()
    widths: Dict[str, int] = layout.get("prediction_widths") or {}
    rows = []
    for sn, dk in sorted(layout["data_splits"].items()):
        rows.append(
            {
                "format_version": MANIFEST_FORMAT_VERSION_HUB_TABULAR,
                "model_split_name": layout["model_split"],
                "task_split_name": sn,
                "original_data_key": dk,
                "prediction_width": int(widths.get(sn, 0)),
            }
        )
    if not rows:
        rows.append(
            {
                "format_version": MANIFEST_FORMAT_VERSION_HUB_TABULAR,
                "model_split_name": layout["model_split"],
                "task_split_name": "",
                "original_data_key": "",
                "prediction_width": 0,
            }
        )
    return pd.DataFrame(rows)


def _tabular_manifest_and_models_features():
    from datasets import Features, Value

    manifest_features = Features(
        {
            "format_version": Value("int64"),
            "model_split_name": Value("string"),
            "task_split_name": Value("string"),
            "original_data_key": Value("string"),
            "prediction_width": Value("int64"),
        }
    )
    models_features = Features(
        {
            "model_idx": Value("int64"),
            "model_name": Value("string"),
        }
    )
    return manifest_features, models_features


def build_tabular_hub_splits(
    data: Dict[str, Any], *, debug: bool = False
) -> Dict[str, Any]:
    """
    Map split_name -> Dataset built from pandas tables (Hub viewer friendly).
    """
    from datasets import Dataset

    layout = build_hub_split_layout(data, debug=debug)
    mf, mof = _tabular_manifest_and_models_features()
    pd = _require_pandas()

    out: Dict[str, Any] = {}

    manifest_df = _build_manifest_dataframe(layout)
    out[MANIFEST_SPLIT] = Dataset.from_pandas(
        manifest_df, preserve_index=False, features=mf
    )

    models_df = pd.DataFrame(
        {
            "model_idx": np.arange(len(data["models"]), dtype=np.int64),
            "model_name": [str(x) for x in data["models"]],
        }
    )
    out[MODELS_SPLIT] = Dataset.from_pandas(
        models_df, preserve_index=False, features=mof
    )

    for sn, dk in sorted(layout["data_splits"].items()):
        tdf = _task_block_to_dataframe(data["data"][dk])
        n_logit_cols = len(_logit_column_names(tdf))
        tf = _task_features_for_num_choices(n_logit_cols)
        out[sn] = Dataset.from_pandas(tdf, preserve_index=False, features=tf)

    return out


def build_model_outputs_dataset_dict(
    data: Dict[str, Any], *, debug: bool = False
) -> Any:
    """In-memory DatasetDict of tabular splits (for local tests; schemas differ per split)."""
    from datasets import DatasetDict

    parts = build_tabular_hub_splits(data, debug=debug)
    return DatasetDict(parts)


def push_model_outputs_to_hub(
    repo_id: str,
    data: Dict[str, Any],
    *,
    token: Optional[str] = None,
    private: bool = False,
    debug: bool = False,
) -> None:
    if "/" not in repo_id:
        raise ValueError(
            f"repo_id must be a full Hub dataset id 'org/name', got: {repo_id!r}"
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )

    splits = build_tabular_hub_splits(data, debug=debug)
    huge: Union[int, str] = 1 << 40
    order = [MANIFEST_SPLIT, MODELS_SPLIT] + sorted(
        k for k in splits if k not in (MANIFEST_SPLIT, MODELS_SPLIT)
    )
    for name in order:
        # One Hub *config* per block; a single split "train" inside each config. Using only
        # Hub splits would force identical features across the whole repo (HF error).
        splits[name].push_to_hub(
            repo_id,
            name,
            split="train",
            private=private,
            token=token,
            max_shard_size=huge,
        )

    _upload_model_outputs_dataset_readme(repo_id, token=token)


def _merge_model_outputs_readme(
    static_readme_text: str, hub_readme_text: str
) -> str:
    """
    Combine the curated dataset card (license, tags, markdown body) with YAML metadata
    produced on the Hub by ``datasets`` (``configs``, ``dataset_info``).

    ``push_to_hub`` writes those keys into README.md; uploading the static card alone
    would drop them and the Hub viewer would only show a subset of configurations.
    """
    from huggingface_hub.repocard import DatasetCard
    from huggingface_hub.repocard_data import DatasetCardData

    static_card = DatasetCard(static_readme_text)
    hub_card = DatasetCard(hub_readme_text)
    merged_dict = {**static_card.data.to_dict(), **hub_card.data.to_dict()}
    merged_data = DatasetCardData(ignore_metadata_errors=True, **merged_dict)
    line_break = "\n"
    raw = (
        f"---{line_break}{merged_data.to_yaml(line_break=line_break)}"
        f"{line_break}---{line_break}{static_card.text}"
    )
    return DatasetCard(raw).content


def _upload_model_outputs_dataset_readme(
    repo_id: str,
    *,
    token: Optional[str] = None,
) -> None:
    """Upload the curated card merged with Hub metadata (configs / dataset_info)."""
    from pathlib import Path

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    root = Path(__file__).resolve().parent.parent
    readme_path = root / "docs" / "model_outputs_readme.md"
    if not readme_path.is_file():
        raise FileNotFoundError(
            f"Dataset README missing (expected {readme_path}). "
            "Add docs/model_outputs_readme.md before pushing."
        )
    static_text = readme_path.read_text(encoding="utf-8")
    try:
        remote_path = hf_hub_download(
            repo_id,
            "README.md",
            repo_type="dataset",
            token=token,
        )
        hub_text = Path(remote_path).read_text(encoding="utf-8")
        payload = _merge_model_outputs_readme(static_text, hub_text)
    except EntryNotFoundError:
        payload = static_text
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=payload.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )


def _manifest_dict_from_row(mrow: Any) -> Dict[str, Any]:
    """Legacy pickle manifest (v2) or old int64/string/json columns."""
    blob_raw = mrow.get(PICKLE_COLUMN)
    if blob_raw is not None:
        blob = _blob_to_bytes(blob_raw)
        if len(blob) > 0:
            obj = pickle.loads(blob)
            if isinstance(obj, dict) and "data_splits" in obj:
                return obj
    if "data_splits_json" in mrow and mrow["data_splits_json"] is not None:
        return {
            "format_version": int(mrow["format_version"]),
            "model_split": str(mrow["model_split"]),
            "data_splits": json.loads(mrow["data_splits_json"]),
        }
    raise ValueError(
        f"Unrecognized manifest split row: keys={list(mrow.keys())}"
    )


def _is_tabular_manifest_dataset(manifest_ds: Any) -> bool:
    return "original_data_key" in manifest_ds.column_names


def reassemble_model_outputs_from_tabular_splits(
    splits: Dict[str, Any]
) -> Dict[str, Any]:
    """Rebuild from split name -> Dataset (in-memory or loaded one split at a time)."""
    pd = _require_pandas()
    manifest_ds = splits[MANIFEST_SPLIT]
    mf = manifest_ds.to_pandas()
    version = int(mf["format_version"].iloc[0])
    if version != MANIFEST_FORMAT_VERSION_HUB_TABULAR:
        raise ValueError(
            f"Expected tabular hub format_version {MANIFEST_FORMAT_VERSION_HUB_TABULAR}, got {version!r}"
        )
    model_split_name = str(mf["model_split_name"].iloc[0])

    models_ds = splits[model_split_name]
    mdf = models_ds.to_pandas().sort_values("model_idx", kind="mergesort")
    models = mdf["model_name"].tolist()

    inner: Dict[str, Any] = {}
    for _, row in mf.iterrows():
        sn = str(row["task_split_name"])
        dk = str(row["original_data_key"])
        if not sn or not dk:
            continue
        full_dim: Optional[int] = None
        if "prediction_width" in row.index:
            pw = row["prediction_width"]
            if pd.notna(pw) and int(pw) > 0:
                full_dim = int(pw)
        inner[dk] = _dataframe_to_task_block(
            splits[sn].to_pandas(), full_answer_dim=full_dim
        )
    return {"models": models, "data": inner}


def reassemble_model_outputs_from_dataset_dict(dsd: Any) -> Dict[str, Any]:
    """Rebuild pickle dict from a loaded DatasetDict (legacy v2 pickle splits on Hub)."""
    mrow = dsd[MANIFEST_SPLIT][0]
    manifest_obj = _manifest_dict_from_row(mrow)
    version = int(manifest_obj["format_version"])
    if version != MANIFEST_FORMAT_VERSION_HUB_PICKLE:
        raise ValueError(
            f"Unsupported legacy pickle manifest format_version: {version!r} "
            f"(expected {MANIFEST_FORMAT_VERSION_HUB_PICKLE})"
        )
    data_splits: Dict[str, str] = dict(manifest_obj["data_splits"])
    model_split = str(manifest_obj["model_split"])

    models = pickle.loads(_blob_to_bytes(dsd[model_split][0][PICKLE_COLUMN]))
    inner: Dict[str, Any] = {}
    for sn, dk in data_splits.items():
        inner[dk] = pickle.loads(_blob_to_bytes(dsd[sn][0][PICKLE_COLUMN]))
    return {"models": models, "data": inner}


def _load_hub_config_train(
    repo_id: str, config_name: str, *, token: Optional[str] = None
) -> Any:
    from datasets import load_dataset

    ds = load_dataset(repo_id, config_name, split="train", token=token)
    return ds


def _download_tabular_from_hub(
    repo_id: str,
    manifest_ds: Any,
    *,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    mf = manifest_ds.to_pandas()
    splits_map: Dict[str, Any] = {MANIFEST_SPLIT: manifest_ds}
    model_split_name = str(mf["model_split_name"].iloc[0])
    splits_map[model_split_name] = _load_hub_config_train(
        repo_id, model_split_name, token=token
    )
    for _, row in mf.iterrows():
        sn = str(row["task_split_name"])
        dk = str(row["original_data_key"])
        if not sn or not dk:
            continue
        if sn not in splits_map:
            splits_map[sn] = _load_hub_config_train(repo_id, sn, token=token)
    return reassemble_model_outputs_from_tabular_splits(splits_map)


def download_model_outputs_from_hub(
    repo_id: str,
    *,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    from datasets import load_dataset

    manifest_ds = None
    try:
        manifest_ds = _load_hub_config_train(
            repo_id, MANIFEST_SPLIT, token=token
        )
    except Exception:
        pass

    if manifest_ds is not None and _is_tabular_manifest_dataset(manifest_ds):
        return _download_tabular_from_hub(repo_id, manifest_ds, token=token)

    dsd = load_dataset(repo_id, token=token)
    return reassemble_model_outputs_from_dataset_dict(dsd)
