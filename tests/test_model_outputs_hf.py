"""
Tests for Hugging Face layout of model_outputs.pickle.

Integration test (HF vs reference pickle) runs only when env vars are set; see docstring below.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from model_outputs_hf import (  # noqa: E402
    DEFAULT_MODEL_OUTPUTS_HF_REPO,
    MANIFEST_FORMAT_VERSION_HUB_TABULAR,
    _merge_model_outputs_readme,
    assert_model_outputs_equal,
    build_manifest,
    build_hub_split_layout,
    build_model_outputs_dataset_dict,
    download_model_outputs_from_hub,
    get_hub_base,
    iter_shard_names,
    merge_shards_to_model_outputs,
    reassemble_model_outputs_from_tabular_splits,
    slice_model_outputs_to_debug_keys,
    split_model_outputs_to_shards,
)


def _minimal_model_outputs() -> dict:
    return {
        "models": ["m1", "m2"],
        "data": {
            "harness_arc_challenge_25": {
                "correctness": np.array([[1.0, 0.0]], dtype=np.float64),
                "predictions": np.zeros((1, 2, 5), dtype=np.float64),
            },
            "harness_hellaswag_10": {
                "correctness": np.array([[0.5, 1.0]], dtype=np.float64),
                "predictions": np.ones((1, 2, 10), dtype=np.float64),
            },
        },
    }


def test_get_hub_base_default_without_env(monkeypatch):
    monkeypatch.delenv("DISCO_MODEL_OUTPUTS_HF_BASE", raising=False)
    assert get_hub_base(None) == DEFAULT_MODEL_OUTPUTS_HF_REPO
    assert get_hub_base("other/repo") == "other/repo"


def test_iter_shard_names_order():
    data = _minimal_model_outputs()
    names = iter_shard_names(data)
    assert names[0] == "models"
    assert names[1:] == sorted(names[1:])


def test_split_merge_roundtrip():
    data = _minimal_model_outputs()
    shards = split_model_outputs_to_shards(data)
    merged = merge_shards_to_model_outputs(shards)
    assert_model_outputs_equal(data, merged)


def test_build_manifest_lists_shards():
    data = _minimal_model_outputs()
    m = build_manifest(data)
    assert m["shards"] == iter_shard_names(data)


def test_pickle_bytes_roundtrip_per_shard():
    """Same bytes workflow as Hub (pickle column would hold these blobs)."""
    data = _minimal_model_outputs()
    shards = split_model_outputs_to_shards(data)
    blobs = {
        k: pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
        for k, v in shards.items()
    }
    restored = {k: pickle.loads(b) for k, b in blobs.items()}
    merged = merge_shards_to_model_outputs(restored)
    assert_model_outputs_equal(data, merged)


def test_hub_split_layout_unique_and_reserved():
    data = _minimal_model_outputs()
    layout = build_hub_split_layout(data)
    assert layout["format_version"] == MANIFEST_FORMAT_VERSION_HUB_TABULAR
    assert layout["model_split"] == "models"
    names = set(layout["data_splits"].keys())
    assert "manifest" not in names
    assert "models" not in names
    assert len(names) == len(layout["data_splits"])
    assert layout["prediction_widths"]["arc_challenge"] == 5
    assert layout["prediction_widths"]["hellaswag"] == 10


def test_trailing_padding_logits_omitted_on_hub_and_restored_on_download():
    p = np.ones((1, 2, 10), dtype=np.float64)
    p[:, :, 5:] = float("-inf")
    data = {
        "models": ["m1", "m2"],
        "data": {
            "harness_hellaswag_10": {
                "correctness": np.array([[0.5, 1.0]], dtype=np.float64),
                "predictions": p,
            }
        },
    }
    dd = build_model_outputs_dataset_dict(data)
    logit_cols = [
        c for c in dd["hellaswag"].to_pandas().columns if c.startswith("logit_")
    ]
    assert len(logit_cols) == 5
    out = reassemble_model_outputs_from_tabular_splits(dict(dd))
    assert_model_outputs_equal(data, out)


def test_datasetdict_reassemble_roundtrip_local():
    data = _minimal_model_outputs()
    dd = build_model_outputs_dataset_dict(data)
    out = reassemble_model_outputs_from_tabular_splits(dict(dd))
    assert_model_outputs_equal(data, out)


def test_debug_datasetdict_only_hellaswag_when_mmlu_absent():
    data = _minimal_model_outputs()
    dd = build_model_outputs_dataset_dict(data, debug=True)
    assert set(dd.keys()) == {"manifest", "models", "hellaswag"}
    assert "arc_challenge" not in dd
    out = reassemble_model_outputs_from_tabular_splits(dict(dd))
    assert out["models"] == data["models"]
    assert set(out["data"].keys()) == {"harness_hellaswag_10"}
    partial = {
        "models": data["models"],
        "data": {"harness_hellaswag_10": data["data"]["harness_hellaswag_10"]},
    }
    assert_model_outputs_equal(partial, out)


def test_merge_readme_keeps_hub_configs_and_static_body():
    """Curated README must not replace datasets-generated ``configs`` (Hub viewer)."""
    static = """---
license: other
tags:
  - disco
pretty_name: DISCO Model Outputs
---

# Curated title

Curated body.
"""
    hub = """---
configs:
  - config_name: manifest
    data_files:
      - split: train
        path: manifest/train-*
  - config_name: hellaswag
    data_files:
      - split: train
        path: hellaswag/train-*
---

auto-generated
"""
    out = _merge_model_outputs_readme(static, hub)
    assert "configs:" in out
    assert "hellaswag" in out
    assert "manifest/train-*" in out
    assert "disco" in out
    assert "# Curated title" in out
    assert "Curated body." in out


def test_slice_model_outputs_to_debug_keys():
    data = _minimal_model_outputs()
    data["data"]["harness_hendrycksTest_abstract_algebra_5"] = {
        "correctness": np.array([[1.0, 0.0]], dtype=np.float64),
        "predictions": np.zeros((1, 2, 5), dtype=np.float64),
    }
    sub = slice_model_outputs_to_debug_keys(data)
    assert sub["models"] == data["models"]
    assert set(sub["data"].keys()) == {
        "harness_hellaswag_10",
        "harness_hendrycksTest_abstract_algebra_5",
    }
    assert "harness_arc_challenge_25" not in sub["data"]


def test_debug_datasetdict_includes_mmlu_abstract_algebra_when_present():
    data = _minimal_model_outputs()
    data["data"]["harness_hendrycksTest_abstract_algebra_5"] = {
        "correctness": np.array([[1.0, 0.0]], dtype=np.float64),
        "predictions": np.zeros((1, 2, 5), dtype=np.float64),
    }
    dd = build_model_outputs_dataset_dict(data, debug=True)
    assert "hellaswag" in dd
    assert "mmlu_abstract_algebra" in dd
    assert "arc_challenge" not in dd
    out = reassemble_model_outputs_from_tabular_splits(dict(dd))
    assert set(out["data"].keys()) == {
        "harness_hellaswag_10",
        "harness_hendrycksTest_abstract_algebra_5",
    }


def _resolve_reference_pickle_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _reference_pickle_search_paths(raw: str) -> list[Path]:
    """
    Resolve env path from repo root, then common typo: model-outputs vs model_outputs basename.
    """
    primary = _resolve_reference_pickle_path(raw)
    candidates = [primary]
    stem = primary.stem
    if stem in ("model-outputs", "model_outputs"):
        alt_stem = (
            "model_outputs" if stem == "model-outputs" else "model-outputs"
        )
        candidates.append(primary.with_name(alt_stem + primary.suffix))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            ordered.append(rp)
    return ordered


@pytest.mark.integration
def test_hf_hub_matches_reference_gdrive_pickle():
    """
    Compare a reference pickle (e.g. from Google Drive) to data reassembled from the Hub.

    Set:
      DISCO_MODEL_OUTPUTS_COMPARE_GDRIVE_PICKLE=path/to/model_outputs.pickle
        (relative paths are resolved from the repository root, not only cwd)
      DISCO_MODEL_OUTPUTS_HF_BASE=org/disco-model-outputs  (optional if default in get_hub_base)

    Set DEBUG=1 to compare only tasks uploaded with ``--debug`` (hellaswag + mmlu abstract
    algebra when present); the Hub dataset must have been pushed with ``--debug``.

    Optional: HF_TOKEN if the datasets are private.
    """
    raw_ref = os.environ.get("DISCO_MODEL_OUTPUTS_COMPARE_GDRIVE_PICKLE")
    if not raw_ref:
        pytest.skip(
            "Set DISCO_MODEL_OUTPUTS_COMPARE_GDRIVE_PICKLE to run this integration test."
        )

    tried = _reference_pickle_search_paths(raw_ref)
    ref_path = next((p for p in tried if p.is_file()), None)
    if ref_path is None:
        pytest.skip(
            "Reference pickle not found. Tried:\n  "
            + "\n  ".join(str(p) for p in tried)
            + f"\n(DISCO_MODEL_OUTPUTS_COMPARE_GDRIVE_PICKLE={raw_ref!r}, repo root={ROOT}). "
            "Typo: use model_outputs.pickle (underscore) not model-outputs.pickle. "
            "Re-run with pytest -rs to show this skip reason."
        )

    try:
        hf_repo = get_hub_base(os.environ.get("DISCO_MODEL_OUTPUTS_HF_BASE"))
    except ValueError as e:
        pytest.skip(str(e))

    with ref_path.open("rb") as handle:
        reference = pickle.load(handle)

    token = os.environ.get("HF_TOKEN")
    from_hub = download_model_outputs_from_hub(hf_repo, token=token)

    if os.environ.get("DEBUG", "").strip() == "1":
        reference = slice_model_outputs_to_debug_keys(reference)

    assert_model_outputs_equal(reference, from_hub)
