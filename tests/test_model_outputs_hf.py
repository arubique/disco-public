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
    MANIFEST_FORMAT_VERSION_HUB_TABULAR,
    _merge_model_outputs_readme,
    assert_model_outputs_equal,
    build_manifest,
    build_hub_split_layout,
    build_model_outputs_dataset_dict,
    download_model_outputs_from_hub,
    iter_shard_names,
    merge_shards_to_model_outputs,
    reassemble_model_outputs_from_tabular_splits,
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


@pytest.mark.integration
def test_hf_hub_matches_reference_gdrive_pickle():
    """
    Compare a reference pickle (obtained from Google Drive via gdown or any mirror)
    to the object reassembled from the Hugging Face Hub.

    Set:
      DISCO_MODEL_OUTPUTS_HF_BASE=org/disco-model-outputs
      DISCO_MODEL_OUTPUTS_COMPARE_GDRIVE_PICKLE=/path/to/model_outputs.pickle

    Optional: HF_TOKEN if the datasets are private.
    """
    ref_path = os.environ.get("DISCO_MODEL_OUTPUTS_COMPARE_GDRIVE_PICKLE")
    hf_repo = os.environ.get("DISCO_MODEL_OUTPUTS_HF_BASE")
    if not ref_path or not hf_repo:
        pytest.skip(
            "Set DISCO_MODEL_OUTPUTS_COMPARE_GDRIVE_PICKLE and DISCO_MODEL_OUTPUTS_HF_BASE "
            "to run Hub vs Google Drive pickle comparison."
        )
    if not os.path.isfile(ref_path):
        pytest.skip(f"Reference pickle not found: {ref_path}")

    with open(ref_path, "rb") as handle:
        reference = pickle.load(handle)

    token = os.environ.get("HF_TOKEN")
    from_hub = download_model_outputs_from_hub(hf_repo, token=token)
    assert_model_outputs_equal(reference, from_hub)
