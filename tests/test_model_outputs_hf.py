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


def test_datasetdict_reassemble_roundtrip_local():
    data = _minimal_model_outputs()
    dd = build_model_outputs_dataset_dict(data)
    out = reassemble_model_outputs_from_tabular_splits(dict(dd))
    assert_model_outputs_equal(data, out)


def test_debug_datasetdict_has_two_splits_only():
    data = _minimal_model_outputs()
    dd = build_model_outputs_dataset_dict(data, debug=True)
    assert set(dd.keys()) == {"manifest", "models"}
    out = reassemble_model_outputs_from_tabular_splits(dict(dd))
    assert out["models"] == data["models"]
    assert out["data"] == {}


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
