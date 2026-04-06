To train DISCO you will need model outputs on MMLU, Hellaswag, Winogrande and Arc datasets.
The recommended source is the Hugging Face Hub (same content as the former Google Drive archive).

Set the full Hub **dataset repository id** once (`namespace/name`):

```
export DISCO_MODEL_OUTPUTS_HF_BASE=your-org/disco-model-outputs
python scripts/download_model_outputs.py
```

This script writes `data/model_outputs.pickle` by loading that dataset. On the Hub, each block is a **subset / configuration** (`manifest`, `models`, `hellaswag`, `mmlu_*`, …), each with a `train` split, so the viewer can use different columns per subset. Tables are **tabular** (pandas-friendly): the manifest lists `task_split_name` → `original_data_key`; `models` has `model_idx` / `model_name`; each task subset has `sample_idx`, `model_idx`, `correctness`, and one column per **non-padding** answer choice (`logit_0` … `logit_{k-1}`): trailing columns that are all `-inf`/null in the pickle are omitted on upload. The manifest row for that task includes `prediction_width` (original last-axis length, e.g. 31) so download restores `-inf` padding to match the Google Drive pickle byte-for-byte in array tests.

To use the legacy Google Drive file instead:

```
python scripts/download_model_outputs.py --source gdrive
```

### Upload model outputs to Hugging Face (maintainers)

After you have `data/model_outputs.pickle`, you can publish shards with:

```
export DISCO_MODEL_OUTPUTS_HF_BASE=your-org/disco-model-outputs
huggingface-cli login   # or export HF_TOKEN=...
python scripts/upload_model_outputs_to_hf.py
```

Use `python scripts/upload_model_outputs_to_hf.py --debug` to push `manifest`, `models`, and only **Hellaswag** plus **MMLU abstract algebra** (`harness_hellaswag_10`, `harness_hendrycksTest_abstract_algebra_5`) when those keys exist in the pickle—omitting all other tasks.

Everything is pushed as **one** dataset repo. The Hub viewer **Subset** menu lists configurations (`manifest`, `models`, `hellaswag`, …); pick a subset then view the `train` split.

The dataset repository **`README.md`** on the Hub is uploaded from [`docs/model_outputs_readme.md`](model_outputs_readme.md) on each push (paper link, construction steps, column layout).

To confirm the Hub copy matches a reference `model_outputs.pickle` (for example one downloaded with `--source gdrive`), install test deps and run:

```
pip install pytest
export DISCO_MODEL_OUTPUTS_HF_BASE=arubique/disco-model-outputs
export DISCO_MODEL_OUTPUTS_COMPARE_GDRIVE_PICKLE=data/model_outputs.pickle
pytest tests/test_model_outputs_hf.py -m integration
```

Paths in `DISCO_MODEL_OUTPUTS_COMPARE_GDRIVE_PICKLE` may be relative to the **repository root** (not only the shell cwd). If the basename is `model-outputs.pickle` but the file on disk is `model_outputs.pickle`, the test tries both. If pytest reports `s` (skipped), run `pytest … -rs` to print the skip reason. If the Hub was uploaded with `--debug`, set `DEBUG=1` so the test slices the reference pickle to the same tasks before comparing.
