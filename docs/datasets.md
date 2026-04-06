To train DISCO you will need model outputs on MMLU, Hellaswag, Winogrande and Arc datasets.
The recommended source is the Hugging Face Hub (same content as the former Google Drive archive).

Set the full Hub **dataset repository id** once (`namespace/name`):

```
export DISCO_MODEL_OUTPUTS_HF_BASE=your-org/disco-model-outputs
python scripts/download_model_outputs.py
```

This script writes `data/model_outputs.pickle` by loading that dataset. On the Hub, each block is a **subset / configuration** (`manifest`, `models`, `hellaswag`, `mmlu_*`, …), each with a `train` split, so the viewer can use different columns per subset. Tables are **tabular** (pandas-friendly): the manifest lists `task_split_name` → `original_data_key`; `models` has `model_idx` / `model_name`; each task subset has `sample_idx`, `model_idx`, `correctness`, and `predictions` (list of floats) per row.

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

Use `python scripts/upload_model_outputs_to_hf.py --debug` to push only the `manifest` and `models` splits (two Hub splits, no benchmark data) for a quick connectivity check.

Everything is pushed as **one** dataset repo. The Hub viewer **Subset** menu lists configurations (`manifest`, `models`, `hellaswag`, …); pick a subset then view the `train` split.

To confirm the Hub copy matches a reference `model_outputs.pickle` (for example one downloaded with `--source gdrive`), install test deps and run:

```
pip install pytest
export DISCO_MODEL_OUTPUTS_HF_BASE=your-org/disco-model-outputs
export DISCO_MODEL_OUTPUTS_COMPARE_GDRIVE_PICKLE=/path/to/model_outputs.pickle
pytest tests/test_model_outputs_hf.py -m integration
```
