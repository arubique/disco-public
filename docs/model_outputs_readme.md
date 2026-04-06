---
license: other
language:
  - en
task_categories:
  - other
tags:
  - disco
  - leaderboard
  - mmlu
  - hellaswag
  - winogrande
  - arc
  - model-evaluation
pretty_name: DISCO Model Outputs (Open LLM Leaderboard)
---

# DISCO model outputs

Tabular release of **per-model, per-item correctness and answer scores** used to train and evaluate [DISCO: Diversifying Sample Condensation for Efficient Model Evaluation](https://huggingface.co/papers/2510.07959). The paper studies cheap benchmark performance prediction from a small subset of evaluation items; this dataset supplies the raw harness-style outputs for MMLU (57 subjects), HellaSwag, Winogrande, ARC, and related tasks from the Open LLM Leaderboard ecosystem.

## Paper

- **Hugging Face Papers:** [2510.07959](https://huggingface.co/papers/2510.07959)  
- **arXiv:** [2510.07959](https://arxiv.org/abs/2510.07959)

## How this dataset is built (source pipeline)

The on-disk artifact in the [DISCO codebase](https://github.com/arubique/disco-public) is `data/model_outputs.pickle`. It can be **downloaded** from the Hub (see `scripts/download_model_outputs.py`) or **rebuilt from Open LLM Leaderboard snapshots** using the same steps as in the project README:

1. **Extended leaderboard snapshot** (tinyBenchmarks-style Open LLM Leaderboard data), on the order of many hours to fetch:  
   `python ./scripts/download_leaderboard.py --lb_type openllm_leaderboard --lb_savepath ./data/lb_raw_extended.pickle`

2. **MMLU-fields snapshot** (additional models / fields), on the order of ~1 hour:  
   `python ./scripts/download_leaderboard.py --lb_type mmlu_fields --lb_savepath ./data/lb_raw.pickle`

3. **Merge and extract** into the ordered pickle consumed by DISCO (~20 minutes):  
   `python scripts/extract_model_outputs_from_raw_data.py`

That pipeline produces `model_outputs.pickle` with a list of model identifiers and, for each harness task, dense arrays of **correctness** and **per-choice scores** (logits / likelihood-style values as stored by the harness). The Hub upload script flattens those arrays into viewer-friendly tables.

## Hub layout (configs and columns)

This repository is a **multi-config** dataset. Each config corresponds to one logical table; within each config the split is named **`train`**.

| Config | Role |
|--------|------|
| `manifest` | Maps Hub config names to original harness keys (`task_split_name` → `original_data_key`). |
| `models` | `model_idx`, `model_name` — one row per model. |
| *task configs* | e.g. `hellaswag`, `mmlu_abstract_algebra`, … — long format: `sample_idx`, `model_idx`, `correctness`, and `logit_0` … `logit_{K-1}` for each answer choice. |

Pick a **subset** in the dataset viewer, then open the **`train`** split to inspect rows.

## Code and documentation

- Repository: [github.com/arubique/disco-public](https://github.com/arubique/disco-public)  
- Hub upload / download helpers: `scripts/model_outputs_hf.py`, `scripts/upload_model_outputs_to_hf.py`, `scripts/download_model_outputs.py`  
- Extra notes: [`docs/datasets.md`](https://github.com/arubique/disco-public/blob/main/docs/datasets.md) (paths relative to the GitHub repo)

## License

This card uses `license: other` because the release aggregates **derived statistics** from public Open LLM Leaderboard–style evaluations; confirm any reuse constraints with the original benchmark and leaderboard terms.

## Citation

If you use this dataset, please cite the DISCO paper (see the Hugging Face Papers page above for bibliographic metadata).
