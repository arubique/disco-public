# Running `scripts/two_stages_v2.py`

This script runs the two-stage pipeline: it picks **anchor questions** from **source** model outputs, builds **embeddings** (PCA on softmax of predictions on those anchors), fits a **RandomForest** regressor from train embeddings to true accuracies, and evaluates **predicted accuracies** on **target** models. Optional flags only control **inputs**, **which anchor-sampling preset** to use, and **where to persist intermediate artifacts**.

Run from the repository root (or ensure `PYTHONPATH` includes the project so `experiments`, `utils`, etc. resolve):

```bash
python scripts/two_stages_v2.py \
  --source_model_outputs_path /path/to/source.pkl \
  --target_model_outputs_path /path/to/target.pkl \
  [--anchors_save_path ...] \
  [--weights_save_path ...] \
  [--transform_save_path ...] \
  [--anchor_preset mmlu_rank|mmlu_mae|hellaswag_rank|hellaswag_mae]
```

## CLI arguments

### `--source_model_outputs_path` (string, default: `None`)

Path to a pickle file with **source** (training pool) model outputs. The script loads it with `load_pickle` and expects the same structure used elsewhere in this codebase, including at least:

- **`predictions`**: used to sample anchors and to build train embeddings.
- **`correctness`**: used to compute per-model **true accuracies** (with scenario balancing via `Scenarios` / `Datapoints`).
- **`Scenarios`**, **`Datapoints`**, **`Models`**: scenario metadata, balancing weights, and model index mapping.

If omitted, `load_pickle(None)` will fail at runtime—you must pass a valid path for a real run.

### `--target_model_outputs_path` (string, default: `None`)

Path to a pickle file with **target** (evaluation) model outputs, same general schema as source (notably **`predictions`** and **`Models`**). Target predictions are sliced to the **same anchor indices** as the source, then embedded with the **same transform** fitted on source anchors. Ground-truth accuracies for printing and metrics are derived from target **`correctness`** (and balancing), analogous to source.

### `--anchors_save_path` (string, default: `None`)

If set, writes a pickle of **`anchor_points_new`**: the integer indices of the **100** anchor items chosen by the primary sampling strategy (see `--anchor_preset`). These indices align rows of `source_outputs["predictions"]` and `target_outputs["predictions"]`. If omitted, anchors are still computed but not saved.

### `--weights_save_path` (string, default: `None`)

If set, writes a pickle of **`fitted_weights`**: nested structure keyed by sampling name and anchor count, holding the fitted **`RandomForestRegressor`** (`n_estimators=100`, keyed as `RandomForestRegressor_100`) used to map embeddings to accuracy. Fitted models may also be cached on disk under a directory derived from the internal `cache_path` passed to `make_fitted_weights_v2` (see script). If omitted, weights are still computed but not dumped to this path.

### `--transform_save_path` (string, default: `None`)

If set, writes a pickle of **`transform_v2`**: the embedding transform returned by `compute_embedding` when fitting on **source** predictions restricted to the chosen anchors (PCA to 256 dimensions after softmax). Target embeddings use this same transform so train and test live in the same space. If omitted, the transform is still used in-process but not saved.

### `--anchor_preset` (string, default: `None`)

Selects **how anchor questions are sampled** (disagreement flavor and optional **guiding models**). The script always uses **100** anchors (`number_items = [100]`); the preset changes **which disagreement sampler string** and **which model indices** guide pairwise disagreement (PDS), not the count.

| Value | Effect |
|--------|--------|
| *(omit or `None`)* | `guiding_models = None`, `sampling_names = ["high-disagreement"]` |
| `mmlu_rank` | 100 MMLU-specific guiding model indices, `high-disagreement@100+nonstratified` |
| `mmlu_mae` | 2 guiding models `(268, 250)`, `high-disagreement@2` |
| `hellaswag_rank` | 10 Hellaswag-specific guiding models, `high-disagreement@10+nonstratified` |
| `hellaswag_mae` | `guiding_models = None`, `high-disagreement+nonstratified` |

The script also runs a **stratified random** anchor draw internally for baselines; that is not controlled by this flag.

---

**Note:** Random seed is fixed via `apply_random_seed(RANDOM_SEED)` from `experiments`. There is no CLI flag for seed, PCA size (fixed at 256), or anchor count (fixed at 100) in the current script.
