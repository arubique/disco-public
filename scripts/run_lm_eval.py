"""
Thin wrapper to run the lm-evaluation-harness CLI from the git submodule in external/.

Example usage (from the repository root):

    python scripts/run_lm_eval.py \\
        --batch_size=8 \\
        --device=cuda:0 \\
        --gen_kwargs=max_gen_toks=128,output_scores=True,return_dict_in_generate=True \\
        --model=hf \\
        --model_args=pretrained=abacusai/MetaMath-bagel-34b-v0.2-c1500,trust_remote_code=True \\
        --num_fewshot=10 \\
        --output_path=/weka/oh/arubinstein17/github/lm-evaluation-harness/output/hellaswag_050126/llama13b_r71 \\
        --tasks=hellaswag_prompts \\
        --log_samples
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_external_on_path() -> None:
    """
    Prepend the `external/lm-evaluation-harness` directory (git submodule)
    to sys.path so that `import lm_eval` resolves to that copy.
    """

    repo_root = Path(__file__).resolve().parents[1]
    submodule_root = repo_root / "external" / "lm-evaluation-harness"

    submodule_str = str(submodule_root)
    if submodule_str not in sys.path:
        sys.path.insert(0, submodule_str)


def main() -> None:
    _ensure_external_on_path()

    # Import after sys.path modification so we pick up the submodule package
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
