#!/usr/bin/env python
"""
Estimate wall-clock time for evaluating 8B LLMs on MMLU.

- Uses Hugging Face models (AutoModelForCausalLM + AutoTokenizer)
- Uses Hugging Face MMLU dataset (cais/mmlu by default)
- Samples a subset of questions, measures generation time
- Optional --full-eval flag runs on every sample and reports accuracy
- Reports:
    * tokens/sec
    * mean / std time per question
    * extrapolated full-MMLU eval time

Install requirements:
    pip install torch transformers datasets accelerate

Run:
    python estimate_mmlu_time.py
"""

import argparse
import re
import time
import math
import random
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# CONFIG SECTION
# -------------------------

# Put your 8B-ish models here (or whatever you want to test)
MODEL_NAMES: List[str] = [
    # Examples; replace with what you actually use
    # "meta-llama/Meta-Llama-3-8B-Instruct",
    # "logicker/SkkuDataScienceGlobal-10.7b",
    # "meta-llama/Llama-2-13b-hf@2023_08_19T22_35_38.117975",
    "meta-math/MetaMath-Mistral-7B"
    # "some-org/some-8b-model",
]

# Total number of MMLU questions to sample for timing per model
NUM_QUESTIONS = (
    200  # 50–200 is a good range; can be smaller if you're impatient
)

# Batch size for generation
BATCH_SIZE = 8

# Max new tokens to generate per question (we just need few tokens to time)
MAX_NEW_TOKENS = 4

# Device: "cuda" or "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Approximate number of questions in full MMLU
# (adjust if you know the exact size for your split/version)
MMLU_FULL_SIZE = 15000

CHOICE_LETTERS = ["A", "B", "C", "D"]

# subset names we know exist in cais/mmlu
SUBJECT_CONFIGS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def extract_choice(text: str) -> Optional[str]:
    """Extract the first standalone multiple-choice letter (A/B/C/D)."""
    if not text:
        return None
    match = re.search(r"\b([ABCD])\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def normalize_subject_name(name: Optional[str]) -> Optional[str]:
    """
    Accepts either a raw MMLU subject (e.g., 'abstract_algebra') or a harness-style
    scenario key like 'harness_hendrycksTest_abstract_algebra_5'. Returns the
    subject name understood by cais/mmlu, or None if not provided.
    """
    if name is None:
        return None
    if name in SUBJECT_CONFIGS or name == "all":
        return name
    match = re.match(r"^harness_hendrycksTest_(.+)_(\d+)$", name)
    if match:
        subject = match.group(1)
        if subject not in SUBJECT_CONFIGS:
            raise ValueError(
                f"Derived subject '{subject}' is not in the supported MMLU subject list."
            )
        return subject
    raise ValueError(
        "Unsupported scenario/subject name. Provide either an MMLU subject such as "
        "'abstract_algebra' or a harness scenario like "
        "'harness_hendrycksTest_abstract_algebra_5'."
    )


# -------------------------
# DATA LOADING
# -------------------------


def load_mmlu_questions(
    num_questions: int,
    full_eval: bool = False,
    subject_name: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Load MMLU (cais/mmlu) from Hugging Face.

    Returns a list of dicts with:
      - prompt: formatted multiple-choice question
      - answer: correct letter (A/B/C/D)

    If `full_eval` is False, randomly sample `num_questions` prompts. If
    `subject_name` is provided, restrict sampling to that subject/scenario.
    """

    subject_filter = normalize_subject_name(subject_name)

    all_examples: List[Dict[str, str]] = []

    if full_eval:
        if subject_filter is None or subject_filter == "all":
            datasets_to_scan = [load_dataset("cais/mmlu", "all", split="test")]
        else:
            datasets_to_scan = [
                load_dataset("cais/mmlu", subject_filter, split="test")
            ]
    else:
        if subject_filter is not None:
            if subject_filter == "all":
                datasets_to_scan = [
                    load_dataset("cais/mmlu", "all", split="test")
                ]
            else:
                datasets_to_scan = [
                    load_dataset("cais/mmlu", subject_filter, split="test")
                ]
        else:
            random_subjects = random.sample(
                SUBJECT_CONFIGS, k=min(10, len(SUBJECT_CONFIGS))
            )
            datasets_to_scan = [
                load_dataset("cais/mmlu", subj, split="test")
                for subj in random_subjects
            ]

    for ds in datasets_to_scan:
        for ex in ds:
            question = ex["question"]
            choices = ex["choices"]  # list of strings
            options_str = "\n".join(
                f"{letter}. {text}"
                for letter, text in zip(CHOICE_LETTERS, choices)
            )
            prompt = (
                f"Q: {question}\n"
                f"{options_str}\n"
                f"Answer with the letter of the correct option (A, B, C, or D)."
            )
            correct_letter = CHOICE_LETTERS[ex["answer"]]
            all_examples.append({"prompt": prompt, "answer": correct_letter})

    if not full_eval and subject_filter not in ("all",):
        random.shuffle(all_examples)
        all_examples = all_examples[:num_questions]

    return all_examples


# -------------------------
# TIMING LOGIC
# -------------------------


def time_model_on_prompts(
    model_name: str,
    examples: List[Dict[str, str]],
    batch_size: int = 8,
    max_new_tokens: int = 4,
    device: str = "cuda",
    collect_accuracy: bool = False,
) -> Dict[str, Any]:
    """
    Load model + tokenizer, run generation on `prompts`, and measure:
      - total wall-clock time
      - per-question times (approximate)
      - tokens/sec (prompt + generated)
    Returns a dict with stats.
    """

    print(f"\n=== Loading model: {model_name} ===")
    if "@" in model_name:
        model_name = model_name.split("@")[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # For many causal LMs, pad_token is not set; reuse eos_token
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    # Warm-up: a few dummy passes so compilation/caching is done
    print("Running warm-up...")
    warmup_texts = ["Warm-up question A", "Warm-up question B"]
    warmup_inputs = tokenizer(
        warmup_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        _ = model.generate(
            **warmup_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    print("Starting timed runs...")
    total_time = 0.0
    total_tokens = 0
    per_question_times: List[float] = []

    num_prompts = len(examples)
    num_correct = 0
    for start_idx in tqdm(range(0, num_prompts, batch_size)):
        # batch_examples = examples[start_idx:start_idx + batch_size]
        batch_examples = {
            "prompt": examples["prompt"][start_idx : start_idx + batch_size],
            "answer": examples["answer"][start_idx : start_idx + batch_size],
            "choices": examples["choices"][start_idx : start_idx + batch_size],
        }
        batch_prompts = [ex for ex in batch_examples["prompt"]]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        batch_size_actual = len(batch_examples)

        with torch.no_grad():
            t0 = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            t1 = time.perf_counter()

        batch_time = t1 - t0
        total_time += batch_time

        # Approximate: count all non-pad tokens in outputs (prompt + generated)
        pad_id = tokenizer.pad_token_id
        tokens_in_batch = int((outputs != pad_id).sum().item())
        total_tokens += tokens_in_batch

        # Approximate per-question time: same time for each sample in the batch
        t_per_sample = batch_time / batch_size_actual
        per_question_times.extend([t_per_sample] * batch_size_actual)

        if collect_accuracy:
            batch_answers = [ex for ex in batch_examples["answer"]]
            generated = outputs[:, inputs["input_ids"].shape[1] :]
            gen_texts = tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
            for gen_text, gold in zip(gen_texts, batch_answers):
                pred = extract_choice(gen_text)
                if pred is not None and pred == gold:
                    num_correct += 1

    # Stats
    per_question_times = per_question_times[:num_prompts]  # safety
    mean_t = float(np.mean(per_question_times))
    std_t = float(np.std(per_question_times))

    tokens_per_sec = (
        total_tokens / total_time if total_time > 0 else float("nan")
    )

    stats = {
        "model_name": model_name,
        "num_questions": num_prompts,
        "total_time_sec": total_time,
        "mean_time_per_question_sec": mean_t,
        "std_time_per_question_sec": std_t,
        "tokens_total": total_tokens,
        "tokens_per_sec": tokens_per_sec,
    }

    if collect_accuracy:
        accuracy = (
            num_correct / num_prompts if num_prompts > 0 else float("nan")
        )
        stats["num_correct"] = num_correct
        stats["accuracy"] = accuracy
    else:
        stats["num_correct"] = None
        stats["accuracy"] = None

    return stats


def pretty_time(seconds: float) -> str:
    """Format seconds as H:MM:SS for human readability."""
    if math.isinf(seconds) or math.isnan(seconds):
        return "N/A"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


# -------------------------
# MAIN
# -------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate timing (and optional accuracy) for MMLU evaluation."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model IDs to evaluate (defaults to MODEL_NAMES list in the script).",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=NUM_QUESTIONS,
        help="Number of questions to sample when not using --full-eval.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size to use during generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Max new tokens to generate per question.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Device identifier for torch (e.g., cuda, cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting subject subsets when sampling.",
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="If set, iterate over every question in cais/mmlu and compute accuracy.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help=(
            "Restrict evaluation to a single MMLU subject or harness scenario "
            "(e.g., 'abstract_algebra' or 'harness_hendrycksTest_abstract_algebra_5')."
        ),
    )
    parser.add_argument(
        "--openllm-scenarios",
        type=str,
        default=None,
        help="OpenLLM scenarios to evaluate.",
    )
    return parser.parse_args()


def load_openllm_scenarios(
    model_name: str,
    scenarios: List[str],
) -> List[Dict[str, str]]:
    """
    Load OpenLLM scenarios from Hugging Face.
    """
    creator, model = tuple(model_name.split("/"))
    if "@" in model:
        model, timestamp = tuple(model.split("@"))
    else:
        timestamp = "latest"

    model_details_id = "open-llm-leaderboard/details_{:}__{:}".format(
        creator, model
    )
    scenarios_list = scenarios.split(",")
    examples = {"prompt": [], "answer": [], "choices": []}
    for scenario in scenarios_list:
        aux = load_dataset(model_details_id, scenario)
        examples["prompt"].extend(list(aux[timestamp]["full_prompt"]))
        examples["answer"].extend(list(aux[timestamp]["gold"]))
        examples["choices"].extend(list(aux[timestamp]["choices"]))
    return examples


def main():
    args = parse_args()
    random.seed(args.seed)

    model_names = args.models if args.models is not None else MODEL_NAMES
    device = args.device

    if args.openllm_scenarios is None:
        print("Loading MMLU prompts...")
        subject_filter = (
            normalize_subject_name(args.scenario)
            if args.scenario is not None
            else None
        )
        examples = load_mmlu_questions(
            num_questions=args.num_questions,
            full_eval=args.full_eval,
            subject_name=subject_filter,
        )
        if subject_filter:
            dataset_note = f" [subject: {subject_filter}]"
        print(f"Loaded {len(examples)} prompts for timing{dataset_note}.\n")
    else:
        examples = None
    dataset_note = " (full dataset)" if args.full_eval else ""

    all_model_stats: List[Dict[str, Any]] = []

    for model_name in model_names:
        if examples is None:
            assert args.openllm_scenarios is not None
            examples = load_openllm_scenarios(
                model_name=model_name,
                scenarios=args.openllm_scenarios,
            )
            print(
                f"Loaded {len(examples)} prompts for {args.openllm_scenarios} of model {model_name}.\n"
            )
        stats = time_model_on_prompts(
            model_name=model_name,
            examples=examples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            collect_accuracy=args.full_eval,
        )

        mean_t_q = stats["mean_time_per_question_sec"]
        t_full_est = (
            stats["total_time_sec"]
            if args.full_eval
            else mean_t_q * MMLU_FULL_SIZE
        )
        stats["full_mmlu_time_est_sec"] = t_full_est

        print("\n--- Results for model:", model_name, "---")
        print(f"Device: {device}")
        print(f"Questions processed: {stats['num_questions']}")
        print(f"Total time: {stats['total_time_sec']:.2f} s")
        print(f"Mean time / question: {mean_t_q:.4f} s")
        print(
            f"Std time / question: {stats['std_time_per_question_sec']:.4f} s"
        )
        print(f"Tokens total (prompt + gen): {stats['tokens_total']}")
        print(f"Throughput: {stats['tokens_per_sec']:.2f} tokens/s")
        if args.full_eval and stats["accuracy"] is not None:
            acc_pct = stats["accuracy"] * 100.0
            print(
                f"Accuracy: {acc_pct:.2f}% "
                f"({stats['num_correct']}/{stats['num_questions']})"
            )
        duration_label = (
            "Actual full MMLU time"
            if args.full_eval
            else f"Estimated full MMLU time (~{MMLU_FULL_SIZE} questions)"
        )
        print(
            f"{duration_label}: {pretty_time(t_full_est)} "
            f"({t_full_est:.1f} s)"
        )

        all_model_stats.append(stats)

    if len(all_model_stats) > 1:
        print("\n=== Aggregate across models ===")
        full_times = np.array(
            [s["full_mmlu_time_est_sec"] for s in all_model_stats], dtype=float
        )
        mean_full = full_times.mean()
        std_full = full_times.std(ddof=1) if len(full_times) > 1 else 0.0

        label = (
            "Mean actual full-MMLU time"
            if args.full_eval
            else "Mean estimated full-MMLU time"
        )
        print(f"{label}: {pretty_time(mean_full)} ({mean_full:.1f} s)")
        pct = (std_full / mean_full * 100.0) if mean_full != 0 else float("nan")
        print(
            f"Std across models:                {std_full:.1f} s "
            f"(~{pct:.1f}% of mean)"
        )


if __name__ == "__main__":
    main()
