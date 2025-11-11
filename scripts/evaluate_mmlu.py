import argparse
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

CHOICE_LETTERS = ["A", "B", "C", "D"]


def build_prompt(question, choices, fewshot_examples=None):
    # fewshot_examples: list of (q, choices, correct_letter)
    parts = []
    if fewshot_examples:
        parts.append(
            "The following are multiple-choice questions (with answers) about many academic subjects.\n"
        )
        for fq, fchoices, fans in fewshot_examples:
            parts.append(
                format_qa_block(fq, fchoices, fans, include_answer=True)
            )
    parts.append(format_qa_block(question, choices, None, include_answer=False))
    return "\n".join(parts)


def format_qa_block(q, choices, answer_letter=None, include_answer=True):
    s = q.strip() + "\n"
    for i, c in enumerate(choices):
        s += f"{CHOICE_LETTERS[i]}. {c.strip()}\n"
    if include_answer and answer_letter is not None:
        s += f"Answer: {answer_letter}\n\n"
    else:
        s += "Answer:"
    return s


def extract_choice(text):
    # Look for standalone A/B/C/D as the first such token
    m = re.search(r"\b([ABCD])\b", text.strip(), re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def get_fewshot(ds_subject, k=5):
    # Take first k examples from the same subject for few-shot
    few = []
    for ex in ds_subject.select(range(min(k, len(ds_subject)))):
        correct = CHOICE_LETTERS[ex["answer"]]
        few.append((ex["question"], ex["choices"], correct))
    return few


def evaluate_mmlu(
    model_id,
    subset="all",
    num_fewshot=5,
    max_samples=None,
    batch_size=4,
    max_new_tokens=4,
    device=None,
):
    print(f"Loading dataset cais/mmlu ({subset})...")
    ds = load_dataset("cais/mmlu", subset, split="test")  # uses MMLU test split

    if max_samples:
        ds = ds.select(range(max_samples))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    total, correct = 0, 0

    # Precompute few-shot by subject (MMLU-style per-subject)
    by_subject = {}
    for i in range(len(ds)):
        subj = ds[i]["subject"]
        by_subject.setdefault(subj, []).append(i)

    for subj, indices in by_subject.items():
        subject_ds = ds.select(indices)
        fewshot = (
            get_fewshot(subject_ds, k=num_fewshot) if num_fewshot > 0 else None
        )

        for start in range(0, len(subject_ds), batch_size):
            batch = subject_ds[start : start + batch_size]

            prompts = [
                build_prompt(q, choices, fewshot_examples=fewshot)
                for q, choices in zip(batch["question"], batch["choices"])
            ]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            gen = out[:, enc["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for ex, gen_text in zip(batch, texts):
                pred = extract_choice(gen_text)
                gold = CHOICE_LETTERS[ex["answer"]]

                if pred == gold:
                    correct += 1
                total += 1

    acc = 100.0 * correct / total
    print(f"MMLU accuracy: {acc:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="MMLU subset, e.g. all, abstract_algebra, etc.",
    )
    parser.add_argument("--num_fewshot", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    evaluate_mmlu(
        model_id=args.model_id,
        subset=args.subset,
        num_fewshot=args.num_fewshot,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
