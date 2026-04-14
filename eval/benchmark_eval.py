"""
Benchmark Evaluation Script for Synnapse-Qwen2.5-3B
====================================================
Evaluates the fine-tuned model against the base Qwen2.5-3B-Instruct
on three standard LLM benchmarks:

  1. MMLU        — 5-shot MCQ, log-likelihood scoring       (Accuracy)
  2. Big-Bench Hard — 3-shot CoT, exact-match generation    (Accuracy)
  3. TruthfulQA  — 0-shot MC1, log-likelihood scoring       (MC1 Accuracy)

Usage:
    python -m eval.benchmark_eval                  # default (200 MMLU, 100 BBH, full TruthfulQA)
    python -m eval.benchmark_eval --samples 20     # quick smoke-test
    python -m eval.benchmark_eval --full            # evaluate on complete datasets
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
FT_MODEL_NAME   = "Wvidit/Qwen-3-grpo"
RESULTS_PATH    = Path(__file__).parent / "benchmark_results.json"

MMLU_CHOICES   = ["A", "B", "C", "D"]

# BBH 3-shot exemplars (generic CoT style — works across subtasks)
BBH_FEW_SHOT = [
    {
        "input":  "If the first two statements are true, is the third statement true?\n"
                  "Statement 1: All dogs are animals.\n"
                  "Statement 2: All animals are living things.\n"
                  "Statement 3: All dogs are living things.",
        "target": "Yes"
    },
    {
        "input":  "Which of the following is a valid conclusion?\n"
                  "Premise 1: If it rains, the ground gets wet.\n"
                  "Premise 2: It rained.\n"
                  "Conclusion: The ground is wet.",
        "target": "Yes"
    },
    {
        "input":  "Choose the most logical completion: The chef cooked the meal and then ___.\n"
                  "(A) served it   (B) ignored it   (C) deleted it",
        "target": "(A)"
    },
]

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks emitted by Qwen models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def load_model_and_tokenizer(model_name: str, device_map: str = "auto"):
    """Load a causal LM + tokenizer with bfloat16 weights."""
    print(f"  Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def build_chat_prompt(system: str, user: str, tokenizer) -> str:
    """Apply the ChatML template that Qwen expects."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def score_completions(prompt: str, completions: list[str], model, tokenizer) -> list[float]:
    """
    Return the mean log-probability that `model` assigns to each candidate
    completion, given `prompt` as prefix.
    """
    scores = []
    for comp in completions:
        full_text = prompt + comp
        enc = tokenizer(full_text, return_tensors="pt").to(model.device)
        prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

        outputs = model(**enc)
        logits  = outputs.logits  # (1, seq_len, vocab)

        # Shift: logits[t] predicts token[t+1]
        shift_logits = logits[0, prompt_len - 1 : -1, :]
        shift_labels = enc["input_ids"][0, prompt_len:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lps = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

        # Mean log-prob (length-normalised)
        scores.append(token_lps.mean().item())
    return scores


@torch.no_grad()
def generate_text(prompt: str, model, tokenizer, max_new_tokens: int = 256) -> str:
    """Generate text from a prompt, stripping think tags."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # greedy for reproducibility
        temperature=1.0,
    )
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    return strip_think_tags(tokenizer.decode(new_tokens, skip_special_tokens=True))


# ────────────────────────────────────────────────────────────────────────────────
# Benchmark 1 – MMLU (5-shot, log-likelihood MCQ)
# ────────────────────────────────────────────────────────────────────────────────

def _build_mmlu_few_shot_prefix(few_shot_examples: list[dict]) -> str:
    """Build the 5-shot exemplar prefix from MMLU examples."""
    lines = []
    for ex in few_shot_examples:
        q = ex["question"]
        choices = [ex[f"choices"][i] for i in range(4)]
        ans_idx = ex["answer"]
        lines.append(
            f"Question: {q}\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
            f"Answer: {MMLU_CHOICES[ans_idx]}"
        )
    return "\n\n".join(lines)


def evaluate_mmlu(model, tokenizer, n_samples: int | None = 200) -> dict:
    """Evaluate MMLU accuracy using log-likelihood scoring."""
    print("\n" + "=" * 60)
    print("BENCHMARK: MMLU (Massive Multitask Language Understanding)")
    print("=" * 60)

    ds = load_dataset("cais/mmlu", "all", split="test")
    ds_dev = load_dataset("cais/mmlu", "all", split="dev")

    # Use first 5 dev examples as few-shot
    few_shot_prefix = _build_mmlu_few_shot_prefix(
        [ds_dev[i] for i in range(min(5, len(ds_dev)))]
    )

    if n_samples is not None:
        ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    correct = 0
    total   = 0
    subject_stats: dict[str, dict] = {}

    for i, ex in enumerate(ds):
        question = ex["question"]
        choices  = [ex["choices"][j] for j in range(4)]
        label    = ex["answer"]  # int 0-3
        subject  = ex.get("subject", "unknown")

        user_text = (
            f"{few_shot_prefix}\n\n"
            f"Question: {question}\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
            f"Answer:"
        )

        prompt = build_chat_prompt(
            "You are a knowledgeable assistant. Answer each multiple-choice question with only the letter of the correct option.",
            user_text,
            tokenizer,
        )

        # Score each option letter
        candidate_scores = score_completions(
            prompt, [f" {c}" for c in MMLU_CHOICES], model, tokenizer
        )
        pred = int(torch.tensor(candidate_scores).argmax().item())

        if pred == label:
            correct += 1

        # Per-subject tracking
        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}
        subject_stats[subject]["total"]   += 1
        subject_stats[subject]["correct"] += int(pred == label)

        total += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ds)}]  running acc = {correct/total:.2%}")

    accuracy = correct / total if total else 0
    print(f"\n  MMLU Accuracy: {correct}/{total} = {accuracy:.2%}")

    return {
        "benchmark": "MMLU",
        "accuracy":  round(accuracy, 4),
        "correct":   correct,
        "total":     total,
        "per_subject": {
            k: round(v["correct"] / v["total"], 4) if v["total"] else 0
            for k, v in subject_stats.items()
        },
    }


# ────────────────────────────────────────────────────────────────────────────────
# Benchmark 2 – Big-Bench Hard (3-shot CoT, exact-match)
# ────────────────────────────────────────────────────────────────────────────────

def _bbh_extract_answer(text: str) -> str:
    """Extract the final answer from BBH CoT output."""
    text = text.strip()
    # Look for "the answer is X" pattern
    m = re.search(r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)", text, re.I)
    if m:
        return m.group(1).strip().rstrip(".")
    # Fallback: last line
    lines = text.strip().splitlines()
    return lines[-1].strip() if lines else ""


def evaluate_bbh(model, tokenizer, n_samples: int | None = 100) -> dict:
    """Evaluate Big-Bench Hard accuracy using few-shot CoT generation."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Big-Bench Hard (BBH)")
    print("=" * 60)

    # Load a BBH subset — boolean_expressions is a reliable subtask
    subtasks_to_try = [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "movie_recommendation",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ]

    all_examples = []
    for subtask in subtasks_to_try:
        try:
            sub_ds = load_dataset("maveriq/bigbenchhard", subtask, split="train")
            for ex in sub_ds:
                all_examples.append({**ex, "_subtask": subtask})
        except Exception:
            continue  # subtask may not be available

    if not all_examples:
        print("  WARNING: Could not load any BBH subtasks. Skipping.")
        return {"benchmark": "BBH", "accuracy": None, "correct": 0, "total": 0}

    import random
    random.seed(42)
    random.shuffle(all_examples)
    if n_samples is not None:
        all_examples = all_examples[:n_samples]

    # Build few-shot prefix
    few_shot_lines = []
    for fs in BBH_FEW_SHOT:
        few_shot_lines.append(
            f"Q: {fs['input']}\n"
            f"A: Let's think step by step. The answer is {fs['target']}."
        )
    few_shot_prefix = "\n\n".join(few_shot_lines)

    correct = 0
    total   = 0
    subtask_stats: dict[str, dict] = {}

    for i, ex in enumerate(all_examples):
        inp    = ex["input"]
        target = ex["target"].strip()
        subtask = ex["_subtask"]

        user_text = (
            f"{few_shot_prefix}\n\n"
            f"Q: {inp}\n"
            f"A: Let's think step by step."
        )

        prompt = build_chat_prompt(
            "You are a helpful assistant. Solve the problem step by step and give a final answer.",
            user_text,
            tokenizer,
        )

        response = generate_text(prompt, model, tokenizer, max_new_tokens=256)
        pred = _bbh_extract_answer(response)

        # Normalize comparison
        is_correct = pred.strip().lower() == target.strip().lower()
        if is_correct:
            correct += 1

        if subtask not in subtask_stats:
            subtask_stats[subtask] = {"correct": 0, "total": 0}
        subtask_stats[subtask]["total"] += 1
        subtask_stats[subtask]["correct"] += int(is_correct)

        total += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(all_examples)}]  running acc = {correct/total:.2%}")

    accuracy = correct / total if total else 0
    print(f"\n  BBH Accuracy: {correct}/{total} = {accuracy:.2%}")

    return {
        "benchmark":    "Big-Bench Hard",
        "accuracy":     round(accuracy, 4),
        "correct":      correct,
        "total":        total,
        "per_subtask": {
            k: round(v["correct"] / v["total"], 4) if v["total"] else 0
            for k, v in subtask_stats.items()
        },
    }


# ────────────────────────────────────────────────────────────────────────────────
# Benchmark 3 – TruthfulQA (MC1, log-likelihood)
# ────────────────────────────────────────────────────────────────────────────────

def evaluate_truthfulqa(model, tokenizer, n_samples: int | None = None) -> dict:
    """Evaluate TruthfulQA MC1 accuracy using log-likelihood scoring."""
    print("\n" + "=" * 60)
    print("BENCHMARK: TruthfulQA (MC1)")
    print("=" * 60)

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")

    if n_samples is not None:
        ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    correct = 0
    total   = 0

    for i, ex in enumerate(ds):
        question = ex["question"]
        # MC1: single correct answer is first, rest are wrong
        mc1_targets  = ex["mc1_targets"]
        choices      = mc1_targets["choices"]
        labels       = mc1_targets["labels"]   # list of 0/1, exactly one 1

        if not choices or sum(labels) == 0:
            continue

        true_idx = labels.index(1)

        user_text = f"Question: {question}\nAnswer:"

        prompt = build_chat_prompt(
            "You are a helpful and truthful assistant. Answer the question accurately.",
            user_text,
            tokenizer,
        )

        # Score each candidate completion
        candidate_scores = score_completions(
            prompt, [f" {c}" for c in choices], model, tokenizer
        )
        pred = int(torch.tensor(candidate_scores).argmax().item())

        if pred == true_idx:
            correct += 1

        total += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ds)}]  running acc = {correct/total:.2%}")

    accuracy = correct / total if total else 0
    print(f"\n  TruthfulQA MC1 Accuracy: {correct}/{total} = {accuracy:.2%}")

    return {
        "benchmark": "TruthfulQA (MC1)",
        "accuracy":  round(accuracy, 4),
        "correct":   correct,
        "total":     total,
    }

# ────────────────────────────────────────────────────────────────────────────────
# Main Evaluation Driver
# ────────────────────────────────────────────────────────────────────────────────

def run_benchmarks(n_mmlu: int | None, n_bbh: int | None, n_tqa: int | None):
    """Run all benchmarks for both models and produce comparison report."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Synnapse Benchmark Evaluation Suite                      ║")
    print("║   Base Model  : Qwen/Qwen2.5-3B-Instruct                  ║")
    print("║   Fine-Tuned  : Wvidit/Qwen-3-grpo                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    all_results = {"timestamp": datetime.now().isoformat(), "models": {}}

    for tag, model_name in [("base", BASE_MODEL_NAME), ("fine_tuned", FT_MODEL_NAME)]:
        print(f"\n{'━' * 60}")
        print(f"  Evaluating: {model_name}  ({tag})")
        print(f"{'━' * 60}")
        model, tokenizer = load_model_and_tokenizer(model_name)

        results = {}
        start = time.time()

        # 1) MMLU
        results["mmlu"] = evaluate_mmlu(model, tokenizer, n_samples=n_mmlu)

        # 2) Big-Bench Hard
        results["bbh"]  = evaluate_bbh(model, tokenizer, n_samples=n_bbh)

        # 3) TruthfulQA
        results["truthfulqa"] = evaluate_truthfulqa(model, tokenizer, n_samples=n_tqa)

        results["total_time_sec"] = round(time.time() - start, 1)
        all_results["models"][tag] = results

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                     BENCHMARK COMPARISON                           ║")
    print("╠══════════════════════════╦════════════════╦════════════════╦════════╣")
    print("║ Benchmark                ║ Base Model     ║ Fine-Tuned     ║ Delta  ║")
    print("╠══════════════════════════╬════════════════╬════════════════╬════════╣")

    benchmarks = [
        ("MMLU",                "mmlu"),
        ("Big-Bench Hard",      "bbh"),
        ("TruthfulQA (MC1)",    "truthfulqa"),
    ]

    for display_name, key in benchmarks:
        base_acc = all_results["models"]["base"][key]["accuracy"]
        ft_acc   = all_results["models"]["fine_tuned"][key]["accuracy"]

        if base_acc is not None and ft_acc is not None:
            delta = ft_acc - base_acc
            delta_str = f"{delta:+.2%}"
        else:
            delta_str = "N/A"

        base_str = f"{base_acc:.2%}" if base_acc is not None else "N/A"
        ft_str   = f"{ft_acc:.2%}"   if ft_acc   is not None else "N/A"

        print(f"║ {display_name:<24} ║ {base_str:>14} ║ {ft_str:>14} ║ {delta_str:>6} ║")

    print("╚══════════════════════════╩════════════════╩════════════════╩════════╝")
    print()

    base_time = all_results["models"]["base"]["total_time_sec"]
    ft_time   = all_results["models"]["fine_tuned"]["total_time_sec"]
    print(f"  Total evaluation time:  base={base_time}s  |  fine-tuned={ft_time}s")
    print()

    # ── Save results ─────────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to: {RESULTS_PATH}")

    return all_results


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Synnapse model on MMLU, Big-Bench Hard, and TruthfulQA"
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Override per-benchmark sample count (for quick testing)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run on the full benchmark datasets (slow!)"
    )
    parser.add_argument(
        "--mmlu-samples",  type=int, default=200,
        help="Number of MMLU samples (default: 200)"
    )
    parser.add_argument(
        "--bbh-samples",   type=int, default=100,
        help="Number of BBH samples (default: 100)"
    )
    parser.add_argument(
        "--tqa-samples",   type=int, default=None,
        help="Number of TruthfulQA samples (default: all)"
    )
    args = parser.parse_args()

    # --samples overrides per-benchmark settings
    if args.samples is not None:
        n_mmlu = n_bbh = n_tqa = args.samples
    elif args.full:
        n_mmlu = n_bbh = n_tqa = None
    else:
        n_mmlu = args.mmlu_samples
        n_bbh  = args.bbh_samples
        n_tqa  = args.tqa_samples

    run_benchmarks(n_mmlu, n_bbh, n_tqa)


if __name__ == "__main__":
    main()
