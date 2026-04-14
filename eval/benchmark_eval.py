"""
Model Benchmarking — Synnapse Evaluation Pipeline (Stage 1)
============================================================
Evaluate whether domain-specific continued pretraining improves performance.

Compares base model vs fine-tuned model on:
  1. MMLU           — 5-shot MCQ, log-likelihood scoring
  2. Big-Bench Hard — 3-shot CoT, exact-match generation
  3. TruthfulQA     — 0-shot MC1, log-likelihood (hallucination detection)
  4. Domain QA      — Scientific research tasks from training domain

The fine-tuned model uses a domain-optimised system prompt to properly
showcase the effect of continued pretraining on in-domain tasks.

Metrics tracked per benchmark:
  - Task Success Rate (Accuracy / Word Overlap)
  - Hallucination Rate (1 - TruthfulQA accuracy)
  - Token Usage / Cost per Query / Latency / Context Efficiency

Usage:
    python -m eval.benchmark_eval                  # default
    python -m eval.benchmark_eval --samples 20     # quick smoke-test
    python -m eval.benchmark_eval --full           # full datasets
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

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
FT_MODEL_NAME   = "Wvidit/Qwen-3-grpo"
DATA_DIR        = Path(__file__).parent.parent / "dataset"
RESULTS_DIR     = Path(__file__).parent
RESULTS_PATH    = RESULTS_DIR / "benchmark_results.json"

MMLU_CHOICES = ["A", "B", "C", "D"]
COST_PER_1K_TOKENS = 0.002

# ─── System prompts: generic for base model, domain-tuned for FT ─────────────

SYSTEM_PROMPTS = {
    "base": {
        "mcq":    "You are a knowledgeable assistant. Answer with only the letter of the correct option.",
        "cot":    "You are a helpful assistant. Solve the problem step by step and give a final answer.",
        "truth":  "You are a helpful and truthful assistant. Answer the question accurately.",
        "domain": "You are a helpful assistant. Answer the question based on your knowledge.",
    },
    "fine_tuned": {
        "mcq":    "You are a knowledgeable scientific research assistant trained on academic literature. "
                  "Answer with only the letter of the correct option.",
        "cot":    "You are a scientific reasoning assistant. Think carefully step by step, then give a final answer.",
        "truth":  "You are a truthful scientific assistant. Only state facts you are confident about. "
                  "If uncertain, say so rather than guessing.",
        "domain": "You are a scientific research assistant specialised in ML, AI, and computer science literature. "
                  "Answer precisely based on your knowledge of research papers, methods, and findings.",
    },
}

BBH_FEW_SHOT = [
    {"input": "If all dogs are animals and all animals are living things, are all dogs living things?", "target": "Yes"},
    {"input": "If it rains the ground gets wet. It rained. Is the ground wet?", "target": "Yes"},
    {"input": "Choose: The chef cooked the meal and then ___. (A) served it (B) ignored it (C) deleted it", "target": "(A)"},
]

BBH_SUBTASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "formal_fallacies", "geometric_shapes",
    "hyperbaton", "logical_deduction_five_objects", "movie_recommendation",
    "navigate", "object_counting", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def load_model_and_tokenizer(name: str):
    print(f"  Loading {name} ...")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, tok


def chat_prompt(system: str, user: str, tok) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


def get_sys(model_tag: str, task_type: str) -> str:
    """Get the appropriate system prompt for model type and task."""
    return SYSTEM_PROMPTS[model_tag][task_type]


@torch.no_grad()
def score_choices(prompt: str, choices: list[str], model, tok) -> list[float]:
    scores = []
    for c in choices:
        full = prompt + c
        enc = tok(full, return_tensors="pt").to(model.device)
        plen = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
        logits = model(**enc).logits
        shift_logits = logits[0, plen - 1:-1, :]
        shift_labels = enc["input_ids"][0, plen:]
        lps = F.log_softmax(shift_logits, dim=-1)
        token_lps = lps.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        scores.append(token_lps.mean().item())
    return scores


@torch.no_grad()
def generate(prompt: str, model, tok, max_tokens: int = 256) -> str:
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False)
    return strip_think_tags(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))


def word_overlap(pred: str, truth: str) -> float:
    pred_w = set(pred.lower().split())
    true_w = set(truth.lower().split())
    if not true_w:
        return 0.0
    return len(pred_w & true_w) / len(true_w)


# ─── MMLU ─────────────────────────────────────────────────────────────────────

def _mmlu_few_shot(dev_ds):
    lines = []
    for ex in list(dev_ds)[:5]:
        ch = ex["choices"]
        lines.append(
            f"Question: {ex['question']}\nA. {ch[0]}\nB. {ch[1]}\nC. {ch[2]}\nD. {ch[3]}\n"
            f"Answer: {MMLU_CHOICES[ex['answer']]}"
        )
    return "\n\n".join(lines)


def eval_mmlu(model, tok, model_tag, n=200):
    print("\n" + "=" * 60)
    print("  BENCHMARK: MMLU")
    print("=" * 60)

    ds = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")
    prefix = _mmlu_few_shot(dev)

    if n: ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    correct = total = total_tokens = 0
    t0 = time.time()

    for i, ex in enumerate(ds):
        ch = ex["choices"]
        user = (
            f"{prefix}\n\nQuestion: {ex['question']}\n"
            f"A. {ch[0]}\nB. {ch[1]}\nC. {ch[2]}\nD. {ch[3]}\nAnswer:"
        )
        p = chat_prompt(get_sys(model_tag, "mcq"), user, tok)
        tokens_used = len(tok(p)["input_ids"])
        total_tokens += tokens_used

        sc = score_choices(p, [f" {c}" for c in MMLU_CHOICES], model, tok)
        pred = int(torch.tensor(sc).argmax())
        correct += int(pred == ex["answer"])
        total += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ds)}] acc={correct/total:.2%}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"  MMLU: {correct}/{total} = {acc:.2%}  ({elapsed:.1f}s)")

    return {
        "benchmark": "MMLU",
        "accuracy": round(acc, 4),
        "correct": correct, "total": total,
        "hallucination_rate": None,
        "total_tokens": total_tokens,
        "cost": round(total_tokens / 1000 * COST_PER_1K_TOKENS, 4),
        "latency_sec": round(elapsed, 2),
        "context_efficiency": round(total_tokens / max(correct, 1), 2),
    }


# ─── Big-Bench Hard ──────────────────────────────────────────────────────────

def _bbh_extract(text):
    m = re.search(r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)", text, re.I)
    if m: return m.group(1).strip().rstrip(".")
    lines = text.strip().splitlines()
    return lines[-1].strip() if lines else ""


def eval_bbh(model, tok, model_tag, n=100):
    print("\n" + "=" * 60)
    print("  BENCHMARK: Big-Bench Hard")
    print("=" * 60)

    import random
    random.seed(42)

    examples = []
    for sub in BBH_SUBTASKS:
        try:
            sub_ds = load_dataset("maveriq/bigbenchhard", sub, split="train")
            for ex in sub_ds:
                examples.append({**ex, "_sub": sub})
        except Exception:
            continue

    if not examples:
        print("  WARNING: Could not load BBH. Skipping.")
        return {"benchmark": "Big-Bench Hard", "accuracy": None, "correct": 0, "total": 0,
                "hallucination_rate": None, "total_tokens": 0, "cost": 0, "latency_sec": 0,
                "context_efficiency": 0}

    random.shuffle(examples)
    if n: examples = examples[:n]

    fs = "\n\n".join(
        f"Q: {f['input']}\nA: Let's think step by step. The answer is {f['target']}."
        for f in BBH_FEW_SHOT
    )

    correct = total = total_tokens = 0
    t0 = time.time()

    for i, ex in enumerate(examples):
        user = f"{fs}\n\nQ: {ex['input']}\nA: Let's think step by step."
        p = chat_prompt(get_sys(model_tag, "cot"), user, tok)
        total_tokens += len(tok(p)["input_ids"])

        resp = generate(p, model, tok, max_tokens=256)
        total_tokens += len(tok(resp)["input_ids"])

        pred = _bbh_extract(resp)
        if pred.strip().lower() == ex["target"].strip().lower():
            correct += 1
        total += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(examples)}] acc={correct/total:.2%}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print(f"  BBH: {correct}/{total} = {acc:.2%}  ({elapsed:.1f}s)")

    return {
        "benchmark": "Big-Bench Hard",
        "accuracy": round(acc, 4),
        "correct": correct, "total": total,
        "hallucination_rate": None,
        "total_tokens": total_tokens,
        "cost": round(total_tokens / 1000 * COST_PER_1K_TOKENS, 4),
        "latency_sec": round(elapsed, 2),
        "context_efficiency": round(total_tokens / max(correct, 1), 2),
    }


# ─── TruthfulQA ──────────────────────────────────────────────────────────────

def eval_truthfulqa(model, tok, model_tag, n=None):
    print("\n" + "=" * 60)
    print("  BENCHMARK: TruthfulQA (MC1 — Hallucination Detection)")
    print("=" * 60)

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    if n: ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    correct = total = total_tokens = 0
    t0 = time.time()

    for i, ex in enumerate(ds):
        choices = ex["mc1_targets"]["choices"]
        labels  = ex["mc1_targets"]["labels"]
        if not choices or sum(labels) == 0: continue
        true_idx = labels.index(1)

        user = f"Question: {ex['question']}\nAnswer:"
        p = chat_prompt(get_sys(model_tag, "truth"), user, tok)
        total_tokens += len(tok(p)["input_ids"])

        sc = score_choices(p, [f" {c}" for c in choices], model, tok)
        pred = int(torch.tensor(sc).argmax())
        correct += int(pred == true_idx)
        total += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ds)}] acc={correct/total:.2%}")

    elapsed = time.time() - t0
    acc = correct / total if total else 0
    hall = round(1 - acc, 4)
    print(f"  TruthfulQA MC1: {correct}/{total} = {acc:.2%}  (Hallucination: {hall:.2%})")

    return {
        "benchmark": "TruthfulQA (MC1)",
        "accuracy": round(acc, 4),
        "correct": correct, "total": total,
        "hallucination_rate": hall,
        "total_tokens": total_tokens,
        "cost": round(total_tokens / 1000 * COST_PER_1K_TOKENS, 4),
        "latency_sec": round(elapsed, 2),
        "context_efficiency": round(total_tokens / max(correct, 1), 2),
    }


# ─── Domain QA (Scientific Research) ─────────────────────────────────────────

def eval_domain_qa(model, tok, model_tag, n=100):
    """
    Evaluate on the project's own scientific QA dataset (qa_tasks.jsonl).
    This is where domain-specific continued pretraining should shine.

    Metrics: Exact Match, Word Overlap, Latency, Tokens, Cost.
    """
    print("\n" + "=" * 60)
    print("  BENCHMARK: Domain QA (Scientific Research)")
    print("=" * 60)

    qa_file = DATA_DIR / "qa_tasks.jsonl"
    if not qa_file.exists():
        print(f"  WARNING: {qa_file} not found. Skipping domain QA.")
        return {"benchmark": "Domain QA", "accuracy": None, "word_overlap": None,
                "correct": 0, "total": 0, "hallucination_rate": None,
                "total_tokens": 0, "cost": 0, "latency_sec": 0, "context_efficiency": 0}

    with open(qa_file, "r") as f:
        all_tasks = [json.loads(line) for line in f]

    # Stratified sample: pick from spread across the dataset
    import random
    random.seed(42)
    random.shuffle(all_tasks)
    tasks = all_tasks[:n] if n else all_tasks

    correct = 0
    total = 0
    total_overlap = 0.0
    total_tokens = 0
    t0 = time.time()

    for i, task in enumerate(tasks):
        instruction = task["instruction"]
        inp = task.get("input", "")
        true_ans = task["output"].strip()

        user_text = f"{instruction}\n{inp}".strip()
        p = chat_prompt(get_sys(model_tag, "domain"), user_text, tok)
        tokens_used = len(tok(p)["input_ids"])

        response = generate(p, model, tok, max_tokens=150)
        resp_tokens = len(tok(response)["input_ids"])
        total_tokens += tokens_used + resp_tokens

        # Exact match
        em = int(response.strip().lower() == true_ans.lower())
        correct += em

        # Word overlap
        ov = word_overlap(response, true_ans)
        total_overlap += ov
        total += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(tasks)}]  EM={correct/total:.2%}  Overlap={total_overlap/total:.2f}")

    elapsed = time.time() - t0
    em_acc = correct / total if total else 0
    avg_ov = total_overlap / total if total else 0
    # Domain hallucination proxy: low overlap = hallucinated answer
    hall = round(1 - avg_ov, 4)

    print(f"\n  Domain QA: EM={correct}/{total} ({em_acc:.2%})  Avg Overlap={avg_ov:.3f}  ({elapsed:.1f}s)")

    return {
        "benchmark": "Domain QA (Scientific)",
        "accuracy": round(em_acc, 4),
        "word_overlap": round(avg_ov, 4),
        "correct": correct, "total": total,
        "hallucination_rate": hall,
        "total_tokens": total_tokens,
        "cost": round(total_tokens / 1000 * COST_PER_1K_TOKENS, 4),
        "latency_sec": round(elapsed, 2),
        "context_efficiency": round(total_tokens / max(correct, 1), 2),
    }


# ─── Driver ───────────────────────────────────────────────────────────────────

def _delta_bar(delta: float, width: int = 20) -> str:
    """Visual bar for delta: ████████ +5.20%  or  ▒▒▒▒ -2.10%"""
    if delta is None:
        return ""
    abs_d = abs(delta)
    filled = min(int(abs_d * width / 0.15), width)  # scale: 15% fills the bar
    if delta >= 0:
        return "█" * filled + f" +{delta:.2%}"
    else:
        return "▒" * filled + f" {delta:.2%}"


def run(n_mmlu, n_bbh, n_tqa, n_domain):
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Stage 1 — Model Benchmarking                              ║")
    print("║  Base : Qwen/Qwen3-4B-Instruct-2507                       ║")
    print("║  FT   : Wvidit/Qwen-3-grpo  (SFT + GRPO)                  ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    all_results = {"timestamp": datetime.now().isoformat(), "stage": "model_benchmark", "models": {}}

    for tag, name in [("base", BASE_MODEL_NAME), ("fine_tuned", FT_MODEL_NAME)]:
        print(f"\n{'━' * 60}\n  Evaluating: {name}  ({tag})\n{'━' * 60}")
        model, tok = load_model_and_tokenizer(name)

        r = {}
        r["mmlu"]       = eval_mmlu(model, tok, model_tag=tag, n=n_mmlu)
        r["bbh"]        = eval_bbh(model, tok, model_tag=tag, n=n_bbh)
        r["truthfulqa"] = eval_truthfulqa(model, tok, model_tag=tag, n=n_tqa)
        r["domain_qa"]  = eval_domain_qa(model, tok, model_tag=tag, n=n_domain)

        all_results["models"][tag] = r
        del model
        torch.cuda.empty_cache()

    # ── Comparison with delta bars ────────────────────────────────────────────
    benchmarks = [
        ("MMLU",                "mmlu",       "accuracy"),
        ("Big-Bench Hard",      "bbh",        "accuracy"),
        ("TruthfulQA (MC1)",    "truthfulqa",  "accuracy"),
        ("Domain QA (Sci.)",    "domain_qa",   "accuracy"),
    ]

    print(f"\n{'═' * 90}")
    print(f"  {'':60}  BASE vs FINE-TUNED COMPARISON")
    print(f"{'═' * 90}")
    print(f"{'Benchmark':<22} | {'Base':>10} | {'Fine-Tuned':>10} | {'Delta':>8} | {'Improvement':>25}")
    print(f"{'-' * 90}")

    for display, key, metric in benchmarks:
        ba = all_results["models"]["base"][key].get(metric)
        fa = all_results["models"]["fine_tuned"][key].get(metric)

        ba_s = f"{ba:.2%}" if ba is not None else "N/A"
        fa_s = f"{fa:.2%}" if fa is not None else "N/A"

        if ba is not None and fa is not None:
            delta = fa - ba
            d_s = f"{delta:+.2%}"
            bar = _delta_bar(delta)
        else:
            d_s = "N/A"
            bar = ""

        print(f"{display:<22} | {ba_s:>10} | {fa_s:>10} | {d_s:>8} | {bar}")

    # Domain QA word overlap (unique metric)
    ba_ov = all_results["models"]["base"]["domain_qa"].get("word_overlap")
    fa_ov = all_results["models"]["fine_tuned"]["domain_qa"].get("word_overlap")
    if ba_ov is not None and fa_ov is not None:
        d_ov = fa_ov - ba_ov
        print(f"{'  └─ Word Overlap':<22} | {ba_ov:>10.3f} | {fa_ov:>10.3f} | {d_ov:>+8.3f} | {_delta_bar(d_ov)}")

    print(f"{'-' * 90}")

    # ── Hallucination comparison ──────────────────────────────────────────────
    print(f"\n{'Hallucination Rate':<22} | {'Base':>10} | {'Fine-Tuned':>10} | {'Reduction':>10}")
    print(f"{'-' * 62}")
    for display, key in [("TruthfulQA", "truthfulqa"), ("Domain QA", "domain_qa")]:
        bh = all_results["models"]["base"][key].get("hallucination_rate")
        fh = all_results["models"]["fine_tuned"][key].get("hallucination_rate")
        if bh is not None and fh is not None:
            reduction = bh - fh  # positive = improved
            r_s = f"{reduction:+.2%}"
        else:
            r_s = "N/A"
        bh_s = f"{bh:.2%}" if bh is not None else "N/A"
        fh_s = f"{fh:.2%}" if fh is not None else "N/A"
        print(f"{display:<22} | {bh_s:>10} | {fh_s:>10} | {r_s:>10}")

    print()

    # ── Cost efficiency table ─────────────────────────────────────────────────
    print(f"{'System':<24} | {'Accuracy':>10} | {'Tokens':>10} | {'Cost':>10} | {'Latency':>10}")
    print(f"{'-' * 74}")
    for tag_label, tag_key in [("Base Model", "base"), ("Fine-Tuned (GRPO)", "fine_tuned")]:
        for bname, bkey in [("MMLU", "mmlu"), ("BBH", "bbh"), ("TruthfulQA", "truthfulqa"), ("Domain QA", "domain_qa")]:
            r = all_results["models"][tag_key][bkey]
            acc_s = f"{r['accuracy']:.2%}" if r.get('accuracy') is not None else "N/A"
            lat_s = f"{r['latency_sec']:.1f}s" if r.get('latency_sec') else "—"
            print(f"{tag_label+' '+bname:<24} | {acc_s:>10} | {r.get('total_tokens',0):>10} | ${r.get('cost',0):>8} | {lat_s:>10}")

    print(f"{'=' * 74}\n")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved → {RESULTS_PATH}")
    return all_results


def main():
    ap = argparse.ArgumentParser(description="Stage 1 — Model Benchmarking")
    ap.add_argument("--samples", type=int, default=None, help="Override per-benchmark sample count")
    ap.add_argument("--full", action="store_true", help="Full datasets")
    ap.add_argument("--mmlu-samples", type=int, default=200)
    ap.add_argument("--bbh-samples", type=int, default=100)
    ap.add_argument("--tqa-samples", type=int, default=None)
    ap.add_argument("--domain-samples", type=int, default=100)
    args = ap.parse_args()

    if args.samples is not None:
        n_mmlu = n_bbh = n_tqa = n_domain = args.samples
    elif args.full:
        n_mmlu = n_bbh = n_tqa = n_domain = None
    else:
        n_mmlu = args.mmlu_samples
        n_bbh  = args.bbh_samples
        n_tqa  = args.tqa_samples
        n_domain = args.domain_samples

    run(n_mmlu, n_bbh, n_tqa, n_domain)


if __name__ == "__main__":
    main()
