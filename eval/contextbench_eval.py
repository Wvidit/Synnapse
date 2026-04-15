"""
ContextBench Benchmarking — Synnapse Evaluation Pipeline (Stage 3)
===================================================================
Evaluate how efficiently the agent manages context under token and cost
constraints using different context policies.

Policies tested:
  A. Naive         — Full context window, FIFO truncation
  B. RAG Retrieval — Top-K chunk retrieval (simulated FAISS)
  C. Compression   — Summarisation + verified-fact caching

Metrics:
  - Task Success Rate
  - Token Usage
  - Cost per Query
  - Latency
  - Context Efficiency (tokens per successful task)

Usage:
    python -m eval.contextbench_eval
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import requests

# ─── Configuration ────────────────────────────────────────────────────────────

AGENT_API = "http://127.0.0.1:1111/agent/run"
RESULTS_PATH = Path(__file__).parent / "contextbench_results.json"
COST_PER_1K_TOKENS = 0.002

# Scientific Research Assistant task dataset
CONTEXT_TASKS = [
    {"query": "Find papers on attention mechanisms that contradict Vaswani.",
     "max_steps": 5, "keywords": ["attention", "contradict"]},
    {"query": "Verify if the graph structure improves the BLEU score.",
     "max_steps": 7, "keywords": ["graph", "bleu", "improve"]},
    {"query": "Summarize recent work on diffusion models and their impact on computer vision.",
     "max_steps": 6, "keywords": ["diffusion", "vision"]},
    {"query": "Explore the citation network around self-supervised learning papers.",
     "max_steps": 5, "keywords": ["citation", "self-supervised"]},
    {"query": "Generate a hypothesis about combining retrieval-augmented generation with knowledge graphs.",
     "max_steps": 8, "keywords": ["hypothesis", "retrieval", "knowledge"]},
    {"query": "Find contradictions in the literature on dropout regularization.",
     "max_steps": 6, "keywords": ["dropout", "contradict"]},
    {"query": "Analyze the relationship between model scale and emergent abilities.",
     "max_steps": 5, "keywords": ["scale", "emergent"]},
    {"query": "Search for papers that improve upon BERT for scientific text understanding.",
     "max_steps": 5, "keywords": ["bert", "scientific"]},
]

# Context policies with different budget configurations
POLICY_CONFIGS = {
    "naive": {
        "label": "Policy A — Naive (Full Context)",
        "context_policy": "naive",
        "token_budget": 8192,
    },
    "rag_retrieval": {
        "label": "Policy B — RAG Retrieval",
        "context_policy": "rag_retrieval",
        "token_budget": 4096,
    },
    "compression_cache": {
        "label": "Policy C — Compression + Cache",
        "context_policy": "compression_cache",
        "token_budget": 2048,
    },
}


# ─── Evaluation ───────────────────────────────────────────────────────────────

def _estimate_tokens(context: list) -> int:
    return sum(len(str(c).split()) for c in context)


def _check_success(resp: dict, task: dict) -> bool:
    if resp.get("status") != "success":
        return False
    ctx = " ".join(str(c) for c in resp.get("final_context", [])).lower()
    if "error" in ctx and "no error" not in ctx:
        return False
    return sum(1 for kw in task["keywords"] if kw in ctx) >= 1


def eval_policy(policy_key: str, config: dict, tasks: list) -> dict:
    """Evaluate a single context policy across all tasks."""
    print(f"\n{'━' * 60}")
    print(f"  {config['label']}")
    print(f"  Budget: {config['token_budget']} tokens")
    print(f"{'━' * 60}")

    successes = 0
    total_tokens = 0
    total_latency = 0.0
    task_results = []

    for task in tasks:
        t0 = time.time()
        try:
            resp = requests.post(AGENT_API, json={
                "query": task["query"],
                "max_steps": task["max_steps"],
                "token_budget": config["token_budget"],
                "context_policy": config["context_policy"],
            }, timeout=60)
            data = resp.json()
            latency = time.time() - t0
        except requests.exceptions.ConnectionError:
            print("  ❌ Agent server not running. Run `python -m agent.server` first.")
            return _empty_result(policy_key, config)
        except Exception:
            data = {"status": "error", "final_context": [], "steps_taken": 0}
            latency = time.time() - t0

        tokens = _estimate_tokens(data.get("final_context", []))
        ok = _check_success(data, task)
        successes += int(ok)
        total_tokens += tokens
        total_latency += latency

        icon = "✅" if ok else "❌"
        print(f"  {icon} {task['query'][:55]:<55}  steps={data.get('steps_taken','?')}  "
              f"tok={tokens}  lat={latency:.1f}s")

        task_results.append({
            "query": task["query"],
            "success": ok,
            "tokens": tokens,
            "steps": data.get("steps_taken", 0),
            "latency_sec": round(latency, 2),
        })

    n = len(tasks)
    acc = successes / n if n else 0
    avg_tok = total_tokens / n if n else 0
    avg_cost = avg_tok / 1000 * COST_PER_1K_TOKENS
    avg_lat = total_latency / n if n else 0
    ctx_eff = total_tokens / max(successes, 1)

    return {
        "policy": policy_key,
        "label": config["label"],
        "accuracy": round(acc, 4),
        "total_tokens": total_tokens,
        "avg_tokens_per_query": round(avg_tok),
        "avg_cost_per_query": round(avg_cost, 4),
        "avg_latency_sec": round(avg_lat, 2),
        "context_efficiency": round(ctx_eff, 2),
        "token_budget": config["token_budget"],
        "tasks_evaluated": n,
        "details": task_results,
    }


def _empty_result(key, config):
    return {
        "policy": key, "label": config["label"],
        "accuracy": None, "total_tokens": 0,
        "avg_tokens_per_query": 0, "avg_cost_per_query": 0,
        "avg_latency_sec": 0, "context_efficiency": 0,
        "token_budget": config["token_budget"],
        "tasks_evaluated": 0, "details": [],
    }


# ─── Driver ───────────────────────────────────────────────────────────────────

def run(n_samples=None):
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Stage 3 — ContextBench Benchmarking                       ║")
    print("║  Context Management Efficiency Evaluation                   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    tasks = CONTEXT_TASKS
    if n_samples is not None:
        tasks = tasks[:n_samples]

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "stage": "contextbench",
        "policies": {},
    }

    for key, config in POLICY_CONFIGS.items():
        r = eval_policy(key, config, tasks)
        all_results["policies"][key] = r

    # ── Results table (required format) ───────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"{'System':<35} | {'Accuracy':>10} | {'Tokens':>8} | {'Cost':>10}")
    print(f"{'-' * 78}")
    for key in POLICY_CONFIGS:
        r = all_results["policies"][key]
        acc = f"{r['accuracy']:.0%}" if r['accuracy'] is not None else "N/A"
        print(f"{r['label']:<35} | {acc:>10} | {r['avg_tokens_per_query']:>8} | "
              f"${r['avg_cost_per_query']:>8.4f}")
    print(f"{'=' * 78}")

    # ── Detailed metrics ──────────────────────────────────────────────────────
    print(f"\n{'Metric':<25}", end="")
    for key in POLICY_CONFIGS:
        print(f" | {POLICY_CONFIGS[key]['label'][:18]:>18}", end="")
    print(f"\n{'-' * 82}")

    metrics = [
        ("Accuracy",            lambda r: f"{r['accuracy']:.2%}" if r['accuracy'] is not None else "N/A"),
        ("Avg Tokens/Query",    lambda r: f"{r['avg_tokens_per_query']}"),
        ("Avg Cost/Query",      lambda r: f"${r['avg_cost_per_query']:.4f}"),
        ("Avg Latency",         lambda r: f"{r['avg_latency_sec']:.2f}s"),
        ("Context Efficiency",  lambda r: f"{r['context_efficiency']:.1f} tok/ok"),
        ("Token Budget",        lambda r: f"{r['token_budget']}"),
    ]

    for name, fmt in metrics:
        print(f"{name:<25}", end="")
        for key in POLICY_CONFIGS:
            print(f" | {fmt(all_results['policies'][key]):>18}", end="")
        print()

    print()

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved → {RESULTS_PATH}")
    return all_results


def main():
    ap = argparse.ArgumentParser(description="Stage 3 — ContextBench Benchmarking")
    ap.add_argument("--samples", type=int, default=None, help="Number of tasks")
    args = ap.parse_args()
    run(n_samples=args.samples)


if __name__ == "__main__":
    main()
