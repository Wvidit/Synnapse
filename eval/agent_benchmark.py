"""
Agent Benchmarking — Synnapse Evaluation Pipeline (Stage 2)
============================================================
Compare a **Baseline LLM Agent** (standard tools only) vs a
**Neurosymbolic Agent** (full Synnapse: KG + Z3 verifier + taxonomy +
hypothesis generation).

Runs against the agent server's /agent/run endpoint with different
context policies and tool configurations.

Metrics:
  - Task Success Rate
  - Hallucination (proxy: answer plausibility score)
  - Token Usage
  - Cost per Query
  - Latency
  - Context Efficiency

Usage:
    python -m eval.agent_benchmark                # default (all tasks)
    python -m eval.agent_benchmark --samples 5    # quick smoke-test
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests

# ─── Configuration ────────────────────────────────────────────────────────────

AGENT_API = "http://127.0.0.1:8000/agent/run"
RESULTS_PATH = Path(__file__).parent / "agent_benchmark_results.json"
COST_PER_1K_TOKENS = 0.002

# Scientific Research Assistant task suite
EVAL_TASKS = [
    # Single-hop retrieval
    {"query": "Find papers on contrastive learning.",
     "type": "single-hop", "expected_tools": ["search_literature"],
     "success_keywords": ["contrastive", "learning", "paper"]},

    {"query": "Find papers about transformer architectures for NLP.",
     "type": "single-hop", "expected_tools": ["search_literature"],
     "success_keywords": ["transformer", "attention", "nlp"]},

    {"query": "Search for recent work on graph neural networks.",
     "type": "single-hop", "expected_tools": ["search_literature"],
     "success_keywords": ["graph", "neural", "network"]},

    # Multi-hop reasoning
    {"query": "Find papers that contradict Vaswani et al. 2017 attention mechanism.",
     "type": "multi-hop", "expected_tools": ["search_literature", "explore_citations"],
     "success_keywords": ["attention", "mechanism", "contradict"]},

    {"query": "Explore citations of papers on reinforcement learning from human feedback.",
     "type": "multi-hop", "expected_tools": ["explore_citations"],
     "success_keywords": ["reinforcement", "human", "feedback"]},

    # Hypothesis generation
    {"query": "Propose a hypothesis about how knowledge graphs improve language model factuality.",
     "type": "hypothesis", "expected_tools": ["search_literature", "generate_hypothesis"],
     "success_keywords": ["hypothesis", "knowledge", "graph"]},

    {"query": "Generate a hypothesis about combining symbolic reasoning with neural networks.",
     "type": "hypothesis", "expected_tools": ["generate_hypothesis"],
     "success_keywords": ["hypothesis", "symbolic", "neural"]},

    # Logic verification (neurosymbolic)
    {"query": "Is the claim that 'graph neural networks improve sample efficiency' logically consistent with recent publications?",
     "type": "verification", "expected_tools": ["verify_logic"],
     "success_keywords": ["consistent", "verified", "logic"]},

    {"query": "Verify whether attention mechanisms contradict recurrent architectures.",
     "type": "verification", "expected_tools": ["verify_logic"],
     "success_keywords": ["consistent", "contradict", "verified"]},

    # Summarization / context compression
    {"query": "Summarize the state of research on diffusion models for image generation.",
     "type": "summarization", "expected_tools": ["search_literature", "summarize_context"],
     "success_keywords": ["diffusion", "image", "generation"]},
]

# Agent configurations
AGENT_CONFIGS = {
    "baseline": {
        "label": "Baseline LLM Agent",
        "context_policy": "naive",
        "token_budget": 8192,
        "max_steps": 5,
        "description": "Standard LLM agent with naive context window (no symbolic tools)",
    },
    "neurosymbolic": {
        "label": "Neurosymbolic Agent (Synnapse)",
        "context_policy": "compression_cache",
        "token_budget": 8192,
        "max_steps": 10,
        "description": "Full Synnapse pipeline: KG, Z3 verifier, taxonomy, compression",
    },
}


# ─── Evaluation Logic ────────────────────────────────────────────────────────

def _estimate_tokens(context: list) -> int:
    """Estimate tokens from context list."""
    return sum(len(str(c).split()) for c in context)


def _check_success(response: dict, task: dict) -> bool:
    """Check if the agent response indicates task success."""
    if response.get("status") != "success":
        return False

    context_str = " ".join(str(c) for c in response.get("final_context", [])).lower()

    # Check for error indicators
    if "error" in context_str and "no error" not in context_str:
        return False

    # Check keyword presence (at least 1 of expected keywords)
    matches = sum(1 for kw in task["success_keywords"] if kw in context_str)
    return matches >= 1


def _plausibility_score(response: dict, task: dict) -> float:
    """Proxy hallucination metric: how plausible is the response?"""
    context_str = " ".join(str(c) for c in response.get("final_context", [])).lower()

    if not context_str.strip():
        return 0.0

    score = 0.0
    kw_matches = sum(1 for kw in task["success_keywords"] if kw in context_str)
    score += min(kw_matches / max(len(task["success_keywords"]), 1), 1.0) * 0.5

    # Penalise empty/error responses
    if len(context_str.split()) > 20:
        score += 0.3
    if "error" not in context_str:
        score += 0.2

    return round(min(score, 1.0), 3)


def eval_agent_config(config_key: str, config: dict, tasks: list) -> dict:
    """Run all tasks against a single agent configuration."""
    print(f"\n{'━' * 60}")
    print(f"  Agent: {config['label']}")
    print(f"  Policy: {config['context_policy']}  |  Budget: {config['token_budget']}  |  Max Steps: {config['max_steps']}")
    print(f"{'━' * 60}")

    results = []
    total_success = 0
    total_tokens = 0
    total_plausibility = 0.0

    for i, task in enumerate(tasks):
        t0 = time.time()
        try:
            resp = requests.post(AGENT_API, json={
                "query": task["query"],
                "max_steps": config["max_steps"],
                "token_budget": config["token_budget"],
                "context_policy": config["context_policy"],
            }, timeout=60)
            data = resp.json()
            latency = time.time() - t0
        except requests.exceptions.ConnectionError:
            print("  ❌ Agent server not running (http://127.0.0.1:8000)")
            print("  Run `python -m agent.server` first.")
            return _empty_result(config_key, config)
        except Exception as e:
            data = {"status": "error", "final_context": [], "steps_taken": 0}
            latency = time.time() - t0

        tokens = _estimate_tokens(data.get("final_context", []))
        success = _check_success(data, task)
        plaus = _plausibility_score(data, task)

        total_success += int(success)
        total_tokens += tokens
        total_plausibility += plaus

        status_icon = "✅" if success else "❌"
        print(f"  {status_icon} [{task['type']:<14}] {task['query'][:60]:<60}  "
              f"steps={data.get('steps_taken', '?')}  tokens={tokens}  lat={latency:.1f}s")

        results.append({
            "query": task["query"],
            "type": task["type"],
            "success": success,
            "plausibility": plaus,
            "steps": data.get("steps_taken", 0),
            "tokens": tokens,
            "latency_sec": round(latency, 2),
        })

    n = len(tasks)
    acc = total_success / n if n else 0
    avg_tokens = total_tokens / n if n else 0
    avg_cost = avg_tokens / 1000 * COST_PER_1K_TOKENS
    avg_lat = sum(r["latency_sec"] for r in results) / n if n else 0
    avg_plaus = total_plausibility / n if n else 0
    ctx_eff = total_tokens / max(total_success, 1)

    return {
        "agent": config_key,
        "label": config["label"],
        "accuracy": round(acc, 4),
        "hallucination_rate": round(1 - avg_plaus, 4),
        "total_tokens": total_tokens,
        "avg_tokens_per_query": round(avg_tokens),
        "avg_cost_per_query": round(avg_cost, 4),
        "avg_latency_sec": round(avg_lat, 2),
        "context_efficiency": round(ctx_eff, 2),
        "tasks_evaluated": n,
        "details": results,
    }


def _empty_result(key, config):
    return {
        "agent": key, "label": config["label"],
        "accuracy": None, "hallucination_rate": None,
        "total_tokens": 0, "avg_tokens_per_query": 0,
        "avg_cost_per_query": 0, "avg_latency_sec": 0,
        "context_efficiency": 0, "tasks_evaluated": 0, "details": [],
    }


# ─── Driver ───────────────────────────────────────────────────────────────────

def run(n_samples=None):
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Stage 2 — Agent Benchmarking                              ║")
    print("║  Baseline LLM vs Neurosymbolic Agent (Synnapse)            ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    tasks = EVAL_TASKS
    if n_samples is not None:
        tasks = tasks[:n_samples]

    all_results = {"timestamp": datetime.now().isoformat(), "stage": "agent_benchmark", "agents": {}}

    for key, config in AGENT_CONFIGS.items():
        r = eval_agent_config(key, config, tasks)
        all_results["agents"][key] = r

    # ── Per-task-type breakdown ─────────────────────────────────────────────
    task_types = sorted(set(t["type"] for t in tasks))

    print(f"\n{'═' * 90}")
    print(f"  AGENT COMPARISON — Baseline vs Neurosymbolic")
    print(f"{'═' * 90}")

    # Summary table
    print(f"\n{'System':<32} | {'Accuracy':>10} | {'Tokens':>8} | {'Cost':>10} | {'Latency':>8} | {'Halluc.':>8}")
    print(f"{'-' * 90}")
    for key in AGENT_CONFIGS:
        r = all_results["agents"][key]
        acc = f"{r['accuracy']:.2%}" if r['accuracy'] is not None else "N/A"
        hall = f"{r['hallucination_rate']:.2%}" if r.get('hallucination_rate') is not None else "N/A"
        print(f"{r['label']:<32} | {acc:>10} | {r['avg_tokens_per_query']:>8} | "
              f"${r['avg_cost_per_query']:>8.4f} | {r['avg_latency_sec']:>6.2f}s | {hall:>8}")

    # Delta row
    bl = all_results["agents"].get("baseline", {})
    ns = all_results["agents"].get("neurosymbolic", {})
    if bl.get("accuracy") is not None and ns.get("accuracy") is not None:
        d_acc = ns["accuracy"] - bl["accuracy"]
        d_tok = ns["avg_tokens_per_query"] - bl["avg_tokens_per_query"]
        d_cost = ns["avg_cost_per_query"] - bl["avg_cost_per_query"]
        d_hall = bl["hallucination_rate"] - ns["hallucination_rate"]
        print(f"{'-' * 90}")
        print(f"{'  Δ Neurosymbolic Advantage':<32} | {d_acc:>+10.2%} | {d_tok:>+8} | "
              f"${d_cost:>+8.4f} |          | {d_hall:>+8.2%}")

    print(f"{'═' * 90}")

    # Per-task-type breakdown
    print(f"\n  Per-Task-Type Accuracy:")
    print(f"  {'Task Type':<18} | {'Baseline':>10} | {'Neurosymbolic':>14} | {'Δ':>8}")
    print(f"  {'-' * 58}")

    for tt in task_types:
        for key in ["baseline", "neurosymbolic"]:
            details = all_results["agents"].get(key, {}).get("details", [])
            tt_tasks = [d for d in details if d["type"] == tt]
            tt_success = sum(1 for d in tt_tasks if d["success"])
            all_results["agents"][key].setdefault("per_type", {})[tt] = {
                "accuracy": round(tt_success / len(tt_tasks), 4) if tt_tasks else 0,
                "count": len(tt_tasks),
            }

        b_acc = all_results["agents"]["baseline"]["per_type"][tt]["accuracy"]
        n_acc = all_results["agents"]["neurosymbolic"]["per_type"][tt]["accuracy"]
        delta = n_acc - b_acc
        bar = "█" * min(int(abs(delta) * 20 / 0.5), 20)
        sign = "+" if delta >= 0 else "-"
        print(f"  {tt:<18} | {b_acc:>10.0%} | {n_acc:>14.0%} | {sign}{abs(delta):.0%} {bar}")

    print()

    # Tool usage summary
    print(f"  Tools Utilised per Agent:")
    for key in AGENT_CONFIGS:
        details = all_results["agents"].get(key, {}).get("details", [])
        total_steps = sum(d.get("steps", 0) for d in details)
        avg_steps = total_steps / len(details) if details else 0
        print(f"    {AGENT_CONFIGS[key]['label']}: avg {avg_steps:.1f} steps/task, "
              f"{total_steps} total tool calls")
    print()

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved → {RESULTS_PATH}")
    return all_results


def main():
    ap = argparse.ArgumentParser(description="Stage 2 — Agent Benchmarking")
    ap.add_argument("--samples", type=int, default=None, help="Number of tasks to evaluate")
    args = ap.parse_args()
    run(n_samples=args.samples)


if __name__ == "__main__":
    main()
