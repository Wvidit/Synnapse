"""
Context Management Policies — Synnapse
========================================
Three context policies that control how the agent manages its
working memory under token and cost constraints.

  A. Naive            — Keep full context, FIFO truncation
  B. RAG Retrieval    — Keyword-relevance-based top-K selection
  C. Compression+Cache — Summarise old context, pin verified facts
"""

from typing import List


# ─── Policy A: Naive (Full Context) ──────────────────────────────────────────

def policy_a_naive(
    new_observation: str,
    history: List[str],
    max_tokens: int = 8192,
    query: str = "",
) -> List[str]:
    """
    Append everything to context. When the token budget is exceeded,
    pop the oldest entries (FIFO).
    """
    history.append(new_observation)

    while _count_tokens(history) > max_tokens and len(history) > 1:
        history.pop(0)

    return history


# ─── Policy B: RAG Retrieval ─────────────────────────────────────────────────

def policy_b_rag(
    new_observation: str,
    history: List[str],
    max_tokens: int = 4096,
    query: str = "",
) -> List[str]:
    """
    Keep only the top-K most relevant chunks based on keyword overlap
    with the query + recency bonus. Simulates FAISS-style retrieval
    without requiring an embedding model.
    """
    history.append(new_observation)

    max_entries = 4

    if len(history) <= max_entries:
        return history

    # Score each entry by relevance to the query + recency
    query_words = set(query.lower().split()) if query else set()

    scored = []
    for idx, entry in enumerate(history):
        entry_words = set(entry.lower().split())
        recency = idx / len(history)  # 0.0 (oldest) → 1.0 (newest)

        # Keyword overlap with query
        if query_words:
            overlap = len(entry_words & query_words) / max(len(query_words), 1)
        else:
            overlap = 0.0

        # Penalise error entries, boost substantive content
        error_penalty = -0.5 if "error" in entry.lower() else 0.0
        content_bonus = 0.3 if len(entry.split()) > 30 else 0.0

        # Boost entries with verified facts or hypotheses
        fact_bonus = 0.4 if any(kw in entry.lower() for kw in
                                ["verified", "consistent", "hypothesis"]) else 0.0

        score = overlap * 0.4 + recency * 0.3 + content_bonus + fact_bonus + error_penalty
        scored.append((score, idx, entry))

    # Keep top-K by score, but preserve original order
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = sorted(scored[:max_entries], key=lambda x: x[1])
    history = [entry for _, _, entry in selected]

    # Trim if still over token budget
    while _count_tokens(history) > max_tokens and len(history) > 1:
        history.pop(0)

    return history


# ─── Policy C: Compression + Verified-Fact Cache ─────────────────────────────

def policy_c_compression(
    new_observation: str,
    history: List[str],
    max_tokens: int = 2048,
    query: str = "",
) -> List[str]:
    """
    Pin verified facts. When context exceeds half the budget,
    compress older entries into a summary while keeping
    verified facts and the most recent entries intact.
    """
    history.append(new_observation)
    total = _count_tokens(history)

    if total <= max_tokens // 2 or len(history) <= 2:
        return history

    # Separate verified facts (pinned) from compressible context
    verified = []
    compressible = []
    recent = history[-2:]  # always keep last 2

    for entry in history[:-2]:
        if _is_verified_fact(entry):
            verified.append(entry)
        else:
            compressible.append(entry)

    if compressible:
        summary = _extractive_summary(compressible, max_words=100)
        history = verified + [f"[CONTEXT SUMMARY]: {summary}"] + recent
    else:
        history = verified + recent

    # Final trim if still over budget
    while _count_tokens(history) > max_tokens and len(history) > 1:
        history.pop(0)

    return history


# ─── Shared utilities ────────────────────────────────────────────────────────

def _count_tokens(entries: List[str]) -> int:
    """Approximate token count from word count."""
    return len(" ".join(entries).split())


def _is_verified_fact(entry: str) -> bool:
    """Check if an entry contains a verified/pinned fact."""
    lower = entry.lower()
    return any(kw in lower for kw in [
        "verified", "consistent", "contradiction", "confirmed",
        "[fact]", "logically valid",
    ])


def _extractive_summary(entries: List[str], max_words: int = 100) -> str:
    """
    Simple extractive summary: take the first sentence from each
    entry until we hit the word limit.
    """
    sentences = []
    word_count = 0

    for entry in entries:
        # Strip the "Action: ...\nObservation: ..." prefix if present
        text = entry
        if "Observation:" in text:
            text = text.split("Observation:", 1)[1].strip()

        # Take first meaningful sentence (skip very short fragments)
        for sent in text.replace("\n", ". ").split(". "):
            sent = sent.strip()
            if len(sent.split()) < 4:
                continue
            sentences.append(sent)
            word_count += len(sent.split())
            if word_count >= max_words:
                break
        if word_count >= max_words:
            break

    return ". ".join(sentences)[:500]


# ─── Standalone runner (legacy) ──────────────────────────────────────────────

def run_contextbench():
    """Simple standalone test of the three policies."""
    print("Running ContextBench standalone test...")

    tasks = [
        {"query": "Find papers on attention mechanisms that contradict Vaswani.", "steps": 5},
        {"query": "Verify if the graph structure improves the BLEU score.", "steps": 7},
    ]

    policies = {
        "naive": policy_a_naive,
        "rag_retrieval": policy_b_rag,
        "compression_cache": policy_c_compression,
    }

    results = []
    for p_name, p_func in policies.items():
        print(f"\nPolicy: {p_name}")
        for t in tasks:
            history: List[str] = []
            total_tokens = 0
            for step in range(t["steps"]):
                obs = f"Observation for step {step}: Found relevant results for '{t['query']}' including papers on transformers, attention, and graph networks."
                history = p_func(obs, history, query=t["query"])
                total_tokens += _count_tokens(history)

            cost = (total_tokens / 1000) * 0.002
            print(f"  {t['query'][:50]:<50}  tokens={total_tokens}  cost=${cost:.4f}  entries={len(history)}")
            results.append({
                "policy": p_name,
                "task": t["query"],
                "cost": cost,
                "tokens": total_tokens,
                "final_entries": len(history),
            })

    print("\nDone.")
    return results


if __name__ == "__main__":
    run_contextbench()
