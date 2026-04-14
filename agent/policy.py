"""
Policy Controller — Synnapse Agent
====================================
Heuristic-based tool selection for the ReAct loop.

The policy is context-policy-aware:
  - "naive":              simple linear tool chains, limited steps
  - "rag_retrieval":      search-heavy, fewer generation steps
  - "compression_cache":  full pipeline with compression + verification
"""

from typing import Dict, Any, List

# ─── Tool chains by intent ────────────────────────────────────────────────────
# Each chain is an ordered list of tools to execute.
# The policy walks through the chain, skipping already-executed tools.

TOOL_CHAINS = {
    "verification": [
        "search_literature",   # 1. Gather evidence first
        "lookup_taxonomy",     # 2. Ground in ontology
        "verify_logic",        # 3. Then run Z3 verifier
        "summarize_context",   # 4. Compress for output
    ],
    "hypothesis": [
        "search_literature",   # 1. Gather domain context
        "explore_citations",   # 2. Explore citation network
        "generate_hypothesis", # 3. Generate novel hypothesis
        "verify_logic",        # 4. Verify logical consistency
        "summarize_context",   # 5. Summarize findings
    ],
    "citation": [
        "search_literature",   # 1. Find the paper first
        "explore_citations",   # 2. Explore citation graph
        "lookup_taxonomy",     # 3. Categorise results
        "summarize_context",   # 4. Compress
    ],
    "summarization": [
        "search_literature",   # 1. Retrieve relevant papers
        "lookup_taxonomy",     # 2. Get taxonomic context
        "summarize_context",   # 3. Compress into summary
    ],
    "default": [
        "search_literature",   # 1. Always start with search
        "summarize_context",   # 2. Compress if large
    ],
}

# ─── Intent detection keywords ───────────────────────────────────────────────

INTENT_KEYWORDS = {
    "verification":  ["verify", "consistent", "logical", "contradict", "valid", "check"],
    "hypothesis":    ["hypothesis", "hypothesize", "generate", "propose", "suggest", "predict"],
    "citation":      ["paper", "arxiv:", "citation", "cite", "explore", "network"],
    "summarization": ["summarize", "summary", "overview", "state of", "survey", "review"],
}


def _detect_intent(query: str) -> str:
    """Detect user intent from query keywords."""
    query_lower = query.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return intent
    return "default"


def _get_chain_for_policy(intent: str, context_policy: str) -> list[str]:
    """
    Adjust the tool chain based on context policy.
    - naive:              limit to first 2 tools (baseline agent)
    - rag_retrieval:      use search-heavy subset
    - compression_cache:  use full chain (neurosymbolic)
    """
    full_chain = TOOL_CHAINS.get(intent, TOOL_CHAINS["default"])

    if context_policy == "naive":
        # Baseline: only basic tools, max 2 steps
        basic_tools = {"search_literature", "summarize_context", "explore_citations"}
        return [t for t in full_chain if t in basic_tools][:2]

    elif context_policy == "rag_retrieval":
        # RAG: search + taxonomy, skip generation/verification
        rag_tools = {"search_literature", "explore_citations", "lookup_taxonomy", "summarize_context"}
        return [t for t in full_chain if t in rag_tools]

    else:
        # compression_cache / neurosymbolic: full chain
        return full_chain


def get_next_action_heuristic(state: Dict[str, Any]) -> str:
    """
    Walk through the tool chain for the detected intent, returning the
    next tool that hasn't been executed yet.
    """
    query = state.get("query", "")
    token_percent = state.get("token_budget_percent", 0.0)
    history = state.get("history", [])
    context_policy = state.get("context_policy", "compression_cache")
    history_str = " ".join(str(h) for h in history).lower()

    # Guard: stop on error in the last action
    if history and "error" in str(history[-1]).lower():
        # Only stop on critical errors, not "no results" type messages
        last = str(history[-1]).lower()
        if any(e in last for e in ["exception", "traceback", "connectionerror", "server not running"]):
            return "stop"

    # Guard: budget exhausted
    if token_percent > 0.85:
        # If we haven't summarized yet, do that first
        if "summarize_context" not in history_str and len(history) > 1:
            return "summarize_context"
        return "stop"

    # Detect intent and get policy-adjusted chain
    intent = _detect_intent(query)
    chain = _get_chain_for_policy(intent, context_policy)

    # Walk chain: return first un-executed tool
    for tool in chain:
        if tool not in history_str:
            return tool

    # Mid-budget compression if we have accumulated context
    if (token_percent > 0.4
            and "summarize_context" not in history_str
            and len(history) > 2):
        return "summarize_context"

    return "stop"


def get_next_action(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Policy Learning Controller.
    Dispatches to the heuristic; can be upgraded to a learned policy later.
    """
    action = get_next_action_heuristic(state)
    return {"tool_name": action}
