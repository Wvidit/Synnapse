from typing import Dict, Any

def get_next_action_heuristic(state: Dict[str, Any]) -> str:
    """
    The heuristic rules requested in the problem statement.
    """
    query = state.get("query", "").lower()
    token_percent = state.get("token_budget_percent", 0.0)
    has_hypothesis = state.get("has_hypothesis", False)
    history = state.get("history", [])
    history_str = " ".join(history)
    
    # Stop if last action errored or budget exhausted
    if len(history) > 0 and "error" in history[-1].lower():
        return "stop"
    if token_percent > 0.8:
        return "stop"
    
    # If user wants to verify logic
    if any(k in query for k in ["verify", "consistent", "logical", "contradict"]):
        if "verify_logic" not in history_str:
            return "verify_logic"
        return "stop"
    
    # If user wants a hypothesis: search → generate → stop
    if any(k in query for k in ["hypothesis", "hypothesize", "generate", "propose", "suggest"]):
        if "search_literature" not in history_str:
            return "search_literature"
        if "generate_hypothesis" not in history_str:
            return "generate_hypothesis"
        return "stop"
    
    # If query mentions a specific paper: explore → stop
    if "paper" in query or "arxiv:" in query or "citation" in query:
        if "explore_citations" not in history_str:
            return "explore_citations"
        return "stop"
    
    # Default: search once → stop
    if "search_literature" not in history_str:
        return "search_literature"
    
    # Budget mid-point: summarize to compress
    if token_percent > 0.4 and "summarize_context" not in history_str:
        return "summarize_context"
    
    # Everything done
    return "stop"

def get_next_action(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Policy Learning Controller: Start with heuristic, then upgrade to BERT-tiny.
    """
    # For now, dispatch to heuristic
    action = get_next_action_heuristic(state)
    return {"tool_name": action}
