from typing import List, Dict, Any
import requests

# Context Management Policies

def policy_a_naive(new_observation: str, history: List[str], max_tokens: int = 8192) -> List[str]:
    """
    Pass the full context window every turn. Keep appending.
    """
    history.append(new_observation)
    # Token pruning only if hard limit hit:
    while len(" ".join(history).split()) > max_tokens and len(history) > 1:
        history.pop(0) # naive FIFO pop
    return history

def policy_b_rag(new_observation: str, history: List[str], max_tokens: int = 4096) -> List[str]:
    """
    Retrieve only top-K chunks rather than carrying everything.
    """
    # In a real impl, we embed the history/new_observation and pull from Faiss.
    # Here we simulate keeping the 3 most recent or "most relevant" chunks.
    history.append(new_observation)
    if len(history) > 3:
        # Mock retrieval: just pick the latest 3
        history = history[-3:]
    return history

def policy_c_compression(new_observation: str, history: List[str], max_tokens: int = 2048) -> List[str]:
    """
    Compression policy + cache verified facts.
    """
    # Extract verified facts if any
    verified_facts = []
    for item in history:
        if "verified" in item.lower():
            verified_facts.append(item)
            
    history.append(new_observation)
    
    # Needs compression?
    total_tokens = len(" ".join(history).split())
    if total_tokens > max_tokens / 2: # Trigger compression early
        # Mocking the summarizer tool call
        compressed = summarize_mock(" ".join(history))
        history = verified_facts + [compressed]
        
    return history

def summarize_mock(text: str) -> str:
    return f"[COMPRESSED]: {text[:100]}..."

def run_contextbench():
    print("Running ContextBench...")
    # Mocking task dataset
    tasks = [
        {"query": "Find papers on attention mechanisms that contradict Vaswani.", "steps": 5},
        {"query": "Verify if the graph structure improves the BLEU score.", "steps": 7}
    ]
    
    policies = {
        "naive": policy_a_naive,
        "rag_retrieval": policy_b_rag,
        "compression_cache": policy_c_compression
    }
    
    results = []
    for p_name, p_func in policies.items():
        print(f"\nEvaluating Policy: {p_name}")
        for t in tasks:
            history = []
            total_tokens_consumed = 0
            for step in range(t["steps"]):
                obs = f"Observation for step {step} of query {t['query']}"
                history = p_func(obs, history)
                total_tokens_consumed += len(" ".join(history).split())
                
            cost = (total_tokens_consumed / 1000) * 0.002
            print(f"Task Complete. Cost: ${cost:.4f}, Tokens: {total_tokens_consumed}")
            results.append({
                "policy": p_name,
                "task": t["query"],
                "cost": cost,
                "tokens": total_tokens_consumed
            })
            
    print("\nBenchmark Complete.")
    return results

if __name__ == "__main__":
    run_contextbench()
