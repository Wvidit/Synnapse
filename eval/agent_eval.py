import requests
import json
import time

API_URL = "http://127.0.0.1:8000/agent/run"

def run_agent_eval():
    print("Running Agent End-to-End Evaluation...")
    
    tasks = [
        {"type": "single-hop", "query": "Find papers on contrastive learning."},
        {"type": "multi-hop", "query": "Find papers that contradict Vaswani et al. 2017 attention mechanism."},
        {"type": "validation", "query": "Is the claim that 'graph neural networks improve sample efficiency' logically consistent with recent publications?"}
    ]
    
    results = []
    
    for task in tasks:
        print(f"\nEvaluating Task: {task['query']}")
        try:
            start_time = time.time()
            resp = requests.post(
                API_URL, 
                json={
                    "query": task["query"],
                    "max_steps": 10,
                    "token_budget": 8192,
                    "context_policy": "compression_cache"
                }
            )
            data = resp.json()
            latency = time.time() - start_time
            
            print(f"Status: {data.get('status')} in {data.get('steps_taken')} steps.")
            
            results.append({
                "task": task['query'],
                "type": task['type'],
                "success": data.get('status') == 'success',
                "steps": data.get('steps_taken'),
                "latency_sec": latency
            })
            
        except requests.exceptions.ConnectionError:
            print("Error: FastAPI server not running on 127.0.0.1:8000")
            print("Please run `python -m agent.server` first.")
            break
            
    print("\nAgent Evaluation Summary:")
    success_count = sum(1 for r in results if r["success"])
    if results:
        print(f"Accuracy: {success_count}/{len(results)} ({(success_count/len(results))*100:.2f}%)")
        print("Note: To collect full benchmark results for the report, run all eval scripts and aggregate outputs.")
        
if __name__ == "__main__":
    run_agent_eval()
