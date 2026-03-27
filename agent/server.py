import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

from .tools import (
    search_literature, 
    explore_citations, 
    generate_hypothesis, 
    verify_logic, 
    summarize_context, 
    lookup_taxonomy
)
from .policy import get_next_action

app = FastAPI(title="Synnapse ReAct Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    max_steps: int = 10
    token_budget: int = 8192
    context_policy: str = "compression_cache"
    
class ActionRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any]

@app.post("/tools/search")
def api_search(req: ActionRequest):
    return search_literature(**req.args)

@app.post("/tools/citations")
def api_citations(req: ActionRequest):
    return explore_citations(**req.args)

@app.post("/tools/hypothesize")
def api_hypothesize(req: ActionRequest):
    return generate_hypothesis(**req.args)

@app.post("/tools/verify")
def api_verify(req: ActionRequest):
    return verify_logic(**req.args)

@app.post("/tools/summarize")
def api_summarize(req: ActionRequest):
    return summarize_context(**req.args)

@app.post("/tools/taxonomy")
def api_taxonomy(req: ActionRequest):
    return lookup_taxonomy(**req.args)

@app.post("/agent/run")
def run_agent(req: QueryRequest):
    """
    Main ReAct loop using the Policy Controller.
    """
    context = []
    current_tokens = 0
    
    for step in range(req.max_steps):
        # 1. Get next action from Policy Controller
        state = {
            "query": req.query,
            "context_length": len(context),
            "token_budget_percent": current_tokens / max(1, req.token_budget),
            "has_hypothesis": any("hypothesis" in str(c) for c in context),
            "history": context
        }
        
        action = get_next_action(state)
        tool_name = action["tool_name"]
        
        if tool_name == "stop":
            break
            
        # 2. Execute action
        try:
            if tool_name == "explore_citations":
                # Extract arxiv ID, strip any "arxiv:" prefix
                import re
                arxiv_match = re.search(r'(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d)?)', req.query, re.IGNORECASE)
                paper_key = arxiv_match.group(1) if arxiv_match else req.query
                # Try with and without version suffix
                observation = explore_citations(paper_id=paper_key, depth=1)
            elif tool_name == "generate_hypothesis":
                context_text = " ".join(context[-3:]) if context else req.query
                observation = generate_hypothesis(context=context_text)
            elif tool_name == "verify_logic":
                observation = verify_logic(hypothesis=req.query, premises=context[-2:] if context else ["No premises."])
            elif tool_name == "summarize_context":
                observation = summarize_context(text=" ".join(context))
                # For Policy C, replace context with summary
                context = [observation]
            elif tool_name == "lookup_taxonomy":
                observation = lookup_taxonomy(query=req.query)
            elif tool_name == "search_literature":
                observation = search_literature(query=req.query, top_k=5)
            else:
                observation = f"Unknown tool: {tool_name}"
        except Exception as e:
            observation = f"Error executing {tool_name}: {e}"
            
        # 3. Update context
        context.append(f"Action: {tool_name}\nObservation: {observation}")
        current_tokens += len(str(observation).split()) # Mock token count
        
    return {
        "status": "success",
        "steps_taken": step + 1,
        "final_context": context,
        "policy_used": req.context_policy
    }

if __name__ == "__main__":
    uvicorn.run("agent.server:app", host="0.0.0.0", port=8000, reload=True)
