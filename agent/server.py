"""
Synnapse ReAct Agent API
=========================
FastAPI server implementing the agent loop with context-policy-aware
tool execution. The context policy from the request controls both
tool selection (via the Policy Controller) and context management.
"""

import re
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


# ─── Direct tool endpoints ────────────────────────────────────────────────────

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


# ─── Context management ──────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Approximate token count from word count."""
    return len(text.split())


def _apply_context_policy(
    context: List[str],
    new_entry: str,
    policy: str,
    token_budget: int,
) -> List[str]:
    """
    Apply the requested context management policy after adding a new entry.

    Policies:
      naive             — Append everything; FIFO pop if over budget.
      rag_retrieval     — Keep only the top-K most relevant entries.
      compression_cache — Compress old entries, cache verified facts.
    """
    context.append(new_entry)
    total = _estimate_tokens(" ".join(context))

    if policy == "naive":
        # FIFO truncation when budget exceeded
        while total > token_budget and len(context) > 1:
            context.pop(0)
            total = _estimate_tokens(" ".join(context))

    elif policy == "rag_retrieval":
        # Keep most relevant: prioritise entries containing query keywords,
        # then most recent.  Keep at most 4 entries.
        max_entries = 4
        if len(context) > max_entries:
            # Score each entry by whether it has substantive content
            scored = []
            for i, c in enumerate(context):
                # Prefer entries with real results, not just action labels
                has_content = len(c.split()) > 30
                is_error = "error" in c.lower()
                score = (0 if is_error else 1) + (1 if has_content else 0) + i * 0.01
                scored.append((score, i, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            context = [c for _, _, c in sorted(scored[:max_entries], key=lambda x: x[1])]

    elif policy == "compression_cache":
        # Cache verified facts; compress the rest when over half budget
        if total > token_budget // 2 and len(context) > 2:
            verified = [c for c in context if "verified" in c.lower() or "consistent" in c.lower()]
            # Keep the last 2 entries + verified facts, compress the middle
            to_compress = context[:-2]
            to_compress = [c for c in to_compress if c not in verified]
            if to_compress:
                compressed_text = " ".join(to_compress)
                # Take the key sentences (first 150 words)
                words = compressed_text.split()
                summary = " ".join(words[:150])
                context = verified + [f"[CONTEXT SUMMARY]: {summary}"] + context[-2:]

    return context


# ─── Tool execution ──────────────────────────────────────────────────────────

def _execute_tool(tool_name: str, query: str, context: List[str]) -> Any:
    """
    Execute a tool and return the observation.
    Tool arguments are derived from the query and accumulated context.
    """
    if tool_name == "search_literature":
        return search_literature(query=query, top_k=5)

    elif tool_name == "explore_citations":
        # Try to extract arxiv ID from query
        arxiv_match = re.search(
            r'(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d)?)', query, re.IGNORECASE
        )
        paper_key = arxiv_match.group(1) if arxiv_match else query
        return explore_citations(paper_id=paper_key, depth=1)

    elif tool_name == "generate_hypothesis":
        context_text = " ".join(context[-3:]) if context else query
        return generate_hypothesis(context=context_text)

    elif tool_name == "verify_logic":
        # Build premises from recent context
        premises = []
        for c in context[-3:]:
            # Extract claim-like sentences from context
            if c and len(c.split()) > 5:
                premises.append(c[:200])
        if not premises:
            premises = ["No premises available."]
        return verify_logic(hypothesis=query, premises=premises)

    elif tool_name == "summarize_context":
        if context:
            return summarize_context(text=" ".join(context))
        return "No context to summarize."

    elif tool_name == "lookup_taxonomy":
        return lookup_taxonomy(query=query)

    else:
        return f"Unknown tool: {tool_name}"


# ─── Final answer synthesis ──────────────────────────────────────────────────

def _synthesize_answer(query: str, context: List[str], tools_used: List[str]) -> str:
    """
    Call the LLM to produce a conversational answer from collected tool observations.
    Falls back to an extractive summary if the LLM is unavailable.
    """
    from .tools import generate_llm_response

    # Build a concise evidence block from tool observations
    evidence_parts = []
    for entry in context:
        # Extract just the observation text (skip "Action: ..." line)
        if "Observation:" in entry:
            obs = entry.split("Observation:", 1)[1].strip()
            # Truncate very long observations
            if len(obs) > 400:
                obs = obs[:400] + "..."
            evidence_parts.append(obs)
        else:
            if len(entry) > 400:
                entry = entry[:400] + "..."
            evidence_parts.append(entry)

    evidence = "\n---\n".join(evidence_parts)

    prompt = (
        "<|im_start|>system\n"
        "You are Synnapse, a neurosymbolic scientific research assistant. "
        "You have just executed a series of research tools to answer the user's question. "
        "Based on the evidence gathered below, provide a clear, helpful, and well-structured answer. "
        "Be conversational but precise. Use bullet points or bold text for key findings. "
        "Do NOT mention tool names or internal steps — just answer naturally.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"My question: {query}\n\n"
        f"Evidence gathered:\n{evidence}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    answer = generate_llm_response(prompt, max_new_tokens=300)

    # If LLM failed, build a simple extractive answer
    if not answer or "not loaded" in answer.lower():
        answer = _extractive_fallback(query, context)

    return answer


def _extractive_fallback(query: str, context: List[str]) -> str:
    """Build a readable answer from tool outputs when the LLM is unavailable."""
    import json

    parts = []
    for entry in context:
        if "Observation:" not in entry:
            continue
        obs_raw = entry.split("Observation:", 1)[1].strip()

        try:
            parsed = json.loads(obs_raw)
        except (json.JSONDecodeError, TypeError):
            parsed = None

        if parsed:
            if isinstance(parsed, dict):
                if "results" in parsed and isinstance(parsed["results"], list):
                    for r in parsed["results"][:5]:
                        parts.append(str(r))
                elif "hypothesis" in parsed:
                    parts.append(f"💡 {parsed['hypothesis']}")
                elif "source" in parsed:
                    line = f"📄 {parsed['source']}"
                    if "connected_papers" in parsed:
                        titles = [p.get("title", "") for p in parsed["connected_papers"][:3]]
                        line += "\n  → " + "\n  → ".join(titles)
                    parts.append(line)
                elif "taxonomy_matches" in parsed:
                    matches = parsed["taxonomy_matches"][:5]
                    parts.append("🌳 Taxonomy:\n" + "\n".join(f"  • {m.get('key', '')}" for m in matches))
                elif "verdict" in parsed:
                    parts.append(f"✅ {parsed['verdict']}")
                elif "error" in parsed:
                    continue  # skip errors
                else:
                    parts.append(str(parsed))
        elif obs_raw and len(obs_raw) > 10:
            parts.append(obs_raw[:300])

    if parts:
        return "\n\n".join(parts)
    return "I searched the literature but couldn't find a strong match for your query. Could you try rephrasing?"


# ─── Main agent loop ─────────────────────────────────────────────────────────

@app.post("/agent/run")
def run_agent(req: QueryRequest):
    """
    Main ReAct loop with context-policy-aware tool selection and management.
    After all tools complete, synthesizes a final conversational answer.
    """
    import json as _json

    context: List[str] = []
    current_tokens = 0
    tools_used: List[str] = []
    step = 0

    for step in range(req.max_steps):
        # 1. Build state for the policy controller
        state = {
            "query": req.query,
            "context_length": len(context),
            "token_budget_percent": current_tokens / max(1, req.token_budget),
            "has_hypothesis": any("hypothesis" in str(c).lower() for c in context),
            "history": context,
            "context_policy": req.context_policy,
        }

        # 2. Get next action from policy
        action = get_next_action(state)
        tool_name = action["tool_name"]

        if tool_name == "stop":
            break

        # 3. Execute the tool
        try:
            observation = _execute_tool(tool_name, req.query, context)
        except Exception as e:
            observation = f"Error executing {tool_name}: {e}"

        tools_used.append(tool_name)

        # 4. Add to context with policy-based management
        obs_str = _json.dumps(observation) if isinstance(observation, (dict, list)) else str(observation)
        new_entry = f"Action: {tool_name}\nObservation: {obs_str}"
        context = _apply_context_policy(
            context, new_entry, req.context_policy, req.token_budget
        )

        # If summarize replaced context, reset token count
        if tool_name == "summarize_context" and req.context_policy == "compression_cache":
            current_tokens = _estimate_tokens(" ".join(context))
        else:
            current_tokens += _estimate_tokens(obs_str)

    # 5. Synthesize a final conversational answer from all tool results
    answer = _synthesize_answer(req.query, context, tools_used)

    return {
        "status": "success",
        "answer": answer,
        "steps_taken": step + 1,
        "final_context": context,
        "tools_used": tools_used,
        "policy_used": req.context_policy,
        "tokens_consumed": current_tokens,
    }


if __name__ == "__main__":
    uvicorn.run("agent.server:app", host="0.0.0.0", port=8000, reload=True)

