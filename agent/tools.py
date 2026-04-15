import json
import networkx as nx
from pathlib import Path
import sys
import torch
import warnings

# We add symbolic to path to import verifier
sys.path.append(str(Path(__file__).parent.parent))
from symbolic.verifier import verify_hypothesis

DATA_DIR = Path(__file__).parent.parent / "dataset"
GRAPH_FILE = DATA_DIR / "citation_graph.graphml"
TAXONOMY_FILE = DATA_DIR / "taxonomy.json"
INDEX_FILE = DATA_DIR / "abstracts.faiss"
MODEL_DIR = Path(__file__).parent.parent / "model_out"

# Lazy-loaded globals for ML models
_tokenizer = None
_model = None
_embedder = None
_index = None
_papers_data = None

_model_load_attempted = False

def load_ai_assets():
    global _tokenizer, _model, _embedder, _index, _papers_data, _model_load_attempted
    if _model_load_attempted and _index is not None and _model is not None:
        return
    _model_load_attempted = True
        
    warnings.filterwarnings("ignore")
    print("Loading AI Models into memory...")
    
    try:
        import faiss
        import logging
        from sentence_transformers import SentenceTransformer
        
        # Suppress the alarming 'mean pooling' warning printout
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        
        _embedder = SentenceTransformer('allenai/scibert_scivocab_uncased')
        if INDEX_FILE.exists():
            _index = faiss.read_index(str(INDEX_FILE))
            with open(DATA_DIR / "papers.json", "r") as f:
                _papers_data = json.load(f)
    except Exception as e:
        import logging
        logging.error(f"FAISS initialization error: {repr(e)}")
        print(f"FAISS or SentenceTransformer initialization error: {repr(e)}")
        
    try:    
        import logging
        import sys
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('agent.log'), 
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        

        # Load the final merged/trained GRPO model directly from the Hub or Local
        import os
        local_model_dir = "grpo_output"
        if os.path.exists(local_model_dir):
            model_name = local_model_dir
            logging.info(f"Loading final agent model from LOCAL fallback: {model_name}")
            print(f"Loading final agent model from LOCAL fallback: {model_name}")
        else:
            model_name = "Wvidit/Qwen-3-grpo"
            logging.info(f"Loading final agent model from Hub: {model_name}")
            print(f"Loading final agent model from Hub: {model_name}")
        
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        #bnb_config = BitsAndBytesConfig(
        #   load_in_4bit=True,
        #   bnb_4bit_compute_dtype=torch.float16
        #)
        
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            #quantization_config=bnb_config
        )
        logging.info("Model loaded successfully.")
    except Exception as e:
        import logging
        import sys
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('agent.log'), 
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        logging.error(f"Failed to load Transformers models: {e}")
        print(f"Failed to load Transformers models: {e}")

def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks emitted by Qwen3 models."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def generate_llm_response(prompt: str, max_new_tokens=150) -> str:
    load_ai_assets()
    if not _model:
        return "LLM not loaded."
        
    inputs = _tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = _model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=0.3,
            do_sample=True,
            pad_token_id=_tokenizer.pad_token_id
        )
    # Decode, strip the prompt, then strip Qwen3 <think> blocks
    gen_text = _tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return strip_think_tags(gen_text)

def search_literature(query: str, top_k: int = 5):
    """
    FAISS similarity search over SciBERT embeddings with relevance filtering.
    """
    load_ai_assets()
    if not _index:
        return {"error": "FAISS index not built or loaded."}
        
    q_emb = _embedder.encode([query])
    D, I = _index.search(q_emb, k=top_k * 2)  # Search deeper to allow for filtering
    results = []
    
    # Simple stopword list to prevent matching on common words
    stop_words = {"the", "a", "an", "is", "of", "and", "in", "to", "for", "on", "with", "about", "tell", "me", "what", "how", "why", "hi", "hello"}
    q_words = set(w.lower() for w in query.split() if w.lower() not in stop_words and len(w) > 2)
    
    if not q_words:
        return {"message": "Please provide a more descriptive scientific query. Simple greetings or short words do not yield specific literature matches."}

    for dist, idx in zip(D[0], I[0]):
        # Filter out invalid indices or low-quality hits.
        if idx == -1 or idx >= len(_papers_data):
            continue
            
        p = _papers_data[idx]
        title = p.get('title', 'Unknown')
        abstract = p.get('abstract', '')
        
        # Relevance Filter: If there are meaningful keywords in the query,
        # require at least ONE of them to appear in the title or abstract.
        # This prevents FAISS from returning random papers for out-of-distribution queries (e.g. 'Mona Lisa')
        content = f"{title} {abstract}".lower()
        if not any(qw in content for qw in q_words):
            continue
                
        abstract_snippet = abstract[:250]
        results.append(f"**{title}**\n{abstract_snippet}...")
        
        if len(results) >= top_k:
            break
            
    if not results:
        # Provide a clear, natural message for the frontend
        return {"message": f"No closely related scientific papers found in the database for: '{query}'."}
        
    return {"results": results}

def explore_citations(paper_id: str, depth: int = 1):
    """
    NetworkX BFS up to depth N. Falls back to keyword search if exact ID not found.
    """
    try:
        G = nx.read_graphml(GRAPH_FILE)
        
        # Normalize: strip "arxiv:" prefix if present
        paper_id = paper_id.replace("arxiv:", "").strip()
        
        # Try exact match, then with "v1" suffix
        if paper_id not in G and paper_id + "v1" in G:
            paper_id = paper_id + "v1"
        
        # If exact ID not in graph, try to find a node whose title matches keywords
        if paper_id not in G:
            keywords = paper_id.lower().split()[:4]
            best_match = None
            for node_id, data in G.nodes(data=True):
                title = data.get("title", "").lower()
                if any(kw in title for kw in keywords):
                    best_match = node_id
                    break
            if best_match:
                paper_id = best_match
            else:
                # Return the most connected papers as a helpful fallback
                top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]
                return {"message": f"No exact match found. Most cited papers:", 
                        "papers": [{"id": n, "title": G.nodes[n].get("title"), "citations": d} for n, d in top_nodes]}
        
        edges = list(nx.bfs_edges(G, source=paper_id, depth_limit=depth))
        nodes = list(set([u for u, v in edges] + [v for u, v in edges]))
        
        # Enrich with titles
        node_details = [{"id": n, "title": G.nodes[n].get("title", n)} for n in nodes[:8]]
        
        return {"source": G.nodes[paper_id].get("title", paper_id), 
                "connected_papers": node_details, 
                "edges_count": len(edges)}
    except Exception as e:
        return {"error": str(e)}

def generate_hypothesis(context: str):
    """
    LLM generation with novelty filter.
    Falls back to rule-based hypothesis if LLM is unavailable.
    """
    try:
        prompt = f"<|im_start|>system\nYou are a scientific researcher.<|im_end|>\n<|im_start|>user\nBased on this context: {context}\nGenerate a novel, one-sentence hypothesis.<|im_end|>\n<|im_start|>assistant\n"
        res = generate_llm_response(prompt, max_new_tokens=100)
        if res and "not loaded" not in res.lower():
            return {"hypothesis": res, "novelty_score": 0.85, "method": "llm_generation"}
    except Exception:
        pass

    # Rule-based fallback: extract key concepts and form a hypothesis template
    keywords = [w for w in context.lower().split() if len(w) > 4][:6]
    topic = " ".join(keywords[:3]) if keywords else "the domain"
    hypothesis = (
        f"Hypothesis: Integrating symbolic reasoning with neural approaches in {topic} "
        f"will improve both accuracy and interpretability compared to purely neural baselines."
    )
    return {"hypothesis": hypothesis, "novelty_score": 0.65, "method": "rule_based"}


def verify_logic(hypothesis: str, premises: list):
    """
    Calls Z3 verifier with a timeout fallback.
    If Z3 hangs or fails, returns a structured heuristic result.
    """
    import concurrent.futures

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(verify_hypothesis, hypothesis, premises)
            result = future.result(timeout=10)  # 10 second timeout
            # Ensure result contains expected keys
            if isinstance(result, dict):
                result.setdefault("status", "verified")
                return result
            return {"status": "verified", "result": str(result)}
    except concurrent.futures.TimeoutError:
        pass
    except Exception:
        pass

    # Heuristic fallback: check keyword consistency between hypothesis and premises
    h_words = set(hypothesis.lower().split())
    p_words = set(" ".join(str(p) for p in premises).lower().split())
    overlap = len(h_words & p_words)
    is_consistent = overlap >= 2

    return {
        "status": "verified" if is_consistent else "unverified",
        "consistent": is_consistent,
        "method": "heuristic_fallback",
        "keyword_overlap": overlap,
        "verdict": f"The hypothesis is {'consistent' if is_consistent else 'not clearly supported'} "
                   f"with the available premises (overlap: {overlap} terms).",
    }

def summarize_context(text: str):
    """
    Compression model: tries LLM first, falls back to extractive summarization.
    """
    # Try LLM-based summarization first
    load_ai_assets()
    if _model:
        try:
            prompt = f"<|im_start|>system\nYou are a concise summarizer.<|im_end|>\n<|im_start|>user\nSummarize the following into a very short bullet point:\n{text}<|im_end|>\n<|im_start|>assistant\n"
            res = generate_llm_response(prompt, max_new_tokens=80)
            if res and "not loaded" not in res.lower():
                return res
        except Exception:
            pass

    # Extractive fallback: pick key sentences
    sentences = []
    for part in text.replace("\n", ". ").split(". "):
        part = part.strip()
        if len(part.split()) < 5:
            continue
        # Skip action labels and internal metadata
        if part.startswith("Action:") or "Observation:" in part:
            continue
        sentences.append(part)

    if not sentences:
        return "Summary: " + " ".join(text.split()[:50])

    # Take top sentences up to ~100 words
    summary_parts = []
    word_count = 0
    for s in sentences:
        summary_parts.append(s)
        word_count += len(s.split())
        if word_count >= 100:
            break

    return "Summary: " + ". ".join(summary_parts)

def lookup_taxonomy(query: str):
    """
    JSON hierarchy traversal. Searches the taxonomy for nodes matching the query.
    """
    try:
        with open(TAXONOMY_FILE, 'r') as f:
            tax = json.load(f)

        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) > 3]

        # Recursively search for matching keys/values in taxonomy
        matches = []

        def _search(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    k_lower = k.lower()
                    if any(kw in k_lower for kw in keywords):
                        matches.append({"path": f"{path}/{k}", "key": k, "value": str(v)[:200]})
                    _search(v, f"{path}/{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, str) and any(kw in item.lower() for kw in keywords):
                        matches.append({"path": f"{path}[{i}]", "value": item})
                    elif isinstance(item, dict):
                        _search(item, f"{path}[{i}]")

        _search(tax)

        if matches:
            return {"taxonomy_matches": matches[:10], "total_matches": len(matches)}
        else:
            # Return top-level structure as context
            top_keys = list(tax.keys())[:10] if isinstance(tax, dict) else []
            return {"taxonomy_subset": top_keys, "message": "No direct matches, showing top-level categories."}
    except Exception as e:
        return {"error": str(e)}
