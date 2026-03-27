import networkx as nx
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "dataset"
GRAPH_FILE = DATA_DIR / "citation_graph.graphml"
OUT_TRIPLES = DATA_DIR / "triples.jsonl"

def generate_triples():
    print(f"Loading graph from {GRAPH_FILE}...")
    try:
        G = nx.read_graphml(GRAPH_FILE)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    print("Generating graph-verbalized triples...")
    triples = []
    
    for u, v in G.edges():
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        
        # We might not have metadata for v if it wasn't scraped directly
        u_title = u_data.get('title', u)
        v_title = v_data.get('title', v)
        
        # Build Alpaca formatted instruction
        instruction = "Synthesize a relationship between the following two research papers based on their citations."
        input_text = f"Paper A: {u_title}\nPaper B: {v_title}"
        output_text = f"The paper '{u_title}' extends the work of '{v_title}' by building upon its foundations. '{v_title}' is highly cited in the field, indicating its foundational relevance."
        
        triples.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
        
    print(f"Generated {len(triples)} triples.")
    print(f"Saving to {OUT_TRIPLES}...")
    with open(OUT_TRIPLES, 'w') as f:
        for t in triples:
            f.write(json.dumps(t) + '\n')
            
    print("Done!")

if __name__ == "__main__":
    generate_triples()
