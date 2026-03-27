import json
import networkx as nx
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "dataset"
PAPERS_JSON = DATA_DIR / "papers.json"
GRAPH_FILE = DATA_DIR / "citation_graph.graphml"

def build_citation_graph():
    print(f"Loading papers from {PAPERS_JSON}...")
    try:
        with open(PAPERS_JSON, 'r') as f:
            papers = json.load(f)
    except FileNotFoundError:
        print("papers.json not found. Run fetch_papers.py first.")
        # Create a dummy list for testing if wanted, or just return.
        return

    G = nx.DiGraph()

    print("Building citation graph...")
    # Add nodes
    for paper in papers:
        # arXiv IDs might have versions like '2310.12345v1', use base if needed
        # but let's just stick to the string
        node_id = paper['arxiv_id']
        G.add_node(
            node_id, 
            title=paper.get('title', ''), 
            abstract=paper.get('abstract', ''), 
            year=paper.get('year', ''), 
            field=paper.get('field', '')
        )

    # Add edges
    edge_count = 0
    for paper in papers:
        u = paper['arxiv_id']
        citations = paper.get('citations', [])
        for cit in citations:
            # We cite papers. If the cited paper is in our dataset, add an edge.
            # Semantic scholar gives us the title and sometimes arxivId of the cited paper.
            v_arxiv = cit.get('arxivId')
            v_s2 = cit.get('paperId')
            
            # Use arxivId if available, else Semantic Scholar ID
            v = f"arxiv:{v_arxiv}" if v_arxiv else f"s2:{v_s2}"
            if not v_arxiv and not v_s2:
                continue
                
            # If we don't strictly require the node to exist before adding the edge:
            # In NetworkX, adding an edge creates the nodes if they don't exist.
            # We only have rich metadata for nodes we explicitly fetched.
            if u != v:
                G.add_edge(u, v)
                edge_count += 1

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    print(f"Saving graph to {GRAPH_FILE}...")
    nx.write_graphml_lxml(G, GRAPH_FILE)
    print("Done!")

if __name__ == "__main__":
    build_citation_graph()
