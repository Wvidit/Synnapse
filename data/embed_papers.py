import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "dataset"
PAPERS_JSON = DATA_DIR / "papers.json"
INDEX_FILE = DATA_DIR / "abstracts.faiss"
ID_MAP_FILE = DATA_DIR / "id_map.json"

def embed_papers():
    print(f"Loading papers from {PAPERS_JSON}...")
    try:
        with open(PAPERS_JSON, 'r') as f:
            papers = json.load(f)
    except FileNotFoundError:
        print("papers.json not found. Run fetch_papers.py first.")
        return

    # model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Lighter
    model_name = "allenai/scibert_scivocab_uncased"
    
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)
    
    abstracts = []
    ids = []
    
    for paper in papers:
        if paper.get('abstract'):
            abstracts.append(paper['abstract'])
            ids.append(paper['arxiv_id'])
            
    if not abstracts:
        print("No abstracts to embed.")
        return

    print(f"Embedding {len(abstracts)} abstracts...")
    embeddings = model.encode(abstracts, show_progress_bar=True)
    
    print("Building FAISS index...")
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype('float32'))
    
    print(f"Saving index to {INDEX_FILE}...")
    faiss.write_index(index, str(INDEX_FILE))
    
    print(f"Saving ID map to {ID_MAP_FILE}...")
    with open(ID_MAP_FILE, 'w') as f:
        json.dump(ids, f)
        
    print("Done!")

if __name__ == "__main__":
    embed_papers()
