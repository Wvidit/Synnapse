import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "dataset"
PAPERS_JSON = DATA_DIR / "papers.json"
TAXONOMY_FILE = DATA_DIR / "taxonomy.json"

def build_taxonomy():
    print(f"Loading papers from {PAPERS_JSON}...")
    try:
        with open(PAPERS_JSON, 'r') as f:
            papers = json.load(f)
    except FileNotFoundError:
        print("papers.json not found. Run fetch_papers.py first.")
        return

    # Base arXiv hierarchy
    taxonomy = {
        "arXiv": {
            "cs.LG": {},
            "cs.AI": {},
            "q-bio.BM": {}
        }
    }

    # Augment with S2 Fields of Study
    print("Augmenting taxonomy with Semantic Scholar fields...")
    for paper in papers:
        arxiv_field = paper.get('field') # cs.LG
        if not arxiv_field:
            continue
            
        if arxiv_field not in taxonomy["arXiv"]:
            taxonomy["arXiv"][arxiv_field] = {}

        s2_fields = paper.get('s2_fields', [])
        if not isinstance(s2_fields, list):
            s2_fields = []
            
        for field in s2_fields:
            # S2 field might be a dict {'category': 'Computer Science', 'source': '...' }
            # Or just strings depending on API version. 
            # Assuming strings based on the fields query 's2FieldsOfStudy'
            category = field.get('category') if isinstance(field, dict) else field
            if category:
                if category not in taxonomy["arXiv"][arxiv_field]:
                    taxonomy["arXiv"][arxiv_field][category] = []
                
                # We could attach papers to the leaves, but let's just build the tree structure.

    print(f"Saving taxonomy to {TAXONOMY_FILE}...")
    with open(TAXONOMY_FILE, 'w') as f:
        json.dump(taxonomy, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    build_taxonomy()
