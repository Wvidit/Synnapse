import arxiv
import requests
import json
import time
from pathlib import Path
from typing import List, Dict

# Target categories: cs.LG (Machine Learning), cs.AI (Artificial Intelligence), q-bio.BM (Biomolecules)
CATEGORIES = ["cs.LG", "cs.AI", "q-bio.BM"]
MAX_RESULTS = 50  # Keep small for testing
DATA_DIR = Path(__file__).parent.parent / "dataset"
DATA_DIR.mkdir(exist_ok=True, parents=True)

def fetch_arxiv_papers(category: str, max_results: int = 100) -> List[Dict]:
    print(f"Fetching {max_results} papers for {category} from arXiv...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    for result in client.results(search):
        papers.append({
            "arxiv_id": result.get_short_id(),
            "title": result.title,
            "abstract": result.summary,
            "year": result.published.year,
            "field": category,
            "published": result.published.isoformat(),
            "authors": [author.name for author in result.authors]
        })
    print(f"Found {len(papers)} papers for {category}.")
    return papers

def enrich_with_semantic_scholar_batch(papers: List[Dict]) -> List[Dict]:
    """
    Uses Semantic Scholar's Batch API to avoid rate limits 
    (which are very strict for individual GET requests without an API key).
    """
    print("Enriching with Semantic Scholar (batch API)...")
    enriched = []
    
    # Semantic Scholar allows up to 500 IDs per batch request.
    # We will chunk into sizes of 50 to be safe.
    batch_size = 50
    
    for i in range(0, len(papers), batch_size):
        chunk = papers[i:i + batch_size]
        
        # Prepare list of arXiv IDs in the format expected by S2
        s2_ids = [f"arXiv:{p['arxiv_id'].split('v')[0]}" for p in chunk]
        
        url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        params = {
            "fields": "citations,citations.paperId,citations.title,citations.authors,citations.year,s2FieldsOfStudy,citations.externalIds"
        }
        
        # We might still get 429 if we call the batch API too frequently. 
        # Add basic exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, params=params, json={"ids": s2_ids})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Process the newly enriched chunk
                    for idx, paper_data in enumerate(data):
                        original_paper = chunk[idx]
                        
                        # API returns null for papers it doesn't recognize
                        if paper_data is None:
                            enriched.append(original_paper)
                            continue
                            
                        original_paper["s2_fields"] = paper_data.get("s2FieldsOfStudy", [])
                        
                        citations = []
                        for c in paper_data.get("citations", []):
                            c_arxiv = c.get("externalIds", {}).get("ArXiv", None)
                            citations.append({
                                "paperId": c.get("paperId"),
                                "arxivId": c_arxiv,
                                "title": c.get("title"),
                                "year": c.get("year")
                            })
                        original_paper["citations"] = citations
                        enriched.append(original_paper)
                    
                    break # Success, break retry loop
                    
                elif response.status_code == 429:
                    wait_time = (attempt + 1) * 3
                    print(f"Rate limited on batch. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error {response.status_code}: {response.text}")
                    # Keep originals without S2 data
                    enriched.extend(chunk)
                    break
                    
            except Exception as e:
                print(f"Exception during batch fetch: {e}")
                enriched.extend(chunk)
                break
                
        # 3 seconds delay between batches to be respectful to the free API limit
        print(f"Processed batch ({len(enriched)}/{len(papers)} Total)")
        time.sleep(3)
        
    return enriched

def main():
    all_papers = []
    for cat in CATEGORIES:
        cat_papers = fetch_arxiv_papers(cat, max_results=MAX_RESULTS) 
        enriched = enrich_with_semantic_scholar_batch(cat_papers)
        all_papers.extend(enriched)
        
    output_file = DATA_DIR / "papers.json"
    with open(output_file, 'w') as f:
        json.dump(all_papers, f, indent=2)
    print(f"Saved {len(all_papers)} enriched papers to {output_file}")

if __name__ == "__main__":
    main()
