import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "dataset"
PAPERS_JSON = DATA_DIR / "papers.json"
OUT_QA = DATA_DIR / "qa_tasks.jsonl"
OUT_HYPOTHESIS = DATA_DIR / "hypothesis.jsonl"

def generate_datasets():
    print(f"Loading papers from {PAPERS_JSON}...")
    try:
        with open(PAPERS_JSON, 'r') as f:
            papers = json.load(f)
    except FileNotFoundError:
        print("papers.json not found.")
        return

    print("Generating hypothesis-conclusion and Research QA pairs...")
    qa_data = []
    hypothesis_data = []
    
    for paper in papers:
        # In a real scenario, this would use GPT-3.5 or Llama to extract hypothesis
        # Here we mock the generation process to create structural data
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        field = paper.get('field', '')
        arxiv_id = paper.get('arxiv_id', '')
        
        if not title or not abstract:
            continue
            
        # 1. Hypothesis Generation
        instruction_h = "Extract the core hypothesis and the supporting findings from the provided abstract."
        output_h = f"Hypothesis: The methods described in '{title}' improve baseline performance.\nConclusion: The experiments confirm significant advantages."
        
        hypothesis_data.append({
            "instruction": instruction_h,
            "input": abstract,
            "output": output_h
        })
        
        # 2. QA Pairs Generation
        questions = [
            (f"What is the main contribution of the paper '{title}'?", f"The main contribution is detailed in its abstract: {abstract[:100]}..."),
            (f"Which field does the ID {arxiv_id} belong to?", f"It belongs to {field}."),
            (f"Can you provide papers related to {field}?", f"Yes, '{title}' is a prominent paper in {field}.")
        ]
        
        q, a = random.choice(questions)
        qa_data.append({
            "instruction": q,
            "input": "",
            "output": a
        })

    print(f"Saving {len(qa_data)} QA pairs to {OUT_QA}...")
    with open(OUT_QA, 'w') as f:
        for t in qa_data:
            f.write(json.dumps(t) + '\n')
            
    print(f"Saving {len(hypothesis_data)} Hypothesis pairs to {OUT_HYPOTHESIS}...")
    with open(OUT_HYPOTHESIS, 'w') as f:
        for t in hypothesis_data:
            f.write(json.dumps(t) + '\n')
            
    print("Done!")

if __name__ == "__main__":
    generate_datasets()
