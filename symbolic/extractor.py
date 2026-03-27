import spacy
from typing import List, Dict

# Assumes en_core_sci_lg is installed
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
try:
    nlp = spacy.load("en_core_sci_lg")
except OSError:
    print("Warning: en_core_sci_lg not found. Using small blank model for fallback.")
    nlp = spacy.blank("en")

def extract_claims(text: str) -> List[Dict]:
    """
    Extracts Subject-Verb-Object triples mapping to predicates
    like causes(X, Y), improves(method, metric), contradicts(A, B).
    """
    doc = nlp(text)
    claims = []
    
    # Very basic SVO extraction logic
    for sent in doc.sents:
        subj = None
        obj = None
        verb = None
        
        for token in sent:
            if "subj" in token.dep_:
                subj = token.text
            if "obj" in token.dep_:
                obj = token.text
            if token.pos_ == "VERB":
                verb = token.lemma_
                
        if subj and verb and obj:
            # Map common verbs to our SMT logic predicates
            if verb in ["cause", "lead", "result"]:
                predicate = "causes"
            elif verb in ["improve", "increase", "enhance"]:
                predicate = "improves"
            elif verb in ["contradict", "conflict", "disprove"]:
                predicate = "contradicts"
            else:
                predicate = "related_to"
                
            claims.append({
                "predicate": predicate,
                "subject": subj.lower(),
                "object": obj.lower(),
                "original_text": sent.text
            })
            
    return claims

def serialize_to_smtlib(claims: List[Dict]) -> List[str]:
    """
    Convert extracted dict to SMT-LIB formatted strings (or simply 
    the variables needed for Z3 python API directly).
    """
    stmts = []
    for claim in claims:
        pred = claim['predicate']
        s = claim['subject'].replace(' ', '_').replace('-', '_')
        o = claim['object'].replace(' ', '_').replace('-', '_')
        
        # Example SMT-LIB style formulation
        # For our Python integrations, returning string representations is fine
        stmts.append(f"{pred}({s}, {o})")
    return stmts

if __name__ == "__main__":
    text = "The novel attention mechanism improves the BLEU score significantly. This contradicts previous theories."
    claims = extract_claims(text)
    print(claims)
    print(serialize_to_smtlib(claims))
