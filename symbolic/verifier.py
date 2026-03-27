from z3 import Solver, Bool, And, Not, sat, unknown
import threading

def run_with_timeout(func, args, timeout_sec):
    result = {"status": "unverified", "error": "timeout"}
    
    def target():
        try:
            res = func(*args)
            result.update(res)
        except Exception as e:
            result["error"] = str(e)
            
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout_sec)
    
    if thread.is_alive():
        print("Z3 verification timed out. Falling back.")
        return {"consistent": False, "status": "unverified_timeout"}
    return result

def verify_logic_core(hypothesis_fol: str, premises: list[str]) -> dict:
    s = Solver()
    
    # In a fully robust implementation, we'd parse the FOL string into Z3 Bools.
    # We mock the compilation for the scope of the scaffold:
    z3_vars = {}
    
    def get_var(name):
        if name not in z3_vars:
            z3_vars[name] = Bool(name)
        return z3_vars[name]

    # For premises like "improves(attention, score)"
    # We just treat the entire predicate as a boolean variable for basic consistency check
    # More advanced logic would use Uninterpreted Functions and Quantifiers.
    
    for p in premises:
        s.add(get_var(p) == True)
        
    # Check if hypothesis contradicts premises
    # Let's say hypothesis is "Not(improves(attention, score))"
    if hypothesis_fol.startswith("Not("):
        inner = hypothesis_fol[4:-1]
        s.add(Not(get_var(inner)) == True)
    else:
        s.add(get_var(hypothesis_fol) == True)
        
    result = s.check()
    
    if result == sat:
        return {"consistent": True, "status": "verified", "model": str(s.model())}
    elif result == unknown:
        return {"consistent": False, "status": "unverified"}
    else:
        return {"consistent": False, "status": "contradiction"}

def verify_hypothesis(hypothesis_fol: str, premises: list[str], timeout: int = 5) -> dict:
    """" Wraps the z3 solver with a timeout guardrail. """
    return run_with_timeout(verify_logic_core, (hypothesis_fol, premises), timeout)

if __name__ == "__main__":
    premises = ["improves(attention, bleu)", "causes(training, convergence)"]
    hypo_consistent = "causes(training, convergence)"
    hypo_contradict = "Not(improves(attention, bleu))"
    
    print("Test Consistent:", verify_hypothesis(hypo_consistent, premises))
    print("Test Contradiction:", verify_hypothesis(hypo_contradict, premises))
