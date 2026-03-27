import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "model_out" / "final_lora"
DATA_DIR = Path(__file__).parent.parent / "dataset"

def evaluate_truthfulqa():
    print("Evaluating TruthfulQA... (Mock Harness)")
    # This would normally use the lm-evaluation-harness
    # os.system(f"python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,peft={MODEL_DIR} --tasks truthfulqa_mc --batch_size 8")
    pass

def evaluate_domain_qa():
    print("Evaluating Domain QA tasks...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    if not MODEL_DIR.exists():
        print(f"Skipping domain QA since LoRA weights {MODEL_DIR} not found.")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )
    
    model = PeftModel.from_pretrained(base_model, str(MODEL_DIR))
    model.eval()
    
    qa_file = DATA_DIR / "qa_tasks.jsonl"
    if not qa_file.exists():
        return
        
    correct = 0
    total = 0
    
    with open(qa_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompt = f"### Instruction:\n{data['instruction']}\n\n### Input:\n{data['input']}\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50)
                
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            # Basic eval logic...
            print(f"Q: {data['instruction'][:50]}...")
            print(f"Model A: {response.strip()}")
            print(f"True A : {data['output']}")
            print("-" * 40)
            
            total += 1
            if total >= 5: # Just test a few
                break
                
if __name__ == "__main__":
    evaluate_truthfulqa()
    evaluate_domain_qa()
