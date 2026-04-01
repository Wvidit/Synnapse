import re
import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
FT_MODEL_NAME = "Wvidit/Synnapse-Qwen2.5-3B"   # Fine-tuned model on HF Hub
DATA_DIR = Path(__file__).parent.parent / "dataset"


def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks emitted by Qwen models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def evaluate_truthfulqa():
    print("Evaluating TruthfulQA... (Mock Harness)")
    pass


def evaluate_domain_qa():
    print("Evaluating Domain QA tasks...")

    print(f"Fine-tuned model: {FT_MODEL_NAME}")

    # Load tokenizer from base model (same tokenizer, fine-tuning doesn't change it)
    ft_tokenizer = AutoTokenizer.from_pretrained(FT_MODEL_NAME)
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    base_model.eval()

    print("Loading fine-tuned model from HF Hub...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        FT_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    ft_model.eval()

    qa_file = DATA_DIR / "qa_tasks.jsonl"
    if not qa_file.exists():
        print(f"Dataset not found: {qa_file}")
        return

    base_correct = 0
    ft_correct = 0
    base_overlap_total = 0
    ft_overlap_total = 0
    total = 0

    def word_overlap(pred, true_ans):
        pred_words = set(pred.lower().split())
        true_words = set(true_ans.lower().split())
        if not true_words:
            return 0.0
        return len(pred_words.intersection(true_words)) / len(true_words)

    def make_chat_prompt(instruction, inp, tokenizer):
        """Format prompt using ChatML template."""
        messages = [
            {"role": "system", "content": "You are a scientific research assistant."},
            {"role": "user", "content": f"{instruction}\n{inp}"},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    with open(qa_file, 'r') as f:
        for line in f:
            data = json.loads(line)

            # Format prompts using ChatML
            base_prompt = make_chat_prompt(data['instruction'], data.get('input', ''), base_tokenizer)
            ft_prompt = make_chat_prompt(data['instruction'], data.get('input', ''), ft_tokenizer)

            base_inputs = base_tokenizer(base_prompt, return_tensors="pt").to(base_model.device)
            ft_inputs = ft_tokenizer(ft_prompt, return_tensors="pt").to(ft_model.device)

            with torch.no_grad():
                base_outputs = base_model.generate(**base_inputs, max_new_tokens=100)
                ft_outputs = ft_model.generate(**ft_inputs, max_new_tokens=100)

            base_response = strip_think_tags(
                base_tokenizer.decode(base_outputs[0][base_inputs.input_ids.shape[1]:], skip_special_tokens=True)
            )
            ft_response = strip_think_tags(
                ft_tokenizer.decode(ft_outputs[0][ft_inputs.input_ids.shape[1]:], skip_special_tokens=True)
            )

            true_ans = data['output'].strip()

            base_em = int(base_response.strip().lower() == true_ans.lower())
            ft_em = int(ft_response.strip().lower() == true_ans.lower())
            base_correct += base_em
            ft_correct += ft_em

            base_ov = word_overlap(base_response, true_ans)
            ft_ov = word_overlap(ft_response, true_ans)
            base_overlap_total += base_ov
            ft_overlap_total += ft_ov

            print(f"Q: {data['instruction'][:100]}...")
            print(f"  Base:      {base_response[:200].strip()} (EM: {base_em}, Overlap: {base_ov:.2f})")
            print(f"  Finetuned: {ft_response[:200].strip()} (EM: {ft_em}, Overlap: {ft_ov:.2f})")
            print(f"  Ground T:  {true_ans[:200]}")
            print("-" * 60)

            total += 1
            if total >= 50:
                break

    print(f"\n{'='*60}")
    print(f"Evaluation Stats (Over {total} samples):")
    print(f"{'='*60}")
    print(f"Base Model     - Exact Match: {base_correct/total:.2%}  |  Avg Word Overlap: {base_overlap_total/total:.2f}")
    print(f"Fine-Tuned     - Exact Match: {ft_correct/total:.2%}  |  Avg Word Overlap: {ft_overlap_total/total:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    evaluate_truthfulqa()
    evaluate_domain_qa()
