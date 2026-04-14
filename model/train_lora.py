import os
import re
import sys
import torch
from pathlib import Path

# All Paths for local execution
MODEL_DIR = Path(__file__).parent.parent / "model_out"
DATA_DIR = Path(__file__).parent.parent / "dataset"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
HUB_MODEL_NAME = "Wvidit/Synnapse-Qwen2.5-3B"


# ─── Reward Functions for GRPO ───────────────────────────────────────────────

def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks emitted by Qwen3 models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_completion_text(completion) -> str:
    """
    Extract plain text from a completion that may be:
    - a plain string
    - a list of message dicts [{"role": "assistant", "content": "..."}]
    """
    if isinstance(completion, str):
        return strip_think_tags(completion)
    if isinstance(completion, list):
        # Extract content from message dicts
        parts = []
        for msg in completion:
            if isinstance(msg, dict) and "content" in msg:
                parts.append(msg["content"])
        return strip_think_tags(" ".join(parts))
    return str(completion)


def format_reward(completions, **kwargs):
    """
    Reward structural compliance of model outputs.
    Scientific responses should contain substantive content, not be empty or
    just repeat the question.
    """
    rewards = []
    for completion in completions:
        text = get_completion_text(completion)
        score = 0.0

        # Base: non-trivial length
        if len(text.split()) >= 5:
            score += 0.3

        # Contains scientific-sounding structure
        if any(kw in text.lower() for kw in [
            "hypothesis", "abstract", "contribution", "belongs to",
            "paper", "method", "result", "experiment", "improve"
        ]):
            score += 0.4

        # Well-formed sentences (ends with period or colon)
        if text.rstrip().endswith(('.', ':', '"', "'")):
            score += 0.3

        rewards.append(score)
    return rewards


def factual_reward(completions, **kwargs):
    """
    Soft word-overlap reward between generated completion and ground-truth.
    Ground-truth outputs are passed via the 'answer' column in the dataset.
    """
    rewards = []
    ground_truths = kwargs.get("answer", [])

    for i, completion in enumerate(completions):
        text = get_completion_text(completion)
        if i < len(ground_truths) and ground_truths[i]:
            gt = ground_truths[i]
            pred_words = set(text.lower().split())
            gt_words = set(gt.lower().split())
            if gt_words:
                overlap = len(pred_words & gt_words) / len(gt_words)
                rewards.append(min(overlap, 1.0))
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def length_reward(completions, **kwargs):
    """
    Penalise too-short or too-long responses.
    Sweet spot: 20–200 tokens.
    """
    rewards = []
    for completion in completions:
        text = get_completion_text(completion)
        n_words = len(text.split())
        if n_words < 5:
            rewards.append(0.0)
        elif n_words < 20:
            rewards.append(0.3)
        elif n_words <= 200:
            rewards.append(1.0)
        elif n_words <= 300:
            rewards.append(0.5)
        else:
            rewards.append(0.2)
    return rewards


# ─── Remote Training Function ──────────────────────────────────────────

def train():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
    from huggingface_hub import login, HfApi

    # Login to HuggingFace using the token from Modal Secrets
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        try:
            user_info = HfApi().whoami(token=hf_token)
            print(f"Logged in to Hugging Face as: {user_info['name']}")
            print(f"Will push to: {HUB_MODEL_NAME}")
        except Exception as e:
            print(f"Warning: Could not verify HF user: {e}")
    else:
        print("WARNING: HF_TOKEN not found in environment. push_to_hub may fail.")

    # Create output directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ─── STAGE 1: Supervised Fine-Tuning ───
    print("=" * 60)
    print("STAGE 1: Supervised Fine-Tuning (SFT)")
    print("=" * 60)

    print("Loading datasets...")
    try:
        ds_triples = load_dataset("json", data_files=str(DATA_DIR / "triples.jsonl"))['train']
        ds_qa = load_dataset("json", data_files=str(DATA_DIR / "qa_tasks.jsonl"))['train']
        ds_hypo = load_dataset("json", data_files=str(DATA_DIR / "hypothesis.jsonl"))['train']

        dataset = concatenate_datasets([ds_triples, ds_qa, ds_hypo])
        dataset = dataset.shuffle(seed=42)

        split_ds = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_ds['train']
        eval_dataset = split_ds['test']
    except Exception as e:
        print(f"Error loading datasets (ensure JSONL files exist): {e}")
        return

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

    def formatting_prompts_func(example):
        if isinstance(example.get('instruction', ''), list):
            output_texts = []
            for i in range(len(example['instruction'])):
                messages = [
                    {"role": "system", "content": "You are a scientific research assistant."},
                    {"role": "user", "content": f"{example['instruction'][i]}\n{example.get('input', [''])[i]}"},
                    {"role": "assistant", "content": example['output'][i]},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                output_texts.append(text)
            return output_texts
        else:
            messages = [
                {"role": "system", "content": "You are a scientific research assistant."},
                {"role": "user", "content": f"{example['instruction']}\n{example.get('input', '')}"},
                {"role": "assistant", "content": example['output']},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    print("Setting up SFT Trainer...")
    training_args = SFTConfig(
        output_dir=str(MODEL_DIR / "sft_model"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,  # Fixed: 1e-4 is way too high for full fine-tuning without LoRA
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=2500,
        max_steps=2500,  # Increased from 1000
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    print("Starting SFT training...")
    trainer.train()

    print("Saving SFT model...")
    sft_path = str(MODEL_DIR / "sft_model")
    trainer.model.save_pretrained(sft_path)
    tokenizer.save_pretrained(sft_path)

    print(f"Pushing SFT model to HF Hub ({HUB_MODEL_NAME}-sft)...")
    trainer.model.push_to_hub(f"{HUB_MODEL_NAME}-sft", private=False)
    tokenizer.push_to_hub(f"{HUB_MODEL_NAME}-sft")
    print(f"✅ SFT model uploaded to https://huggingface.co/{HUB_MODEL_NAME}-sft")
    
    # ─── End of SFT ───────────────────────────────────────────────────────────
    print("✅ SFT complete.")


# ─── STAGE 2: GRPO (Separate Function) ─────────────────────────────────

def train_grpo():
    """
    GRPO RL fine-tuning stage.
    Loads the SFT model directly from HF Hub so it can be run independently
    of the SFT stage (as long as the SFT model has already been pushed).
    """
    import gc
    import os
    from datasets import load_dataset, concatenate_datasets
    from trl import GRPOTrainer, GRPOConfig
    from huggingface_hub import login, HfApi
    from transformers import AutoTokenizer

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Authenticate
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        try:
            user_info = HfApi().whoami(token=hf_token)
            print(f"Logged in to HF as: {user_info['name']}")
        except Exception as e:
            print(f"Warning: Could not verify HF user: {e}")
    else:
        print("WARNING: HF_TOKEN not set. push_to_hub may fail.")

    SFT_HUB = f"{HUB_MODEL_NAME}-sft"
    print(f"Loading SFT model from HF Hub: {SFT_HUB}")
    # Load base tokenizer to avoid corrupted extra_special_tokens in SFT Hub repo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Qwen-specific fix: Ensure pad_token is set correctly
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = "<|endoftext|>"
    
    print("Preparing GRPO dataset...")
    try:
        ds_qa = load_dataset("json", data_files=str(DATA_DIR / "qa_tasks.jsonl"))['train']
        ds_hypo = load_dataset("json", data_files=str(DATA_DIR / "hypothesis.jsonl"))['train']
        grpo_dataset = concatenate_datasets([ds_qa, ds_hypo]).shuffle(seed=42)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    def make_grpo_prompt(example):
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        example["prompt"] = [
            {"role": "system", "content": "You are a scientific research assistant."},
            {"role": "user", "content": f"{instruction}\n{inp}".strip()},
        ]
        example["answer"] = example.get("output", "")
        return example

    grpo_dataset = grpo_dataset.map(make_grpo_prompt)
    grpo_dataset = grpo_dataset.remove_columns(
        [c for c in grpo_dataset.column_names if c not in ["prompt", "answer"]]
    )

    print("Configuring GRPO Trainer...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    grpo_config = GRPOConfig(
        output_dir=str(MODEL_DIR / "grpo_output"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=10,
        max_steps=1000,
        bf16=True,
        max_grad_norm=0.3,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_completion_length=256,
        optim="paged_adamw_8bit",    # Better memory handling on single GPU
        # Generation overrides
        temperature=0.9,
        top_p=0.9,
    )

    trainer_grpo = GRPOTrainer(
        model=SFT_HUB,    # Load directly from HF Hub
        args=grpo_config,
        train_dataset=grpo_dataset,
        reward_funcs=[format_reward, factual_reward, length_reward],
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer_grpo.train()

    print(f"Pushing GRPO model to HF Hub ({HUB_MODEL_NAME})...")
    trainer_grpo.model.push_to_hub(HUB_MODEL_NAME, private=False)
    tokenizer.push_to_hub(HUB_MODEL_NAME)
    print(f"✅ GRPO complete. Model at https://huggingface.co/{HUB_MODEL_NAME}")


# ─── Local Entrypoints ────────────────────────────────────────────────────────

def main():
    """Run full pipeline: SFT then GRPO."""
    print("Running SFT...")
    train()
    print("Running GRPO...")
    train_grpo()


def grpo_only():
    """Run GRPO only (requires SFT model already on HF Hub)."""
    print(f"Starting GRPO from SFT checkpoint {HUB_MODEL_NAME}-sft ...")
    train_grpo()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "grpo":
        grpo_only()
    else:
        main()

