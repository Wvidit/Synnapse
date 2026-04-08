"""
local_grpo.py
=============
Runs SFT and GRPO reinforcement fine-tuning locally, adapted from the Modal pipeline.
Uses pure PyTorch execution.
"""

import os
import re
import sys
import site
import argparse
from pathlib import Path

# Fix 'libcudart.so.12' missing errors by native-linking pip's bundled CUDA runtime
for p in site.getsitepackages() + [site.getusersitepackages()]:
    cuda_path = os.path.join(p, "nvidia", "cuda_runtime", "lib")
    if os.path.exists(cuda_path):
        os.environ["LD_LIBRARY_PATH"] = cuda_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# Monkey-patch to prevent TRL from natively crashing on a broken local vLLM installation
sys.modules['vllm'] = None

import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from huggingface_hub import login, HfApi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"          # target base model
SFT_HUB    = "Wvidit/Synnapse-Qwen3-4B-sft"   # SFT checkpoint on HF Hub
GRPO_HUB   = "Wvidit/Synnapse-Qwen3-4B"        # final GRPO model destination
DATA_DIR   = Path("dataset")                   # local relative path
OUTPUT_DIR_SFT = Path("sft_output")
OUTPUT_DIR_GRPO = Path("grpo_output")


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _get_text(completion) -> str:
    if isinstance(completion, str):
        return _strip_think(completion)
    if isinstance(completion, list):
        parts = [m["content"] for m in completion if isinstance(m, dict) and "content" in m]
        return _strip_think(" ".join(parts))
    return str(completion)


def format_reward(completions, **kwargs):
    """Reward structural compliance."""
    rewards = []
    for c in completions:
        text  = _get_text(c)
        score = 0.0
        if len(text.split()) >= 5:
            score += 0.3
        if any(kw in text.lower() for kw in [
            "hypothesis", "abstract", "contribution", "belongs to",
            "paper", "method", "result", "experiment", "improve",
        ]):
            score += 0.4
        if text.rstrip().endswith((".", ":", '"', "'")):
            score += 0.3
        rewards.append(score)
    return rewards


def factual_reward(completions, **kwargs):
    """Soft word-overlap reward."""
    rewards        = []
    ground_truths  = kwargs.get("answer", [])
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt   = ground_truths[i] if i < len(ground_truths) else None
        if gt:
            pred_words = set(text.lower().split())
            gt_words   = set(gt.lower().split())
            overlap    = len(pred_words & gt_words) / len(gt_words) if gt_words else 0.0
            rewards.append(min(overlap, 1.0))
        else:
            rewards.append(0.0)
    return rewards


def length_reward(completions, **kwargs):
    """Penalise responses that are too short or excessively long."""
    rewards = []
    for c in completions:
        n = len(_get_text(c).split())
        if   n < 5:   rewards.append(0.0)
        elif n < 20:  rewards.append(0.3)
        elif n <= 200: rewards.append(1.0)
        elif n <= 300: rewards.append(0.5)
        else:          rewards.append(0.2)
    return rewards


class HeartbeatCallback(transformers.TrainerCallback):
    """Prints a heartbeat message every step to show we are alive."""
    def on_step_begin(self, args, state, control, **kwargs):
        print(f"\n--- STEP {state.global_step+1}/{state.max_steps} STARTING ---")
    def on_step_end(self, args, state, control, **kwargs):
        print(f"--- STEP {state.global_step} FINISHED ---\n")


# ---------------------------------------------------------------------------
# SFT Training
# ---------------------------------------------------------------------------

def run_sft():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print(f"Loading datasets from {DATA_DIR} ...")
    ds_triples = load_dataset("json", data_files=str(DATA_DIR / "triples.jsonl"))["train"]
    ds_qa      = load_dataset("json", data_files=str(DATA_DIR / "qa_tasks.jsonl"))["train"]
    ds_hypo    = load_dataset("json", data_files=str(DATA_DIR / "hypothesis.jsonl"))["train"]

    dataset = concatenate_datasets([ds_triples, ds_qa, ds_hypo]).shuffle(seed=42)
    split_ds = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split_ds["train"]
    eval_ds  = split_ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def sft_formatting_func(example):
        messages = [
            {"role": "system", "content": "You are a scientific research assistant."},
            {"role": "user", "content": f"{example['instruction']}\n{example.get('input', '')}".strip()},
            {"role": "assistant", "content": example['output']},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    OUTPUT_DIR_SFT.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR_SFT),
        per_device_train_batch_size=2,          # Reduced for local execution
        per_device_eval_batch_size=2,           # Reduced for local execution
        gradient_accumulation_steps=16,         # Increased to compensate
        learning_rate=1e-5,
        logging_steps=10,                       # More frequent local logging
        eval_strategy="steps",
        eval_steps=500,
        max_steps=2500,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_length=2048,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        formatting_func=sft_formatting_func,
    )

    print("Starting local SFT training...")
    trainer.train()

    print(f"Pushing SFT model to HF Hub: {SFT_HUB}")
    trainer.model.push_to_hub(SFT_HUB, private=False)
    tokenizer.push_to_hub(SFT_HUB)
    print("SFT complete.")


# ---------------------------------------------------------------------------
# GRPO Training
# ---------------------------------------------------------------------------

def run_grpo():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages for CUDA asserts

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        try:
            info = HfApi().whoami(token=hf_token)
            print(f"Logged in to HF as: {info['name']}")
        except Exception as e:
            print(f"Warning: Could not verify HF user: {e}")
    else:
        print("WARNING: HF_TOKEN not set. push_to_hub will likely fail.")

    # -- Tokeniser ------------------------------------------------------------
    print(f"Loading tokeniser from base: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"pad_token={tokenizer.pad_token!r}  pad_token_id={tokenizer.pad_token_id}")
    
    if hf_token:
        print(f"Pushing clean tokeniser back to {SFT_HUB} to fix potential crashes...")
        tokenizer.push_to_hub(SFT_HUB, private=False)

    # -- Dataset --------------------------------------------------------------
    print(f"Loading GRPO dataset from {DATA_DIR} ...")
    try:
        ds_qa   = load_dataset("json", data_files=str(DATA_DIR / "qa_tasks.jsonl"))["train"]
        ds_hypo = load_dataset("json", data_files=str(DATA_DIR / "hypothesis.jsonl"))["train"]
        grpo_dataset = concatenate_datasets([ds_qa, ds_hypo]).shuffle(seed=42)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    def make_prompt(example):
        instruction = example.get("instruction", "")
        inp         = example.get("input", "")
        example["prompt"] = [
            {"role": "system",  "content": "You are a scientific research assistant."},
            {"role": "user",    "content": f"{instruction}\n{inp}".strip()},
        ]
        example["answer"] = example.get("output", "")
        return example

    grpo_dataset = grpo_dataset.map(make_prompt)
    keep_cols    = {"prompt", "answer"}
    drop_cols    = [c for c in grpo_dataset.column_names if c not in keep_cols]
    grpo_dataset = grpo_dataset.remove_columns(drop_cols)
    print(f"GRPO dataset size: {len(grpo_dataset)} examples")

    # -- GRPO Config ----------------------------------------------------------
    OUTPUT_DIR_GRPO.mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=str(OUTPUT_DIR_GRPO),
        per_device_train_batch_size=1,          # Minimal batch size for local consumer GPUs
        gradient_accumulation_steps=8,          # Compensate for smaller batch size
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        max_steps=1500,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        optim="adamw_torch",
        num_generations=2,           # G in GRPO: reduced for local GPU limitations
        max_completion_length=128,   
        temperature=0.7,
        use_vllm=False,              # Disabled by default for broader local compatibility
        logging_steps=1,             
        save_steps=500,
        save_total_limit=2,
        report_to="none",            
    )

    # -- Load model explicitly in bfloat16 ------------------------------------
    print(f"Loading SFT model from HF Hub in float16: {SFT_HUB}")
    model = AutoModelForCausalLM.from_pretrained(
        SFT_HUB,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # -- Trainer --------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=grpo_dataset,
        reward_funcs=[format_reward, factual_reward, length_reward],
        processing_class=tokenizer,
        callbacks=[HeartbeatCallback()], 
    )

    # -- Train ----------------------------------------------------------------
    print("Starting local GRPO training...")
    trainer.train()

    # -- Push to Hub ----------------------------------------------------------
    print(f"Pushing GRPO model to HF Hub: {GRPO_HUB}")
    trainer.model.push_to_hub(GRPO_HUB, private=False)
    tokenizer.push_to_hub(GRPO_HUB)
    print(f"GRPO complete. Final model at https://huggingface.co/{GRPO_HUB}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Synnapse SFT + GRPO locally without Modal.")
    parser.add_argument("--sft-only", action="store_true", help="Run only the SFT stage.")
    parser.add_argument("--grpo-only", action="store_true", help="Run only the GRPO stage.")
    args = parser.parse_args()

    if args.grpo_only:
        print("Skipping SFT. Running GRPO only...")
        run_grpo()
    elif args.sft_only:
        print("Running SFT only...")
        run_sft()
    else:
        print("Running Pipeline locally: SFT -> GRPO")
        print("To run parts individually, use `--sft-only` or `--grpo-only` flags.")
        run_sft()
        run_grpo()
