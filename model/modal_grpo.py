"""
modal_grpo.py
=============
Runs GRPO reinforcement fine-tuning of the Synnapse SFT model on Modal with
an H100 GPU. The job is capped at 2 hours via Modal's `timeout` parameter.

Prerequisites
-------------
1. Install the Modal CLI and authenticate:
       pip install modal
       modal token new

2. Create a Modal Secret called "huggingface-secret" that exposes HF_TOKEN:
       modal secret create huggingface-secret HF_TOKEN=hf_...

3. Make sure the SFT model has been pushed to HF Hub at:
       Wvidit/Synnapse-Qwen2.5-3B-sft

Run
---
    modal run model/modal_grpo.py
"""

import os
import re
from pathlib import Path

import modal
import transformers

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Pin PyTorch 2.5.1 because vLLM 0.7.2 STRICTLY requires it.
    .pip_install(
        "torch==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.48.3",    # PINNED to avoid vision_config bug in 4.49.0 with vLLM
        "trl==0.14.0",             # PINNED to avoid PyTorch 2.6 constraint in newer TRL
        "vllm==0.7.2",             # PINNED for 10x faster generation
        "peft>=0.14.0",
        "datasets>=2.20.0",
        "accelerate>=1.3.0",
        "bitsandbytes>=0.45.0",
        "huggingface_hub>=0.25.0",
        "sentencepiece",
        "packaging",
    )
    # Bake the local dataset/ directory into the image.
    .add_local_dir(
        local_path=str(Path(__file__).parent.parent / "dataset"),
        remote_path="/root/dataset",
    )
)

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app = modal.App("synnapse-grpo", image=image)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"          # base tokeniser source
SFT_HUB    = "Wvidit/Synnapse-Qwen2.5-3B-sft"   # SFT checkpoint on HF Hub
GRPO_HUB   = "Wvidit/Synnapse-Qwen2.5-3B"        # final GRPO model destination
DATA_DIR   = Path("/root/dataset")
OUTPUT_DIR = Path("/root/grpo_output")
TWO_HOURS  = 60*60*4 # seconds

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
# SFT Modal function
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=TWO_HOURS,
    ephemeral_disk=512 * 1024,
)
def run_sft():
    import torch
    import os
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig
    from huggingface_hub import login, HfApi

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("Loading datasets from /root/dataset ...")
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
                {"role": "user", "content": f"{example['instruction']}\n{example.get('input', '')}"},
                {"role": "assistant", "content": example['output']},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    SFT_OUT_DIR = Path("/root/sft_output")
    SFT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(SFT_OUT_DIR),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=2500,
        max_steps=2500,
        bf16=True,
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

    print("Starting SFT training... (timeout = 2 hours)")
    trainer.train()

    print(f"Pushing SFT model to HF Hub: {SFT_HUB}")
    trainer.model.push_to_hub(SFT_HUB, private=False)
    tokenizer.push_to_hub(SFT_HUB)
    print("SFT complete.")


# ---------------------------------------------------------------------------
# GRPO Modal function
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=TWO_HOURS,
    ephemeral_disk=512 * 1024,  # 512 GiB – Modal minimum; fits model weights + outputs
)
def run_grpo():
    import torch
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoTokenizer
    from trl import GRPOTrainer, GRPOConfig
    from huggingface_hub import login, HfApi

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages for CUDA asserts

    # -- Auth -----------------------------------------------------------------
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
    # SFT outputs sometimes corrupt 'extra_special_tokens' into a list in the
    # tokenizer_config.json. This crashes vLLM when it tries to load the model.
    # To fix this, we forcefully push the clean base tokenizer to the SFT repo
    # right before vLLM initializes!
    print(f"Loading tokeniser from base: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"pad_token={tokenizer.pad_token!r}  pad_token_id={tokenizer.pad_token_id}")
    
    if hf_token:
        print(f"Pushing clean tokeniser back to {SFT_HUB} to fix vLLM crash...")
        tokenizer.push_to_hub(SFT_HUB, private=False)


    # -- Dataset --------------------------------------------------------------
    print("Loading GRPO dataset from mounted /root/dataset ...")
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        # Batch sizing – H100 has 80 GB; 3B model in bf16 is ~6 GB so we can
        # afford a large effective batch with gradient accumulation.
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,          # reduced to 2 so progress bar updates faster
        # Learning schedule
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        # Training duration – we rely on Modal's 2-hour timeout as the hard
        # cap; max_steps provides a safety upper bound beyond that.
        max_steps=1500,
        # Precision / memory — train in bf16 on H100 for speed
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        # GRPO-specific
        num_generations=4,           # G in GRPO
        max_completion_length=128,   # Shorter to be even faster
        temperature=0.7,
        # vLLM Acceleration
        use_vllm=True,               # Use vLLM for MASSIVELY faster generation
        vllm_device="cuda:0",
        vllm_gpu_memory_utilization=0.4, # Reserve 40% memory for vLLM engine
        # Logging / saving
        logging_steps=1,             # update loss every step!
        save_steps=500,
        save_total_limit=2,
        report_to="none",            # disable wandb unless you set it up
    )

    # -- Load model explicitly in fp32 ----------------------------------------
    # Passing a model-name string to GRPOTrainer lets it choose the dtype,
    # which defaults to the model config's torch_dtype (often bf16/fp16).
    # Loading in bfloat16 explicitly fixes precision issues natively with vLLM.
    from transformers import AutoModelForCausalLM
    print(f"Loading SFT model from HF Hub in bfloat16: {SFT_HUB}")
    model = AutoModelForCausalLM.from_pretrained(
        SFT_HUB,
        torch_dtype=torch.bfloat16,
        device_map=None,             # GRPOTrainer handles device mapping
    )

    # -- Trainer --------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=grpo_dataset,
        reward_funcs=[format_reward, factual_reward, length_reward],
        processing_class=tokenizer,
        callbacks=[HeartbeatCallback()], # Print status every step
    )

    # -- Train ----------------------------------------------------------------
    print("Starting GRPO training... (timeout = 2 hours)")
    trainer.train()

    # -- Push to Hub ----------------------------------------------------------
    print(f"Pushing GRPO model to HF Hub: {GRPO_HUB}")
    trainer.model.push_to_hub(GRPO_HUB, private=False)
    tokenizer.push_to_hub(GRPO_HUB)
    print(f"GRPO complete. Final model at https://huggingface.co/{GRPO_HUB}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(sft_only: bool = False, grpo_only: bool = False):
    """
    Run the Synnapse SFT + GRPO training pipeline.
    Flags:
        --sft-only   Run only the SFT stage
        --grpo-only  Run only the GRPO stage
    """
    if grpo_only:
        print("Skipping SFT. Running GRPO only...")
        run_grpo.remote()
    elif sft_only:
        print("Running SFT only...")
        run_sft.remote()
    else:
        print("Running Pipeline: SFT -> GRPO")
        print("To run parts individually, use `--sft-only` or `--grpo-only` flags.")
        # Running sequentially gives each step a fresh container and fresh VRAM
        run_sft.remote()
        run_grpo.remote()
