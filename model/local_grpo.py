"""
local_grpo.py
=============
Runs SFT and GRPO reinforcement fine-tuning locally, optimised for the
NVIDIA DGX Spark GB10 Superchip (128 GB unified CPU+GPU memory).
Uses pure PyTorch execution — no Modal, no vLLM required.
"""

import os
import re
import sys
import site
import argparse
from pathlib import Path
from plot_metrics import MetricsLogger

# ---------------------------------------------------------------------------
# CUDA / library setup
# ---------------------------------------------------------------------------

# Fix 'libcudart.so.12' missing errors by native-linking pip's bundled CUDA runtime
for p in site.getsitepackages() + [site.getusersitepackages()]:
    cuda_path = os.path.join(p, "nvidia", "cuda_runtime", "lib")
    if os.path.exists(cuda_path):
        os.environ["LD_LIBRARY_PATH"] = (
            cuda_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )

# DGX Spark: allow PyTorch to use unified memory aggressively
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# Monkey-patch to prevent TRL from crashing on a missing local vLLM install
sys.modules["vllm"] = None

import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from huggingface_hub import login, HfApi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME      = "Qwen/Qwen3-4B-Instruct-2507"    # base model
SFT_HUB         = "Wvidit/Qwen3-4B"   # SFT checkpoint on HF Hub
GRPO_HUB        = "Wvidit/Synnapse-Qwen3-4B"        # final GRPO model destination
DATA_DIR        = Path("dataset")                   # local dataset directory
OUTPUT_DIR_SFT  = Path("sft_output")
OUTPUT_DIR_GRPO = Path("grpo_output")

# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks before scoring."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _get_text(completion) -> str:
    """Normalise a completion to a plain string."""
    if isinstance(completion, str):
        return _strip_think(completion)
    if isinstance(completion, list):
        parts = [
            m["content"]
            for m in completion
            if isinstance(m, dict) and "content" in m
        ]
        return _strip_think(" ".join(parts))
    return str(completion)


def format_reward(completions, **kwargs):
    """Reward structural / domain compliance."""
    rewards = []
    for c in completions:
        text  = _get_text(c)
        score = 0.0
        if len(text.split()) >= 5:
            score += 0.3
        if any(
            kw in text.lower()
            for kw in [
                "hypothesis", "abstract", "contribution", "belongs to",
                "paper", "method", "result", "experiment", "improve",
            ]
        ):
            score += 0.4
        if text.rstrip().endswith((".", ":", '"', "'")):
            score += 0.3
        rewards.append(score)
    return rewards


def factual_reward(completions, **kwargs):
    """Soft word-overlap reward against the ground-truth answer."""
    rewards       = []
    ground_truths = kwargs.get("answer", [])
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
    """Penalise completions that are too short or excessively long."""
    rewards = []
    for c in completions:
        n = len(_get_text(c).split())
        if   n < 5:    rewards.append(0.0)
        elif n < 20:   rewards.append(0.3)
        elif n <= 200: rewards.append(1.0)
        elif n <= 300: rewards.append(0.5)
        else:          rewards.append(0.2)
    return rewards


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class HeartbeatCallback(transformers.TrainerCallback):
    """Prints a heartbeat every step so long runs stay observable."""
    def on_step_begin(self, args, state, control, **kwargs):
        print(f"\n--- STEP {state.global_step + 1}/{state.max_steps} STARTING ---")

    def on_step_end(self, args, state, control, **kwargs):
        print(f"--- STEP {state.global_step} FINISHED ---\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hf_login() -> bool:
    """Login to HF Hub if a token is available. Returns True on success."""
    hf_token = "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"
    if not hf_token:
        print("WARNING: HF_TOKEN not set. push_to_hub will likely fail.")
        return False
    login(token=hf_token)
    try:
        info = HfApi().whoami(token=hf_token)
        print(f"Logged in to HF Hub as: {info['name']}")
    except Exception as e:
        print(f"Warning: could not verify HF user: {e}")
    return True


def _load_model(repo: str) -> AutoModelForCausalLM:
    """
    Load a causal LM in bfloat16 pinned to GPU 0.

    device_map={"": 0} pins every parameter to a single device, which avoids
    the fragmented allocation that device_map="auto" can produce on unified
    memory systems like the DGX Spark GB10.
    """
    print(f"Loading model from: {repo}")
    return AutoModelForCausalLM.from_pretrained(
        repo,
        device_map={"": 0},        # pin to GPU 0 — optimal for unified memory
        torch_dtype=torch.bfloat16, # Blackwell-native precision; avoids fp16 instability
    )


# ---------------------------------------------------------------------------
# SFT Training
# ---------------------------------------------------------------------------

def run_sft():
    _hf_login()

    # -- Dataset --------------------------------------------------------------
#   print(f"Loading SFT datasets from {DATA_DIR} ...")
#   ds_triples = load_dataset("json", data_files=str(DATA_DIR / "triples.jsonl"))["train"]
#   ds_qa      = load_dataset("json", data_files=str(DATA_DIR / "qa_tasks.jsonl"))["train"]
#   ds_hypo    = load_dataset("json", data_files=str(DATA_DIR / "hypothesis.jsonl"))["train"]
#
#   dataset  = concatenate_datasets([ds_triples, ds_qa, ds_hypo]).shuffle(seed=42)
#   split_ds = dataset.train_test_split(test_size=0.1, seed=42)
#   train_ds = split_ds["train"]
#   eval_ds  = split_ds["test"]
#   print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")
#
#   # -- Tokeniser ------------------------------------------------------------
#   tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#   if tokenizer.pad_token is None:
#       tokenizer.pad_token = tokenizer.eos_token
#
#   def sft_formatting_func(example):
#       messages = [
#           {"role": "system",    "content": "You are a scientific research assistant."},
#           {"role": "user",      "content": f"{example['instruction']}\n{example.get('input', '')}".strip()},
#           {"role": "assistant", "content": example["output"]},
#       ]
#       return tokenizer.apply_chat_template(
#           messages, tokenize=False, add_generation_prompt=False
#       )
#
#   # -- Model ----------------------------------------------------------------
#   model = _load_model(MODEL_NAME)
#
#   # -- Training config ------------------------------------------------------
#   OUTPUT_DIR_SFT.mkdir(parents=True, exist_ok=True)
#
#   training_args = SFTConfig(
#       output_dir=str(OUTPUT_DIR_SFT),
#       # ── Batch / memory ────────────────────────────────────────────────────
#       per_device_train_batch_size=8,   # 128 GB unified mem — fill it
#       per_device_eval_batch_size=8,
#       gradient_accumulation_steps=4,   # effective batch = 32; matches original 2×16
#       gradient_checkpointing=False,    # 128 GB — no need to trade compute for memory
#       max_length=4096,                 # doubled from 2048; room available
#       # ── Precision ─────────────────────────────────────────────────────────
#       fp16=False,
#       bf16=True,                       # Blackwell tensor-core native
#       # ── Optimiser ─────────────────────────────────────────────────────────
#       optim="adamw_torch_fused",       # paged_adamw_8bit is a discrete-VRAM workaround
#       learning_rate=1e-5,
#       max_grad_norm=0.3,
#       lr_scheduler_type="cosine",
#       warmup_steps=50,
#       # ── Schedule / logging ────────────────────────────────────────────────
#       eval_steps=1000,
#       max_steps=10000,
#       logging_steps=10,
#       eval_strategy="steps",
#       report_to="none",
#   )
#
#   trainer = SFTTrainer(
#       model=model,
#       train_dataset=train_ds,
#       eval_dataset=eval_ds,
#       args=training_args,
#       formatting_func=sft_formatting_func,
#   )
#
#   print("Starting SFT training...")
#   trainer.train()

    print(f"Pushing SFT model to HF Hub: {SFT_HUB}")
    trainer.model.push_to_hub(SFT_HUB, private=False)
    tokenizer.push_to_hub(SFT_HUB)
    print("SFT complete.")


# ---------------------------------------------------------------------------
# GRPO Training
# ---------------------------------------------------------------------------

def run_grpo():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # clearer CUDA assert messages

    logged_in = _hf_login()
    if not logged_in:
        print("WARNING: proceeding without HF login; push_to_hub will fail.")

    # -- Tokeniser ------------------------------------------------------------
    print(f"Loading tokeniser from base: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"pad_token={tokenizer.pad_token!r}  pad_token_id={tokenizer.pad_token_id}")

    if logged_in:
        # Push a clean tokeniser back to the SFT hub to prevent downstream crashes
        print(f"Pushing clean tokeniser to {SFT_HUB} ...")
        tokenizer.push_to_hub(SFT_HUB, private=False)

    # -- Dataset --------------------------------------------------------------
    print(f"Loading GRPO dataset from {DATA_DIR} ...")
    try:
        ds_qa   = load_dataset("json", data_files=str(DATA_DIR / "qa_tasks.jsonl"))["train"]
        ds_hypo = load_dataset("json", data_files=str(DATA_DIR / "hypothesis.jsonl"))["train"]
        grpo_dataset = concatenate_datasets([ds_qa, ds_hypo]).shuffle(seed=42)
    except Exception as e:
        print(f"Error loading GRPO datasets: {e}")
        return

    def make_prompt(example):
        example["prompt"] = [
            {"role": "system", "content": "You are a scientific research assistant."},
            {"role": "user",   "content": f"{example.get('instruction', '')}\n{example.get('input', '')}".strip()},
        ]
        example["answer"] = example.get("output", "")
        return example

    grpo_dataset = grpo_dataset.map(make_prompt)
    drop_cols    = [c for c in grpo_dataset.column_names if c not in {"prompt", "answer"}]
    grpo_dataset = grpo_dataset.remove_columns(drop_cols)
    print(f"GRPO dataset size: {len(grpo_dataset)} examples")

    # -- Training config ------------------------------------------------------
    OUTPUT_DIR_GRPO.mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=str(OUTPUT_DIR_GRPO),
        # ── Batch / memory ────────────────────────────────────────────────────
        per_device_train_batch_size=4,   # was 1 on consumer GPU
        gradient_accumulation_steps=4,   # effective batch = 16
        gradient_checkpointing=False,    # 128 GB — not needed
        num_generations=8,               # was 2; more rollouts → stronger reward signal
        max_completion_length=512,       # was 128; room for richer reasoning chains
        # ── Precision ─────────────────────────────────────────────────────────
        fp16=False,
        bf16=True,                       # Blackwell native
        # ── Optimiser ─────────────────────────────────────────────────────────
        optim="adamw_torch",       # replaces paged_adamw_8bit
        learning_rate=5e-6,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        # ── Generation ────────────────────────────────────────────────────────
        temperature=0.7,
        use_vllm=False,
        # ── Schedule / logging ────────────────────────────────────────────────
        max_steps=1500,
        logging_steps=1,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )

    # -- Model ----------------------------------------------------------------
    model = _load_model(SFT_HUB)

    # -- Trainer --------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=grpo_dataset,
        reward_funcs=[format_reward, factual_reward, length_reward],
        processing_class=tokenizer,
        callbacks=[HeartbeatCallback(), MetricsLogger("grpo_metrics.png")],
    )

    # -- Train ----------------------------------------------------------------
    print("Starting GRPO training...")
    trainer.train()

    # -- Push to Hub ----------------------------------------------------------
    print(f"Pushing GRPO model to HF Hub: {GRPO_HUB}")
    trainer.model.push_to_hub(GRPO_HUB, private=False)
    tokenizer.push_to_hub(GRPO_HUB)
    print(f"GRPO complete. Final model → https://huggingface.co/{GRPO_HUB}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Synnapse SFT + GRPO locally on a DGX Spark."
    )
    parser.add_argument("--sft-only",  action="store_true", help="Run only the SFT stage.")
    parser.add_argument("--grpo-only", action="store_true", help="Run only the GRPO stage.")
    args = parser.parse_args()

    if args.grpo_only:
        print("Skipping SFT. Running GRPO only...")
        run_grpo()
    elif args.sft_only:
        print("Running SFT only...")
        run_sft()
    else:
        print("Running full pipeline: SFT → GRPO")
        print("Tip: use --sft-only or --grpo-only to run stages individually.")
#        run_sft()
        run_grpo()
