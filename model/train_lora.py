import os
import torch
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_DIR = Path(__file__).parent.parent / "model_out"
DATA_DIR = Path(__file__).parent.parent / "dataset"

def train():
    # Model details
    model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
    # Can substitute with a smaller open model for local testing if needed
    
    print("Loading datasets...")
    # Load and combine the three datasets
    try:
        ds_triples = load_dataset("json", data_files=str(DATA_DIR / "triples.jsonl"))['train']
        ds_qa = load_dataset("json", data_files=str(DATA_DIR / "qa_tasks.jsonl"))['train']
        ds_hypo = load_dataset("json", data_files=str(DATA_DIR / "hypothesis.jsonl"))['train']
        
        # We can implement a 60/20/20 split logic here by duplicating or sampling
        # For this scaffolding, we just concatenate them
        dataset = concatenate_datasets([ds_triples, ds_qa, ds_hypo])
        dataset = dataset.shuffle(seed=42)
    except Exception as e:
        print(f"Error loading datasets (ensure JSONL files exist): {e}")
        return

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # QLoRA: Load in 4-bit to fit easily on an 8GB RTX 4060 Laptop GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Format Alpaca function
    def formatting_prompts_func(example):
        if isinstance(example.get('instruction', ''), list):
            output_texts = []
            for i in range(len(example['instruction'])):
                text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction'][i]}\n\n### Input:\n{example.get('input', [''])[i]}\n\n### Response:\n{example['output'][i]}<|eot_id|>"
                output_texts.append(text)
            return output_texts
        else:
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction']}\n\n### Input:\n{example.get('input', '')}\n\n### Response:\n{example['output']}<|eot_id|>"

    print("Setting up Trainer...")
    training_args = SFTConfig(
        output_dir=str(MODEL_DIR),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=1000, # Use num_train_epochs=3 for full run
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_steps=10,
        lr_scheduler_type="constant",
        gradient_checkpointing=True,
        max_length=2048,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.model.save_pretrained(str(MODEL_DIR / "final_lora"))
    tokenizer.save_pretrained(str(MODEL_DIR / "final_lora"))
    print("Done!")

if __name__ == "__main__":
    train()
