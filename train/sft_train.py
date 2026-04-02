"""
counseLLM — SFT Training Script (Stage 1)

Fine-tunes Llama 3.1 8B Instruct on mental health counseling data
using QLoRA (4-bit quantization + LoRA adapters).

Usage:
    # Local (uses configs/sft_config.yaml by default)
    python train/sft_train.py

    # Custom config
    python train/sft_train.py --config configs/sft_config.yaml

    # On Modal
    modal run train/sft_train.py
"""

import argparse
import torch
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sft_config.yaml"


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="counseLLM SFT Training")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve paths relative to project root
    base_model = cfg["model"]["name"]
    train_file = PROJECT_ROOT / cfg["data"]["train_file"]
    val_file = PROJECT_ROOT / cfg["data"]["val_file"]
    output_dir = PROJECT_ROOT / cfg["data"]["output_dir"]

    print(f"{'='*60}")
    print(f"counseLLM — SFT Training (Stage 1)")
    print(f"Config: {args.config}")
    print(f"Base model: {base_model}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # ---- Load tokenizer ----
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Load datasets ----
    print("Loading datasets...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(val_file),
        },
    )
    print(f"  Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

    # ---- Quantization config (4-bit) ----
    q_cfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_cfg["load_in_4bit"],
        bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
    )

    # ---- Load model ----
    print("Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=cfg["model"]["attn_implementation"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
    )
    model = prepare_model_for_kbit_training(model)

    # ---- LoRA config ----
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Training config ----
    t_cfg = cfg["training"]
    l_cfg = cfg["logging"]
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        max_seq_length=t_cfg["max_seq_length"],
        bf16=t_cfg["bf16"],
        gradient_checkpointing=t_cfg["gradient_checkpointing"],
        seed=t_cfg["seed"],
        logging_steps=l_cfg["logging_steps"],
        eval_strategy=l_cfg["eval_strategy"],
        eval_steps=l_cfg["eval_steps"],
        save_strategy=l_cfg["save_strategy"],
        save_steps=l_cfg["save_steps"],
        save_total_limit=l_cfg["save_total_limit"],
        load_best_model_at_end=l_cfg["load_best_model_at_end"],
        metric_for_best_model=l_cfg["metric_for_best_model"],
        greater_is_better=l_cfg["greater_is_better"],
        report_to=l_cfg["report_to"],
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # ---- Train ----
    print("\nStarting SFT training...")
    trainer.train()

    # ---- Save ----
    final_dir = output_dir / "final"
    print("\nSaving final adapter...")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"\nSFT training complete! Adapter saved to: {final_dir}")


if __name__ == "__main__":
    main()
