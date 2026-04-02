"""
counseLLM — Merge Adapters & Push to HuggingFace Hub

Merges the DPO LoRA adapter back into the base model and optionally
pushes the final merged model to HuggingFace Hub.

Usage:
    # Merge only
    python train/merge_and_push.py

    # Merge and push
    python train/merge_and_push.py --push --hub-repo your-username/conseLLM
"""

import argparse
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SFT_ADAPTER_DIR = Path(__file__).resolve().parent.parent / "outputs" / "sft" / "final"
DPO_ADAPTER_DIR = Path(__file__).resolve().parent.parent / "outputs" / "dpo" / "final"
MERGED_DIR = Path(__file__).resolve().parent.parent / "outputs" / "merged"


def main():
    parser = argparse.ArgumentParser(description="Merge counseLLM adapters and optionally push to Hub")
    parser.add_argument("--push", action="store_true", help="Push merged model to HuggingFace Hub")
    parser.add_argument("--hub-repo", type=str, default="your-username/conseLLM",
                        help="HuggingFace Hub repo ID")
    parser.add_argument("--sft-only", action="store_true",
                        help="Merge only SFT adapter (skip DPO)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"counseLLM — Merge Adapters")
    print(f"{'='*60}\n")

    # ---- Load base model (full precision for merging) ----
    print("Loading base model (full precision)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # ---- Merge SFT adapter ----
    print(f"Merging SFT adapter from {SFT_ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(model, str(SFT_ADAPTER_DIR))
    model = model.merge_and_unload()

    if not args.sft_only:
        # ---- Merge DPO adapter ----
        print(f"Merging DPO adapter from {DPO_ADAPTER_DIR}...")
        model = PeftModel.from_pretrained(model, str(DPO_ADAPTER_DIR))
        model = model.merge_and_unload()

    # ---- Save merged model ----
    output_dir = MERGED_DIR / ("sft-only" if args.sft_only else "sft-dpo")
    print(f"\nSaving merged model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved!")

    # ---- Push to Hub ----
    if args.push:
        repo_id = args.hub_repo
        if args.sft_only:
            repo_id += "-sft-only"
        print(f"\nPushing to HuggingFace Hub: {repo_id}...")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print(f"Pushed to: https://huggingface.co/{repo_id}")

    print("\nDone!")


if __name__ == "__main__":
    main()
