"""
counseLLM - Push LoRA adapters to HuggingFace Hub (local)

Uploads the SFT and DPO LoRA adapter folders to HuggingFace Hub from a local
machine. Use this when you can't run the Modal-based `push_adapters` function
(e.g. spend limit reached) - first download the folders from the Modal volume
with `modal volume get`, then run this script.

By default, both adapters are uploaded as subfolders (`sft/` and `dpo/`) of the
existing merged-model repo `Wothmag07/counseLLM`, so a single repo holds the
merged model at the root plus both adapters in subfolders.

Usage:
    # 1. Download adapters from Modal volume (CLI download, no compute spend)
    modal volume get counsellm-data counseLLM/outputs/sft/final ./outputs/sft/
    modal volume get counsellm-data counseLLM/outputs/dpo/final ./outputs/dpo/

    # 2. Set HF token with write scope
    #    PowerShell: $env:HF_TOKEN = "hf_xxx"
    #    bash:       export HF_TOKEN=hf_xxx

    # 3. Push
    python train/push_adapters_local.py
    python train/push_adapters_local.py --skip-sft  # push DPO only
    python train/push_adapters_local.py --repo OtherOwner/other-repo
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SFT_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "sft" / "final"
DPO_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "dpo" / "final"

DEFAULT_REPO = "Wothmag07/counseLLM"
SFT_SUBFOLDER = "sft"
DPO_SUBFOLDER = "dpo"


def adapter_readme(stage: str, repo_id: str, subfolder: str) -> str:
    return f"""---
library_name: peft
base_model: {BASE_MODEL}
license: apache-2.0
language:
- en
tags:
- lora
- peft
- llama-3.1
- mental-health
- counseling
- {stage.lower()}
---

# counseLLM - {stage} LoRA adapter

LoRA adapter for `{BASE_MODEL}` produced by the **{stage}** stage of the
[counseLLM](https://github.com/wothmag07/counseLLM) training pipeline.

This adapter lives in the `{subfolder}/` subfolder of
[{repo_id}](https://huggingface.co/{repo_id}). The fully merged model is at the
repo root.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "{BASE_MODEL}", torch_dtype="bfloat16", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL}")
model = PeftModel.from_pretrained(base, "{repo_id}", subfolder="{subfolder}")
```

## Training

- Method: QLoRA (4-bit NF4) on NVIDIA H100
- Stage: {stage}
- Pipeline and configs: see the [counseLLM repository](https://github.com/wothmag07/counseLLM)

## Intended use

Research and educational use only. **Not a substitute for professional mental
health care.** If you are in crisis, contact the
[988 Suicide & Crisis Lifeline](https://988lifeline.org/).
"""


def push_one(
    api: HfApi,
    stage: str,
    folder: Path,
    repo_id: str,
    subfolder: str,
):
    if not folder.exists():
        print(f"[{stage}] SKIPPING - {folder} not found")
        return
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required if not (folder / f).exists()]
    if missing:
        print(f"[{stage}] SKIPPING - missing files: {missing}")
        return

    readme_path = folder / "README.md"
    readme_path.write_text(
        adapter_readme(stage, repo_id, subfolder), encoding="utf-8"
    )
    print(f"[{stage}] Wrote model card to {readme_path}")

    print(f"[{stage}] Uploading {folder} -> {repo_id}/{subfolder} ...")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=subfolder,
        commit_message=f"Upload counseLLM {stage} LoRA adapter to {subfolder}/",
    )
    print(f"[{stage}] Done: https://huggingface.co/{repo_id}/tree/main/{subfolder}")


def main():
    parser = argparse.ArgumentParser(
        description="Push counseLLM LoRA adapters to HF Hub as subfolders of an existing repo"
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Target HF repo (must already exist)")
    parser.add_argument("--sft-subfolder", default=SFT_SUBFOLDER)
    parser.add_argument("--dpo-subfolder", default=DPO_SUBFOLDER)
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-dpo", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set. Export your write-scoped HF token first.")
    login(token=token)
    api = HfApi()

    # Verify the target repo exists (we're not creating it - it already holds the merged model)
    try:
        api.repo_info(repo_id=args.repo, repo_type="model")
    except Exception as e:
        raise SystemExit(f"Repo {args.repo} not accessible with this token: {e}")

    if not args.skip_sft:
        push_one(api, "SFT", SFT_ADAPTER_DIR, args.repo, args.sft_subfolder)
    if not args.skip_dpo:
        push_one(api, "DPO", DPO_ADAPTER_DIR, args.repo, args.dpo_subfolder)

    print(f"\nAll requested adapters pushed to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
