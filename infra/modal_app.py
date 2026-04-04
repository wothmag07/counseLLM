"""
counseLLM — Modal GPU Deployment

Runs the full training pipeline (data prep → SFT → DPO → merge → eval)
on Modal's cloud GPUs.

Usage:
    # Run full pipeline
    modal run infra/modal_app.py::train_full_pipeline

    # Run individual stages
    modal run infra/modal_app.py::prepare_data
    modal run infra/modal_app.py::train_sft
    modal run infra/modal_app.py::train_dpo
    modal run infra/modal_app.py::merge_model
    modal run infra/modal_app.py::run_eval

    # Deploy Chainlit app
    modal deploy infra/modal_app.py
"""

import modal
from pathlib import Path

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("counseLLM")

# Persistent volume for storing datasets, checkpoints, and model weights
volume = modal.Volume.from_name("counsellm-data", create_if_missing=True)
VOLUME_PATH = "/data"
PROJECT_ON_VOLUME = "/data/counseLLM"

# Path to local project root (parent of infra/)
LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Docker image with all dependencies + local project files baked in
training_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch==2.5.0",
        "transformers==4.46.3",
        "trl==0.12.2",
        "peft==0.14.0",
        "bitsandbytes==0.44.1",
        "datasets==2.21.0",
        "accelerate==1.1.1",
        "pyyaml>=6.0",
        "sentencepiece",
        "protobuf",
        "wandb",
        "huggingface_hub",
        "python-dotenv",
        "bert-score",
        "rouge-score",
        "openai",
    )
    .run_commands("pip install wheel setuptools")
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .add_local_dir(str(LOCAL_PROJECT_ROOT / "data"), remote_path="/app/data")
    .add_local_dir(str(LOCAL_PROJECT_ROOT / "configs"), remote_path="/app/configs")
    .add_local_dir(str(LOCAL_PROJECT_ROOT / "eval"), remote_path="/app/eval")
    .add_local_dir(str(LOCAL_PROJECT_ROOT / "train"), remote_path="/app/train")
    .add_local_dir(str(LOCAL_PROJECT_ROOT / "app"), remote_path="/app/chat")
)

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.30.0",
        "chainlit>=1.0.0",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
    )
    .add_local_dir(str(LOCAL_PROJECT_ROOT / "app"), remote_path="/app/chat")
)

# Secrets
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")
openai_secret = modal.Secret.from_name("openai-secret")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hf_login():
    """Authenticate with HuggingFace using the injected secret."""
    import os
    from huggingface_hub import login

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        print("Logged in to HuggingFace Hub")
    else:
        print("WARNING: HF_TOKEN not found — gated model downloads may fail")


def sync_to_volume():
    """Copy baked-in project files to persistent volume."""
    import shutil
    from pathlib import Path

    dst = Path(PROJECT_ON_VOLUME)
    dst.mkdir(parents=True, exist_ok=True)

    for folder in ["data", "configs", "eval", "train"]:
        src = Path("/app") / folder
        if src.exists():
            shutil.copytree(src, dst / folder, dirs_exist_ok=True)


# ---------------------------------------------------------------------------
# Data Preparation (no GPU needed)
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret, wandb_secret],
    timeout=1800,
)
def prepare_data(
    skip_psych8k: bool = True, dpo_sample_size: int = 2000, min_rating_gap: int = 3
):
    """Download and prepare SFT + DPO datasets."""
    import subprocess
    from pathlib import Path

    hf_login()
    sync_to_volume()
    project_dir = Path(PROJECT_ON_VOLUME)

    # Run SFT data prep
    cmd = ["python", str(project_dir / "data" / "prepare_sft_data.py")]
    if skip_psych8k:
        cmd.append("--skip-psych8k")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Run DPO data prep
    cmd = [
        "python",
        str(project_dir / "data" / "prepare_dpo_data.py"),
        "--sample-size",
        str(dpo_sample_size),
        "--min-rating-gap",
        str(min_rating_gap),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Cache base model on volume so eval/training don't re-download
    print("Caching base model to volume...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        "meta-llama/Llama-3.1-8B-Instruct",
        local_dir=str(Path(PROJECT_ON_VOLUME) / "models" / "base"),
    )

    volume.commit()
    print("Data preparation complete!")


# ---------------------------------------------------------------------------
# SFT Training
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret, wandb_secret],
    timeout=21600,
)
def train_sft():
    """Stage 1: Supervised Fine-Tuning with QLoRA."""
    import yaml
    import torch
    from pathlib import Path
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    hf_login()
    sync_to_volume()
    project_dir = Path(PROJECT_ON_VOLUME)

    with open(project_dir / "configs" / "sft_config.yaml") as f:
        cfg = yaml.safe_load(f)

    base_model = cfg["model"]["name"]
    train_file = project_dir / "data" / "sft_train.jsonl"
    val_file = project_dir / "data" / "sft_val.jsonl"
    output_dir = project_dir / "outputs" / "sft"

    print(f"Base model: {base_model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(val_file),
        },
    )
    print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

    q_cfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_cfg["load_in_4bit"],
        bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=cfg["model"]["attn_implementation"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
    )
    model = prepare_model_for_kbit_training(model)

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

    t_cfg = cfg["training"]
    l_cfg = cfg["logging"]
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=float(t_cfg["learning_rate"]),
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        max_length=t_cfg["max_length"],
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

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    volume.commit()
    print(f"SFT complete! Saved to {final_dir}")


# ---------------------------------------------------------------------------
# DPO Training
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret, wandb_secret],
    timeout=7200,
)
def train_dpo():
    """Stage 2: Direct Preference Optimization."""
    import yaml
    import torch
    from pathlib import Path
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOConfig, DPOTrainer

    hf_login()
    sync_to_volume()
    project_dir = Path(PROJECT_ON_VOLUME)

    with open(project_dir / "configs" / "dpo_config.yaml") as f:
        cfg = yaml.safe_load(f)

    base_model = cfg["model"]["name"]
    sft_adapter_dir = project_dir / "outputs" / "sft" / "final"
    train_file = project_dir / "data" / "dpo_train.jsonl"
    val_file = project_dir / "data" / "dpo_val.jsonl"
    output_dir = project_dir / "outputs" / "dpo"

    print(f"Base model: {base_model}")
    print(f"SFT adapter: {sft_adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(val_file),
        },
    )
    print(f"DPO Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

    q_cfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_cfg["load_in_4bit"],
        bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=cfg["model"]["attn_implementation"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
    )

    print("Merging SFT adapter...")
    model = PeftModel.from_pretrained(model, str(sft_adapter_dir))
    model = model.merge_and_unload()
    model = prepare_model_for_kbit_training(model)

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

    t_cfg = cfg["training"]
    l_cfg = cfg["logging"]
    training_args = DPOConfig(
        output_dir=str(output_dir),
        beta=cfg["dpo"]["beta"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=float(t_cfg["learning_rate"]),
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        max_length=t_cfg["max_length"],
        max_prompt_length=t_cfg["max_prompt_length"],
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

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    volume.commit()
    print(f"DPO complete! Saved to {final_dir}")


# ---------------------------------------------------------------------------
# Merge Model
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret, wandb_secret],
    timeout=3600,
)
def merge_model(push_to_hub: bool = False, hub_repo: str = "Wothmag07/counseLLM"):
    """Merge LoRA adapters into base model and optionally push to Hub."""
    import torch
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    hf_login()
    project_dir = Path(PROJECT_ON_VOLUME)
    base_model = "meta-llama/Llama-3.1-8B-Instruct"
    sft_adapter = project_dir / "outputs" / "sft" / "final"
    dpo_adapter = project_dir / "outputs" / "dpo" / "final"
    merged_dir = project_dir / "outputs" / "merged" / "sft-dpo"

    print("Loading base model (full precision for merge)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("Merging SFT adapter...")
    model = PeftModel.from_pretrained(model, str(sft_adapter))
    model = model.merge_and_unload()

    if dpo_adapter.exists():
        print("Merging DPO adapter...")
        model = PeftModel.from_pretrained(model, str(dpo_adapter))
        model = model.merge_and_unload()

    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    volume.commit()

    if push_to_hub:
        print(f"Pushing to {hub_repo}...")
        model.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)
        print(f"Done! https://huggingface.co/{hub_repo}")

    print("Merge complete!")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret, wandb_secret, openai_secret],
    timeout=7200,
)
def run_eval(models: str = None):
    """Run automated evaluation on all model variants."""
    import os
    import subprocess
    from pathlib import Path

    if models is None:
        models = ["base", "sft", "dpo"]
    else:
        models = models.split(",")

    hf_login()
    sync_to_volume()
    project_dir = Path(PROJECT_ON_VOLUME)

    # Stage 1: Automated metrics (perplexity, BERTScore, ROUGE-L, etc.)
    cmd = [
        "python",
        str(project_dir / "eval" / "evaluate.py"),
        "--models",
        *models,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()

    # Stage 2: LLM-as-Judge (OpenAI)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("\nRunning LLM-as-Judge evaluation...")
        cmd = [
            "python",
            str(project_dir / "eval" / "llm_judge.py"),
            "--provider", "openai",
            "--api-key", api_key,
            "--from-results", str(project_dir / "results" / "eval_results.json"),
        ]
        print(f"Running: {' '.join(cmd[:6])} --api-key ***")
        subprocess.run(cmd, check=True)
        volume.commit()
    else:
        print("\nSkipping LLM-as-Judge: OPENAI_API_KEY not set")

    print("Evaluation complete!")


# ---------------------------------------------------------------------------
# Full Pipeline (orchestrator — no GPU needed)
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret, wandb_secret],
    timeout=21600,
)
def train_full_pipeline():
    """Run the complete pipeline: data prep → SFT → DPO → merge."""
    print("=" * 60)
    print("counseLLM — Full Training Pipeline")
    print("=" * 60)

    prepare_data.remote()
    print("\n[1/4] Data preparation complete\n")

    train_sft.remote()
    print("\n[2/4] SFT training complete\n")

    train_dpo.remote()
    print("\n[3/4] DPO training complete\n")

    merge_model.remote()
    print("\n[4/4] Model merge complete\n")

    print("=" * 60)
    print("Pipeline complete! Model saved to volume.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Chainlit Web App (for deployment)
# ---------------------------------------------------------------------------


@app.function(
    image=inference_image,
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret, wandb_secret],
)
@modal.concurrent(max_inputs=5)
@modal.asgi_app()
def web_app():
    """Deploy counseLLM as a web app on Modal."""
    from chainlit.utils import mount_chainlit
    from fastapi import FastAPI

    fastapi_app = FastAPI()
    mount_chainlit(app=fastapi_app, target="/app/chat/app.py")
    return fastapi_app
