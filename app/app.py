"""
counseLLM — Chainlit Chat Interface

A conversational mental health counseling chatbot powered by
the counseLLM fine-tuned model (Llama 3.1 8B + SFT + DPO).

Usage:
    chainlit run app/app.py
"""

import torch
import chainlit as cl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel
from threading import Thread
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Check Modal volume path first, then local project path
VOLUME_ROOT = Path("/data/counseLLM")
_root = VOLUME_ROOT if VOLUME_ROOT.exists() else PROJECT_ROOT

# Try merged model first, fall back to base + adapters
MERGED_MODEL_DIR = _root / "outputs" / "merged" / "sft-dpo"
SFT_ADAPTER_DIR = _root / "outputs" / "sft" / "final"
DPO_ADAPTER_DIR = _root / "outputs" / "dpo" / "final"

SYSTEM_PROMPT = (
    "You are a mental health counselor providing supportive, empathetic guidance. "
    "Respond by first acknowledging the person's feelings, then explore their "
    "situation with open-ended questions. Use techniques like reflective listening, "
    "validation, and gentle reframing. Keep responses warm, conversational, and "
    "non-judgmental. For crisis situations involving self-harm or suicide, "
    "prioritize safety by encouraging the person to contact a crisis helpline "
    "or emergency services."
)

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model():
    """Load the counseLLM model — tries merged first, then adapters, then base."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Option 1: Merged model exists
    if MERGED_MODEL_DIR.exists():
        print(f"Loading merged model from {MERGED_MODEL_DIR}")
        model = AutoModelForCausalLM.from_pretrained(
            str(MERGED_MODEL_DIR),
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(MERGED_MODEL_DIR))

    # Option 2: Base model + adapters
    elif SFT_ADAPTER_DIR.exists():
        print(f"Loading base model + adapters")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        # Merge SFT
        model = PeftModel.from_pretrained(model, str(SFT_ADAPTER_DIR))
        model = model.merge_and_unload()

        # Merge DPO if available
        if DPO_ADAPTER_DIR.exists():
            model = PeftModel.from_pretrained(model, str(DPO_ADAPTER_DIR))
            model = model.merge_and_unload()
            print("Loaded: base + SFT + DPO")
        else:
            print("Loaded: base + SFT (no DPO adapter found)")

    # Option 3: Base model only (no fine-tuning done yet)
    else:
        print(f"No fine-tuned model found. Loading base model: {BASE_MODEL}")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Chainlit app
# ---------------------------------------------------------------------------

model, tokenizer = None, None


@cl.on_chat_start
async def on_chat_start():
    """Called when a new chat session starts."""
    global model, tokenizer

    # Load model on first session
    if model is None:
        msg = cl.Message(content="Loading counseLLM model... This may take a minute.")
        await msg.send()
        model, tokenizer = load_model()
        msg.content = "Model loaded. How can I help you today?"
        await msg.update()
    else:
        await cl.Message(
            content="Welcome back. I'm here to listen — feel free to share what's on your mind."
        ).send()

    # Initialize conversation history in session
    cl.user_session.set(
        "history",
        [
            {"role": "system", "content": SYSTEM_PROMPT},
        ],
    )


@cl.on_message
async def on_message(message: cl.Message):
    """Called when the user sends a message."""
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    # Tokenize with chat template
    input_text = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Streaming generation
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "do_sample": True,
        "repetition_penalty": REPETITION_PENALTY,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }

    # Run generation in a separate thread (non-blocking)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream tokens to the UI
    response_msg = cl.Message(content="")
    await response_msg.send()

    full_response = ""
    for token in streamer:
        full_response += token
        await response_msg.stream_token(token)

    await response_msg.update()

    # Update history
    history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("history", history)
