"""
counseLLM — SFT Data Preparation Script

Downloads, cleans, and merges 5 mental health counseling datasets into a
unified Llama 3.1 chat-template format for supervised fine-tuning.

Datasets:
    1. ShenLab/MentalChat16K         (~16K, synthetic + interview, MIT)
    2. Estwld/empathetic_dialogues_llm (~10K subsample, real human, Apache 2.0)
    3. EmoCareAI/Psych8k             (~8K, real therapist, CC-BY-NC-SA, gated)
    4. nbertagnolli/counsel-chat      (~940 after dedup, real therapist)
    5. thu-coai/esconv               (~910, real human + strategy labels, CC-BY-NC)

Output:
    - data/sft_train.jsonl
    - data/sft_val.jsonl
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a mental health counselor providing supportive, empathetic guidance. "
    "Respond by first acknowledging the person's feelings, then explore their "
    "situation with open-ended questions. Use techniques like reflective listening, "
    "validation, and gentle reframing. Keep responses warm, conversational, and "
    "non-judgmental. For crisis situations involving self-harm or suicide, "
    "prioritize safety by encouraging the person to contact a crisis helpline "
    "or emergency services."
)

SEED = 42
EMPATHETIC_DIALOGUES_SUBSAMPLE = 10_000
VAL_RATIO = 0.1

OUTPUT_DIR = Path(__file__).parent
TRAIN_FILE = OUTPUT_DIR / "sft_train.jsonl"
VAL_FILE = OUTPUT_DIR / "sft_val.jsonl"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_single_turn(user_msg: str, assistant_msg: str, source: str) -> dict:
    """Format a single-turn conversation."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": assistant_msg.strip()},
        ],
        "source": source,
    }


def format_multi_turn(turns: list[dict], source: str) -> dict:
    """Format a multi-turn conversation.

    Args:
        turns: List of {"role": "user"|"assistant", "content": str}
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(turns)
    return {
        "messages": messages,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Dataset processors
# ---------------------------------------------------------------------------

def process_mentalchat16k() -> list[dict]:
    """ShenLab/MentalChat16K — single-turn, instruction/input/output format."""
    print("[1/5] Loading ShenLab/MentalChat16K...")
    ds = load_dataset("ShenLab/MentalChat16K", split="train")
    examples = []
    for row in ds:
        user_msg = (row["input"] or "").strip()
        assistant_msg = (row["output"] or "").strip()
        if not user_msg or not assistant_msg:
            continue
        examples.append(format_single_turn(user_msg, assistant_msg, "mentalchat16k"))
    print(f"  -> {len(examples)} examples")
    return examples


def process_empathetic_dialogues(subsample: int = EMPATHETIC_DIALOGUES_SUBSAMPLE) -> list[dict]:
    """Estwld/empathetic_dialogues_llm — multi-turn conversations with emotion labels."""
    print("[2/5] Loading Estwld/empathetic_dialogues_llm...")
    ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train")

    all_examples = []
    for row in ds:
        conversations = row["conversations"]
        if not conversations or len(conversations) < 2:
            continue

        turns = []
        valid = True
        for turn in conversations:
            role = turn["role"]
            content = turn["content"].strip()
            if role not in ("user", "assistant") or not content:
                valid = False
                break
            turns.append({"role": role, "content": content})

        if valid and len(turns) >= 2:
            all_examples.append(format_multi_turn(turns, "empathetic_dialogues"))

    # Subsample for balance
    if len(all_examples) > subsample:
        random.seed(SEED)
        all_examples = random.sample(all_examples, subsample)

    print(f"  -> {len(all_examples)} examples (subsampled from {len(ds)})")
    return all_examples


def process_psych8k() -> list[dict]:
    """EmoCareAI/Psych8k — real therapist multi-turn dialogues (gated)."""
    print("[3/5] Loading EmoCareAI/Psych8k (gated — requires HF approval)...")
    try:
        ds = load_dataset("EmoCareAI/Psych8k", split="train")
    except Exception as e:
        print(f"  -> SKIPPED: Could not load Psych8k ({e})")
        print("     Apply for access at: https://huggingface.co/datasets/EmoCareAI/Psych8k")
        return []

    examples = []
    for row in ds:
        # Psych8k has multi-turn conversations — detect format
        # Try common column patterns
        if "conversations" in row and isinstance(row["conversations"], list):
            turns = []
            for turn in row["conversations"]:
                role = turn.get("role", turn.get("from", ""))
                content = turn.get("content", turn.get("value", "")).strip()
                # Normalize roles
                if role in ("human", "patient", "client", "user"):
                    role = "user"
                elif role in ("gpt", "therapist", "counselor", "assistant"):
                    role = "assistant"
                else:
                    continue
                if content:
                    turns.append({"role": role, "content": content})
            if len(turns) >= 2:
                examples.append(format_multi_turn(turns, "psych8k"))

        elif "input" in row and "output" in row:
            user_msg = (row["input"] or "").strip()
            assistant_msg = (row["output"] or "").strip()
            if user_msg and assistant_msg:
                examples.append(format_single_turn(user_msg, assistant_msg, "psych8k"))

    print(f"  -> {len(examples)} examples")
    return examples


def process_counsel_chat() -> list[dict]:
    """nbertagnolli/counsel-chat — real therapist Q&A, deduplicated by question."""
    print("[4/5] Loading nbertagnolli/counsel-chat...")
    ds = load_dataset("nbertagnolli/counsel-chat", split="train")

    # Group answers by questionID, pick highest-upvoted per question
    question_answers = defaultdict(list)
    for row in ds:
        qid = row["questionID"]
        question_answers[qid].append(row)

    examples = []
    for qid, answers in question_answers.items():
        best = max(answers, key=lambda x: x["upvotes"])
        title = (best["questionTitle"] or "").strip()
        text = (best["questionText"] or "").strip()

        # Combine title + text if both exist
        if title and text:
            user_msg = f"{title}\n\n{text}"
        else:
            user_msg = title or text

        assistant_msg = (best["answerText"] or "").strip()

        if not user_msg or not assistant_msg:
            continue

        examples.append(format_single_turn(user_msg, assistant_msg, "counsel_chat"))

    print(f"  -> {len(examples)} examples (deduplicated from {len(ds)} rows, {len(question_answers)} unique questions)")
    return examples


def process_esconv() -> list[dict]:
    """thu-coai/esconv — real human emotional support conversations with strategy labels."""
    print("[5/5] Loading thu-coai/esconv...")
    ds = load_dataset("thu-coai/esconv", split="train")

    examples = []
    for row in ds:
        # Single 'text' column containing JSON string
        try:
            data = json.loads(row["text"]) if isinstance(row["text"], str) else row["text"]
        except (json.JSONDecodeError, TypeError):
            continue

        dialog = data.get("dialog", [])
        if not dialog or len(dialog) < 2:
            continue

        turns = []
        for turn in dialog:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "").strip()
            if not text:
                continue
            if speaker == "usr":
                role = "user"
            elif speaker == "sys":
                role = "assistant"
            else:
                continue
            turns.append({"role": role, "content": text})

        # Ensure conversation starts with user and alternates properly
        if turns and turns[0]["role"] == "user" and len(turns) >= 2:
            examples.append(format_multi_turn(turns, "esconv"))

    print(f"  -> {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def validate_example(example: dict) -> bool:
    """Basic validation for a formatted example."""
    messages = example.get("messages", [])
    if len(messages) < 3:  # system + at least user + assistant
        return False
    if messages[0]["role"] != "system":
        return False
    # Check that user and assistant messages exist
    roles = [m["role"] for m in messages[1:]]
    if "user" not in roles or "assistant" not in roles:
        return False
    # Check no empty content
    if any(not m["content"].strip() for m in messages):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT dataset for counseLLM")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="Validation split ratio")
    parser.add_argument("--empathetic-subsample", type=int, default=EMPATHETIC_DIALOGUES_SUBSAMPLE,
                        help="Number of empathetic dialogue examples to subsample")
    parser.add_argument("--skip-psych8k", action="store_true",
                        help="Skip Psych8k dataset (if you don't have access)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Process each dataset
    all_examples = []
    all_examples.extend(process_mentalchat16k())
    all_examples.extend(process_empathetic_dialogues(subsample=args.empathetic_subsample))

    if not args.skip_psych8k:
        all_examples.extend(process_psych8k())
    else:
        print("[3/5] Psych8k SKIPPED (--skip-psych8k flag)")

    all_examples.extend(process_counsel_chat())
    all_examples.extend(process_esconv())

    # Validate
    valid_examples = [ex for ex in all_examples if validate_example(ex)]
    dropped = len(all_examples) - len(valid_examples)
    if dropped > 0:
        print(f"\nDropped {dropped} invalid examples during validation")

    # Print source distribution
    source_counts = defaultdict(int)
    for ex in valid_examples:
        source_counts[ex["source"]] += 1
    print(f"\n{'='*50}")
    print(f"Total valid examples: {len(valid_examples)}")
    print(f"{'='*50}")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = count / len(valid_examples) * 100
        print(f"  {source:30s} {count:>6d}  ({pct:.1f}%)")

    # Shuffle and split
    random.shuffle(valid_examples)
    val_size = int(len(valid_examples) * args.val_ratio)
    val_examples = valid_examples[:val_size]
    train_examples = valid_examples[val_size:]

    print(f"\nTrain: {len(train_examples)} | Val: {len(val_examples)}")

    # Write to JSONL
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def write_jsonl(filepath: Path, data: list[dict]):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                # Remove 'source' field from output (only used for tracking)
                output_item = {"messages": item["messages"]}
                f.write(json.dumps(output_item, ensure_ascii=False) + "\n")
        print(f"Written: {filepath} ({len(data)} examples)")

    write_jsonl(TRAIN_FILE, train_examples)
    write_jsonl(VAL_FILE, val_examples)

    # Also write a metadata file
    metadata = {
        "total_examples": len(valid_examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "source_distribution": dict(sorted(source_counts.items(), key=lambda x: -x[1])),
        "system_prompt": SYSTEM_PROMPT,
        "datasets": {
            "mentalchat16k": "ShenLab/MentalChat16K",
            "empathetic_dialogues": "Estwld/empathetic_dialogues_llm",
            "psych8k": "EmoCareAI/Psych8k",
            "counsel_chat": "nbertagnolli/counsel-chat",
            "esconv": "thu-coai/esconv",
        },
    }
    meta_path = OUTPUT_DIR / "sft_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Written: {meta_path}")

    print("\nDone! SFT data preparation complete.")


if __name__ == "__main__":
    main()
