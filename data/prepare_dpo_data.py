"""
counseLLM — DPO Data Preparation Script

Downloads and processes the PsychoCounsel-Preference dataset into chosen/rejected
pairs for Direct Preference Optimization training.

Dataset:
    Psychotherapy-LLM/PsychoCounsel-Preference (36.7K, CC-BY-NC-4.0)
    - 7 rating dimensions: empathy, relevance, clarity, safety, exploration, autonomy, staging
    - Pairs are sampled by rating gap to maximize preference signal

Output:
    - data/dpo_train.jsonl
    - data/dpo_val.jsonl
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset

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
SAMPLE_SIZE = 2000
VAL_RATIO = 0.1
MAX_PAIRS_PER_QUESTION = 3

RATING_DIMENSIONS = [
    "empathy", "relevance", "clarity", "safety",
    "exploration", "autonomy", "staging",
]

OUTPUT_DIR = Path(__file__).parent
TRAIN_FILE = OUTPUT_DIR / "dpo_train.jsonl"
VAL_FILE = OUTPUT_DIR / "dpo_val.jsonl"


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def compute_rating_gap(row: dict) -> float:
    """Compute the total rating gap between chosen and rejected responses."""
    chosen_sum = sum(row.get(f"chosen_{dim}_rating", 0) for dim in RATING_DIMENSIONS)
    rejected_sum = sum(row.get(f"rejected_{dim}_rating", 0) for dim in RATING_DIMENSIONS)
    return chosen_sum - rejected_sum


def format_dpo_example(row: dict, rating_gap: float) -> dict:
    """Format a single DPO preference pair."""
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["question"].strip()},
    ]

    return {
        "prompt": prompt_messages,
        "chosen": [{"role": "assistant", "content": row["chosen"].strip()}],
        "rejected": [{"role": "assistant", "content": row["rejected"].strip()}],
        "rating_gap": rating_gap,
        "chosen_model": row.get("chosen_model", ""),
        "rejected_model": row.get("rejected_model", ""),
        "question_id": row.get("ID", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare DPO dataset for counseLLM")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE,
                        help="Number of preference pairs to sample")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO,
                        help="Validation split ratio")
    parser.add_argument("--max-pairs-per-question", type=int, default=MAX_PAIRS_PER_QUESTION,
                        help="Max preference pairs per unique question")
    parser.add_argument("--min-rating-gap", type=float, default=0,
                        help="Minimum rating gap to include a pair (0 = no filter)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load dataset
    print("Loading Psychotherapy-LLM/PsychoCounsel-Preference...")
    ds = load_dataset("Psychotherapy-LLM/PsychoCounsel-Preference", split="train")
    print(f"  -> {len(ds)} total rows")

    # Compute rating gaps and filter
    scored_rows = []
    for row in ds:
        question = (row.get("question") or "").strip()
        chosen = (row.get("chosen") or "").strip()
        rejected = (row.get("rejected") or "").strip()

        if not question or not chosen or not rejected:
            continue

        # Skip pairs where chosen == rejected
        if chosen == rejected:
            continue

        gap = compute_rating_gap(row)

        # Apply minimum rating gap filter
        if gap < args.min_rating_gap:
            continue

        scored_rows.append((gap, row))

    print(f"  -> {len(scored_rows)} valid pairs after filtering")

    # Sort by rating gap (descending) — highest signal first
    scored_rows.sort(key=lambda x: x[0], reverse=True)

    # Print rating gap distribution
    gaps = [g for g, _ in scored_rows]
    print(f"\nRating gap distribution:")
    print(f"  Min: {min(gaps):.0f} | Max: {max(gaps):.0f} | "
          f"Mean: {sum(gaps)/len(gaps):.1f} | Median: {sorted(gaps)[len(gaps)//2]:.0f}")

    # Sample with question diversity constraint
    question_counts = defaultdict(int)
    sampled = []

    for gap, row in scored_rows:
        qid = row.get("ID", row.get("question", "")[:50])
        if question_counts[qid] >= args.max_pairs_per_question:
            continue
        question_counts[qid] += 1
        sampled.append(format_dpo_example(row, gap))
        if len(sampled) >= args.sample_size:
            break

    print(f"\nSampled {len(sampled)} pairs from {len(question_counts)} unique questions")
    print(f"  Avg pairs per question: {len(sampled)/len(question_counts):.1f}")

    # Print rating gap stats for sampled pairs
    sampled_gaps = [ex["rating_gap"] for ex in sampled]
    print(f"  Sampled rating gap — Min: {min(sampled_gaps):.0f} | "
          f"Max: {max(sampled_gaps):.0f} | Mean: {sum(sampled_gaps)/len(sampled_gaps):.1f}")

    # Print model distribution
    chosen_models = defaultdict(int)
    rejected_models = defaultdict(int)
    for ex in sampled:
        chosen_models[ex["chosen_model"]] += 1
        rejected_models[ex["rejected_model"]] += 1
    print(f"\n  Chosen model distribution:")
    for model, count in sorted(chosen_models.items(), key=lambda x: -x[1])[:5]:
        print(f"    {model:30s} {count:>5d}")
    print(f"  Rejected model distribution:")
    for model, count in sorted(rejected_models.items(), key=lambda x: -x[1])[:5]:
        print(f"    {model:30s} {count:>5d}")

    # Shuffle and split
    random.shuffle(sampled)
    val_size = int(len(sampled) * args.val_ratio)
    val_examples = sampled[:val_size]
    train_examples = sampled[val_size:]

    print(f"\nTrain: {len(train_examples)} | Val: {len(val_examples)}")

    # Write to JSONL
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def write_jsonl(filepath: Path, data: list[dict]):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                # Remove metadata fields from output
                output_item = {
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                }
                f.write(json.dumps(output_item, ensure_ascii=False) + "\n")
        print(f"Written: {filepath} ({len(data)} examples)")

    write_jsonl(TRAIN_FILE, train_examples)
    write_jsonl(VAL_FILE, val_examples)

    # Write metadata
    metadata = {
        "total_source_rows": len(ds),
        "valid_pairs_after_filter": len(scored_rows),
        "sampled_pairs": len(sampled),
        "train_pairs": len(train_examples),
        "val_pairs": len(val_examples),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "max_pairs_per_question": args.max_pairs_per_question,
        "min_rating_gap": args.min_rating_gap,
        "unique_questions": len(question_counts),
        "system_prompt": SYSTEM_PROMPT,
        "rating_dimensions": RATING_DIMENSIONS,
        "dataset": "Psychotherapy-LLM/PsychoCounsel-Preference",
        "license": "CC-BY-NC-4.0",
    }
    meta_path = OUTPUT_DIR / "dpo_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Written: {meta_path}")

    print("\nDone! DPO data preparation complete.")


if __name__ == "__main__":
    main()
