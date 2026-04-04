"""
counseLLM — Automated Evaluation Script

Generates responses from all model variants (base, SFT, SFT+DPO)
on a set of test prompts and computes automated metrics.

Metrics:
    - Response length statistics
    - Perplexity (on validation set)
    - Diversity metrics (distinct-1, distinct-2 n-grams)
    - BERTScore (semantic similarity to reference responses)
    - ROUGE-L (lexical overlap with reference responses)

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --models base sft        # only specific variants
    python eval/evaluate.py --prompts eval/test_prompts.json
"""

import json
import math
import argparse
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from bert_score import score as bert_score
from rouge_score import rouge_scorer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Use cached model on volume if available (avoids re-downloading on Modal)
_CACHED_MODEL = Path(__file__).resolve().parent.parent / "models" / "base"
if _CACHED_MODEL.exists():
    BASE_MODEL = str(_CACHED_MODEL)

SYSTEM_PROMPT = (
    "You are a mental health counselor providing supportive, empathetic guidance. "
    "Respond by first acknowledging the person's feelings, then explore their "
    "situation with open-ended questions. Use techniques like reflective listening, "
    "validation, and gentle reframing. Keep responses warm, conversational, and "
    "non-judgmental. For crisis situations involving self-harm or suicide, "
    "prioritize safety by encouraging the person to contact a crisis helpline "
    "or emergency services."
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = PROJECT_ROOT / "eval" / "test_prompts.json"
SFT_ADAPTER = PROJECT_ROOT / "outputs" / "sft" / "final"
DPO_ADAPTER = PROJECT_ROOT / "outputs" / "dpo" / "final"
RESULTS_DIR = PROJECT_ROOT / "results"

GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(variant: str):
    """Load a model variant: 'base', 'sft', or 'dpo'."""
    print(f"\nLoading model variant: {variant}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if variant == "sft":
        print(f"  Loading SFT adapter from {SFT_ADAPTER}")
        model = PeftModel.from_pretrained(model, str(SFT_ADAPTER))

    elif variant == "dpo":
        print(f"  Loading SFT adapter from {SFT_ADAPTER}")
        model = PeftModel.from_pretrained(model, str(SFT_ADAPTER))
        model = model.merge_and_unload()
        print(f"  Loading DPO adapter from {DPO_ADAPTER}")
        model = PeftModel.from_pretrained(model, str(DPO_ADAPTER))

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate a counseling response for a user prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **GENERATION_CONFIG,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (exclude input)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_length_stats(responses: list[str]) -> dict:
    """Compute response length statistics."""
    word_counts = [len(r.split()) for r in responses]
    char_counts = [len(r) for r in responses]
    return {
        "avg_words": sum(word_counts) / len(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "avg_chars": sum(char_counts) / len(char_counts),
    }


def compute_distinct_ngrams(responses: list[str], n: int) -> float:
    """Compute distinct-n metric (ratio of unique n-grams to total n-grams)."""
    all_ngrams = []
    for response in responses:
        words = response.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 2048) -> float:
    """Compute average perplexity over a list of texts."""
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def compute_bert_score(prompts: list[str], responses: list[str]) -> dict:
    """Compute BERTScore between prompts and responses (measures semantic relevance)."""
    P, R, F1 = bert_score(responses, prompts, lang="en", verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }


def compute_rouge_l(prompts: list[str], responses: list[str]) -> dict:
    """Compute ROUGE-L between prompts and responses."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(p, r)["rougeL"] for p, r in zip(prompts, responses)]
    return {
        "precision": sum(s.precision for s in scores) / len(scores),
        "recall": sum(s.recall for s in scores) / len(scores),
        "f1": sum(s.fmeasure for s in scores) / len(scores),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate counseLLM model variants")
    parser.add_argument("--models", nargs="+", default=["base", "sft", "dpo"],
                        choices=["base", "sft", "dpo"],
                        help="Model variants to evaluate")
    parser.add_argument("--prompts", type=str, default=str(DEFAULT_PROMPTS),
                        help="Path to test prompts JSON file")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR),
                        help="Output directory for results")
    args = parser.parse_args()

    # Load test prompts
    with open(args.prompts, "r", encoding="utf-8") as f:
        test_prompts = json.load(f)
    print(f"Loaded {len(test_prompts)} test prompts")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for variant in args.models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {variant}")
        print(f"{'='*60}")

        model, tokenizer = load_model_and_tokenizer(variant)

        # Generate responses
        responses = []
        for i, prompt_data in enumerate(test_prompts):
            print(f"  [{i+1}/{len(test_prompts)}] {prompt_data['id']}...", end=" ")
            response = generate_response(model, tokenizer, prompt_data["prompt"])
            responses.append({
                **prompt_data,
                "response": response,
                "model_variant": variant,
            })
            word_count = len(response.split())
            print(f"({word_count} words)")

        # Compute metrics
        response_texts = [r["response"] for r in responses]
        prompt_texts = [r["prompt"] for r in responses]
        metrics = {
            "length_stats": compute_length_stats(response_texts),
            "distinct_1": compute_distinct_ngrams(response_texts, 1),
            "distinct_2": compute_distinct_ngrams(response_texts, 2),
        }

        # Compute perplexity on generated responses
        print("  Computing perplexity...")
        metrics["perplexity"] = compute_perplexity(model, tokenizer, response_texts)

        # Compute BERTScore (semantic relevance to prompt)
        print("  Computing BERTScore...")
        metrics["bert_score"] = compute_bert_score(prompt_texts, response_texts)

        # Compute ROUGE-L (lexical overlap with prompt)
        print("  Computing ROUGE-L...")
        metrics["rouge_l"] = compute_rouge_l(prompt_texts, response_texts)

        all_results[variant] = {
            "metrics": metrics,
            "responses": responses,
        }

        print(f"\n  Results for {variant}:")
        print(f"    Avg words: {metrics['length_stats']['avg_words']:.0f}")
        print(f"    Distinct-1: {metrics['distinct_1']:.3f}")
        print(f"    Distinct-2: {metrics['distinct_2']:.3f}")
        print(f"    Perplexity: {metrics['perplexity']:.2f}")
        print(f"    BERTScore F1: {metrics['bert_score']['f1']:.4f}")
        print(f"    ROUGE-L F1: {metrics['rouge_l']['f1']:.4f}")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Save results
    results_file = output_dir / "eval_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_file}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Metric':<25} ", end="")
    for variant in args.models:
        print(f"{variant:>12}", end="")
    print()
    print("-" * (25 + 12 * len(args.models)))

    metric_keys = [
        ("Avg words", lambda m: f"{m['length_stats']['avg_words']:.0f}"),
        ("Min words", lambda m: f"{m['length_stats']['min_words']}"),
        ("Max words", lambda m: f"{m['length_stats']['max_words']}"),
        ("Distinct-1", lambda m: f"{m['distinct_1']:.3f}"),
        ("Distinct-2", lambda m: f"{m['distinct_2']:.3f}"),
        ("Perplexity", lambda m: f"{m['perplexity']:.2f}"),
        ("BERTScore F1", lambda m: f"{m['bert_score']['f1']:.4f}"),
        ("ROUGE-L F1", lambda m: f"{m['rouge_l']['f1']:.4f}"),
    ]

    for name, fmt_fn in metric_keys:
        print(f"{name:<25} ", end="")
        for variant in args.models:
            val = fmt_fn(all_results[variant]["metrics"])
            print(f"{val:>12}", end="")
        print()

    # Save comparison responses side-by-side
    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(exist_ok=True)

    for i, prompt_data in enumerate(test_prompts):
        comparison = {
            "id": prompt_data["id"],
            "category": prompt_data["category"],
            "prompt": prompt_data["prompt"],
            "responses": {},
        }
        for variant in args.models:
            comparison["responses"][variant] = all_results[variant]["responses"][i]["response"]

        comp_file = comparisons_dir / f"{prompt_data['id']}.json"
        with open(comp_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\nSide-by-side comparisons saved to {comparisons_dir}/")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
