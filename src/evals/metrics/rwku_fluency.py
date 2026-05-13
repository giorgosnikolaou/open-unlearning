"""RWKU fluency metric: bigram/trigram Shannon entropy on model generations."""

import logging
import math
from collections import Counter

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from evals.metrics.base import unlearning_metric

logger = logging.getLogger("evaluator")


def _ngram_entropy(tokens, n):
    """Compute Shannon entropy over the n-gram frequency distribution."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


@unlearning_metric(name="rwku_fluency")
def rwku_fluency(model, **kwargs):
    """Compute bigram and trigram entropy from model generations.

    Generates responses for each sample in the dataset (AlpacaEval-style
    instructions), then computes average bigram and trigram Shannon entropy
    over word-level n-gram distributions. Higher entropy indicates more
    diverse/fluent text.

    Returns:
        Dict with agg_value (average of bi+trigram entropy), bigram_entropy,
        and trigram_entropy.
    """
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    batch_size = kwargs.get("batch_size", 4)
    generation_args = kwargs.get("generation_args", {})

    if hasattr(generation_args, "items"):
        generation_args = OmegaConf.to_container(generation_args, resolve=True)

    max_new_tokens = generation_args.pop("max_new_tokens", 512)
    generation_args.pop("stopwords", None)

    model.eval()
    bigram_entropies = []
    trigram_entropies = []

    # data is an HFDataset — iterate in simple batches
    max_samples = kwargs.get("max_samples", None)
    num_samples = min(len(data), max_samples) if max_samples else len(data)
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Computing fluency"):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_items = [data[i] for i in range(start_idx, end_idx)]

        # Build prompts from the instruction field
        instructions = []
        for item in batch_items:
            # utility_fluency has 'instruction' column
            instruction = item.get("instruction", item.get("question", ""))
            instructions.append(instruction)

        # Tokenize
        inputs = tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                **generation_args,
            )

        # Decode only newly generated tokens
        for i in range(len(batch_items)):
            gen_ids = outputs[i, inputs["input_ids"].shape[1] :]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Word-level tokenization for n-gram entropy
            words = gen_text.split()
            bigram_entropies.append(_ngram_entropy(words, 2))
            trigram_entropies.append(_ngram_entropy(words, 3))

    avg_bigram = float(np.mean(bigram_entropies)) if bigram_entropies else 0.0
    avg_trigram = float(np.mean(trigram_entropies)) if trigram_entropies else 0.0
    avg_entropy = (avg_bigram + avg_trigram) / 2.0

    return {
        "agg_value": avg_entropy,
        "bigram_entropy": avg_bigram,
        "trigram_entropy": avg_trigram,
    }
