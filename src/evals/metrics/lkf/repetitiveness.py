"""LKF Repetitiveness Metric using n-gram entropy."""
import json
import logging
import os
from typing import Dict, Any, List, Optional, Literal

import torch
import numpy as np
import numpy.typing as npt
import scipy.stats.mstats
import nltk
from datasets import DatasetDict

from evals.metrics.base import unlearning_metric
from evals.metrics.lkf.inference_helper import generate_completions


logger = logging.getLogger(__name__)


def n_gram_entropy(
    gen_texts: List[str],
    agg: Literal["arith", "geom"] = "arith"
) -> float:
    """Compute average n-gram entropy across generated texts.

    Args:
        gen_texts: List of generated text strings.
        agg: Aggregation method - "arith" for arithmetic mean, "geom" for geometric mean.

    Returns:
        The aggregated n-gram entropy value.
    """
    assert agg in ["arith", "geom"], f"agg must be 'arith' or 'geom', got {agg}"

    entropies = [compute_n_gram_entropy(txt) for txt in gen_texts]
    result = (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropies)
    return float(result)


def compute_n_gram_entropy(
    sentence: str,
    ns: Optional[List[int]] = None,
    weights: Optional[List[float]] = None,
    agg: Literal["arith", "geom"] = "arith"
) -> float:
    """Compute weighted n-gram entropy for a single sentence.

    Args:
        sentence: The input sentence.
        ns: List of n-gram orders to compute. Defaults to [2, 3].
        weights: Weights for each n-gram order. Defaults to [2/3, 4/3].
        agg: Aggregation method for combining n-gram entropies.

    Returns:
        The weighted n-gram entropy value.
    """
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"], f"agg must be 'arith' or 'geom', got {agg}"

    entropy_list: List[float] = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs: npt.NDArray[np.float64] = np.array([freq for _, freq in fdist.items()])
        if len(freqs) == 0:
            entropy_list.append(0.0)
            continue
        freqs = freqs / freqs.sum()
        entropy = np.sum(-freqs * np.log(freqs) / np.log(2))
        entropy_list.append(float(entropy))

    weighted_entropies: npt.NDArray[np.float64] = np.array(entropy_list) * np.array(weights)
    result = (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(weighted_entropies)
    return float(result)


def compute_freq(sentence: str, n: int = 2) -> nltk.FreqDist:
    """Compute frequency distribution of n-grams in a sentence.

    Args:
        sentence: The input sentence.
        n: The n-gram order.

    Returns:
        Frequency distribution of n-grams.
    """
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


@unlearning_metric(name="repetitiveness_lkf")
def repetitiveness_lkf(model, **kwargs) -> Dict[str, Any]:
    """Evaluate text repetitiveness using n-gram entropy.

    This metric measures how repetitive the model's generations are by computing
    n-gram entropy. Higher entropy indicates more diverse (less repetitive) text.

    Args:
        model: The language model to evaluate.
        **kwargs: Additional arguments including:
            - tokenizer: Model tokenizer
            - data: Dataset to evaluate on
            - batch_size: Batch size for generation (default: 1)
            - num_samples: Maximum samples to evaluate (default: 1000)
            - output_dir: Optional directory to save results

    Returns:
        Dict containing:
            - agg_value: Scaled entropy score (entropy * 100)
            - entropy: Raw n-gram entropy value
            - num_samples: Number of samples evaluated
    """
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    batch_size = kwargs.get("batch_size", 1)
    num_samples = kwargs.get("num_samples", 1000)
    output_dir = kwargs.get("output_dir")
    task_name = kwargs.get("task_name", "model")

    # Check if we should also run baseline evaluation
    baseline_model_path = kwargs.get("baseline_model_path")
    run_baseline = baseline_model_path is not None

    # Download NLTK punkt tokenizer if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    tokenizer.padding_side = 'left'
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    prompts: List[str] = []
    questions: List[Dict[str, Any]] = []

    # Prepare prompts
    for ix, sample in enumerate(data):
        if ix >= num_samples:
            break

        instruction = sample.get('instruction', sample.get('question', ''))
        prompt = f'Instruction: {instruction}\n'

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(formatted_prompt)
        questions.append(sample)

    # Generate completions
    terminators: List[List[int]] = [
        [tokenizer.eos_token_id],
    ]
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != tokenizer.unk_token_id:
            terminators.append([eot_id])

    logger.info(f"Generating {len(prompts)} completions for repetitiveness evaluation")
    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=128,
        batch_size=batch_size,
        do_sample=False,
        stop_id_sequences=terminators
    )

    # Compute entropy
    entropy = n_gram_entropy(outputs)
    scaled_entropy = entropy * 100

    logger.info(f"Repetitiveness entropy: {scaled_entropy:.4f}")

    # Optionally save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Prepare results
        for answer, question in zip(outputs, questions):
            question['prediction'] = answer

        output_result = {
            'entropy': scaled_entropy,
            'num_samples': len(outputs),
            'results': questions,
        }

        # Save as repetitiveness_results.json (standard format)
        result_file = os.path.join(output_dir, "repetitiveness_results.json")
        with open(result_file, 'w') as f:
            json.dump(output_result, f, indent=4)
        logger.info(f"Saved repetitiveness results to: {result_file}")

        # Also save in winrate-compatible format: {task_name}.jsonl
        rep_file = os.path.join(output_dir, f"{task_name}.jsonl")
        with open(rep_file, 'w') as f:
            json.dump(output_result, f, indent=4)
        logger.info(f"Saved winrate-compatible results to: {rep_file}")

    # Run baseline evaluation if requested and not already done
    if run_baseline and output_dir:
        baseline_rep_file = os.path.join(output_dir, "pretrained.jsonl")

        if os.path.exists(baseline_rep_file):
            logger.info(f"Baseline repetitiveness results already exist at: {baseline_rep_file}")
        else:
            logger.info("Running baseline repetitiveness evaluation...")
            logger.info(f"Loading baseline model: {baseline_model_path}")

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Get token if available
            hf_token = os.environ.get("HF_TOKEN")
            tokenizer_kwargs = {"token": hf_token} if hf_token else {}
            model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
            if hf_token:
                model_kwargs["token"] = hf_token

            baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_path, **tokenizer_kwargs)
            if baseline_tokenizer.pad_token is None:
                baseline_tokenizer.pad_token = baseline_tokenizer.eos_token

            baseline_model = AutoModelForCausalLM.from_pretrained(baseline_model_path, **model_kwargs)

            # Use the exact same prompts/questions that were used for the unlearned model
            logger.info(f"Using {len(questions)} prompts from unlearned model evaluation")

            # Prepare baseline prompts
            baseline_tokenizer.padding_side = 'left'
            baseline_model.generation_config.pad_token_id = baseline_tokenizer.pad_token_id

            baseline_prompts: List[str] = []
            for sample in questions:
                instruction = sample.get('instruction', sample.get('question', ''))
                prompt = f'Instruction: {instruction}\n'
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = baseline_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                baseline_prompts.append(formatted_prompt)

            # Generate completions for baseline
            terminators: List[List[int]] = [[baseline_tokenizer.eos_token_id]]
            if hasattr(baseline_tokenizer, 'convert_tokens_to_ids'):
                eot_id = baseline_tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot_id is not None and eot_id != baseline_tokenizer.unk_token_id:
                    terminators.append([eot_id])

            logger.info(f"Generating {len(baseline_prompts)} completions for baseline")
            baseline_outputs = generate_completions(
                model=baseline_model,
                tokenizer=baseline_tokenizer,
                prompts=baseline_prompts,
                max_new_tokens=128,
                batch_size=batch_size,
                do_sample=False,
                stop_id_sequences=terminators
            )

            # Compute baseline entropy
            baseline_entropy = n_gram_entropy(baseline_outputs)
            baseline_scaled_entropy = baseline_entropy * 100

            logger.info(f"Baseline repetitiveness entropy: {baseline_scaled_entropy:.4f}")

            # Prepare baseline results
            baseline_questions = questions.copy()
            for answer, question in zip(baseline_outputs, baseline_questions):
                question['prediction'] = answer

            baseline_output_result = {
                'entropy': baseline_scaled_entropy,
                'num_samples': len(baseline_outputs),
                'results': baseline_questions,
            }

            # Save baseline results
            with open(baseline_rep_file, 'w') as f:
                json.dump(baseline_output_result, f, indent=4)
            logger.info(f"Saved baseline repetitiveness results to: {baseline_rep_file}")

            # Clean up baseline model from memory
            del baseline_model
            del baseline_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    tokenizer.padding_side = 'right'

    return {
        "agg_value": scaled_entropy,
        "entropy": entropy,
        "num_samples": len(outputs),
    }