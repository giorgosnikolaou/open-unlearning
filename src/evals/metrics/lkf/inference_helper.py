# Source: https://github.com/allenai/open-instruct/blob/main/eval/utils.py
import logging
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import tqdm
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer, 
    StoppingCriteria,
    StoppingCriteriaList
)

logger = logging.getLogger(__name__)


# class KeyWordsCriteria(StoppingCriteria):
#     def __init__(self, stop_id_sequences):
#         assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
#         self.stop_sequences = stop_id_sequences

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         sequences_should_be_stopped = []
#         for i in range(input_ids.shape[0]):
#             sequence_should_be_stopped = False
#             for stop_sequence in self.stop_sequences:
#                 if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
#                     sequence_should_be_stopped = True
#                     break
#             sequences_should_be_stopped.append(sequence_should_be_stopped)
#         return all(sequences_should_be_stopped)

class KeyWordsCriteria(StoppingCriteria):
    """Stopping criteria based on sequences of token IDs.

    This implementation returns a per-sample BoolTensor, allowing each sequence
    in the batch to stop independently when it reaches a stop token.

    Attributes:
        stop_sequences: List of token ID sequences that trigger stopping.
    """

    stop_sequences: List[List[int]]

    def __init__(self, stop_id_sequences: List[List[int]]) -> None:
        """Initialize the stopping criteria.

        Args:
            stop_id_sequences: List of token ID sequences to match.

        Raises:
            TypeError: If stop_id_sequences is not a list of lists of ints.
        """
        if (
            not isinstance(stop_id_sequences, list)
            or len(stop_id_sequences) == 0
            or not all(isinstance(seq, list) for seq in stop_id_sequences)
            or not all(isinstance(x, int) for seq in stop_id_sequences for x in seq)
        ):
            raise TypeError("stop_id_sequences should be a list of list of ints")
        self.stop_sequences = stop_id_sequences

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any
    ) -> torch.BoolTensor:
        """Check if generation should stop for each sequence in the batch.

        Args:
            input_ids: Generated token IDs of shape (batch_size, seq_len).
            scores: Token scores.
            **kwargs: Additional arguments.

        Returns:
            BoolTensor of shape (batch_size,) indicating which sequences should stop.
        """
        batch_size = input_ids.size(0)
        stop_mask: torch.BoolTensor = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=input_ids.device
        ) # type: ignore

        for idx in range(batch_size):
            for stop_sequence in self.stop_sequences:
                if input_ids[idx, -len(stop_sequence):].tolist() == stop_sequence:
                    stop_mask[idx] = True
                    break

        # NOTE:
        # The original implementation returned a single bool via `all(...)`, which caused
        # generation to stop only when *all* sequences in the batch had reached a stop token
        # (i.e., global batch-level stopping).
        #
        # Here, we return a per-sample BoolTensor instead, so each sequence can stop
        # independently. This prevents finished samples from continuing to generate
        # unnecessary tokens and matches modern Transformers batched generation behavior.
        return stop_mask



@torch.no_grad()
def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    batch_size: int = 1,
    stop_id_sequences: Optional[List[List[int]]] = None,
    add_special_tokens: bool = True,
    disable_tqdm: bool = False,
    **generation_kwargs: Any
) -> List[str]:
    """Generate text completions for a list of prompts.

    Args:
        model: The language model to use for generation.
        tokenizer: The tokenizer for the model.
        prompts: List of prompt strings.
        batch_size: Number of prompts to process in parallel.
        stop_id_sequences: Optional list of token ID sequences that trigger stopping.
        add_special_tokens: Whether to add special tokens during tokenization.
        disable_tqdm: Whether to disable the progress bar.
        **generation_kwargs: Additional arguments passed to model.generate().

    Returns:
        List of generated completion strings (without the prompts).
    """
    generations: List[str] = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        )
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

        try:
            batch_outputs: torch.LongTensor = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=(
                    StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) 
                    if stop_id_sequences else 
                    None
                ),
                **generation_kwargs
            ) # type: ignore
            
            # The stopping criteria may not remove all stop sequences
            # due to batch-level processing, so remove any remaining ones.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.size(0)):
                    for token_idx in range(batch_input_ids.size(1), batch_outputs.size(1)):
                        if any(
                            batch_outputs[output_idx, token_idx: token_idx + len(stop_sequence)].tolist() == stop_sequence
                            for stop_sequence in stop_id_sequences
                        ):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id # type: ignore
                            break

            # Remove the prompt from the output
            # Re-encode the prompt to ensure special tokens are treated consistently
            batch_outputs_text = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts_text = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # Duplicate the prompts to match the number of return sequences
            batch_prompts_expanded = [
                prompt for prompt in batch_prompts_text for _ in range(num_return_sequences)
            ]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts_expanded, batch_outputs_text)
            ]
        except Exception as e:
            logger.error(f"Error when generating completions for batch: {e}")
            logger.error(f"Batch prompts: {batch_prompts}")
            logger.warning("Using empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(batch_prompts) // num_return_sequences) # type: ignore

    assert len(generations) == len(prompts) * num_return_sequences, (
        f"Expected {len(prompts) * num_return_sequences} generations, got {len(generations)}"
    )
    return generations


@torch.no_grad()
def get_next_word_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    candidate_token_ids: Optional[List[int]] = None,
    batch_size: int = 1,
    return_token_predictions: bool = False,
    add_special_tokens: bool = True,
    disable_tqdm: bool = False
) -> Tuple[List[Any], List[List[float]]]:
    """Get next-word predictions for a list of prompts.

    Args:
        model: The language model to use.
        tokenizer: The tokenizer for the model.
        prompts: List of prompt strings.
        candidate_token_ids: Optional list of token IDs to restrict predictions to.
        batch_size: Number of prompts to process in parallel.
        return_token_predictions: If True, return tokens instead of indices.
        add_special_tokens: Whether to add special tokens during tokenization.
        disable_tqdm: Whether to disable the progress bar.

    Returns:
        Tuple of (predictions, probabilities) where predictions are either
        token strings (if return_token_predictions=True) or indices (ints).
    """
    predictions: List[Any] = []
    probs: List[List[float]] = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        )
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits.float()[:, -1, :]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)

        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices.tolist())
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts)) # type: ignore

    assert len(predictions) == len(prompts), (
        f"Expected {len(prompts)} predictions, got {len(predictions)}"
    )
    return predictions, probs


@torch.no_grad()
def score_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    scoring_examples: List[Dict[str, Any]],
    batch_size: int = 1,
    aggregation: str = "sum",
    disable_tqdm: bool = False
) -> Dict[str, Dict[str, float]]:
    """Score completions using log probabilities.

    Args:
        model: The language model to use for scoring.
        tokenizer: The tokenizer for the model.
        scoring_examples: List of dicts with 'prompt' and 'completions' keys.
        batch_size: Number of examples to process in parallel.
        aggregation: How to aggregate log probs ("sum", "mean", or "max").
        disable_tqdm: Whether to disable the progress bar.

    Returns:
        Nested dict mapping prompt -> completion -> score.

    Raises:
        ValueError: If aggregation method is invalid.
    """
    # Unroll the scoring examples
    unrolled_examples: List[Dict[str, str]] = []
    for scoring_example in scoring_examples:
        prompt = scoring_example["prompt"]
        for completion in scoring_example["completions"]:
            unrolled_examples.append({
                "prompt": prompt,
                "completion": completion
            })

    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(unrolled_examples), desc="Scoring Completions")

    scores: List[float] = []
    for i in range(0, len(unrolled_examples), batch_size):
        batch_prompts = [example["prompt"] for example in unrolled_examples[i:i + batch_size]]
        batch_examples = [
            (example["prompt"] if example["prompt"][-1] in ["\n", " "] else example["prompt"] + " ")
            + example["completion"] for example in unrolled_examples[i:i + batch_size]
        ]
        tokenized_batch = tokenizer(batch_examples, padding="longest", return_tensors="pt")
        if model.device.type == "cuda":
            tokenized_batch = {
                key: value.cuda() for key, value in tokenized_batch.items()
            }
        tokenized_batch.pop("token_type_ids", None)
        outputs = model(**tokenized_batch)

        for example_idx, (prompt, example) in enumerate(zip(batch_prompts, batch_examples)):
            tokenized_prompt = tokenizer(prompt, padding=False, return_tensors="pt").input_ids.squeeze(0)
            tokenized_example = tokenizer(example, padding=False, return_tensors="pt").input_ids.squeeze(0)
            completion_ids = tokenized_example[len(tokenized_prompt):]

            # Get the logits for the entire example, removing the padding logits
            if tokenizer.padding_side == "right":
                example_logits = outputs.logits.float()[example_idx, :len(tokenized_example), :]
            else:
                example_logits = outputs.logits.float()[example_idx, -len(tokenized_example):, :]

            # Get the logits for the completion portion
            # Note: shift index left by 1 because logits are computed for the next token
            completion_logits = example_logits[len(tokenized_prompt) - 1:len(tokenized_example) - 1, :]
            completion_log_probs = torch.log_softmax(completion_logits, dim=-1)[
                range(len(completion_ids)), completion_ids
            ]

            if aggregation == "sum":
                score = completion_log_probs.sum().item()
            elif aggregation == "mean":
                score = completion_log_probs.mean().item()
            elif aggregation == "max":
                score = completion_log_probs.max().item()
            else:
                raise ValueError(f"Invalid aggregation method: {aggregation}")
            scores.append(score)

        if not disable_tqdm:
            progress.update(len(batch_examples)) # type: ignore

    # Roll up the scores
    rolled_up_scores: Dict[str, Dict[str, float]] = {}
    for unrolled_example, score in zip(unrolled_examples, scores):
        prompt = unrolled_example["prompt"]
        completion = unrolled_example["completion"]
        if prompt not in rolled_up_scores:
            rolled_up_scores[prompt] = {}
        rolled_up_scores[prompt][completion] = score

    return rolled_up_scores


def dynamic_import_function(function_path: str) -> Callable[..., Any]:
    """Dynamically import a function from a path string.

    Args:
        function_path: Path string (e.g., "module.submodule.my_function").

    Returns:
        The imported function.

    Raises:
        ValueError: If the path doesn't contain at least one dot.
        AttributeError: If the function doesn't exist in the module.
    """
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function
