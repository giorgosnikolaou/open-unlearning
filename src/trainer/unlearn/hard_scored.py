"""Trainers that use pre-computed binary token masks ("hard scores") for
token-level unlearning.

Supported scoring methods:
    - ``gt``: Ground-truth important tokens (Arrow dataset with character ranges)
    - ``seul``: SEUL log-probability threshold (arxiv 2402.05813)
    - ``su_llm``: SU-LLM probability divergence (arxiv 2506.00876)
    - ``su_ngram``: SU-Ngram n-gram divergence (arxiv 2506.00876)
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Any, Dict

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch import nn
from transformers import AutoModelForCausalLM

from data.Cache import SampleCache
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.sb import SelfBalancingGradDiff

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


# ============================================================================
# Post-processing helpers (ported from importance-scoring repo)
# ============================================================================


def _aggregate_spans(
    selected: torch.Tensor,
    mask: torch.Tensor,
    span_distance: int,
) -> torch.Tensor:
    """Fill gaps of size <= *span_distance* between selected tokens (SEUL)."""
    result = selected.bool()
    indices = torch.where(result & mask)[0]
    if indices.numel() < 2:
        return (result & mask).float()
    for start, end in zip(indices[:-1], indices[1:]):
        if (end - start - 1).item() <= span_distance:
            result[start : end + 1] = True
    return (result & mask).float()


def _expand_context(
    selected: torch.Tensor,
    mask: torch.Tensor,
    context_size: int,
) -> torch.Tensor:
    """Context expansion around selected tokens (SU Algorithm 1)."""
    if context_size <= 0:
        return (selected.bool() & mask).float()
    T = selected.numel()
    c = context_size
    result = selected.bool()
    tentative: set[int] = set()
    for i in range(T):
        if not mask[i]:
            tentative.discard(i)
            continue
        if selected[i]:
            for j in range(max(0, i - c), min(T, i + c + 1)):
                tentative.add(j)
        elif i in tentative:
            tentative.discard(i)
    for idx in tentative:
        if mask[idx]:
            result[idx] = True
    return (result & mask).float()


# ============================================================================
# GT mask computation (range-based format)
# ============================================================================


def _build_formatted_text(
    question: str,
    answer: str,
    template_config: Dict[str, Any],
    tokenizer: Any,
) -> tuple[str, str]:
    """Reconstruct the full formatted text and prompt text
    using the same logic as ``preprocess_chat_instance``.

    Returns ``(full_text, prompt_text)`` as strings.
    """
    if template_config["apply_chat_template"]:
        chat: list[dict[str, str]] = []
        system_prompt = template_config.get("system_prompt", None)
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        chat.append({"role": "user", "content": question})
        chat.append({"role": "assistant", "content": answer})
        date_str = template_config.get("date_string", None)
        date_info = {"date_string": date_str} if date_str is not None else {}
        full_text: str = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False, **date_info
        )
        prompt_text: str = tokenizer.apply_chat_template(
            chat[:-1], tokenize=False, add_generation_prompt=True, **date_info
        )
    else:
        prompt_text = ""
        spst = template_config.get("system_prompt_with_special_tokens", None)
        if spst:
            prompt_text += spst
        prompt_text += (
            template_config["user_start_tag"]
            + question
            + template_config["user_end_tag"]
            + template_config["asst_start_tag"]
        )
        full_text = prompt_text + answer

    return full_text, prompt_text


def _compute_single_gt_mask(
    question: str,
    answer: str,
    gt_ranges: list[tuple[int, int]],
    tokenizer: Any,
    template_config: Dict[str, Any],
    max_length: int,
) -> torch.Tensor:
    """Compute a binary mask ``(T-1,)`` for a single sample from GT character ranges."""

    if not gt_ranges:
        # No important tokens → return zeros; caller will determine T
        # We use a placeholder that will be padded/truncated by HardScorer
        return torch.zeros(1, dtype=torch.float)

    full_text, prompt_text = _build_formatted_text(
        question, answer, template_config, tokenizer
    )

    # Find where the raw answer text begins in full_text
    answer_start = full_text.find(answer)
    if answer_start == -1:
        # Fallback: search near the end of prompt_text
        answer_start = full_text.find(answer, max(0, len(prompt_text) - 20))
    if answer_start == -1:
        logger.warning(
            "Could not locate answer in formatted text; returning empty mask"
        )
        return torch.zeros(1, dtype=torch.float)

    # Tokenize with offset mapping
    encoded = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    offsets = encoded["offset_mapping"]
    T = len(encoded["input_ids"])

    # Number of prompt tokens (for filtering answer-only positions)
    prompt_encoded = tokenizer(
        prompt_text, add_special_tokens=True, max_length=max_length, truncation=True
    )
    n_prompt = len(prompt_encoded["input_ids"])

    # Absolute character ranges
    abs_ranges = [(answer_start + s, answer_start + e) for s, e in gt_ranges]

    # Build mask: mask[t] = 1 if token at position t+1 is important
    mask = torch.zeros(max(T - 1, 1), dtype=torch.float)
    for t in range(T - 1):
        # Only consider answer tokens (position t+1 >= n_prompt)
        if t + 1 < n_prompt:
            continue
        tok_start, tok_end = offsets[t + 1]
        if tok_start == tok_end:
            continue
        for abs_start, abs_end in abs_ranges:
            if tok_start < abs_end and abs_start < tok_end:
                mask[t] = 1.0
                break

    return mask


def _load_importance_set(gt_path: str) -> dict[int, list[tuple[int, int]]]:
    """Load GT importance ranges from an Arrow dataset on disk.

    The dataset is expected to have a ``target_words`` column with JSON-encoded
    lists of ``{"start": int, "end": int, ...}`` character-range dicts (same
    format as produced by ``importance-scoring``'s GPT annotation pipeline).
    """
    ds = load_from_disk(gt_path)
    importance_set: dict[int, list[tuple[int, int]]] = {
        idx: [(pair["start"], pair["end"]) for pair in json.loads(target_words)]
        for idx, target_words in enumerate(ds["target_words"])
    }
    return importance_set


def compute_gt_masks(
    forget_dataset: Any,
    gt_path: str,
    tokenizer: Any,
    template_args: Dict[str, Any],
    max_length: int,
) -> dict[int, torch.Tensor]:
    """Compute GT binary masks for all samples in the forget dataset.

    Args:
        forget_dataset: ``QADataset`` (the forget split).
        gt_path: Path to an Arrow dataset directory with ``target_words``
            column (JSON character ranges).
        tokenizer: The tokenizer.
        template_args: Chat template config dict.
        max_length: Max sequence length.
    """
    gt_data = _load_importance_set(gt_path)
    masks: dict[int, torch.Tensor] = {}

    for idx in range(len(forget_dataset)):
        sample = forget_dataset[idx]
        index = sample["index"]
        if isinstance(index, torch.Tensor):
            index = index.item()
        gt_ranges = gt_data.get(index, [])

        question = forget_dataset.data[idx][forget_dataset.question_key]
        answer = forget_dataset.data[idx][forget_dataset.answer_key]

        mask = _compute_single_gt_mask(
            question, answer, gt_ranges, tokenizer, template_args, max_length
        )
        masks[index] = mask

    n_total = len(masks)
    n_with_gt = sum(1 for m in masks.values() if m.sum() > 0)
    logger.info(
        "GT masks: %d/%d samples have at least one important token", n_with_gt, n_total
    )
    return masks


# ============================================================================
# SEUL mask computation
# ============================================================================


def compute_seul_masks(
    model: Any,
    forget_dataset: Any,
    device: torch.device,
    alpha: float | None = None,
    k: float = 0.0,
    span_distance: int = 2,
) -> dict[int, torch.Tensor]:
    """Compute SEUL binary masks (log-probability threshold).

    Thresholding modes (SEUL prefers LOW scores → selects unusual tokens):
        - Explicit: *alpha* is set → ``selected = (log_probs < alpha)``
        - Adaptive: *alpha* is ``None`` → ``threshold = mean + k * std`` per sample
    """
    masks: dict[int, torch.Tensor] = {}
    model.eval()

    for idx in range(len(forget_dataset)):
        sample = forget_dataset[idx]
        batch = {
            key: val.unsqueeze(0).to(device)
            for key, val in sample.items()
            if isinstance(val, torch.Tensor)
        }
        with torch.no_grad():
            cache = SampleCache.from_forward(model, batch)

        log_probs = -cache.token_loss.squeeze(0)  # (T-1,)
        labels_mask = batch["labels"].squeeze(0)[1:] != IGNORE_INDEX

        if alpha is not None:
            selected = (log_probs < alpha) & labels_mask
        else:
            valid_scores = log_probs[labels_mask]
            if valid_scores.numel() == 0:
                selected = torch.zeros_like(labels_mask)
            else:
                threshold = valid_scores.mean() + k * valid_scores.std()
                selected = (log_probs < threshold) & labels_mask

        if span_distance > 0:
            selected = _aggregate_spans(selected.float(), labels_mask, span_distance)

        index = sample["index"]
        if isinstance(index, torch.Tensor):
            index = index.item()
        masks[index] = selected.float().cpu()

    _log_mask_stats(masks, "SEUL")
    return masks


# ============================================================================
# SU-LLM mask computation
# ============================================================================


def _load_ref_model(
    ref_model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Load a reference model from a HuggingFace path."""
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path, torch_dtype=dtype
    ).to(device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    return ref_model


def compute_su_llm_masks(
    model: Any,
    forget_dataset: Any,
    device: torch.device,
    ref_model_path: str,
    gamma: float | None = None,
    k: float = 0.0,
    context_size: int = 2,
) -> dict[int, torch.Tensor]:
    """Compute SU-LLM binary masks (probability divergence).

    Thresholding modes (SU prefers HIGH divergence):
        - Explicit: *gamma* is set → ``selected = (divergence > gamma)``
        - Adaptive: *gamma* is ``None`` → ``threshold = mean + k * std`` per sample
    """
    ref_model = _load_ref_model(ref_model_path, device, model.dtype)
    masks: dict[int, torch.Tensor] = {}
    model.eval()

    for idx in range(len(forget_dataset)):
        sample = forget_dataset[idx]
        batch = {
            key: val.unsqueeze(0).to(device)
            for key, val in sample.items()
            if isinstance(val, torch.Tensor)
        }
        with torch.no_grad():
            cache = SampleCache.from_forward(model, batch)
            ref_cache = SampleCache.from_forward(ref_model, batch)

        p = (-cache.token_loss).exp().squeeze(0)
        p_ref = (-ref_cache.token_loss).exp().squeeze(0)
        divergence = (p_ref - p).abs()

        labels_mask = batch["labels"].squeeze(0)[1:] != IGNORE_INDEX

        if gamma is not None:
            selected = (divergence > gamma) & labels_mask
        else:
            valid_divs = divergence[labels_mask]
            if valid_divs.numel() == 0:
                selected = torch.zeros_like(labels_mask)
            else:
                threshold = valid_divs.mean() + k * valid_divs.std()
                selected = (divergence > threshold) & labels_mask

        if context_size > 0:
            selected = _expand_context(selected.float(), labels_mask, context_size)

        index = sample["index"]
        if isinstance(index, torch.Tensor):
            index = index.item()
        masks[index] = selected.float().cpu()

    del ref_model
    torch.cuda.empty_cache()
    _log_mask_stats(masks, "SU-LLM")
    return masks


# ============================================================================
# SU-Ngram mask computation
# ============================================================================

NgramProbs = dict[tuple[int, ...], dict[int, float]]


def _build_ngram_probs(dataset: Any, ngram_n: int = 3) -> NgramProbs:
    """Build n-gram probability table from a QA dataset.

    Only answer tokens (where ``labels != IGNORE_INDEX``) are counted as
    targets; their preceding context may include template/question tokens.
    """
    counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        input_ids = sample["input_ids"].tolist()
        labels = sample["labels"].tolist()
        for i in range(len(input_ids)):
            if labels[i] == IGNORE_INDEX:
                continue
            if i < ngram_n - 1:
                continue
            context = tuple(input_ids[i - ngram_n + 1 : i])
            counts[context][input_ids[i]] += 1
    probs: NgramProbs = {}
    for ctx, ctr in counts.items():
        total = sum(ctr.values())
        probs[ctx] = {tok: c / total for tok, c in ctr.items()}
    return probs


def compute_su_ngram_masks(
    forget_dataset: Any,
    retain_dataset: Any,
    tokenizer: Any,
    device: torch.device,
    gamma: float | None = None,
    k: float = 0.0,
    ngram_n: int = 3,
    context_size: int = 2,
) -> dict[int, torch.Tensor]:
    """Compute SU-Ngram binary masks (n-gram probability divergence).

    Same adaptive/explicit thresholding as SU-LLM.
    """
    logger.info("Building n-gram tables (n=%d) ...", ngram_n)
    forget_ngram = _build_ngram_probs(forget_dataset, ngram_n)
    retain_ngram = _build_ngram_probs(retain_dataset, ngram_n)
    logger.info(
        "N-gram tables built: %d forget contexts, %d retain contexts",
        len(forget_ngram),
        len(retain_ngram),
    )

    masks: dict[int, torch.Tensor] = {}
    for idx in range(len(forget_dataset)):
        sample = forget_dataset[idx]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        T = len(labels) - 1

        labels_mask = labels[1:] != IGNORE_INDEX
        divergence = torch.zeros(T)

        tokens = input_ids.tolist()
        label_list = labels.tolist()
        for t in range(T):
            if label_list[t + 1] == IGNORE_INDEX:
                continue
            target = tokens[t + 1]
            start = max(0, t + 1 - ngram_n + 1)
            context = tuple(tokens[start : t + 1])
            p_forget = forget_ngram.get(context, {}).get(target, 0.0)
            p_retain = retain_ngram.get(context, {}).get(target, 0.0)
            divergence[t] = abs(p_forget - p_retain)

        if gamma is not None:
            selected = (divergence > gamma) & labels_mask
        else:
            valid_divs = divergence[labels_mask]
            if valid_divs.numel() == 0:
                selected = torch.zeros_like(labels_mask)
            else:
                threshold = valid_divs.mean() + k * valid_divs.std()
                selected = (divergence > threshold) & labels_mask

        if context_size > 0:
            selected = _expand_context(selected.float(), labels_mask, context_size)

        index = sample["index"]
        if isinstance(index, torch.Tensor):
            index = index.item()
        masks[index] = selected.float()

    _log_mask_stats(masks, "SU-Ngram")
    return masks


# ============================================================================
# Common helpers
# ============================================================================


def _log_mask_stats(masks: dict[int, torch.Tensor], method: str) -> None:
    if not masks:
        return
    fracs = [m.sum().item() / max(m.numel(), 1) for m in masks.values()]
    mean_frac = sum(fracs) / len(fracs)
    n_nonempty = sum(1 for f in fracs if f > 0)
    logger.info(
        "%s masks: %d samples, %d non-empty, mean %.1f%% tokens selected",
        method,
        len(masks),
        n_nonempty,
        mean_frac * 100,
    )


def _compute_masks(
    scoring_method: str,
    scoring_args: dict[str, Any],
    model: Any,
    forget_dataset: Any,
    retain_dataset: Any | None,
    tokenizer: Any,
    template_args: Dict[str, Any],
    max_length: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Dispatch mask computation to the appropriate method."""
    if scoring_method == "gt":
        return compute_gt_masks(
            forget_dataset,
            scoring_args["gt_path"],
            tokenizer,
            template_args,
            max_length,
        )
    elif scoring_method == "seul":
        return compute_seul_masks(
            model,
            forget_dataset,
            device,
            alpha=scoring_args.get("alpha"),
            k=scoring_args.get("k", 0.0),
            span_distance=scoring_args.get("span_distance", 2),
        )
    elif scoring_method == "su_llm":
        return compute_su_llm_masks(
            model,
            forget_dataset,
            device,
            ref_model_path=scoring_args["ref_model_path"],
            gamma=scoring_args.get("gamma"),
            k=scoring_args.get("k", 0.0),
            context_size=scoring_args.get("context_size", 2),
        )
    elif scoring_method == "su_ngram":
        if retain_dataset is None:
            raise ValueError("su_ngram scoring requires a retain dataset")
        return compute_su_ngram_masks(
            forget_dataset,
            retain_dataset,
            tokenizer,
            device,
            gamma=scoring_args.get("gamma"),
            k=scoring_args.get("k", 0.0),
            ngram_n=scoring_args.get("ngram_n", 3),
            context_size=scoring_args.get("context_size", 2),
        )
    else:
        raise ValueError(f"Unknown scoring method: {scoring_method}")


# ============================================================================
# HardScoredGradDiff — naive GradDiff with hard masks
# ============================================================================


class HardScoredGradDiff(UnlearnTrainer):
    """GradDiff with pre-computed binary token masks.

    Forget loss is computed only over masked (selected) tokens.
    Retain loss is standard NLL.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 1.0,
        scoring_method: str = "gt",
        scoring_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.alpha_loss = alpha
        self.gamma_loss = gamma
        self.scoring_method = scoring_method
        self.scoring_args = dict(scoring_args) if scoring_args else {}
        self._token_masks: dict[int, torch.Tensor] = {}

    def train(self, **kwargs: Any):
        self._token_masks = _compute_masks(
            scoring_method=self.scoring_method,
            scoring_args=self.scoring_args,
            model=self.model,
            forget_dataset=self.train_dataset.forget,
            retain_dataset=getattr(self.train_dataset, "retain", None),
            tokenizer=self.tokenizer,
            template_args=self.template_args,
            max_length=getattr(self.args, "max_length", 512) or 512,
            device=self.model.device,
        )
        self.data_collator.index = "index"
        return super().train(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = {
            "input_ids": inputs["forget"]["input_ids"],
            "attention_mask": inputs["forget"]["attention_mask"],
            "labels": inputs["forget"]["labels"],
        }
        forget_outputs = model(**forget_inputs)

        # Per-token loss
        logits = forget_outputs.logits
        labels = forget_inputs["labels"]
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
        token_loss = loss_fn(shifted_logits.transpose(-1, -2), shifted_labels)

        # Apply binary masks
        indices = inputs["forget"]["index"]
        B, T = token_loss.shape
        batch_masks: list[torch.Tensor] = []
        for idx in indices:
            idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
            m = self._token_masks[idx_val]
            if m.shape[0] < T:
                m = F.pad(m, (0, T - m.shape[0]))
            elif m.shape[0] > T:
                m = m[:T]
            batch_masks.append(m)
        masks = torch.stack(batch_masks).to(token_loss.device)

        # ── DEBUG: qualitative mask check ──
        # if self.state.global_step < 2:
        #     RED = "\033[91m"
        #     RESET = "\033[0m"
        #     for i, idx in enumerate(indices):
        #         idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
        #         ids = forget_inputs["input_ids"][i]
        #         labs = shifted_labels[i].cpu()
        #         m = batch_masks[i].cpu()
        #         parts = []
        #         for t in range(1,T):
        #             if labs[t] == IGNORE_INDEX:
        #                 continue
        #             tok = self.tokenizer.decode([ids[t]])
        #             if m[t-1] > 0.5:
        #                 parts.append(f"{RED}{tok}{RESET}")
        #             else:
        #                 parts.append(tok)
        #         n_sel = int(m[labs != IGNORE_INDEX].sum())
        #         n_valid = int((labs != IGNORE_INDEX).sum())
        #         print(f"MASK [idx={idx_val}] {n_sel}/{n_valid} selected")
        #         print("".join(parts))
        #         print()
        # exit(0)

        # Masked forget loss (mean over selected tokens), negated
        n_selected = masks.sum().clamp(min=1)
        forget_loss = -(token_loss * masks).sum() / n_selected

        # Retain loss
        retain_inputs = {
            "input_ids": inputs["retain"]["input_ids"],
            "attention_mask": inputs["retain"]["attention_mask"],
            "labels": inputs["retain"]["labels"],
        }
        retain_loss = model(**retain_inputs).loss

        loss = self.gamma_loss * forget_loss + self.alpha_loss * retain_loss
        return (loss, forget_outputs) if return_outputs else loss


# ============================================================================
# HardScoredSBGradDiff — SB objective with hard masks
# ============================================================================


class HardScoredSBGradDiff(SelfBalancingGradDiff):
    """SelfBalancingGradDiff with pre-computed binary masks instead of a
    learned scorer.

    Uses ``HardScorer`` (set via config) + ``NoOpScorerTrainer``.
    The saturation formula ``g_t * Pr[s_t|s_{<t}]^{beta * g_t}`` is computed
    by the existing ``reweighted_NLL`` function.
    """

    def __init__(
        self,
        scoring_method: str = "gt",
        scoring_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.scoring_method = scoring_method
        self.scoring_args = dict(scoring_args) if scoring_args else {}

    def pack_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Override to preserve ``index`` for HardScorer lookup."""
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
            "index": inputs["index"],
        }

    def train(self, **kwargs: Any):
        masks = _compute_masks(
            scoring_method=self.scoring_method,
            scoring_args=self.scoring_args,
            model=self.model,
            forget_dataset=self.train_dataset.forget,
            retain_dataset=getattr(self.train_dataset, "retain", None),
            tokenizer=self.tokenizer,
            template_args=self.template_args,
            max_length=getattr(self.args, "max_length", 512) or 512,
            device=self.model.device,
        )
        self.scorer.set_masks(masks)
        self.data_collator.index = "index"
        return super().train(**kwargs)
