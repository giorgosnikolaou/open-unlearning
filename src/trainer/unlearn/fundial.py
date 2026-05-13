"""FUNDIAL: Focused UNDIAL.

Variant of UNDIAL (Dong et al., NAACL 2025, arxiv 2402.10052) that applies the
self-distillation forget loss only to tokens that fall inside spaCy-detected
noun chunks (``mask_type="noun"``) or named-entity spans
(``mask_type="entity"``). Token masks are precomputed once at the start of
training and looked up by sample ``index`` at each step.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from trainer.unlearn.hard_scored import _compute_single_gt_mask, _log_mask_stats
from trainer.unlearn.undial import UNDIAL
from trainer.utils import compute_fundial_loss

logger = logging.getLogger(__name__)


class FUNDIAL(UNDIAL):
    """UNDIAL restricted to noun or named-entity tokens via spaCy."""

    def __init__(
        self,
        mask_type: str = "noun",
        spacy_model: str = "en_core_web_sm",
        empty_mask_fallback: str = "zero",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if mask_type not in {"noun", "entity"}:
            raise ValueError(f"mask_type must be 'noun' or 'entity', got {mask_type!r}")
        if empty_mask_fallback not in {"zero", "uniform"}:
            raise ValueError(
                "empty_mask_fallback must be 'zero' or 'uniform', "
                f"got {empty_mask_fallback!r}"
            )
        self.mask_type = mask_type
        self.spacy_model = spacy_model
        self.empty_mask_fallback = empty_mask_fallback
        self._token_masks: dict[int, torch.Tensor] = {}
        self._nlp = None

    def _load_spacy(self) -> None:
        if self._nlp is not None:
            return
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "FUNDIAL requires spaCy. Install with `pip install spacy` and "
                f"download the model with `python -m spacy download {self.spacy_model}`."
            ) from e
        try:
            if self.mask_type == "entity":
                disable = ["tagger", "parser", "lemmatizer", "attribute_ruler"]
            else:
                disable = ["ner", "lemmatizer"]
            self._nlp = spacy.load(self.spacy_model, disable=disable)
        except OSError as e:
            raise OSError(
                f"spaCy model '{self.spacy_model}' not found. Download with: "
                f"python -m spacy download {self.spacy_model}"
            ) from e

    def _extract_spans(self, text: str) -> list[tuple[int, int]]:
        self._load_spacy()
        doc = self._nlp(text)
        if self.mask_type == "entity":
            return [(ent.start_char, ent.end_char) for ent in doc.ents]
        return [(chunk.start_char, chunk.end_char) for chunk in doc.noun_chunks]

    def _build_token_masks(self) -> None:
        forget_dataset = self.train_dataset.forget
        max_length = getattr(self.args, "max_length", 512) or 512

        masks: dict[int, torch.Tensor] = {}
        for idx in range(len(forget_dataset)):
            sample = forget_dataset[idx]
            index = sample["index"]
            if isinstance(index, torch.Tensor):
                index = index.item()

            question = forget_dataset.data[idx][forget_dataset.question_key]
            answer = forget_dataset.data[idx][forget_dataset.answer_key]
            spans = self._extract_spans(answer)

            mask = _compute_single_gt_mask(
                question, answer, spans, self.tokenizer, self.template_args, max_length
            )
            if mask.sum() == 0 and self.empty_mask_fallback == "uniform":
                labels = sample["labels"]
                shifted = labels[1:]
                mask = (shifted != -100).float()
            masks[index] = mask

        self._token_masks = masks
        _log_mask_stats(masks, f"FUNDIAL-{self.mask_type}")

    def train(self, **kwargs: Any):
        self._build_token_masks()
        self.data_collator.index = "index"
        return super().train(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = {
            "input_ids": inputs["forget"]["input_ids"],
            "attention_mask": inputs["forget"]["attention_mask"],
            "labels": inputs["forget"]["labels"],
        }

        indices = inputs["forget"]["index"]
        B, T = forget_inputs["labels"].shape
        T_shift = T - 1
        batch_masks: list[torch.Tensor] = []
        for idx in indices:
            idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
            m = self._token_masks[idx_val]
            if m.shape[0] < T_shift:
                m = F.pad(m, (0, T_shift - m.shape[0]))
            elif m.shape[0] > T_shift:
                m = m[:T_shift]
            batch_masks.append(m)
        mask = torch.stack(batch_masks)

        forget_loss, forget_outputs = compute_fundial_loss(
            model, self.ref_model, forget_inputs, self.beta, mask
        )

        retain_inputs = {
            "input_ids": inputs["retain"]["input_ids"],
            "attention_mask": inputs["retain"]["attention_mask"],
            "labels": inputs["retain"]["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
