from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

Model = nn.Module

__all__ = [
    'Cache',
    'SequenceWiseLoss',
    'SequenceWiseNLL',
    'SequenceWiseDPO'
]

@dataclass
class Cache:
    outputs: CausalLMOutputWithPast
    token_loss: torch.Tensor
    hidden_states: torch.Tensor   # (B, T-1, H)
    logits: torch.Tensor          # (B, T-1)
    shifted_labels: torch.Tensor  # (B, T-1)

    @classmethod
    def from_forward(cls, model: Model, inputs: dict[str, Any]) -> Self:
        inputs = dict(inputs)
        inputs["output_hidden_states"] = True

        outputs: CausalLMOutputWithPast = model(**inputs)

        logits = outputs.logits
        labels = inputs["labels"]

        shifted_labels = labels[..., 1:].contiguous()
        logits = logits[..., :-1, :].contiguous()

        hidden_states = outputs.hidden_states[-1][:, :-1, :] # type: ignore

        # Recompute instead of using `outputs.loss` because it is an aggregate.
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss: torch.Tensor = loss_fn(
            logits.transpose(-1, -2), shifted_labels
        )

        return cls(
            outputs=outputs,
            token_loss=token_loss,
            hidden_states=hidden_states,
            logits=logits,
            shifted_labels=shifted_labels,
        )


class SequenceWiseLoss(BaseModel, ABC):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    _counter: int = PrivateAttr(default=0)
    ignore_label: int = -100
    log_every_n: int = 25

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    def _mask(self, labels: torch.Tensor):
        return labels != self.ignore_label

    def _uniform_scores(self, cache: "Cache"):
        mask = self._mask(cache.shifted_labels)
        den = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return mask.float() / den

    def _fixed_value_scores(self, cache: "Cache", value: float):
        mask = self._mask(cache.shifted_labels)
        return torch.full_like(cache.token_loss, value).masked_fill(~mask, 0.0)

    def _log_scores(self, scores: torch.Tensor, mask: torch.Tensor):
        self._counter += 1
        if self._counter % self.log_every_n != 0:
            return

        valid = scores[mask]
        print(f"Min : {valid.min().item():.4f}")
        print(f"Max : {valid.max().item():.4f}")
        print(f"Mean: {valid.mean().item():.4f}")
        print(f"STD : {valid.std().item():.4f}")


class SequenceWiseNLL(SequenceWiseLoss):
    invert_probabilities: bool = False
    use_softmax: bool = True
    per_sequence_loss: bool = False

    def __call__(
        self, 
        cache: Cache, 
        uniform_scores: bool = False,
        scorer: nn.Module | None = None,
        scorer_requires_grad: bool = False,
        skip_softmax: bool = False,
        invert_probabilities: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        
        mask = self._mask(cache.shifted_labels)

        if uniform_scores:
            scores = self._uniform_scores(cache)
        elif scorer is None:
            scores = self._fixed_value_scores(cache, 1.0)
        else:
            # TODO: Maybe always detach?
            scorer_input = cache.hidden_states
            if scorer_requires_grad:
                scorer_input = scorer_input.detach()

            scorer_ctx = torch.enable_grad() if scorer_requires_grad else torch.no_grad()
            with scorer_ctx:
                scores = scorer(scorer_input)

                if self.invert_probabilities or invert_probabilities:
                    scores = 1 - scores

                if self.use_softmax and not skip_softmax:
                    scores = torch.softmax(scores.masked_fill(~mask, -float("inf")), dim=1)

            scores = scores.masked_fill(~mask, 0.0)

        token_loss = cache.token_loss.detach() if scorer_requires_grad else cache.token_loss
        weighted_loss = self.weight_and_reduce(token_loss, scores, mask)

        return weighted_loss, scores, mask
    
    def weight_and_reduce(self, loss: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor | None = None):
        loss = loss * scores

        if not self.per_sequence_loss and mask is not None:
            return loss[mask].mean()
        
        # Supports DPO-like implementations
        return loss.sum(dim=-1)
    

class SequenceWiseDPO(SequenceWiseLoss):
    win_loss_fn: SequenceWiseNLL | None = None
    lose_loss_fn: SequenceWiseNLL | None = None

    beta: float = Field(ge=0, default=0.1)

    @model_validator(mode="after")
    def check_configs(self):
        if self.win_loss_fn is None and self.lose_loss_fn is None:
            raise ValueError(
                "At least one of `win_loss_fn` or `lose_loss_fn` must be provided."
            )
        return self


    def __call__(
        self, 
        win_cache: Cache | None = None, 
        win_cache_ref: Cache | None = None, 
        lose_cache: Cache | None = None, 
        lose_cache_ref: Cache | None = None,
        scorer: nn.Module | None = None,
        scorer_requires_grad: bool = False
    ):
        win_log_ratio, lose_log_ratio = 0.0, 0.0

        # Here, where we pass "alternate" samples, there are no important tokens by definition.
        # So we just use uniform scores so they are in the same scale as the lose samples.
        if (
            self.win_loss_fn is not None and
            win_cache is not None and
            win_cache_ref is not None
        ):
            win_loss, _, _ = self.win_loss_fn(
                cache=win_cache, 
                uniform_scores=scorer is not None
            )
            with torch.no_grad():
                win_ref_loss, _, _ = self.win_loss_fn(
                    cache=win_cache_ref, 
                    uniform_scores=scorer is not None
                )
            win_log_ratio = -(win_loss - win_ref_loss) # [B]

        if (
            self.lose_loss_fn is not None and
            lose_cache is not None and
            lose_cache_ref is not None
        ):
            lose_loss, scores, _ = self.lose_loss_fn(
                cache=lose_cache, 
                scorer=scorer,
                scorer_requires_grad=scorer_requires_grad
            )
            lose_ref_loss = self.lose_loss_fn.weight_and_reduce(
                loss=lose_cache_ref.token_loss, 
                scores=scores
            )
            lose_log_ratio = -(lose_loss - lose_ref_loss) # [B]

        loss = -2 / self.beta * F.logsigmoid(self.beta * (win_log_ratio - lose_log_ratio)).mean() # type: ignore
        return loss

