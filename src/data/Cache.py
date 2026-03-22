from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

__all__ = [
    'Cache',
    'SampleCache',
    'SampleCacheWithHidden',
    'SampleCacheWithLogits',
    'TIDPOCache',
    'TIDPOCacheWithHidden',
]

def _run_forward(
    model: PreTrainedModel,
    inputs: dict[str, Any],
    *,
    output_hidden_states: bool = False,
    ignore_index: int = -100,
) -> tuple[CausalLMOutputWithPast, torch.Tensor]:
    inputs = dict(inputs)
    if output_hidden_states:
        inputs["output_hidden_states"] = True

    outputs: CausalLMOutputWithPast = model(**inputs)

    logits = outputs.logits # (B, T, |V|)
    labels = inputs["labels"] # (B, T)

    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()

    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
    token_loss: torch.Tensor = loss_fn(
        shifted_logits.transpose(-1, -2), shifted_labels
    )

    return outputs, token_loss

@dataclass
class SampleCache:
    inputs: dict[str, Any]
    outputs: CausalLMOutputWithPast
    token_loss: torch.Tensor # (B, T-1); the t'th token correspond to the suprisal of seeing the (t+1)'th token

    @classmethod
    def from_forward(cls, model: PreTrainedModel, inputs: dict[str, Any]) -> Self:
        outputs, token_loss = _run_forward(model, inputs)
        return cls(inputs=inputs, outputs=outputs, token_loss=token_loss)


@dataclass
class SampleCacheWithLogits(SampleCache):
    logits: torch.Tensor # (B, T, V) on model device, graph alive

    @classmethod
    def from_forward(cls, model: PreTrainedModel, inputs: dict[str, Any]) -> Self:
        outputs, token_loss = _run_forward(model, inputs)
        logits = outputs.logits # (B, T, V)
        return cls(inputs=inputs, outputs=outputs, token_loss=token_loss, logits=logits)


@dataclass
class SampleCacheWithHidden(SampleCache):
    hidden_states: torch.Tensor # (B, T, V) on model device, graph alive

    @classmethod
    def from_forward(cls, model: PreTrainedModel, inputs: dict[str, Any], *, layer: int = -1) -> Self:
        outputs, token_loss = _run_forward(model, inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer] # type: ignore
        return cls(inputs=inputs, outputs=outputs, token_loss=token_loss, hidden_states=hidden_states)
    
@dataclass
class TIDPOCache(SampleCacheWithLogits):
    importance: torch.Tensor | None = None

@dataclass
class TIDPOCacheWithHidden(TIDPOCache):
    hidden_states: torch.Tensor | None = None

    @classmethod
    def from_forward(
        cls,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        *,
        layer: int = -1,
    ) -> Self:
        outputs, token_loss = _run_forward(model, inputs, output_hidden_states=True)
        return cls(
            inputs=inputs,
            outputs=outputs,
            token_loss=token_loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states[layer],  # type: ignore
        )

Cache = SampleCache | SampleCacheWithLogits | SampleCacheWithHidden | TIDPOCache | TIDPOCacheWithHidden