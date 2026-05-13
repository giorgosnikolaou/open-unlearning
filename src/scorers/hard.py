from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from data.Cache import SampleCache
from scorers.base import TokenImportanceScorer, WeightResult


class HardScorer(TokenImportanceScorer[SampleCache]):
    """Scorer that returns pre-computed binary masks.

    Masks are set externally via ``set_masks()`` before training starts.
    Each mask is a 1-D float tensor of shape ``(T-1,)`` aligned with
    ``token_loss``, keyed by the dataset sample index.
    """

    cache_cls = SampleCache

    def __init__(self) -> None:
        super().__init__()
        self._masks: dict[int, torch.Tensor] = {}
        self._current_indices: torch.Tensor | None = None

    def set_masks(self, masks: dict[int, torch.Tensor]) -> None:
        self._masks = masks

    # -- forward hooks: strip "index" before model(**inputs), restore after --

    def prepare_forward(
        self,
        model: Any,
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._current_indices = inputs.get("index")
        return {k: v for k, v in inputs.items() if k != "index"}

    def after_forward(self, cache: SampleCache, **kwargs: Any) -> SampleCache:
        if self._current_indices is not None:
            cache.inputs["index"] = self._current_indices
        return cache

    # -- scoring: look up pre-computed binary masks --

    def score(self, cache: SampleCache, **kwargs: Any) -> WeightResult:
        indices = cache.inputs["index"]  # (B,)
        valid_mask = self._build_valid_mask(cache)  # (B, T-1)
        T = valid_mask.shape[-1]

        batch_masks: list[torch.Tensor] = []
        for idx in indices:
            m = self._masks[idx.item()]
            if m.shape[0] < T:
                m = F.pad(m, (0, T - m.shape[0]))
            elif m.shape[0] > T:
                m = m[:T]
            batch_masks.append(m)

        weights = torch.stack(batch_masks).to(cache.token_loss.device)
        weights = self._apply_mask(weights, valid_mask)
        return WeightResult(weights=weights, valid_mask=valid_mask)
