from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data.Cache import SampleCacheWithHidden
from scorers.base import TokenImportanceScorer, WeightResult

logger = logging.getLogger(__name__)


class VectorScorer(TokenImportanceScorer[SampleCacheWithHidden]):
    """Token importance scorer using a single learnable vector.

    Scores tokens via: score_t = <theta, h_t> + offset
    where h_t is the hidden state at position t and theta is a
    learnable vector in R^D.

    Zero initialization with offset=1.0 gives initial scores of 1.0.
    The offset is a plain float (not learnable) and is zeroed by the
    projection step in ScorerTrainerProjected.
    """

    cache_cls = SampleCacheWithHidden

    def __init__(
        self,
        input_dimension: int,
        score_offset: float = 1.0,
        pretrained_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(input_dimension))
        self.score_offset = score_offset

        if pretrained_path is not None:
            state_dict = torch.load(
                pretrained_path, weights_only=True, map_location="cpu"
            )
            self.load_state_dict(state_dict)

    def score(self, cache: SampleCacheWithHidden, **kwargs: Any) -> WeightResult:
        mask = self._build_valid_mask(cache)

        hidden_states = cache.hidden_states  # (B, T, D)
        hidden_states = hidden_states[:, 1:, :]  # (B, T-1, D)
        hidden_states = hidden_states.detach()

        # (B, T-1, D) @ (D,) -> (B, T-1)
        scores = hidden_states.to(self.theta.dtype) @ self.theta + self.score_offset

        scores = self._apply_mask(scores, mask)

        return WeightResult(weights=scores, valid_mask=mask)
