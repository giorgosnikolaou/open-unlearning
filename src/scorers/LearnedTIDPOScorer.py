from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel

from data.Cache import TIDPOCacheWithHidden
from scorers.base import TokenImportanceScorer, WeightResult
from scorers.LearnedScorer import Scorer, ScorerConfig
from scorers.TIDPOScorer import TIDPOScorer

logger = logging.getLogger(__name__)


class LearnedTIDPOScorer(TokenImportanceScorer[TIDPOCacheWithHidden]):
    """Learned MLP scorer trained with TIDPO importance as target signal.

    Composes a learnable MLP (Scorer) with a TIDPOScorer for gradient-based
    importance extraction. During caching, a single forward pass captures
    both hidden states (MLP input) and logits + embeddings (for TIDPO).
    The MLP scores are used in the main loss; TIDPO importance is used
    as the training signal for the scorer trainer.
    """

    cache_cls = TIDPOCacheWithHidden

    def __init__(
        self,
        cfg: ScorerConfig,
        pretrained_path: Path | None = None,
        lam: float = 0.5,
        prior_mean: float = 2.0,
        prior_std: float = 4.0,
        zero_adjacent: bool = False,
    ) -> None:
        super().__init__()
        self.scorer = Scorer(cfg)

        if pretrained_path is not None:
            state_dict = torch.load(
                pretrained_path, weights_only=True, map_location="cpu"
            )
            self.load_state_dict(state_dict)

        self._tidpo = TIDPOScorer(
            lam=lam,
            prior_mean=prior_mean,
            prior_std=prior_std,
            zero_adjacent=zero_adjacent,
        )

    def cache(self, model: PreTrainedModel, inputs: dict[str, Any], **kwargs):
        with torch.enable_grad():
            return super().cache(model, inputs, **kwargs)

    def prepare_forward(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._tidpo.prepare_forward(model, inputs, **kwargs)

    def after_forward(self, cache: TIDPOCacheWithHidden, **kwargs: Any) -> None:
        self._tidpo.after_forward(cache, **kwargs)

    def cleanup_forward(self) -> None:
        self._tidpo.cleanup_forward()

    def score(self, cache: TIDPOCacheWithHidden, **kwargs: Any) -> WeightResult:
        mask = self._build_valid_mask(cache)

        hidden_states = cache.hidden_states
        assert hidden_states is not None, "hidden_states not set on cache"

        hidden_states = hidden_states[:, 1:, :].detach()
        weights = self.scorer(hidden_states)
        weights = self._apply_mask(weights, mask)

        return WeightResult(weights=weights, valid_mask=mask)
