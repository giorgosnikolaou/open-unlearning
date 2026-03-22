from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedModel

from data.Cache import TIDPOCache
from scorers.base import TokenImportanceScorer, WeightResult


class TIDPOScorer(TokenImportanceScorer[TIDPOCache]):
    """TI-DPO: gradient attribution for token importance.

    Computes per-token importance as the L1 norm of the gradient of the
    max logit at the last valid position with respect to each token
    embedding, then combines with a Gaussian positional prior.

    ``I_i = ||nabla_{e_i} max(logits[last_pos])||_1``
    ``W = lambda * I_norm + (1 - lambda) * P_prior``

    Reference: "Token-Importance Guided Direct Preference Optimization"
    (arxiv 2505.19653)
    """

    cache_cls = TIDPOCache

    def __init__(
        self,
        lam: float = 0.5,
        prior_mean: float = 2.0,
        prior_std: float = 4.0,
        zero_adjacent: bool = False,
    ) -> None:
        super().__init__()
        self.lam = lam
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.zero_adjacent = zero_adjacent

        # Per-forward transient state (set by prepare_forward, consumed by after_forward)
        self._embeddings: torch.Tensor | None = None
        self._hook_handle: Any = None


    def cache(self, model: PreTrainedModel, inputs: dict[str, Any], **kwargs):
        with torch.enable_grad():
            return super().cache(model, inputs, **kwargs)

    # ------------------------------------------------------------------
    # prepare_forward / after_forward / cleanup_forward
    # ------------------------------------------------------------------

    def prepare_forward(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Register an embedding hook that captures (but does not modify) the output."""

        def _hook(
            _module: torch.nn.Module,
            _inp: Any,
            output: torch.Tensor,
        ) -> None:
            self._embeddings = output

        self._hook_handle = model.get_input_embeddings().register_forward_hook(_hook)
        return inputs

    def after_forward(self, cache: TIDPOCache, **_kwargs: Any) -> None:
        """Extract gradient importance and store on the cache."""
        try:
            cache.importance = self._extract_importance(cache)
        finally:
            self._cleanup()

    def cleanup_forward(self) -> None:
        self._cleanup()

    # ------------------------------------------------------------------
    # score — combine gradient importance with positional prior
    # ------------------------------------------------------------------

    @staticmethod
    def compute_weights(
        importance: torch.Tensor,
        mask: torch.Tensor,
        *,
        lam: float = 0.5,
        prior_mean: float = 2.0,
        prior_std: float = 4.0,
    ) -> torch.Tensor:
        """Recompute TI-DPO weights from cached raw importance.

        Parameters
        ----------
        importance : (B, T-1) raw gradient importance tensor.
        mask : (B, T-1) bool valid-token mask.
        lam, prior_mean, prior_std :
            same semantics as ``TIDPOScorer.__init__``.

        Returns
        -------
        weights : (B, T-1) tensor, unit-mean over valid tokens, zero elsewhere.
        """
        mask_f = mask.float()
        T = mask.size(1)
        sequence_length = float(T + 1)  # mask is shifted (T-1), full seq is T

        # Clamp and sum-normalize importance per sample
        imp = importance.clamp(min=0) * mask_f
        importance_norm = imp / imp.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Gaussian positional prior (shared across batch)
        positions = torch.arange(T, device=mask.device, dtype=torch.float32)
        mu = (sequence_length - 1) / prior_mean
        sigma = max(sequence_length / prior_std, 1.0)

        normalized_positions = (positions - mu) / sigma
        normalized_positions = normalized_positions ** 2

        prior = torch.exp(-0.5 * normalized_positions) * mask_f
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Combine, invert, and normalize to unit mean over valid tokens per sample
        weights = lam * importance_norm + (1.0 - lam) * prior

        inv = torch.zeros_like(weights)
        inv[mask] = 1.0 / weights[mask].clamp(min=1e-8)
        
        valid_count = mask_f.sum(dim=-1, keepdim=True).clamp(min=1)
        weights = inv / inv.sum(dim=-1, keepdim=True).clamp(min=1e-8) * valid_count
        
        return weights.masked_fill(~mask, 0.0)

    def score(
        self,
        cache: TIDPOCache,
        **_kwargs: Any,
    ) -> WeightResult:
        if cache.importance is None:
            raise RuntimeError(
                "No importance on cache. The cache() → after_forward() "
                "flow must run before score()."
            )

        mask = self._build_valid_mask(cache)
        weights = self.compute_weights(
            cache.importance, mask,
            lam=self.lam,
            prior_mean=self.prior_mean,
            prior_std=self.prior_std,
        )
        return WeightResult(weights=weights, valid_mask=mask)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_importance(self, cache: TIDPOCache) -> torch.Tensor:
        """Compute gradient importance from the live computation graph.

        Returns (B, T-1) importance tensor.
        """
        if self._embeddings is None:
            raise RuntimeError(
                "No embeddings captured. Call prepare_forward() before "
                "from_forward()."
            )

        logits = cache.logits  # (B, T, V), graph alive
        emb = self._embeddings  # (B, T, H), on computation graph
        B, T, _ = logits.shape

        # Per-sample last valid position
        attn_mask = cache.inputs.get("attention_mask")
        if attn_mask is not None:
            last_pos = attn_mask.sum(dim=1).long() - 1  # (B,)
        else:
            last_pos = torch.full((B,), T - 1, device=logits.device, dtype=torch.long)

        # Target: sum of per-sample max logits at last valid position
        # grad(sum_b f_b(emb_b), emb) gives correct per-sample gradients
        per_sample_logits = logits[torch.arange(B, device=logits.device), last_pos]  # (B, V)
        target = per_sample_logits.max(dim=-1).values.sum()  # scalar

        # Gradient w.r.t. embeddings → L1 norm per token position
        grads = torch.autograd.grad(
            outputs=target,
            inputs=emb,
            retain_graph=True,
            create_graph=False,
        )[0]  # (B, T, H)

        importance = grads.abs().sum(dim=-1)  # (B, T)
        importance = importance[:, :-1].detach()  # shift to (B, T-1)

        if self.zero_adjacent:
            adj_pos = last_pos - 1  # (B,)
            valid = (adj_pos >= 0) & (adj_pos < importance.shape[1])
            if valid.any():
                batch_idx = torch.where(valid)[0]
                importance[batch_idx, adj_pos[batch_idx]] = 0.0

        return importance

    def _cleanup(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._embeddings = None


