from __future__ import annotations

import logging
import math
from abc import abstractmethod
from typing import Literal

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.optim import SGD  # type: ignore
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    LambdaLR, 
    LinearLR,
    LRScheduler, 
    SequentialLR
)
from torch.optim.optimizer import Optimizer
from transformers import PreTrainedModel

from data.Cache import Cache
from scorers.base import TokenImportanceScorer
from trainer.utils import reweighted_NLL, reweighted_NLL_correct, reweighted_softmax_NLL

Model = nn.Module

__all__ = [
    'NoOpScorerTrainer',
    'ScorerTrainer',
    'ScorerTrainerGradDiff',
    'ScorerTrainerGradDiffSpread',
    'ScorerTrainerGradDiffSoftmaxSpread',
    'ScorerTrainerProjected',
    'ScorerTrainerSBDPO',
    'ScorerTrainerSBGradDiff',
    'ScorerTrainerSBGradDiffCorrect',
    'ScorerTrainerSBNPO',
    'ScorerTrainerSBSimNPO',
    'ScorerTrainerSBWGA',
    'ScorerTrainerTIDPO',
]



class ScorerOptimConfig(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    lr: float = Field(1e-2, gt=0, le=1, description="Learning Rate (in (0, 1])")
    weight_decay: float = Field(1e-4, gt=0, le=1, description="Weight Decay (in (0, 1])")
    momentum: float = Field(0.9, gt=0, le=1, description="Momentum (in (0, 1])")

    update_every_n_steps: int = Field(10, gt=0, description="Update the scorer every \"n\" model optimizer steps (> 0)")
    grad_clip: float | None = Field(1.0, gt=0, description="Clip scorer gradient (> 0). Set to None to disable.")
    loss_reduction: Literal["mean", "sum"] = Field("mean", description="Loss reduction method")

    scheduler: Literal["none", "linear", "cosine"] = "none"
    warmup_ratio: float = Field(0.0, ge=0.0, le=1.0)
    total_steps: int | None = Field(None, gt=0, description="Total scorer *update* steps (not model steps).")


class ScorerTrainer:
    def __init__(
        self, 
        scorer: TokenImportanceScorer,
        optim_cfg: ScorerOptimConfig, 
        max_steps: int,
        accumulation_steps: int = 1,
        backwards_every_step: bool = False,
        accelerator: Accelerator | None = None,
        embed_grad: bool = False
    ):
        self.scorer = scorer
        self.optim_cfg = optim_cfg
        self.accumulation_steps = accumulation_steps
        self.effective_batches = accumulation_steps
        if backwards_every_step:
            self.effective_batches *= optim_cfg.update_every_n_steps
        self.accelerator = accelerator
        self.backwards_every_step = backwards_every_step
        self.embed_grad = embed_grad
        self._call_count = 0

        self.total_steps = (
            (max_steps + self.optim_cfg.update_every_n_steps - 1) // 
            self.optim_cfg.update_every_n_steps
        )

        self.optimizer: Optimizer = self.build_optimizer(optim_cfg)
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler: LRScheduler | None = self.build_scheduler(
            self.optimizer, 
            optim_cfg, 
            total_steps=self.total_steps
        )

        if self.accelerator is not None:
            # Scorer is a small MLP — skip DeepSpeed wrapping, keep fp32
            # for numerical stability. accelerator.backward() and
            # clip_grad_norm_() still work on unwrapped models.
            self.scorer = self.scorer.float().to(self.accelerator.device)

    def build_optimizer(self, cfg: ScorerOptimConfig) -> Optimizer:
        return SGD(
            self.scorer.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    def build_scheduler(
        self,
        optimizer: Optimizer,
        cfg: ScorerOptimConfig,
        *,
        total_steps: int | None = None,
    ) -> LRScheduler | None:
        if cfg.scheduler == "none":
            return None

        # Resolve total update steps (scheduler steps are scorer *updates*, not model steps)
        resolved_total = cfg.total_steps if cfg.total_steps is not None else total_steps
        if resolved_total is None:
            raise ValueError("`total_steps` must be provided (or `cfg.total_steps` set) when scheduler != 'none'.")
        if resolved_total <= 0:
            raise ValueError("total_steps must be > 0.")

        warmup = int(math.floor(cfg.warmup_ratio * resolved_total))
        warmup = max(0, min(warmup, resolved_total - 1))
        main_steps = resolved_total - warmup

        warmup_sched: LRScheduler | None = None
        if warmup > 0:
            warmup_sched = LambdaLR(
                optimizer,
                lr_lambda=lambda step: step / max(1, warmup),
            )

        if cfg.scheduler == "linear":
            main_sched: LRScheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=main_steps,
            )
        elif cfg.scheduler == "cosine":
            main_sched = CosineAnnealingLR(
                optimizer,
                T_max=main_steps,
                eta_min=0.0, # type: ignore # Pylance is indicating `eta_min` should be an int which is wrong.
            )
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

        if warmup_sched is not None:
            return SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[warmup],
            )

        return main_sched

    def _backwards_step(self, *caches: Cache):
        self.scorer.requires_grad_(True)

        loss = self.compute_loss(*caches)

        # TODO: Is `retain_graph` really needed?
        if getattr(self, "accelerator", None) is not None:
            self.accelerator.backward(loss, retain_graph=True) # type: ignore
        else:
            loss.backward(retain_graph=True)

        self.scorer.requires_grad_(False)

    def _update_step(self):
        if self.optim_cfg.grad_clip is not None:
            base = self.accelerator or torch.nn.utils
            base.clip_grad_norm_(self.scorer.parameters(), self.optim_cfg.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.scheduler is not None:
            self.scheduler.step()

    def step(self, global_step: int, *caches: Cache):
        self._call_count += 1
        at_accum_boundary = self._call_count % self.accumulation_steps == 0

        wants_update = self.is_update_step(global_step)

        # Backward on all micro-batches during an update window (or always if backwards_every_step)
        do_backward = self.backwards_every_step or wants_update
        # Only optimizer.step() at the last micro-batch of the accumulation window
        do_update = wants_update and at_accum_boundary

        if do_backward:
            self._backwards_step(*caches)

        if do_update:
            self._update_step()
    
    def is_update_step(self, global_step: int):
        return (global_step + 1) % self.optim_cfg.update_every_n_steps == 0

    def _log_scores(self, scores: torch.Tensor, mask: torch.Tensor, prefix: str = ""):
        valid = scores[mask]
        selected_mask = (scores > 0.5) & mask  # (B, T)
        total_counts = mask.sum(dim=1).float().clamp(min=1)  # (B,)

        means, stds = [], []
        for i in range(scores.size(0)):
            sel = scores[i][mask[i]]
            if sel.numel() > 0:
                means.append(sel.mean())
                stds.append(sel.std() if sel.numel() > 1 else torch.tensor(0.0, device=sel.device))

        mean_val = torch.stack(means).mean().item() if means else 0.0
        std_val = torch.stack(stds).mean().item() if stds else 0.0
        pct_selected = (selected_mask.sum(dim=1).float() / total_counts).mean().item() * 100

        logger.info(
            "%sscores — min=%.4f max=%.4f mean=%.4f std=%.4f selected=%5.2f%%",
            prefix,
            valid.min().item(), valid.max().item(),
            mean_val, std_val, pct_selected,
        )

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class NoOpScorerTrainer:
    """Scorer trainer for scorers with no learnable parameters (e.g. TIDPOScorer)."""

    def __init__(self, **kwargs):
        pass

    def step(self, global_step: int, *caches):
        pass

    def is_update_step(self, global_step: int):
        return False


class ScorerTrainerGradDiff_working(ScorerTrainer):
    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.3,
        lambda_l2: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2

    def _entropy_loss(
        self, 
        scores: torch.Tensor, 
        mask: torch.Tensor,
        epsilon: float = 1e-6
    ):
        scores = scores.clone()[mask]

        g = scores.clamp(epsilon, 1 - epsilon)
        entropy = -(g * g.log() + (1 - g) * (1 - g).log())

        return entropy.mean()

    def _population_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        budget: float = 0.2
    ):
        # scores: (B, T), mask: (B, T)
        hard = (scores > 0.5).float()
        selected = hard.detach() - scores.detach() + scores  # STE trick
        selected = selected * mask.float()

        # Per-sequence mean of selected tokens
        seq_counts = mask.sum(dim=1).clamp(min=1)  # (B,)
        seq_means = selected.sum(dim=1) / seq_counts  # (B,)

        return ((seq_means - budget) ** 2).mean()

    def _l2_loss(self) -> torch.Tensor:
        return sum(p.pow(2).sum() for p in self.scorer.parameters()) # type: ignore
    
    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache
    ):
        # Forget term: (1 - g) * nll_forget -> pushes g toward 1 on high loss tokens
        forget_loss, f_scores, f_mask = reweighted_NLL(
            cache=forget_cache,
            scorer=self.scorer,
            scorer_requires_grad=True,
            invert_probabilities=True,
            # normalize_token_loss=True,
        )

        # Retain counter-term: (1 - g) * nll_retain -> pushes g toward 1 on high loss tokens
        retain_loss, r_scores, r_mask = reweighted_NLL(
            cache=retain_cache,
            scorer=self.scorer,
            scorer_requires_grad=True,
            invert_probabilities=True,
            # normalize_token_loss=True,
        )

        forget_entropy = self._entropy_loss(f_scores, f_mask)
        forget_population = self._population_loss(f_scores, f_mask, budget=self.budget)
        
        retain_entropy = self._entropy_loss(r_scores, r_mask)
        retain_population = self._population_loss(r_scores, r_mask, budget=self.budget)

        l2 = self._l2_loss()

        loss = (
            retain_loss + forget_loss
            + self.lambda_entropy * (forget_entropy + retain_entropy)
            + self.lambda_population * (forget_population + retain_population)
            + self.lambda_l2 * l2
        )

        logger.info(
            "forget=%.3f f_entropy=%.3f f_budget=%.3f retain=%.3f r_entropy=%.3f r_budget=%.3f l2=%.3f total=%.3f",
            forget_loss.item(), forget_entropy.item(), forget_population.item(),
            retain_loss.item(), retain_entropy.item(), retain_population.item(),
            l2.item(), loss.item()
        )
        
        self._log_scores(f_scores, f_mask, prefix="forget ")
        self._log_scores(r_scores, r_mask, prefix="retain ")

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss


class ScorerTrainerGradDiff(ScorerTrainer):
    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.3,
        lambda_l2: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2

    def _entropy_loss(
        self, 
        scores: torch.Tensor, 
        mask: torch.Tensor,
        epsilon: float = 1e-6
    ):
        scores = scores.clone()[mask]

        g = scores.clamp(epsilon, 1 - epsilon)
        entropy = -(g * g.log() + (1 - g) * (1 - g).log())

        return entropy.mean()

    def _population_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        budget: float = 0.2
    ):
        # scores: (B, T), mask: (B, T)
        hard = (scores > 0.5).float()
        selected = hard.detach() - scores.detach() + scores  # STE trick
        selected = selected * mask.float()

        # Per-sequence mean of selected tokens
        seq_counts = mask.sum(dim=1).clamp(min=1)  # (B,)
        seq_means = selected.sum(dim=1) / seq_counts  # (B,)

        return ((seq_means - budget) ** 2).mean()

    def _l2_loss(self) -> torch.Tensor:
        return sum(p.pow(2).sum() for p in self.scorer.parameters()) # type: ignore
    
    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache
    ):
        # Forget term: (1 - g) * nll_forget -> pushes g toward 1 on high loss tokens
        forget_loss, f_scores, f_mask = reweighted_NLL(
            cache=forget_cache,
            scorer=self.scorer,
            scorer_requires_grad=True,
            invert_probabilities=True,
            # normalize_token_loss=True,
        )

        forget_entropy = self._entropy_loss(f_scores, f_mask)
        forget_population = self._population_loss(f_scores, f_mask, budget=self.budget)

        l2 = self._l2_loss()

        loss = (
            forget_loss
            + self.lambda_entropy * forget_entropy
            + self.lambda_population * forget_population
            + self.lambda_l2 * l2
        )

        logger.info(
            "forget=%.3f f_entropy=%.3f f_budget=%.3f l2=%.3f total=%.3f",
            forget_loss.item(), forget_entropy.item(), forget_population.item(),
            l2.item(), loss.item()
        )
        
        self._log_scores(f_scores, f_mask, prefix="forget ")

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss



class ScorerTrainerSBGradDiff(ScorerTrainer):
    """Matched-objective scorer trainer for SelfBalancingGradDiff.

    Forget objective: -g * sat(g*nll) * nll via reweighted_NLL(beta).
    No retain term. Same penalties as ScorerTrainerGradDiff.
    """

    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.3,
        lambda_l2: float = 0.01,
        beta: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2
        self.beta = beta

    def _entropy_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        epsilon: float = 1e-6,
    ):
        scores = scores.clone()[mask]
        g = scores.clamp(epsilon, 1 - epsilon)
        entropy = -(g * g.log() + (1 - g) * (1 - g).log())
        return entropy.mean()

    def _population_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        budget: float = 0.2,
    ):
        hard = (scores > 0.5).float()
        selected = hard.detach() - scores.detach() + scores
        selected = selected * mask.float()
        seq_counts = mask.sum(dim=1).clamp(min=1)
        seq_means = selected.sum(dim=1) / seq_counts
        return ((seq_means - budget) ** 2).mean()

    def _l2_loss(self) -> torch.Tensor:
        return sum(p.pow(2).sum() for p in self.scorer.parameters())  # type: ignore

    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache,
    ):
        forget_loss, f_scores, f_mask = reweighted_NLL(
            cache=forget_cache,
            scorer=self.scorer,
            scorer_requires_grad=True,
            beta=self.beta,
        )
        forget_loss = -forget_loss

        forget_entropy = self._entropy_loss(f_scores, f_mask)
        forget_population = self._population_loss(f_scores, f_mask, budget=self.budget)
        l2 = self._l2_loss()

        loss = (
            forget_loss
            + self.lambda_entropy * forget_entropy
            + self.lambda_population * forget_population
            + self.lambda_l2 * l2
        )

        logger.info(
            "forget=%.3f f_entropy=%.3f f_budget=%.3f l2=%.3f total=%.3f",
            forget_loss.item(), forget_entropy.item(), forget_population.item(),
            l2.item(), loss.item(),
        )

        self._log_scores(f_scores, f_mask, prefix="forget ")

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss


class ScorerTrainerSBGradDiffCorrect(ScorerTrainer):
    """Matched-objective scorer trainer for SBGradDiffCorrect.

    Forget objective: -sat(nll) * g * nll via reweighted_NLL_correct(beta).
    Otherwise identical to ScorerTrainerSBGradDiff.
    """

    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.3,
        lambda_l2: float = 0.01,
        beta: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2
        self.beta = beta

    def _entropy_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        epsilon: float = 1e-6,
    ):
        scores = scores.clone()[mask]
        g = scores.clamp(epsilon, 1 - epsilon)
        entropy = -(g * g.log() + (1 - g) * (1 - g).log())
        return entropy.mean()

    def _population_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        budget: float = 0.2,
    ):
        hard = (scores > 0.5).float()
        selected = hard.detach() - scores.detach() + scores
        selected = selected * mask.float()
        seq_counts = mask.sum(dim=1).clamp(min=1)
        seq_means = selected.sum(dim=1) / seq_counts
        return ((seq_means - budget) ** 2).mean()

    def _l2_loss(self) -> torch.Tensor:
        return sum(p.pow(2).sum() for p in self.scorer.parameters())  # type: ignore

    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache,
    ):
        forget_loss, f_scores, f_mask = reweighted_NLL_correct(
            cache=forget_cache,
            scorer=self.scorer,
            scorer_requires_grad=True,
            beta=self.beta,
        )
        forget_loss = -forget_loss

        forget_entropy = self._entropy_loss(f_scores, f_mask)
        forget_population = self._population_loss(f_scores, f_mask, budget=self.budget)
        l2 = self._l2_loss()

        loss = (
            forget_loss
            + self.lambda_entropy * forget_entropy
            + self.lambda_population * forget_population
            + self.lambda_l2 * l2
        )

        logger.info(
            "forget=%.3f f_entropy=%.3f f_budget=%.3f l2=%.3f total=%.3f",
            forget_loss.item(), forget_entropy.item(), forget_population.item(),
            l2.item(), loss.item(),
        )

        self._log_scores(f_scores, f_mask, prefix="forget ")

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss


class _SBScoredMatchedMixin:
    """Shared regularizer helpers for matched-objective scorer trainers.

    Mirrors the helpers on ScorerTrainerSBGradDiff so the new NPO/SimNPO/WGA
    matched trainers can reuse them without duplication.
    """

    def _entropy_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        epsilon: float = 1e-6,
    ):
        scores = scores.clone()[mask]
        g = scores.clamp(epsilon, 1 - epsilon)
        entropy = -(g * g.log() + (1 - g) * (1 - g).log())
        return entropy.mean()

    def _population_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        budget: float = 0.2,
    ):
        hard = (scores > 0.5).float()
        selected = hard.detach() - scores.detach() + scores
        selected = selected * mask.float()
        seq_counts = mask.sum(dim=1).clamp(min=1)
        seq_means = selected.sum(dim=1) / seq_counts
        return ((seq_means - budget) ** 2).mean()

    def _l2_loss(self) -> torch.Tensor:
        return sum(p.pow(2).sum() for p in self.scorer.parameters())  # type: ignore


class ScorerTrainerSBNPO(_SBScoredMatchedMixin, ScorerTrainer):
    """Matched-objective scorer trainer for SelfBalancingNPO.

    Forget objective: -2/β · logsigmoid(β · (lose_scored_nll - lose_ref_nll)).mean()
    where lose_scored_nll uses scorer scores with rescale (matches main NPO loss).
    Requires `forget_cache.ref_nll` set by the main trainer (per-sequence ref NLL).
    """

    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.3,
        lambda_l2: float = 0.01,
        beta: float = 0.1,
        score_scale: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2
        self.beta = beta
        self.score_scale = score_scale

    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache,
    ):
        ref_nll = getattr(forget_cache, "ref_nll", None)
        if ref_nll is None:
            raise RuntimeError(
                "ScorerTrainerSBNPO requires `forget_cache.ref_nll` to be set by "
                "the main trainer (SelfBalancingNPO must hoist lose_ref_nll before scorer step)."
            )

        with torch.enable_grad():
            scores, mask = self.scorer.score(forget_cache)

        weights = self.score_scale * scores
        seq_lengths = mask.sum(dim=-1, keepdim=True).float().clamp(min=1)
        weight_sums = (weights * mask.float()).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights * (seq_lengths / weight_sums)
        weighted = forget_cache.token_loss.detach() * weights * mask.float()
        lose_scored_nll = weighted.sum(dim=-1)

        forget_loss = -2 / self.beta * F.logsigmoid(
            self.beta * (lose_scored_nll - ref_nll)
        ).mean()

        forget_entropy = self._entropy_loss(scores, mask)
        forget_population = self._population_loss(scores, mask, budget=self.budget)
        l2 = self._l2_loss()

        loss = (
            forget_loss
            + self.lambda_entropy * forget_entropy
            + self.lambda_population * forget_population
            + self.lambda_l2 * l2
        )

        logger.info(
            "forget=%.3f f_entropy=%.3f f_budget=%.3f l2=%.3f total=%.3f",
            forget_loss.item(), forget_entropy.item(), forget_population.item(),
            l2.item(), loss.item(),
        )

        self._log_scores(scores, mask, prefix="forget ")  # type: ignore[attr-defined]

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss


class ScorerTrainerSBDPO(_SBScoredMatchedMixin, ScorerTrainer):
    """Matched-objective scorer trainer for SelfBalancingDPO.

    Forget objective: -2/β · logsigmoid(β · (win_log_ratio - lose_log_ratio)).mean()
    where lose_log_ratio = -(lose_scored_nll - lose_ref_nll) carries scorer grad
    via lose_scored_nll, while win_log_ratio is treated as a constant offset.
    Requires `forget_cache.ref_nll` and `forget_cache.win_log_ratio` set by the
    main trainer.
    """

    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.3,
        lambda_l2: float = 0.01,
        beta: float = 0.1,
        score_scale: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2
        self.beta = beta
        self.score_scale = score_scale

    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache,
    ):
        ref_nll = getattr(forget_cache, "ref_nll", None)
        win_log_ratio = getattr(forget_cache, "win_log_ratio", None)
        if ref_nll is None or win_log_ratio is None:
            raise RuntimeError(
                "ScorerTrainerSBDPO requires `forget_cache.ref_nll` and "
                "`forget_cache.win_log_ratio` to be set by the main trainer "
                "(SelfBalancingDPO must hoist both before scorer step)."
            )

        with torch.enable_grad():
            scores, mask = self.scorer.score(forget_cache)

        weights = self.score_scale * scores
        seq_lengths = mask.sum(dim=-1, keepdim=True).float().clamp(min=1)
        weight_sums = (weights * mask.float()).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights * (seq_lengths / weight_sums)
        weighted = forget_cache.token_loss.detach() * weights * mask.float()
        lose_scored_nll = weighted.sum(dim=-1)

        lose_log_ratio = -(lose_scored_nll - ref_nll)
        forget_loss = -2 / self.beta * F.logsigmoid(
            self.beta * (win_log_ratio - lose_log_ratio)
        ).mean()

        forget_entropy = self._entropy_loss(scores, mask)
        forget_population = self._population_loss(scores, mask, budget=self.budget)
        l2 = self._l2_loss()

        loss = (
            forget_loss
            + self.lambda_entropy * forget_entropy
            + self.lambda_population * forget_population
            + self.lambda_l2 * l2
        )

        logger.info(
            "forget=%.3f f_entropy=%.3f f_budget=%.3f l2=%.3f total=%.3f",
            forget_loss.item(), forget_entropy.item(), forget_population.item(),
            l2.item(), loss.item(),
        )

        self._log_scores(scores, mask, prefix="forget ")  # type: ignore[attr-defined]

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss


class ScorerTrainerSBSimNPO(_SBScoredMatchedMixin, ScorerTrainer):
    """Matched-objective scorer trainer for SelfBalancingSimNPO.

    Forget objective: -2/β · logsigmoid(β · (scored_nll/seq_len - δ)).mean()
    where scored_nll is the unrescaled per-sequence scored NLL.
    """

    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.3,
        lambda_l2: float = 0.01,
        beta: float = 4.5,
        delta: float = 0.0,
        score_scale: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2
        self.beta = beta
        self.delta = delta
        self.score_scale = score_scale

    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache,
    ):
        with torch.enable_grad():
            scores, mask = self.scorer.score(forget_cache)

        weights = self.score_scale * scores
        weighted = forget_cache.token_loss.detach() * weights * mask.float()
        scored_nll = weighted.sum(dim=-1)
        seq_lengths = mask.sum(dim=-1).float().clamp(min=1)

        forget_loss = scored_nll / seq_lengths - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

        forget_entropy = self._entropy_loss(scores, mask)
        forget_population = self._population_loss(scores, mask, budget=self.budget)
        l2 = self._l2_loss()

        loss = (
            forget_loss
            + self.lambda_entropy * forget_entropy
            + self.lambda_population * forget_population
            + self.lambda_l2 * l2
        )

        logger.info(
            "forget=%.3f f_entropy=%.3f f_budget=%.3f l2=%.3f total=%.3f",
            forget_loss.item(), forget_entropy.item(), forget_population.item(),
            l2.item(), loss.item(),
        )

        self._log_scores(scores, mask, prefix="forget ")  # type: ignore[attr-defined]

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss


class ScorerTrainerSBWGA(_SBScoredMatchedMixin, ScorerTrainer):
    """Matched-objective scorer trainer for SelfBalancingWGA.

    Forget objective: -(exp(-nll)^(β · score_scale · g) · nll)[mask].mean()
    Same formula as compute_scored_wga_loss but with grad flowing through `g`.
    """

    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.3,
        lambda_l2: float = 0.01,
        beta: float = 1.0,
        score_scale: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2
        self.beta = beta
        self.score_scale = score_scale

    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache,
    ):
        with torch.enable_grad():
            scores, mask = self.scorer.score(forget_cache)

        token_loss = forget_cache.token_loss.detach()
        weight_ce = ((-token_loss).exp().detach()) ** (
            self.beta * self.score_scale * scores
        )
        forget_loss = -(weight_ce * token_loss)[mask].mean()

        forget_entropy = self._entropy_loss(scores, mask)
        forget_population = self._population_loss(scores, mask, budget=self.budget)
        l2 = self._l2_loss()

        loss = (
            forget_loss
            + self.lambda_entropy * forget_entropy
            + self.lambda_population * forget_population
            + self.lambda_l2 * l2
        )

        logger.info(
            "forget=%.3f f_entropy=%.3f f_budget=%.3f l2=%.3f total=%.3f",
            forget_loss.item(), forget_entropy.item(), forget_population.item(),
            l2.item(), loss.item(),
        )

        self._log_scores(scores, mask, prefix="forget ")  # type: ignore[attr-defined]

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss


class ScorerTrainerTIDPO(ScorerTrainer):
    """Scorer trainer using TIDPO importance as training signal.

    Loss = (1-g) * tidpo_weight instead of (1-g) * nll.
    Pushes g toward 1 on tokens with high TIDPO weight.
    """

    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.3,
        lambda_l2: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2

    def _entropy_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        epsilon: float = 1e-6,
    ):
        g = scores.clone()[mask].clamp(epsilon, 1 - epsilon)
        entropy = -(g * g.log() + (1 - g) * (1 - g).log())
        return entropy.mean()

    def _population_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        budget: float = 0.2,
    ):
        hard = (scores > 0.5).float()
        selected = hard.detach() - scores.detach() + scores  # STE
        selected = selected * mask.float()
        seq_counts = mask.sum(dim=1).clamp(min=1)
        seq_means = selected.sum(dim=1) / seq_counts
        return ((seq_means - budget) ** 2).mean()

    def _l2_loss(self) -> torch.Tensor:
        return sum(p.pow(2).sum() for p in self.scorer.parameters())  # type: ignore

    def _tidpo_weights(self, cache: Cache, mask: torch.Tensor) -> torch.Tensor:
        from scorers.TIDPOScorer import TIDPOScorer

        importance = cache.importance  # type: ignore
        assert importance is not None, (
            "Cache has no importance. Ensure LearnedTIDPOScorer was used."
        )

        tidpo = self.scorer._tidpo  # type: ignore
        weights = TIDPOScorer.compute_weights(
            importance, mask,
            lam=tidpo.lam,
            prior_mean=tidpo.prior_mean,
            prior_std=tidpo.prior_std,
        )
        return weights.detach()

    def _reweighted_tidpo(self, cache: Cache) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (1-g) * tidpo_weight."""
        with torch.enable_grad():
            scores, mask = self.scorer.score(cache)
            inverted = torch.where(mask, 1 - scores, torch.zeros_like(scores))

        tidpo_weights = self._tidpo_weights(cache, mask)
        loss = (inverted * tidpo_weights)[mask].mean()

        return loss, scores, mask

    def compute_loss(self, forget_cache: Cache, retain_cache: Cache):
        forget_loss, f_scores, f_mask = self._reweighted_tidpo(forget_cache)
        retain_loss, r_scores, r_mask = self._reweighted_tidpo(retain_cache)

        forget_entropy = self._entropy_loss(f_scores, f_mask)
        forget_population = self._population_loss(f_scores, f_mask, budget=self.budget)

        retain_entropy = self._entropy_loss(r_scores, r_mask)
        retain_population = self._population_loss(r_scores, r_mask, budget=self.budget)

        l2 = self._l2_loss()

        loss = (
            retain_loss + forget_loss
            + self.lambda_entropy * (forget_entropy + retain_entropy)
            + self.lambda_population * (forget_population + retain_population)
            + self.lambda_l2 * l2
        )

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss






class ScorerTrainerGradDiffComp(ScorerTrainer):
    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.3,
        lambda_l2: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2

    def _entropy_loss(
        self, 
        scores: torch.Tensor, 
        mask: torch.Tensor,
        epsilon: float = 1e-6
    ):
        scores = scores.clone()[mask]

        g = scores.clamp(epsilon, 1 - epsilon)
        entropy = -(g * g.log() + (1 - g) * (1 - g).log())

        return entropy.mean()

    def _population_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        budget: float = 0.2
    ):
        # scores: (B, T), mask: (B, T)
        hard = (scores > 0.5).float()
        selected = hard.detach() - scores.detach() + scores  # STE trick
        selected = selected * mask.float()

        # Per-sequence mean of selected tokens
        seq_counts = mask.sum(dim=1).clamp(min=1)  # (B,)
        seq_means = selected.sum(dim=1) / seq_counts  # (B,)

        return ((seq_means - budget) ** 2).mean()

    def _l2_loss(self) -> torch.Tensor:
        return sum(p.pow(2).sum() for p in self.scorer.parameters()) # type: ignore
    
    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache
    ):
        # Forget term: (1 - g) * nll_forget -> pushes g toward 1 on high loss tokens
        forget_loss, f_scores, f_mask = reweighted_NLL(
            cache=forget_cache,
            scorer=self.scorer,
            scorer_requires_grad=True,
            invert_probabilities=True,
        )

        # Retain counter-term: (1 - g) * nll_retain -> pushes g toward 1 on high loss tokens
        retain_loss, r_scores, r_mask = reweighted_NLL(
            cache=retain_cache,
            scorer=self.scorer,
            scorer_requires_grad=True,
            invert_probabilities=False,
        )

        forget_entropy = self._entropy_loss(f_scores, f_mask)
        forget_population = self._population_loss(f_scores, f_mask, budget=self.budget)
        
        retain_entropy = self._entropy_loss(r_scores, r_mask)
        retain_population = self._population_loss(r_scores, r_mask, budget=self.budget)

        l2 = self._l2_loss()

        loss = (
            retain_loss + forget_loss
            + self.lambda_entropy * (forget_entropy + retain_entropy)
            + self.lambda_population * (forget_population + retain_population)
            + self.lambda_l2 * l2
        )

        # logger.info(
        #     "forget=%.3f f_entropy=%.3f f_budget=%.3f retain=%.3f r_entropy=%.3f r_budget=%.3f l2=%.3f total=%.3f",
        #     forget_loss.item(), forget_entropy.item(), forget_population.item(),
        #     retain_loss.item(), retain_entropy.item(), retain_population.item(),
        #     l2.item(), loss.item()
        # )
        
        # self._log_scores(f_scores, f_mask)
        # self._log_scores(r_scores, r_mask)

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss





class ScorerTrainerGradDiffSpread(ScorerTrainer):
    """Scorer trainer with within-sequence spread regularization.

    Uses either:
    - "histogram": soft-binned histogram entropy (maximize spread across [0,1])
    - "rbf": pairwise RBF repulsion (push scores apart)
    """

    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.3,
        lambda_l2: float = 0.01,
        entropy_mode: Literal["histogram", "rbf"] = "histogram",
        n_bins: int = 10,
        sigma: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2
        self.entropy_mode = entropy_mode
        self.n_bins = n_bins
        self.sigma = sigma

    def _histogram_entropy_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Negative histogram entropy — minimizing this maximizes spread across [0,1]."""
        centers = torch.linspace(0, 1, self.n_bins, device=scores.device)
        two_sigma_sq = 2 * self.sigma ** 2
        total = torch.tensor(0.0, device=scores.device)
        count = 0

        for i in range(scores.size(0)):
            g = scores[i][mask[i]]
            if g.numel() < 2:
                continue
            dists = -(g.unsqueeze(-1) - centers) ** 2 / two_sigma_sq   # (N, K)
            memberships = F.softmax(dists, dim=-1)                      # (N, K)
            histogram = memberships.mean(dim=0)                         # (K,)
            entropy = -(histogram * (histogram + eps).log()).sum()
            total = total - entropy  # negate: minimize loss = maximize entropy
            count += 1

        return total / max(count, 1)

    def _rbf_repulsion_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise RBF repulsion — minimizing pushes scores apart."""
        two_sigma_sq = 2 * self.sigma ** 2
        total = torch.tensor(0.0, device=scores.device)
        count = 0

        for i in range(scores.size(0)):
            g = scores[i][mask[i]]
            if g.numel() < 2:
                continue
            diffs = g.unsqueeze(0) - g.unsqueeze(1)                     # (N, N)
            K = torch.exp(-diffs ** 2 / two_sigma_sq)
            n = g.numel()
            total = total + K.triu(diagonal=1).sum() / (n * (n - 1) / 2)
            count += 1

        return total / max(count, 1)

    def _entropy_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.entropy_mode == "histogram":
            return self._histogram_entropy_loss(scores, mask)
        else:
            return self._rbf_repulsion_loss(scores, mask)

    def _population_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        budget: float = 0.2,
    ) -> torch.Tensor:
        hard = (scores > 0.5).float()
        selected = hard.detach() - scores.detach() + scores  # STE
        selected = selected * mask.float()
        seq_counts = mask.sum(dim=1).clamp(min=1)
        seq_means = selected.sum(dim=1) / seq_counts
        return ((seq_means - budget) ** 2).mean()

    def _l2_loss(self) -> torch.Tensor:
        return sum(p.pow(2).sum() for p in self.scorer.parameters())  # type: ignore

    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache,
    ) -> torch.Tensor:
        forget_loss, f_scores, f_mask = reweighted_NLL(
            cache=forget_cache,
            scorer=self.scorer,
            scorer_requires_grad=True,
            invert_probabilities=True,
            normalize_token_loss=True,
        )

        retain_loss, r_scores, r_mask = reweighted_NLL(
            cache=retain_cache,
            scorer=self.scorer,
            scorer_requires_grad=True,
            # invert_probabilities=True,
            normalize_token_loss=True,
        )

        forget_entropy = self._entropy_loss(f_scores, f_mask)
        forget_population = self._population_loss(f_scores, f_mask, budget=self.budget)

        retain_entropy = self._entropy_loss(r_scores, r_mask)
        retain_population = self._population_loss(r_scores, r_mask, budget=self.budget)

        l2 = self._l2_loss()

        loss = (
            retain_loss + forget_loss
            + self.lambda_entropy * (forget_entropy + retain_entropy)
            + self.lambda_population * (forget_population + retain_population)
            + self.lambda_l2 * l2
        )

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss


class ScorerTrainerGradDiffSoftmaxSpread(ScorerTrainer):
    """Like ScorerTrainerGradDiffSpread but uses reweighted_softmax_NLL."""

    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.3,
        lambda_l2: float = 0.01,
        entropy_mode: Literal["histogram", "rbf"] = "histogram",
        n_bins: int = 10,
        sigma: float = 0.1,
        beta: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2
        self.entropy_mode = entropy_mode
        self.n_bins = n_bins
        self.sigma = sigma
        self.beta = beta

    def _histogram_entropy_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Negative histogram entropy — minimizing this maximizes spread across [0,1]."""
        centers = torch.linspace(0, 1, self.n_bins, device=scores.device)
        two_sigma_sq = 2 * self.sigma ** 2
        total = torch.tensor(0.0, device=scores.device)
        count = 0

        for i in range(scores.size(0)):
            g = scores[i][mask[i]]
            if g.numel() < 2:
                continue
            dists = -(g.unsqueeze(-1) - centers) ** 2 / two_sigma_sq
            memberships = F.softmax(dists, dim=-1)
            histogram = memberships.mean(dim=0)
            entropy = -(histogram * (histogram + eps).log()).sum()
            total = total - entropy
            count += 1

        return total / max(count, 1)

    def _rbf_repulsion_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise RBF repulsion — minimizing pushes scores apart."""
        two_sigma_sq = 2 * self.sigma ** 2
        total = torch.tensor(0.0, device=scores.device)
        count = 0

        for i in range(scores.size(0)):
            g = scores[i][mask[i]]
            if g.numel() < 2:
                continue
            diffs = g.unsqueeze(0) - g.unsqueeze(1)
            K = torch.exp(-diffs ** 2 / two_sigma_sq)
            n = g.numel()
            total = total + K.triu(diagonal=1).sum() / (n * (n - 1) / 2)
            count += 1

        return total / max(count, 1)

    def _entropy_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.entropy_mode == "histogram":
            return self._histogram_entropy_loss(scores, mask)
        else:
            return self._rbf_repulsion_loss(scores, mask)

    def _population_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        budget: float = 0.2,
    ) -> torch.Tensor:
        hard = (scores > 0.5).float()
        selected = hard.detach() - scores.detach() + scores
        selected = selected * mask.float()
        seq_counts = mask.sum(dim=1).clamp(min=1)
        seq_means = selected.sum(dim=1) / seq_counts
        return ((seq_means - budget) ** 2).mean()

    def _l2_loss(self) -> torch.Tensor:
        return sum(p.pow(2).sum() for p in self.scorer.parameters())  # type: ignore

    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache,
    ) -> torch.Tensor:
        forget_loss, f_scores, _, f_mask = reweighted_softmax_NLL(
            cache=forget_cache,
            scorer=self.scorer,
            scorer_requires_grad=True,
            invert_probabilities=True,
            # normalize_token_loss=True,
        )

        retain_loss, r_scores, _, r_mask = reweighted_softmax_NLL(
            cache=retain_cache,
            scorer=self.scorer,
            scorer_requires_grad=True,
            # invert_probabilities=True,
            # normalize_token_loss=True,
        )

        forget_entropy = self._entropy_loss(f_scores, f_mask)
        forget_population = self._population_loss(f_scores, f_mask, budget=self.budget)

        retain_entropy = self._entropy_loss(r_scores, r_mask)
        retain_population = self._population_loss(r_scores, r_mask, budget=self.budget)

        l2 = self._l2_loss()

        loss = (
            retain_loss + forget_loss
            + self.lambda_entropy * (forget_entropy + retain_entropy)
            + self.lambda_population * (forget_population + retain_population)
            + self.lambda_l2 * l2
        )

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss


class ScorerTrainerProjected(ScorerTrainerGradDiff):
    """ScorerTrainerGradDiff with halfspace projection on the scorer vector.

    After each optimizer step, projects theta to satisfy:
        <theta, h_bar_F> <= alpha + epsilon

    where h_bar_F is the mean hidden state of forget data, accumulated
    across gradient accumulation micro-batches. Also zeros the scorer's
    score_offset on every projection step.
    """

    def __init__(
        self,
        proj_alpha: float = 0.0,
        proj_epsilon: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proj_alpha = proj_alpha
        self.proj_epsilon = proj_epsilon

        self._h_sum: torch.Tensor | None = None
        self._h_count: int = 0

    def _accumulate_forget_hidden(self, forget_cache: Cache) -> None:
        """Add per-sample mean forget hidden states to the running sum."""
        hidden = forget_cache.hidden_states[:, 1:, :].detach().float()  # type: ignore  # (B, T-1, D)
        mask = self.scorer._build_valid_mask(forget_cache)  # (B, T-1)

        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        per_sample = (hidden * mask.float().unsqueeze(-1)).sum(dim=1) / counts  # (B, D)
        batch_sum = per_sample.sum(dim=0)  # (D,)

        if self._h_sum is None:
            self._h_sum = batch_sum
        else:
            self._h_sum = self._h_sum + batch_sum
        self._h_count += hidden.size(0)

    def compute_loss(self, forget_cache: Cache, retain_cache: Cache) -> torch.Tensor:
        self._accumulate_forget_hidden(forget_cache)
        return super().compute_loss(forget_cache, retain_cache)

    def _project_theta(self) -> None:
        """Project theta onto halfspace <theta, h_bar_F> <= b and zero the offset."""
        if self._h_sum is None or self._h_count == 0:
            return

        h_bar = self._h_sum / self._h_count  # (D,)
        b = self.proj_alpha + self.proj_epsilon

        theta = self.scorer.theta  # type: ignore  # nn.Parameter
        dot = torch.dot(theta.data.float(), h_bar)

        if dot.item() > b:
            h_norm_sq = torch.dot(h_bar, h_bar).clamp(min=1e-12)
            theta.data -= ((dot - b) / h_norm_sq) * h_bar.to(theta.dtype)

            logger.info(
                "projection: dot=%.4f b=%.4f violation=%.4f ||h_bar||=%.4f",
                dot.item(), b, (dot.item() - b), h_norm_sq.sqrt().item(),
            )

        # Zero the scorer offset
        self.scorer.score_offset = 0.0  # type: ignore

        # Reset accumulators
        self._h_sum = None
        self._h_count = 0

    def _update_step(self) -> None:
        super()._update_step()
        self._project_theta()
