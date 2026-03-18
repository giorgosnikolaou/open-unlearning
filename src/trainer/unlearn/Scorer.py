from __future__ import annotations

import logging
import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Self

logger = logging.getLogger(__name__)

import torch
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
import torch.nn.functional as F
from trainer.unlearn.SequenceWiseLoss import *
from transformers import PreTrainedModel

Model = nn.Module

__all__ = [
    'ScorerTrainer'
]

# -----------------------------
# Scorer & Configs
# -----------------------------

class ScorerConfig(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    input_dimension: int = Field(2048, gt=0, description="Hidden dimension size (> 0)")
    layers: int = Field(1, ge=0, description="Number of layers (>= 0)")
    scale_factor: float = Field(1.0, gt=0, description="Loss scale factor (> 0)")
    use_sigmoid: bool = True
    zero_init: bool = True


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


class Scorer(nn.Sequential):
    def __init__(self, cfg: ScorerConfig):
        self._config = cfg

        modules: list[nn.Module] = []

        in_dim = out_dim = cfg.input_dimension

        for i in range(cfg.layers - 1):
            in_dim = max(1, int(cfg.input_dimension / (cfg.scale_factor ** i)))
            out_dim = max(1, int(cfg.input_dimension / (cfg.scale_factor ** (i + 1))))

            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.GELU())

        projection = nn.Linear(out_dim, 1)
        if cfg.zero_init:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

        modules.append(projection)

        if cfg.use_sigmoid:
            modules.append(nn.Sigmoid())

        # Initialize Sequential with the built modules
        super().__init__(*modules)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input).squeeze(-1)



# -----------------------------
# Trainers
# -----------------------------

class ScorerTrainer:
    def __init__(
        self,
        cfg: ScorerConfig, 
        optim_cfg: ScorerOptimConfig, 
        max_steps: int,
        accumulation_steps: int = 1,
        backwards_every_step: bool = False,
        accelerator: Accelerator | None = None,
        embed_grad: bool = False
    ):
        self.cfg = cfg
        self.optim_cfg = optim_cfg
        self.effective_batches = accumulation_steps
        if backwards_every_step:
            self.effective_batches *= optim_cfg.update_every_n_steps
        self.accelerator = accelerator
        self.backwards_every_step = backwards_every_step
        self.embed_grad = embed_grad

        self.total_steps = (
            (max_steps + self.optim_cfg.update_every_n_steps - 1) // 
            self.optim_cfg.update_every_n_steps
        )

        self.model = Scorer(cfg)
        self.optimizer: Optimizer = self.build_optimizer(optim_cfg)
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler: LRScheduler | None = self.build_scheduler(
            self.optimizer, 
            optim_cfg, 
            total_steps=self.total_steps
        )

        if self.accelerator is not None:
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )

    def build_optimizer(self, cfg: ScorerOptimConfig) -> Optimizer:
        return SGD(
            self.model.parameters(),
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
        self.model.requires_grad_(True)

        loss = self.compute_loss(*caches)

        # TODO: Is `retain_graph` really needed?
        if getattr(self, "accelerator", None) is not None:
            self.accelerator.backward(loss, retain_graph=True) # type: ignore
        else:
            loss.backward(retain_graph=True)

        self.model.requires_grad_(False)

    def _update_step(self):
        if self.optim_cfg.grad_clip is not None:
            base = self.accelerator or torch.nn.utils
            base.clip_grad_norm_(self.model.parameters(), self.optim_cfg.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.scheduler is not None:
            self.scheduler.step()

    def step(self, global_step: int, *caches: Cache):
        is_update_step = self.is_update_step(global_step)
        is_backwards_step = self.backwards_every_step or is_update_step

        if is_backwards_step:
            self._backwards_step(*caches)

        if is_update_step:
            self._update_step()
    
    def is_update_step(self, global_step: int):
        return (global_step + 1) % self.optim_cfg.update_every_n_steps == 0

    def _log_scores(self, scores: torch.Tensor, mask: torch.Tensor):
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
            "scores — min=%.4f max=%.4f mean=%.4f std=%.4f selected=%5.2f%%",
            valid.min().item(), valid.max().item(),
            mean_val, std_val, pct_selected,
        )

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class ScorerTrainerGradDiff(ScorerTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = SequenceWiseNLL()

    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache
    ):
        forget_loss, scores, mask = self.loss(
            cache=forget_cache,
            scorer=self.model,
            scorer_requires_grad=True,
        )
        # loss = -forget_loss
        loss = forget_loss

        self._log_scores(scores, mask)

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss



class ScorerTrainerRegularizedNonCompeting(ScorerTrainerGradDiff):
    def __init__(
        self,
        rho_ent: float = 1.0,
        rho_pop: float = 1.0,
        alpha_budget: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rho_ent = rho_ent
        self.rho_pop = rho_pop
        self.alpha_budget = alpha_budget

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


    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache
    ):
        # Forget term: (1 - g) * nll_forget -> pushes g toward 1 on high loss tokens
        forget_loss, f_scores, f_mask = self.loss(
            cache=forget_cache,
            scorer=self.model,
            scorer_requires_grad=True,
            
            skip_softmax=True,
            invert_probabilities=True,
        )

        # Retain counter-term: (1 - g) * nll_retain -> pushes g toward 1 on high loss tokens
        retain_loss, r_scores, r_mask = self.loss(
            cache=retain_cache,
            scorer=self.model,
            scorer_requires_grad=True,

            skip_softmax=True,
            invert_probabilities=True,
        )

        forget_entropy = self._entropy_loss(f_scores, f_mask)
        forget_population = self._population_loss(f_scores, f_mask, budget=self.alpha_budget)
        
        retain_entropy = self._entropy_loss(r_scores, r_mask)
        retain_population = self._population_loss(r_scores, r_mask, budget=self.alpha_budget)

        loss = (
            retain_loss + forget_loss
            + self.rho_ent * (forget_entropy + retain_entropy)
            + self.rho_pop * (forget_population + retain_population)
        )

        logger.info(
            "forget=%.3f f_entropy=%.3f f_budget=%.3f retain=%.3f r_entropy=%.3f r_budget=%.3f total=%.3f",
            forget_loss.item(), forget_entropy.item(), forget_population.item(), 
            retain_loss.item(), retain_entropy.item(), retain_population.item(), 
            loss.item()
        )
        
        self._log_scores(f_scores, f_mask)
        self._log_scores(r_scores, r_mask)

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss


class ScorerTrainerRegularizedCompeting(ScorerTrainerGradDiff):
    def __init__(
        self,
        rho_ent: float = 1.0,
        rho_pop: float = 1.0,
        alpha_budget: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rho_ent = rho_ent
        self.rho_pop = rho_pop
        self.alpha_budget = alpha_budget

    def _entropy_loss(
        self, 
        scores: torch.Tensor, 
        mask: torch.Tensor,
        epsilon: float = 1e-6
    ):
        scores_ = scores[mask]

        g = scores_.clamp(epsilon, 1 - epsilon)
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


    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache
    ):
        # Forget term: (1 - g) * nll_forget -> pushes g toward 1 on high loss tokens
        forget_loss, f_scores, f_mask = self.loss(
            cache=forget_cache,
            scorer=self.model,
            scorer_requires_grad=True,
            
            skip_softmax=True,
            invert_probabilities=True,

            normalize_token_loss=True,
        )

        # Retain counter-term: g * nll_retain -> pushes g toward 0 on high loss tokens
        retain_loss, r_scores, r_mask = self.loss(
            cache=retain_cache,
            scorer=self.model,
            scorer_requires_grad=True,

            skip_softmax=True,

            normalize_token_loss=True,
        )

        forget_entropy = self._entropy_loss(f_scores, f_mask)
        forget_population = self._population_loss(f_scores, f_mask, budget=self.alpha_budget)
        
        retain_entropy = self._entropy_loss(r_scores, r_mask)
        retain_population = self._population_loss(r_scores, r_mask, budget=1-self.alpha_budget)


        loss = (
            retain_loss + forget_loss
            + self.rho_ent * (forget_entropy + retain_entropy)
            + self.rho_pop * (forget_population + retain_population)
        )

        logger.info(
            "forget=%.3f f_entropy=%.3f f_budget=%.3f retain=%.3f r_entropy=%.3f r_budget=%.3f total=%.3f",
            forget_loss.item(), forget_entropy.item(), forget_population.item(), 
            retain_loss.item(), retain_entropy.item(), retain_population.item(), 
            loss.item()
        )
        
        self._log_scores(f_scores, f_mask)
        self._log_scores(r_scores, r_mask)

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss


class ScorerTrainerRegularizedGrad(ScorerTrainerGradDiff):
    def __init__(
        self,
        rho_ent: float = 1.0,
        rho_pop: float = 1.0,
        alpha_budget: float = 0.3,
        embed_grad: bool = True,
        # TIDPO parameters
        lam: float = 0.5,
        prior_mean: float = 2.0,
        prior_std: float = 4.0,
        prior_scope: str = "sequence",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rho_ent = rho_ent
        self.rho_pop = rho_pop
        self.alpha_budget = alpha_budget
        self.embed_grad = embed_grad
        self.lam = lam
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.prior_scope = prior_scope

    @staticmethod
    def _compute_tidpo_weights(
        importance: torch.Tensor,
        mask: torch.Tensor,
        seq_len: float,
        *,
        lam: float = 0.5,
        prior_mean: float = 2.0,
        prior_std: float = 4.0,
        prior_scope: str = "sequence",
    ) -> torch.Tensor:
        """Compute TI-DPO weights with inverse_scores=True.

        Replicates TIDPOScorer.compute_weights from importance-scoring.
        Returns (T-1,) tensor, unit-mean over valid tokens, zero elsewhere.
        """
        mask_f = mask.float()

        imp = importance.clamp(min=0) * mask_f
        importance_norm = imp / imp.sum().clamp(min=1e-8)

        T_shifted = mask.numel()
        pos = torch.arange(T_shifted, device=mask.device, dtype=torch.float32)

        if prior_scope == "answer":
            valid_idx = torch.where(mask)[0]
            if valid_idx.numel() > 0:
                first_valid = valid_idx[0].float()
                answer_len = float(valid_idx.numel())
            else:
                first_valid = torch.tensor(0.0, device=mask.device)
                answer_len = 1.0
            center = first_valid + (answer_len - 1) / prior_mean
            sigma = max(answer_len / prior_std, 1.0)
        else:
            center = (seq_len - 1) / prior_mean
            sigma = max(seq_len / prior_std, 1.0)

        prior = torch.exp(-0.5 * ((pos - center) / sigma) ** 2) * mask_f
        prior = prior / prior.sum().clamp(min=1e-8)

        valid_count = mask_f.sum().clamp(min=1)
        weights = lam * importance_norm + (1.0 - lam) * prior

        # Inverse scores
        inv = torch.zeros_like(weights)
        inv[mask] = 1.0 / weights[mask].clamp(min=1e-8)
        weights = inv

        weights = weights / weights.sum().clamp(min=1e-8) * valid_count
        return weights.masked_fill(~mask, 0.0)

    def step(self, global_step: int, *caches: Cache):
        forget_cache, retain_cache = caches

        if self.is_update_step(global_step):
            for cache in (forget_cache, retain_cache):
                embed: torch.Tensor = cache.embed_input  # type: ignore  # (B, T, H)
                original_logits = cache.outputs.logits  # (B, T, V) unshifted
                B = original_logits.shape[0]

                # Per-sample last valid position from attention_mask
                attn_mask = cache.attention_mask
                if attn_mask is not None:
                    last_positions = (attn_mask.sum(dim=1) - 1).long()  # (B,)
                else:
                    last_positions = torch.full(
                        (B,), original_logits.shape[1] - 1,
                        dtype=torch.long, device=original_logits.device,
                    )

                # TIDPO: gradient of max logit at last valid position only
                last_logits = original_logits[torch.arange(B, device=original_logits.device), last_positions]  # (B, V)
                target = last_logits.max(dim=-1).values.sum()

                (grads,) = torch.autograd.grad(
                    outputs=target,
                    inputs=embed,
                    retain_graph=True,
                    create_graph=False,
                )
                cache.gradients = grads[:, :-1, :].detach().clone()  # (B, T-1, H)

        ScorerTrainer.step(self, global_step, *caches)

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

    def _seq_lens(self, cache: Cache) -> list[float]:
        """Per-sample sequence lengths from attention_mask, fallback to T."""
        B = cache.outputs.logits.shape[0]
        T = cache.outputs.logits.shape[1]
        if cache.attention_mask is not None:
            return [float(cache.attention_mask[b].sum().item()) for b in range(B)]
        return [float(T)] * B

    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache
    ):
        forget_mask = self.loss._mask(forget_cache.shifted_labels)
        B = forget_cache.gradients.shape[0]  # type: ignore
        forget_seq_lens = self._seq_lens(forget_cache)
        forget_tidpo = torch.zeros_like(forget_cache.token_loss)
        for b in range(B):
            importance_b = forget_cache.gradients[b].abs().sum(dim=-1)  # type: ignore
            forget_tidpo[b] = self._compute_tidpo_weights(
                importance_b, forget_mask[b], forget_seq_lens[b],
                lam=self.lam, prior_mean=self.prior_mean,
                prior_std=self.prior_std, prior_scope=self.prior_scope,
            )

        with torch.enable_grad():
            forget_scores = self.model(forget_cache.hidden_states.detach()).masked_fill(~forget_mask, 0.0)
        forget_loss = (forget_scores * forget_tidpo)[forget_mask].mean()

        retain_mask = self.loss._mask(retain_cache.shifted_labels)
        retain_seq_lens = self._seq_lens(retain_cache)
        retain_tidpo = torch.zeros_like(retain_cache.token_loss)
        for b in range(B):
            importance_b = retain_cache.gradients[b].abs().sum(dim=-1)  # type: ignore
            retain_tidpo[b] = self._compute_tidpo_weights(
                importance_b, retain_mask[b], retain_seq_lens[b],
                lam=self.lam, prior_mean=self.prior_mean,
                prior_std=self.prior_std, prior_scope=self.prior_scope,
            )

        with torch.enable_grad():
            retain_scores = self.model(retain_cache.hidden_states.detach()).masked_fill(~retain_mask, 0.0)
        retain_loss = (retain_scores * retain_tidpo)[retain_mask].mean()

        forget_entropy = self._entropy_loss(forget_scores, forget_mask)
        forget_population = self._population_loss(forget_scores, forget_mask, budget=self.alpha_budget)

        retain_entropy = self._entropy_loss(retain_scores, retain_mask)
        retain_population = self._population_loss(retain_scores, retain_mask, budget=self.alpha_budget)

        loss = (
            retain_loss + forget_loss
            + self.rho_ent * (forget_entropy + retain_entropy)
            + self.rho_pop * (forget_population + retain_population)
        )

        logger.info(
            "forget=%.3f f_entropy=%.3f f_budget=%.3f retain=%.3f r_entropy=%.3f r_budget=%.3f total=%.3f",
            forget_loss.item(), forget_entropy.item(), forget_population.item(),
            retain_loss.item(), retain_entropy.item(), retain_population.item(),
            loss.item()
        )

        self._log_scores(forget_scores, forget_mask)
        self._log_scores(retain_scores, retain_mask)

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss
    


class ScorerTrainerRegularized(ScorerTrainerGradDiff):
    def __init__(
        self,
        rho_ent: float = 1.0,
        rho_pop: float = 1.0,
        alpha_budget: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rho_ent = rho_ent
        self.rho_pop = rho_pop
        self.alpha_budget = alpha_budget

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


    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache
    ):
        # Forget term: (1 - g) * nll_forget -> pushes g toward 1 on high loss tokens
        forget_loss, f_scores, f_mask = self.loss(
            cache=forget_cache,
            scorer=self.model,
            scorer_requires_grad=True,
            
            skip_softmax=True,
            invert_probabilities=True,
        )

        # Retain counter-term: (1 - g) * nll_retain -> pushes g toward 1 on high loss tokens
        retain_loss, r_scores, r_mask = self.loss(
            cache=retain_cache,
            scorer=self.model,
            scorer_requires_grad=True,

            skip_softmax=True,
            invert_probabilities=True,
        )

        forget_entropy = self._entropy_loss(f_scores, f_mask)
        forget_population = self._population_loss(f_scores, f_mask, budget=self.alpha_budget)
        
        retain_entropy = self._entropy_loss(r_scores, r_mask)
        retain_population = self._population_loss(r_scores, r_mask, budget=self.alpha_budget)

        loss = (
            retain_loss + forget_loss
            + self.rho_ent * (forget_entropy + retain_entropy)
            + self.rho_pop * (forget_population + retain_population)
        )

        logger.info(
            "forget=%.3f f_entropy=%.3f f_budget=%.3f retain=%.3f r_entropy=%.3f r_budget=%.3f total=%.3f",
            forget_loss.item(), forget_entropy.item(), forget_population.item(), 
            retain_loss.item(), retain_entropy.item(), retain_population.item(), 
            loss.item()
        )
        
        self._log_scores(f_scores, f_mask)
        self._log_scores(r_scores, r_mask)

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss



