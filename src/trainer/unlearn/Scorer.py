from __future__ import annotations

import math
from abc import abstractmethod
from typing import Literal

import torch
from accelerate import Accelerator
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.optim import SGD # type: ignore
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    LambdaLR, 
    LinearLR,
    LRScheduler, 
    SequentialLR
)
from torch.optim.optimizer import Optimizer

from trainer.unlearn.SequenceWiseLoss import *

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
        accelerator: Accelerator | None = None
    ):
        self.cfg = cfg
        self.optim_cfg = optim_cfg
        self.effective_batches = accumulation_steps
        if backwards_every_step:
            self.effective_batches *= optim_cfg.update_every_n_steps
        self.accelerator = accelerator
        self.backwards_every_step = backwards_every_step

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
        is_update_step = (global_step + 1) % self.optim_cfg.update_every_n_steps == 0
        is_backwards_step = self.backwards_every_step or is_update_step

        if is_backwards_step:
            self._backwards_step(*caches)

        if is_update_step:
            self._update_step()

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
        loss = -forget_loss

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss
