from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self, Type

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
from transformers import TrainerCallback
from transformers.modeling_outputs import CausalLMOutputWithPast

from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import compute_kl_divergence

Model = nn.Module


# -----------------------------
# Configs
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

    def build_optimizer(self, model: nn.Module):
        return SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def build_scheduler(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
    ) -> LRScheduler | None:
        if self.scheduler == "none":
            return None

        total_steps = self.total_steps or total_steps

        if total_steps <= 0:
            raise ValueError("max_model_steps must be > 0 to derive total_steps.")

        warmup = int(math.floor(self.warmup_ratio * total_steps))
        warmup = max(0, min(warmup, total_steps - 1))

        # No need for error checking here since Pydantic makes sure warmup_ration \in [0,1]
        main_steps = total_steps - warmup

        # Warmup: linear 0 -> 1 multiplier
        warmup_sched: LRScheduler | None = None
        if warmup > 0:
            warmup_sched = LambdaLR(
                optimizer,
                lr_lambda=lambda step: step / max(1, warmup),
            )

        # We dont need to have an else statement since 
        # `self.scheduler` \in ["none", "linear", "cosine"] is ensured by pydantic
        main_sched: LRScheduler = None # type: ignore
        if self.scheduler == "linear":
            main_sched = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=main_steps,
            )
        elif self.scheduler == "cosine":
            main_sched = CosineAnnealingLR(
                optimizer,
                T_max=main_steps,
                eta_min=0.0, # type: ignore # Pylance is indicating `eta_min` should be an int which is wrong.
            )

        # Combine warmup + main
        if warmup_sched is not None:
            return SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[warmup],
            )

        return main_sched


# -----------------------------
# Models
# -----------------------------

class Scorer(nn.Sequential):
    def __init__(
        self,
        input_dimension: int,
        layers: int = 1,
        scale_factor: float = 1.0,
        use_sigmoid: bool = True,
        zero_init: bool = True,
    ):
        # Configuration validation is handled here.
        self._config = ScorerConfig(
            input_dimension=input_dimension,
            layers=layers,
            scale_factor=scale_factor,
            use_sigmoid=use_sigmoid,
            zero_init=zero_init,
        )

        modules: list[nn.Module] = []

        in_dim = out_dim = input_dimension

        for i in range(layers - 1):
            in_dim = max(1, int(input_dimension / (scale_factor ** i)))
            out_dim = max(1, int(input_dimension / (scale_factor ** (i + 1))))

            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.GELU())

        projection = nn.Linear(out_dim, 1)
        if zero_init:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        modules.append(projection)

        if use_sigmoid:
            modules.append(nn.Sigmoid())

        # Initialize Sequential with the built modules
        super().__init__(*modules)

    @property
    def config(self) -> ScorerConfig:
        return self._config

    @classmethod
    def from_config(cls, cfg: ScorerConfig) -> Self:
        return cls(
            input_dimension=cfg.input_dimension,
            layers=cfg.layers,
            scale_factor=cfg.scale_factor,
            use_sigmoid=cfg.use_sigmoid,
            zero_init=cfg.zero_init,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input).squeeze(-1)



# -----------------------------
# Data containers
# -----------------------------

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

        hidden_states = outputs.hidden_states[-1][:, :-1, :]  # type: ignore

        # Recompute instead of using `outputs.loss` because it is an aggregate
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


# -----------------------------
# Losses / objectives
# -----------------------------

class SequenceWiseLosses:
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        self.counter = 0

    def NLL(
        self,
        cache: Cache,
        scorer: nn.Module | None,
        scorer_requires_grad: bool = False,
        stop_grad_through_scores: bool = True,
        use_softmax: bool = False,
        invert_probabilities: bool = False,
    ):
        token_loss = cache.token_loss
        hidden_states = cache.hidden_states
        shifted_labels = cache.shifted_labels
        mask = shifted_labels != -100

        if scorer is None:
            scores = torch.ones_like(token_loss) 
            scores = scores.masked_fill(~mask, 0.0)
            scores /= mask.sum(dim=1, keepdim=True)
            return (
                (token_loss * scores)[mask].mean(), 
                scores, 
                mask
            )
            # return (
            #     token_loss[mask].mean(), 
            #     torch.ones_like(token_loss), 
            #     mask
            # )

        hs_for_scorer = (
            hidden_states.detach()
            if (scorer_requires_grad or stop_grad_through_scores) else
            hidden_states
        )

        scorer_ctx = torch.enable_grad() if scorer_requires_grad else torch.no_grad()
        with scorer_ctx:
            scores = scorer(hs_for_scorer)

            if invert_probabilities:
                scores = 1 - scores

            if use_softmax:
                # counts = mask.sum(dim=1, keepdim=True)  # (#valid tokens per row)
                # probs = probs.masked_fill(~mask, 0.0)
                # probs = torch.where(counts > 0, probs, torch.zeros_like(probs))

                # scores = probs * counts

                scores = torch.softmax(scores.masked_fill(~mask, -float("inf")), dim=1)

            self.counter += 1
            if self.counter % 25 == 0:
                print(f'Min : {scores[mask].min().item():.4f}')
                print(f'Max : {scores[mask].max().item():.4f}')
                print(f'Mean: {scores[mask].mean().item():.4f}')
                print(f'STD : {scores[mask].std().item():.4f}')
        
        scores = scores.masked_fill(~mask, 0.0)
        weighted_loss = (token_loss * scores)[mask].mean()
        return weighted_loss, scores, mask


# -----------------------------
# Callbacks
# -----------------------------

class ScorerUpdateCallback(TrainerCallback):
    def __init__(
        self,
        scorer: nn.Module,
        scorer_optimizer: Optimizer,
        scorer_scheduler: LRScheduler | None,
        update_every_n_steps: int,
        scorer_grad_clip: float | None,
        accelerator: Accelerator | None
    ):
        super().__init__()
        self.scorer = scorer
        
        self.scorer_optimizer = scorer_optimizer
        self.scorer_optimizer.zero_grad(set_to_none=True)

        self.scorer_scheduler = scorer_scheduler
        
        self.update_every_n_steps = update_every_n_steps
        self.scorer_grad_clip = scorer_grad_clip

        self.accelerator = accelerator

    def on_optimizer_step(self, args, state, control, **kwargs):  # type: ignore
        # HuggingFace global_step is incremented after the optimizer step.
        next_step = state.global_step + 1

        # If there are no gradients, stepping does nothing.
        has_any_grad = any(p.grad is not None for p in self.scorer.parameters())

        if not has_any_grad or next_step % self.update_every_n_steps != 0:
            return control

        if self.scorer_grad_clip is not None:
            base = self.accelerator or torch.nn.utils
            base.clip_grad_norm_(self.scorer.parameters(), self.scorer_grad_clip)

        self.scorer_optimizer.step()
        self.scorer_optimizer.zero_grad(set_to_none=True)

        if self.scorer_scheduler is not None:
            lr_before = self.scorer_optimizer.param_groups[0]["lr"]
            print(f"[Scorer] LR before step: {lr_before:.6e}")

            self.scorer_scheduler.step()

            lr_after = self.scorer_optimizer.param_groups[0]["lr"]
            print(f"[Scorer] LR after step:  {lr_after:.6e}")


        return control


# -----------------------------
# Trainer
# -----------------------------

class SelfBalancing(UnlearnTrainer):
    def __init__(
        self,

        gamma: float,
        alpha: float,
        use_softmax: bool,

        scorer_cfg: ScorerConfig,
        scorer_optim_cfg: ScorerOptimConfig,

        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.gamma = gamma
        self.alpha = alpha
        self.use_softmax = use_softmax

        self.scorer_cfg = scorer_cfg
        self.scorer_optim_cfg = scorer_optim_cfg

        self.loss = SequenceWiseLosses()

        device = getattr(self, "accelerator", self.model).device
        
        self.scorer = Scorer.from_config(scorer_cfg).to(device)
        
        self.scorer_optimizer: Optimizer = scorer_optim_cfg.build_optimizer(self.scorer)
        self.scorer_scheduler: LRScheduler | None = scorer_optim_cfg.build_scheduler(
            self.scorer_optimizer, 
            total_steps=self.compute_total_scorer_steps()
        )

        if self.accelerator is not None:
            self.scorer, self.scorer_optimizer, self.scorer_scheduler = self.accelerator.prepare(
                self.scorer, 
                self.scorer_optimizer,
                self.scorer_scheduler
            )

        self.scorer_optim_cfg = scorer_optim_cfg
        self.add_callback(
            ScorerUpdateCallback(
                self.scorer,
                self.scorer_optimizer,
                self.scorer_scheduler,
                scorer_optim_cfg.update_every_n_steps,
                scorer_optim_cfg.grad_clip,
                self.accelerator
            )
        )
        self.effective_batches = (
            # scorer_optim_cfg.update_every_n_steps * # Uncomment if we backprop on every batch
            getattr(self.args, "gradient_accumulation_steps", 1)
        )

    def compute_total_scorer_steps(self):
        train_loader = self.get_train_dataloader()
        
        num_update_steps_per_epoch = len(train_loader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        
        max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)

        self.scorer_total_update_steps = (
            (max_steps + self.scorer_optim_cfg.update_every_n_steps - 1) // 
            self.scorer_optim_cfg.update_every_n_steps
        )

        return self.scorer_total_update_steps

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        output_dir_path: Path = Path(output_dir or self.args.output_dir)  # type: ignore
        output_dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.scorer.state_dict(), output_dir_path / "scorer.pt")

    def _scorer_backward(self, model: nn.Module, forget_cache: Cache, retain_cache: Cache):
        if (self.state.global_step + 1) % self.scorer_optim_cfg.update_every_n_steps != 0:
            return

        model.requires_grad_(False)
        self.scorer.requires_grad_(True)

        ####################
        # Implementation 1 #
        ####################
        # forget_loss_s, _, _ = self.loss.NLL(
        #     forget_cache, 
        #     self.scorer, 
        #     scorer_requires_grad=True, 
        #     use_softmax=self.use_softmax
        # )
        # retain_loss_s, _, _ = self.loss.NLL(
        #     retain_cache, 
        #     self.scorer, 
        #     scorer_requires_grad=True,
        #     use_softmax=self.use_softmax
        # )
        # scorer_objective = self.alpha * retain_loss_s - self.gamma * forget_loss_s


        ####################
        # Implementation 2 #
        ####################
        forget_loss_s, scores, mask = self.loss.NLL(
            forget_cache, 
            self.scorer, 
            scorer_requires_grad=True, 
            use_softmax=self.use_softmax
        )
        # scorer_objective = - self.gamma * forget_loss_s
        # print(mask.sum(dim=-1))
        # print(scores.sum(dim=-1) / mask.sum(dim=-1))
        # exit(0)
        # mean_scores = scores.sum(dim=-1) / mask.sum(dim=-1)
        # scorer_objective = -forget_loss_s + 10 * F.mse_loss(
        #     mean_scores, 
        #     0.4 * torch.ones_like(mean_scores),
        #     reduction='sum'
        # )
        scorer_objective = -forget_loss_s

        ####################
        # Implementation 3 #
        ####################
        # forget_loss_s, _, _ = self.loss.NLL(
        #     forget_cache, 
        #     self.scorer, 
        #     scorer_requires_grad=True, 
        #     use_softmax=self.use_softmax
        # )
        # retain_loss_s_on_forget, _, _ = self.loss.NLL(
        #     forget_cache, 
        #     self.scorer, 
        #     scorer_requires_grad=True,
        #     use_softmax=self.use_softmax,
        #     invert_probabilities=True,
        # )
        # scorer_objective = retain_loss_s_on_forget - forget_loss_s

        if self.scorer_optim_cfg.loss_reduction == "mean":
            scorer_objective = scorer_objective / self.effective_batches

        if getattr(self, "accelerator", None):
            self.accelerator.backward(scorer_objective, retain_graph=True)
        else:
            scorer_objective.backward(retain_graph=True)

        model.requires_grad_(True)
        self.scorer.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        forget_inputs = inputs["forget"]
        retain_inputs = inputs["retain"]

        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }

        forget_cache = Cache.from_forward(model, forget_inputs)
        retain_cache = Cache.from_forward(model, retain_inputs)

        self._scorer_backward(model, forget_cache, retain_cache)

        forget_loss, _, _ = self.loss.NLL(
            forget_cache, 
            self.scorer, 
            scorer_requires_grad=False, 
            use_softmax=self.use_softmax
        )
        retain_loss, _, _ = self.loss.NLL(retain_cache, None, scorer_requires_grad=False)
        # retain_loss, _, _ = self.loss.NLL(
        #     retain_cache, 
        #     self.scorer, 
        #     scorer_requires_grad=False, 
        #     use_softmax=self.use_softmax,
        #     invert_probabilities=True
        # )
        loss = self.alpha * retain_loss - self.gamma * forget_loss

        return (loss, forget_cache.outputs) if return_outputs else loss
