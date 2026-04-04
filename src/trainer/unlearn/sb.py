from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.optim import SGD
from transformers import PreTrainedModel

from data.Cache import Cache
from scorers.base import TokenImportanceScorer
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.Scorer import *
from trainer.utils import reweighted_NLL

logger = logging.getLogger(__name__)

Model = PreTrainedModel


@dataclass
class ScorerPretrainConfig:
    epochs: int = 0
    lr: float = 0.05
    freeze_after: bool = False


class SelfBalancing(UnlearnTrainer):
    def __init__(
        self,
        scorer: TokenImportanceScorer,
        scorer_trainer: Callable[..., Any],
        scorer_pretrain: ScorerPretrainConfig | None = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.scorer = scorer
        self.scorer_pretrain_cfg = scorer_pretrain or ScorerPretrainConfig()
        self._freeze_scorer = False

        self.scorer_trainer = scorer_trainer(
            scorer=self.scorer,
            accumulation_steps=getattr(self.args, "gradient_accumulation_steps", 1),
            max_steps=self.compute_max_steps(),
            accelerator=self.accelerator
        )

    def compute_max_steps(self):
        train_loader = self.get_train_dataloader()
        
        num_update_steps_per_epoch = len(train_loader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        
        return math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model
    
    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        output_dir_path = Path(output_dir or self.args.output_dir)  # type: ignore
        output_dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.scorer.state_dict(), output_dir_path / "scorer.pt")

    def pack_inputs(self, inputs: dict[str, Any]):
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
    
    def prepare_caches(self, model: Model, *inputs: dict[str, Any]) -> tuple[Cache, ...]:
        return tuple(
            self.scorer.cache(model, _inputs)
            for _inputs in inputs
        )

    def train(self, **kwargs):
        if self.scorer_pretrain_cfg.epochs > 0:
            self._pretrain_scorer()
        if self.scorer_pretrain_cfg.freeze_after:
            self.scorer.requires_grad_(False)
            self._freeze_scorer = True
        return super().train(**kwargs)

    @torch.no_grad()
    def _pretrain_scorer(self):
        cfg = self.scorer_pretrain_cfg
        logger.info(
            "Pretraining scorer for %d epoch(s) (lr=%s, freeze_after=%s)",
            cfg.epochs, cfg.lr, cfg.freeze_after,
        )

        # Freeze model — no model gradients needed
        was_training = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        pretrain_opt = SGD(
            self.scorer.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

        dataloader = self.get_train_dataloader()

        for epoch in range(cfg.epochs):
            for step, inputs in enumerate(dataloader):
                inputs = self._prepare_inputs(inputs)
                forget_inputs = self.pack_inputs(inputs["forget"])
                retain_inputs = self.pack_inputs(inputs["retain"])

                forget_cache, retain_cache = self.prepare_caches(
                    self.model, forget_inputs, retain_inputs
                )

                # Enable grad only for scorer params
                with torch.enable_grad():
                    self.scorer.requires_grad_(True)
                    loss = self.scorer_trainer.compute_loss(forget_cache, retain_cache)
                    loss.backward()
                    self.scorer.requires_grad_(False)

                if self.scorer_trainer.optim_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.scorer.parameters(),
                        self.scorer_trainer.optim_cfg.grad_clip,
                    )
                pretrain_opt.step()
                pretrain_opt.zero_grad(set_to_none=True)

            logger.info("Scorer pretrain epoch %d/%d complete", epoch + 1, cfg.epochs)

        # Restore model
        for p in self.model.parameters():
            p.requires_grad_(True)
        if was_training:
            self.model.train()

class SelfBalancingGradDiff(SelfBalancing):
    def __init__(
        self, *,

        alpha: float,
        gamma: float,
        beta: float = 5.0,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        
    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache = self.prepare_caches(model, forget_inputs, retain_inputs)

        if not self._freeze_scorer:
            self.scorer_trainer.step(self.state.global_step, forget_cache, retain_cache)

        forget_loss, scores, mask = reweighted_NLL(
            cache=forget_cache,
            scorer=self.scorer,
            beta=self.beta,
        )
        forget_loss = -forget_loss

        retain_loss: torch.Tensor = retain_cache.outputs.loss # type: ignore
        loss = self.alpha * retain_loss + self.gamma * forget_loss

        return (loss, forget_cache.outputs) if return_outputs else loss



