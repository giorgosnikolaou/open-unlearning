from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from transformers import PreTrainedModel

from data.Cache import Cache
from scorers.base import TokenImportanceScorer
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.Scorer import *
from trainer.utils import reweighted_NLL

Model = PreTrainedModel


class SelfBalancing(UnlearnTrainer):
    def __init__(
        self,
        scorer: TokenImportanceScorer,
        scorer_trainer: Callable[..., Any],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.scorer = scorer

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

class SelfBalancingGradDiff(SelfBalancing):
    def __init__(
        self, *, 

        alpha: float,
        gamma: float,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        
    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache = self.prepare_caches(model, forget_inputs, retain_inputs)

        self.scorer_trainer.step(self.state.global_step, forget_cache, retain_cache)

        # scorer.step may backward through the model graph as a side effect -- clear those gradients
        # self.optimizer.zero_grad()

        forget_loss, scores, mask = reweighted_NLL(
            cache=forget_cache, 
            scorer=self.scorer, 
            beta=5.0,
        )
        forget_loss = -forget_loss

        retain_loss: torch.Tensor = retain_cache.outputs.loss # type: ignore
        loss = self.alpha * retain_loss + self.gamma * forget_loss

        return (loss, forget_cache.outputs) if return_outputs else loss



