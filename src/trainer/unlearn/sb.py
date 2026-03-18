from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.Scorer import *
from trainer.unlearn.SequenceWiseLoss import *

Model = nn.Module

__all__ = [
    'SelfBalancingGradDiff',
    'SelfBalancingNPO',
    'SelfBalancingDPO',
]

class SelfBalancing(UnlearnTrainer):
    def __init__(
        self,
        scorer_trainer: Callable[..., Any],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.scorer: ScorerTrainer = scorer_trainer(
            accumulation_steps=getattr(self.args, "gradient_accumulation_steps", 1),
            max_steps=self.compute_max_steps(),
            accelerator=self.accelerator
        )

        if not isinstance(self.scorer, ScorerTrainer):  # subclasses OK
            raise TypeError(f"Expected ScorerTrainer (or subclass), got {type(self.scorer)}")

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
        output_dir_path: Path = Path(output_dir or self.args.output_dir) # type: ignore
        output_dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.scorer.model.state_dict(), output_dir_path / "scorer.pt")

    def pack_inputs(self, inputs: dict[str, Any]):
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
    
    # def prepare_caches(self, model: Model, *inputs: dict[str, Any]) -> tuple[Cache, ...]:
    #     return tuple(
    #         Cache.from_forward(model, _inputs)
    #         for _inputs in inputs
    #     )

    def prepare_caches(self, model: Model, *inputs: dict[str, Any], embed_grad: bool = False) -> tuple[Cache, ...]:
        return tuple(
            Cache.from_forward(model, _inputs, embed_grad=embed_grad)
            for _inputs in inputs
        )

class SelfBalancingGradDiff(SelfBalancing):
    def __init__(
        self, *, 

        alpha: float,
        gamma: float,

        forget_loss: SequenceWiseNLL, 
        retain_loss: SequenceWiseNLL,

        **kwargs
    ):
        super().__init__(**kwargs)

        self.gamma = gamma
        self.alpha = alpha
        
        self.forget_loss = forget_loss
        self.retain_loss = retain_loss
        
    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache = self.prepare_caches(
            model, 
            forget_inputs, 
            retain_inputs, 
            embed_grad=self.scorer.is_update_step(self.state.global_step) or self.scorer.embed_grad
        )
        self.scorer.step(self.state.global_step, forget_cache, retain_cache)

        # forget_loss, scores, mask = self.forget_loss(cache=forget_cache, scorer=self.scorer.model)
        # forget_loss, scores, mask = self.forget_loss(
        #     cache=forget_cache, 
        #     scorer=self.scorer.model, 
        #     skip_softmax=True,
        #     scorer_input=forget_cache.gradients
        # )
        forget_loss, scores, mask = self.forget_loss(
            cache=forget_cache, 
            scorer=self.scorer.model, 
            skip_softmax=True,
            scorer_input=forget_cache.gradients,
            # beta=1.0,
            beta=5.0,
        )
        forget_loss = -forget_loss
        # retain_loss, _, _ = self.retain_loss(
        #     cache=retain_cache, 
        #     scorer=None,
        #     # uniform_scores=True
        #     uniform_scores=False
        # )
        retain_loss: torch.Tensor = retain_cache.outputs.loss # type: ignore
        loss = self.alpha * retain_loss + self.gamma * forget_loss

        return (loss, forget_cache.outputs) if return_outputs else loss


class SelfBalancingNPO(SelfBalancingGradDiff):
    def __init__(self, *, forget_loss: SequenceWiseDPO, **kwargs):
        super().__init__(forget_loss=forget_loss, **kwargs) # type: ignore
        self.forget_loss = forget_loss # Override for correct typing
        self.ref_model = self._prepare_ref_model(self.model)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache = self.prepare_caches(model, forget_inputs, retain_inputs)
        
        with torch.no_grad():
            forget_cache_ref = Cache.from_forward(self.ref_model, forget_inputs)
        
        self.scorer.step(self.state.global_step, forget_cache, retain_cache)

        forget_loss = self.forget_loss(
            win_cache=None,
            win_cache_ref=None,
            lose_cache=forget_cache,
            lose_cache_ref=forget_cache_ref,
            scorer=self.scorer.model,
            scorer_requires_grad=False,
        )
        retain_loss, _, _ = self.retain_loss(cache=retain_cache, uniform_scores=False)
        # retain_loss, _, _ = self.retain_loss(cache=retain_cache, uniform_scores=True)
        loss = self.alpha * retain_loss + self.gamma * forget_loss

        del retain_cache

        # The existing implementation of NPO returned the tuple `(alternate_cache.outputs, forget_cache.outputs)`.
        # The reason is not evident and, therefore, only return the outputs of the forget samples.
        return (loss, forget_cache.outputs) if return_outputs else loss
    

class SelfBalancingDPO(SelfBalancingGradDiff):
    def __init__(self, *, forget_loss: SequenceWiseDPO, **kwargs):
        super().__init__(forget_loss=forget_loss, **kwargs) # type: ignore
        self.forget_loss = forget_loss # Override for correct typing
        self.ref_model = self._prepare_ref_model(self.model)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"]["original"])
        alternate_inputs = self.pack_inputs(inputs["forget"]["alternate"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache, alternate_cache = self.prepare_caches(
            model, 
            forget_inputs, 
            retain_inputs, 
            alternate_inputs
        )

        with torch.no_grad():
            forget_cache_ref, alternate_cache_ref = self.prepare_caches(
                self.ref_model, 
                forget_inputs, 
                alternate_inputs
            )

        self.scorer.step(self.state.global_step, forget_cache, retain_cache)

        forget_loss = self.forget_loss(
            win_cache=alternate_cache,
            win_cache_ref=alternate_cache_ref,
            lose_cache=forget_cache,
            lose_cache_ref=forget_cache_ref,
            scorer=self.scorer.model,
            scorer_requires_grad=False,
        )
        retain_loss, _, _ = self.retain_loss(cache=retain_cache, uniform_scores=False)
        # retain_loss, _, _ = self.retain_loss(cache=retain_cache, uniform_scores=True)
        loss = self.alpha * retain_loss + self.gamma * forget_loss

        # The existing implementation of DPO returned the tuple `(alternate_cache.outputs, forget_cache.outputs)`.
        # The reason is not evident and, therefore, only return the outputs of the forget samples.
        return (loss, forget_cache.outputs) if return_outputs else loss


