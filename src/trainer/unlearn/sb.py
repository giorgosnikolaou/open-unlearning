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
from transformers import PreTrainedModel, TrainerCallback

from data.Cache import Cache
from scorers.base import TokenImportanceScorer
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.Scorer import *
from trainer.utils import (
    compute_batch_nll,
    compute_kl_divergence,
    compute_scored_batch_nll,
    compute_scored_fundial_loss,
    compute_inverted_scored_wga_loss,
    compute_inverted_wga_loss,
    compute_scored_wga_loss,
    jensun_retain_loss,
    reweighted_NLL,
    reweighted_NLL_correct,
    scored_jensun_multitok_loss,
)

import torch.nn.functional as F

logger = logging.getLogger(__name__)

Model = PreTrainedModel


@dataclass
class ScorerPretrainConfig:
    epochs: int = 0
    lr: float = 0.05
    freeze_after: bool = False
    scorer_save_every_steps: int | None = None


class ScorerCheckpointCallback(TrainerCallback):
    def __init__(self, scorer: nn.Module, save_every_steps: int):
        self.scorer = scorer
        self.save_every_steps = save_every_steps

    def _save(self, output_dir: str, tag: str):
        out = Path(output_dir) / "scorer_checkpoints"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.scorer.state_dict(), out / f"step_{tag}.pt")

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if state.global_step > 0 and state.global_step % self.save_every_steps == 0:
            self._save(args.output_dir, str(state.global_step))

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._save(args.output_dir, "final")


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

        save_every = self.scorer_pretrain_cfg.scorer_save_every_steps
        if save_every is not None and self.accelerator.is_main_process:
            out = Path(self.args.output_dir) / "scorer_checkpoints"
            out.mkdir(parents=True, exist_ok=True)
            torch.save(self.scorer.state_dict(), out / "step_0.pt")
            self.add_callback(ScorerCheckpointCallback(self.scorer, save_every))

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


class SBGradDiffMatchedNoRetain(SelfBalancingGradDiff):
    """SBGradDiffMatched without the retain term in the main loss.

    Retain dataset is still loaded (so the scorer trainer keeps its caches),
    but the main objective drops alpha * retain_loss. Used as a baseline to
    isolate the scorer's effect on the forget objective.
    """

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

        loss = self.gamma * forget_loss

        return (loss, forget_cache.outputs) if return_outputs else loss


class ScoredGradDiff(SelfBalancing):
    """GradDiff with learned scorer reweighting — no saturation.

    forget_loss = -(token_loss * scores)[mask].mean()
    """

    def __init__(self, *, alpha: float, gamma: float, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])
        forget_cache, retain_cache = self.prepare_caches(model, forget_inputs, retain_inputs)

        if not self._freeze_scorer:
            self.scorer_trainer.step(self.state.global_step, forget_cache, retain_cache)

        with torch.no_grad():
            scores, mask = self.scorer.score(forget_cache)

        forget_loss = -(forget_cache.token_loss * scores)[mask].mean()
        retain_loss: torch.Tensor = retain_cache.outputs.loss  # type: ignore
        loss = self.alpha * retain_loss + self.gamma * forget_loss

        return (loss, forget_cache.outputs) if return_outputs else loss


class _SBGradDiffMatchedJointBase(SelfBalancing):
    """Joint variant of SBGradDiffMatched: scorer parameters are optimized via the
    main loss in a single backward pass (no separate scorer optimizer / objective).

    Regularizers (entropy, population, l2) are added to the main loss so they still
    constrain the scorer. The configured `scorer_trainer` should be a NoOp (it is
    not used for stepping); we still construct it to satisfy `SelfBalancing.__init__`.
    Scorer parameters are added to the main HF optimizer as a separate parameter
    group with `scorer_lr`.

    Subclasses choose which weighted-NLL math is used: vanilla (g · sat(g·nll) · nll)
    or "correct" (sat(nll) · g · nll).
    """

    use_correct: bool = False

    def __init__(
        self, *,
        alpha: float,
        gamma: float,
        beta: float = 5.0,
        scorer_lr: float = 0.05,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.2,
        lambda_l2: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.scorer_lr = scorer_lr
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2

        # NoOpScorerTrainer doesn't move scorer to device; do it here so the
        # joint compute_loss can pass cuda hidden_states through it.
        if self.accelerator is not None:
            self.scorer = self.scorer.float().to(self.accelerator.device)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        decay_parameters = self.get_decay_parameter_names(opt_model)

        named = list(opt_model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in named if n in decay_parameters and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in named if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
            },
            {
                "params": [p for p in self.scorer.parameters() if p.requires_grad],
                "lr": self.scorer_lr,
                "weight_decay": 0.0,  # l2 is folded into the loss via lambda_l2
            },
        ]

        optimizer_cls, optimizer_kwargs = type(self).get_optimizer_cls_and_kwargs(
            self.args, opt_model
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _entropy_reg(self, scores: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
        g = scores[mask].clamp(eps, 1 - eps)
        return -(g * g.log() + (1 - g) * (1 - g).log()).mean()

    def _population_reg(self, scores: torch.Tensor, mask: torch.Tensor):
        # Straight-through hard-selection estimator
        hard = (scores > 0.5).float()
        selected = hard.detach() - scores.detach() + scores
        selected = selected * mask.float()
        seq_counts = mask.sum(dim=1).clamp(min=1)
        seq_means = selected.sum(dim=1) / seq_counts
        return ((seq_means - self.budget) ** 2).mean()

    def _l2_reg(self):
        return sum(p.pow(2).sum() for p in self.scorer.parameters())

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache = self.prepare_caches(model, forget_inputs, retain_inputs)

        # Joint: do NOT call scorer_trainer.step (it is a NoOp anyway).
        # Compute the reweighted NLL with both model and scorer gradients flowing.
        scores, mask = self.scorer.score(forget_cache)  # scorer grad enabled by default ctx
        token_loss = forget_cache.token_loss            # model grad preserved (no detach)

        if self.use_correct:
            saturation = (-token_loss).exp().detach() ** self.beta
            weighted = (token_loss * saturation * scores)[mask].mean()
        else:
            weighted = token_loss * scores
            saturation = (-weighted).exp().detach() ** self.beta
            weighted = (weighted * saturation)[mask].mean()
        forget_loss = -weighted

        retain_loss: torch.Tensor = retain_cache.outputs.loss  # type: ignore

        entropy_reg = self._entropy_reg(scores, mask)
        population_reg = self._population_reg(scores, mask)
        l2_reg = self._l2_reg()

        loss = (
            self.gamma * forget_loss
            + self.alpha * retain_loss
            + self.lambda_entropy * entropy_reg
            + self.lambda_population * population_reg
            + self.lambda_l2 * l2_reg
        )

        return (loss, forget_cache.outputs) if return_outputs else loss


class SBGradDiffMatchedJoint(_SBGradDiffMatchedJointBase):
    use_correct = False


class SBGradDiffCorrectMatchedJoint(_SBGradDiffMatchedJointBase):
    use_correct = True


class SBGradDiffCorrect(SelfBalancing):
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

        forget_loss, scores, mask = reweighted_NLL_correct(
            cache=forget_cache,
            scorer=self.scorer,
            beta=self.beta,
        )
        forget_loss = -forget_loss

        retain_loss: torch.Tensor = retain_cache.outputs.loss # type: ignore
        loss = self.alpha * retain_loss + self.gamma * forget_loss

        return (loss, forget_cache.outputs) if return_outputs else loss


class SelfBalancingDPO(SelfBalancing):
    """Scorer-adjusted DPO: scores applied to lose (original) NLL before DPO log-ratio."""

    def __init__(
        self, *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        beta: float = 0.1,
        score_scale: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.score_scale = score_scale
        self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_original = self.pack_inputs(inputs["forget"]["original"])
        alternate_inputs = self.pack_inputs(inputs["forget"]["alternate"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        # Cache forget_original + retain for scorer training
        forget_cache, retain_cache = self.prepare_caches(model, forget_original, retain_inputs)

        # Hoisted: compute lose-side ref NLL and full win-side log-ratio before the
        # scorer step so the matched scorer trainer can mirror the main DPO loss.
        # win_log_ratio is detached for the scorer (it's a constant offset there)
        # but kept live for the main loss so model grads still flow through win_nll.
        with torch.no_grad():
            lose_ref_nll, _ = compute_batch_nll(self.ref_model, forget_original)
        win_nll, _ = compute_batch_nll(model, alternate_inputs)
        with torch.no_grad():
            win_ref_nll, _ = compute_batch_nll(self.ref_model, alternate_inputs)
        win_log_ratio = -(win_nll - win_ref_nll)

        forget_cache.ref_nll = lose_ref_nll
        forget_cache.win_log_ratio = win_log_ratio.detach()

        if not self._freeze_scorer:
            self.scorer_trainer.step(self.state.global_step, forget_cache, retain_cache)

        # Lose side (original): scored NLL, rescaled so weights sum to T
        lose_scored_nll, _, _ = compute_scored_batch_nll(
            forget_cache, self.scorer, self.score_scale, rescale=True
        )
        lose_log_ratio = -(lose_scored_nll - lose_ref_nll)

        forget_loss = -2 / self.beta * F.logsigmoid(
            self.beta * (win_log_ratio - lose_log_ratio)
        ).mean()

        retain_loss: torch.Tensor = retain_cache.outputs.loss  # type: ignore
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_cache.outputs) if return_outputs else loss


class SelfBalancingNPO(SelfBalancing):
    """Scorer-adjusted NPO: scores applied to forget NLL before NPO log-ratio."""

    def __init__(
        self, *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        beta: float = 0.1,
        score_scale: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.score_scale = score_scale
        self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache = self.prepare_caches(model, forget_inputs, retain_inputs)

        # Hoisted: compute ref NLL once and expose to scorer trainer (matched scorer needs it).
        with torch.no_grad():
            lose_ref_nll, _ = compute_batch_nll(self.ref_model, forget_inputs)
        forget_cache.ref_nll = lose_ref_nll

        if not self._freeze_scorer:
            self.scorer_trainer.step(self.state.global_step, forget_cache, retain_cache)

        # Scored NPO loss (DPO with win=None), rescaled so weights sum to T
        lose_scored_nll, _, _ = compute_scored_batch_nll(
            forget_cache, self.scorer, self.score_scale, rescale=True
        )
        lose_log_ratio = -(lose_scored_nll - lose_ref_nll)

        forget_loss = -2 / self.beta * F.logsigmoid(
            self.beta * (0.0 - lose_log_ratio)
        ).mean()

        retain_loss: torch.Tensor = retain_cache.outputs.loss  # type: ignore
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_cache.outputs) if return_outputs else loss


class SelfBalancingSimNPO(SelfBalancing):
    """Scorer-adjusted SimNPO: scored NLL normalized by seq length, logsigmoid transform."""

    def __init__(
        self, *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        beta: float = 4.5,
        delta: float = 0.0,
        score_scale: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.delta = delta
        self.score_scale = score_scale

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache = self.prepare_caches(model, forget_inputs, retain_inputs)

        if not self._freeze_scorer:
            self.scorer_trainer.step(self.state.global_step, forget_cache, retain_cache)

        scored_nll, _, mask = compute_scored_batch_nll(
            forget_cache, self.scorer, self.score_scale
        )
        seq_lengths = mask.sum(dim=-1).float().clamp(min=1)
        forget_loss = scored_nll / seq_lengths - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

        retain_loss: torch.Tensor = retain_cache.outputs.loss  # type: ignore
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_cache.outputs) if return_outputs else loss


TOK_EQ_SB = {
    'SelfBalancingJensUn': [[2822, 4623], [1939, 2969]],
}


class SelfBalancingJensUn(SelfBalancing):
    """Scorer-adjusted JensUn: per-position JS-div reweighted by scorer scores."""

    def __init__(
        self, *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        retain_loss_type: str = "JensUn",
        score_scale: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.retain_loss_type = retain_loss_type
        self.score_scale = score_scale

        # Determine target tokens (Llama vs Phi)
        idx = int("Phi" in self.model.config._name_or_path)
        self.tok_id = TOK_EQ_SB[self.__class__.__name__][idx]

        self.ref_model = None
        if retain_loss_type in ("KL", "JensUn"):
            self.ref_model = self._prepare_ref_model(self.model)

    def _compute_retain_loss(self, model, retain_inputs):
        if self.retain_loss_type == "NLL":
            return model(**retain_inputs).loss
        elif self.retain_loss_type == "KL":
            kl, _ = compute_kl_divergence(model, self.ref_model, retain_inputs)
            return kl
        elif self.retain_loss_type == "JensUn":
            js, _ = jensun_retain_loss(model, self.ref_model, retain_inputs)
            return js
        else:
            raise NotImplementedError(f"{self.retain_loss_type} not implemented")

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache = self.prepare_caches(model, forget_inputs, retain_inputs)

        if not self._freeze_scorer:
            self.scorer_trainer.step(self.state.global_step, forget_cache, retain_cache)

        forget_loss = scored_jensun_multitok_loss(
            forget_cache, self.scorer, self.tok_id, self.score_scale, rescale=True
        )

        retain_loss = self._compute_retain_loss(model, retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_cache.outputs) if return_outputs else loss


class SelfBalancingWGA(SelfBalancing):
    """Scorer-adjusted WGA: exp(-loss)^(beta * score_scale * g_t) instead of exp(-loss)^beta."""

    def __init__(
        self, *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        beta: float = 1.0,
        score_scale: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.score_scale = score_scale

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache = self.prepare_caches(model, forget_inputs, retain_inputs)

        if not self._freeze_scorer:
            self.scorer_trainer.step(self.state.global_step, forget_cache, retain_cache)

        forget_loss = compute_scored_wga_loss(
            forget_cache, self.scorer, self.beta, self.score_scale
        )

        retain_loss: torch.Tensor = retain_cache.outputs.loss  # type: ignore
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_cache.outputs) if return_outputs else loss


class SelfBalancingWGAInverted(SelfBalancing):
    """WGA with inverted-score exponent: exp(-loss)^(beta * (1 - g_t))."""

    def __init__(
        self, *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        beta: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache = self.prepare_caches(model, forget_inputs, retain_inputs)

        if not self._freeze_scorer:
            self.scorer_trainer.step(self.state.global_step, forget_cache, retain_cache)

        forget_loss = compute_inverted_wga_loss(
            forget_cache, self.scorer, self.beta
        )

        retain_loss: torch.Tensor = retain_cache.outputs.loss  # type: ignore
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_cache.outputs) if return_outputs else loss


class SelfBalancingGradDiffInverted(SelfBalancing):
    """GradDiff with inverted-score exponent: g_t * exp(-loss)^(beta * (1 - g_t))."""

    def __init__(
        self, *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        beta: float = 5.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache = self.prepare_caches(model, forget_inputs, retain_inputs)

        if not self._freeze_scorer:
            self.scorer_trainer.step(self.state.global_step, forget_cache, retain_cache)

        forget_loss = compute_inverted_scored_wga_loss(
            forget_cache, self.scorer, self.beta
        )

        retain_loss: torch.Tensor = retain_cache.outputs.loss  # type: ignore
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_cache.outputs) if return_outputs else loss


class SelfBalancingFUNDIAL(SelfBalancing):
    """FUNDIAL with the learned scorer replacing the spaCy hard noun/entity mask.

    Uses UNDIAL's beta-suppressed teacher distribution; weights per-token CE by
    the scorer's continuous scores instead of FUNDIAL's binary mask.
    """

    def __init__(
        self, *,
        alpha: float = 0.0,
        gamma: float = 1.0,
        beta: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache = self.prepare_caches(
            model, forget_inputs, retain_inputs
        )

        if not self._freeze_scorer:
            self.scorer_trainer.step(self.state.global_step, forget_cache, retain_cache)

        forget_loss = compute_scored_fundial_loss(
            cache=forget_cache,
            ref_model=self.ref_model,
            scorer=self.scorer,
            forget_inputs=forget_inputs,
            beta=self.beta,
        )

        retain_loss: torch.Tensor = retain_cache.outputs.loss  # type: ignore
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_cache.outputs) if return_outputs else loss
