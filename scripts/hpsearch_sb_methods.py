#!/usr/bin/env python
"""
Bayesian hyperparameter optimization for scorer-adjusted (SelfBalancing) unlearning methods.

Searches over the same hyperparameters as the base methods (lr, gamma, alpha,
method-specific params) plus the scorer learning rate. The first trial for each
method uses the optimal parameters found for the base method via
hpsearch_bayesian.py (from hyperparam/tofu_forget10/bayesian_summary.json).

Objective: maximize (retain_extraction_strength - extraction_strength)

Usage:
    python scripts/hpsearch_sb_methods.py --methods SBDPO --n-trials 20 --gpus 0 --resume
    python scripts/hpsearch_sb_methods.py --methods SBNPO SBWGA --n-trials 20 --gpus 1 --resume
    python scripts/hpsearch_sb_methods.py --methods SBDPO --dry-run
"""

import argparse
import fcntl
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import optuna

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

MODEL = "Llama-3.2-1B-Instruct"
FORGET_SPLIT = "forget10"
RETAIN_SPLIT = "retain90"
HOLDOUT_SPLIT = "holdout10"

DEFAULT_NUM_EPOCHS = 10

OBJECTIVE_DIRECTION = "maximize"
DATASET_NAME = f"tofu_{FORGET_SPLIT}"
HYPERPARAM_DIR = "hyperparam"
OPTUNA_DB = "sqlite:///scripts/hpsearch_sb_bayesian.db"
OPTUNA_DB = "sqlite:///scripts/hpsearch_sb_bayesian_1.db"
OPTUNA_DB = "sqlite:///scripts/hpsearch_sb_bayesian_2.db"
OPTUNA_DB = "sqlite:///scripts/hpsearch_sb_bayesian_3.db"

# Only compute extraction_strength + retain_extraction_strength
EVAL_METRIC_OVERRIDES = [
    "~eval.tofu.metrics.model_utility",
    "~eval.tofu.metrics.privleak",
    "~eval.tofu.metrics.exact_memorization",
]

# Fixed scorer overrides (same as Scorer entry in hpsearch_bayesian.py)
SCORER_OVERRIDES = [
    "trainer.method_args.scorer.cfg.input_dimension=2048",
    "trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=5",
    "+trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear",
    "trainer.method_args.scorer_trainer.lambda_entropy=1",
    "trainer.method_args.scorer_trainer.lambda_population=10",
    "trainer.method_args.scorer_trainer.budget=0.2",
    "trainer.method_args.scorer_trainer.lambda_l2=1",
]

# Hard-scored methods: pre-computed binary masks, no learned scorer
HARD_GT_PATH = "data/gpt-selected-tokens-tofu/forget10_with_common_words_gpt"
HARD_REF_MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"

HARD_GT_OVERRIDES = [
    f"trainer.method_args.scoring_args.gt_path={HARD_GT_PATH}",
]
HARD_SU_LLM_OVERRIDES = [
    f"trainer.method_args.scoring_args.ref_model_path={HARD_REF_MODEL_PATH}",
]

# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hpsearch_sb")

# ─────────────────────────────────────────────────────────────────────
# Method config dataclass
# ─────────────────────────────────────────────────────────────────────


@dataclass
class MethodConfig:
    """Configuration for a single SB method's Bayesian search."""

    name: str  # Display name / folder name (e.g., "SBDPO")
    experiment: str  # Hydra experiment config path
    suggest_params: Callable  # fn(optuna.Trial) -> dict[str, Any] of Hydra overrides
    initial_params: dict[str, Any]  # Params for study.enqueue_trial()
    extra_overrides: list[str] = field(default_factory=list)
    trainer_name: str | None = None  # Hydra trainer config name


# ─────────────────────────────────────────────────────────────────────
# Per-method suggest functions and initial params
# (Search ranges match the base methods from hpsearch_bayesian.py;
#  initial params come from bayesian_summary.json best_params)
# ─────────────────────────────────────────────────────────────────────


def _suggest_sbdpo(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.2, 0.5, log=True)
    scorer_lr = trial.suggest_float("scorer_lr", 5e-3, 5e-1, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
        "trainer.method_args.scorer_trainer.optim_cfg.lr": scorer_lr,
    }


# From bayesian_summary.json DPO best_params
SBDPO_INITIAL = {
    "lr": 2.273879162545032e-05,
    "gamma": 3.802291708012508,
    "alpha": 0.15072794715140778,
    "beta": 0.20959534853013898,
    "scorer_lr": 0.05,
}


def _suggest_sbnpo(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.05, 0.2, log=True)
    scorer_lr = trial.suggest_float("scorer_lr", 5e-3, 5e-1, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
        "trainer.method_args.scorer_trainer.optim_cfg.lr": scorer_lr,
    }


# From bayesian_summary.json NPO best_params
SBNPO_INITIAL = {
    "lr": 2.5997940107407718e-05,
    "gamma": 0.12339839231169926,
    "alpha": 4.094787173883685,
    "beta": 0.09891887855017602,
    "scorer_lr": 0.05,
}


def _suggest_sbsimnpo(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 2.0, 3.0)
    delta = trial.suggest_float("delta", 0.0, 2.0)
    scorer_lr = trial.suggest_float("scorer_lr", 5e-3, 5e-1, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
        "trainer.method_args.delta": delta,
        "trainer.method_args.scorer_trainer.optim_cfg.lr": scorer_lr,
    }


# From bayesian_summary.json SimNPO best_params
SBSIMNPO_INITIAL = {
    "lr": 2.0816657948687112e-05,
    "gamma": 1.4862885617791477,
    "alpha": 1.284870475110989,
    "beta": 2.822464050600963,
    "delta": 0.032550905821775444,
    "scorer_lr": 0.05,
}


def _suggest_sbjensun(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 5e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    scorer_lr = trial.suggest_float("scorer_lr", 5e-3, 5e-1, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.scorer_trainer.optim_cfg.lr": scorer_lr,
    }


# From bayesian_summary.json JensUn best_params
SBJENSUN_INITIAL = {
    "lr": 3.9827334615368274e-05,
    "gamma": 0.8170564885709993,
    "alpha": 0.7942528090626316,
    "scorer_lr": 0.05,
}


def _suggest_sbwga(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.1, 5.0, log=True)
    scorer_lr = trial.suggest_float("scorer_lr", 5e-3, 5e-1, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
        "trainer.method_args.scorer_trainer.optim_cfg.lr": scorer_lr,
    }


# From bayesian_summary.json WGA best_params
SBWGA_INITIAL = {
    "lr": 1.5731290014419735e-05,
    "gamma": 1.1575818188103137,
    "alpha": 0.7865293167255417,
    "beta": 2.1353608191419418,
    "scorer_lr": 0.05,
}

# ─────────────────────────────────────────────────────────────────────
# Hard-scored method suggest functions & initial params
# (No scorer_lr — masks are pre-computed, not learned)
# ─────────────────────────────────────────────────────────────────────


def _suggest_hard_graddiff(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
    }


# Based on GradDiff best_params (no scorer_lr, no beta)
HARD_GRADDIFF_INITIAL = {
    "lr": 1.900226110275561e-05,
    "gamma": 0.11636609561681736,
    "alpha": 0.7972014659988814,
}


def _suggest_hard_sbgraddiff(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.1, 10.0, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
    }


# Based on GradDiff best_params + default beta
HARD_SBGRADDIFF_INITIAL = {
    "lr": 1.900226110275561e-05,
    "gamma": 0.11636609561681736,
    "alpha": 0.7972014659988814,
    "beta": 5.0,
}


# ─────────────────────────────────────────────────────────────────────
# Scored method (learned scorer, no saturation, fixed scorer_lr)
# ─────────────────────────────────────────────────────────────────────


def _suggest_scored_graddiff(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
    }


# Initial params from GradDiff best (no beta, no scorer_lr)
SCORED_GRADDIFF_INITIAL = {
    "lr": 1.900226110275561e-05,
    "gamma": 0.11636609561681736,
    "alpha": 0.7972014659988814,
}


# ─────────────────────────────────────────────────────────────────────
# Method registry
# ─────────────────────────────────────────────────────────────────────

def _suggest_sbwga_inverted(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 5e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 1, 5.0, log=True)
    scorer_lr = trial.suggest_float("scorer_lr", 5e-3, 5e-1, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
        "trainer.method_args.scorer_trainer.optim_cfg.lr": scorer_lr,
        # "trainer.method_args.scorer_trainer.optim_cfg.lr": 0.05,
    }


# From bayesian_summary.json WGA best_params
SBWGA_INVERTED_INITIAL = {
    "lr": 1.5731290014419735e-05,
    "gamma": 1.1575818188103137,
    "alpha": 0.7865293167255417,
    "beta": 2.1353608191419418,
    "scorer_lr": 0.05,
}


def _suggest_sbgraddiff(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.1, 10.0, log=True)
    scorer_lr = trial.suggest_float("scorer_lr", 5e-3, 5e-1, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
        "trainer.method_args.scorer_trainer.optim_cfg.lr": scorer_lr,
    }

def _suggest_sbgraddiff_no_slr(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.1, 10.0, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta
    }


def _suggest_sbgraddiff_matched(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.1, 10.0, log=True)
    scorer_lr = trial.suggest_float("scorer_lr", 1e-2, 5e-1, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
        "trainer.method_args.scorer_trainer.optim_cfg.lr": scorer_lr,
    }


def _suggest_sbgraddiff_matched_joint(trial: optuna.Trial) -> dict[str, Any]:
    """Joint variant: scorer_lr is a flat method_arg (NoOp scorer_trainer)."""
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.1, 10.0, log=True)
    scorer_lr = trial.suggest_float("scorer_lr", 1e-2, 5e-1, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
        "trainer.method_args.scorer_lr": scorer_lr,
    }

# From bayesian_summary.json GradDiff best_params (drop retain_loss_type, add beta default)
SBGRADDIFF_INITIAL = {
    "lr": 1.900226110275561e-05,
    "gamma": 0.11636609561681736,
    "alpha": 0.7972014659988814,
    "beta": 5.0,
    "scorer_lr": 0.05,
}


def _suggest_sbgraddiff_matched_no_alpha(trial: optuna.Trial) -> dict[str, Any]:
    """NoRetain variant: drops alpha (no retain_loss term in main objective)."""
    lr = trial.suggest_float("lr", 1e-8, 1e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.1, 10.0, log=True)
    scorer_lr = trial.suggest_float("scorer_lr", 1e-2, 5e-1, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.beta": beta,
        "trainer.method_args.scorer_trainer.optim_cfg.lr": scorer_lr,
    }


SBGRADDIFF_NO_ALPHA_INITIAL = {
    "lr": SBGRADDIFF_INITIAL["lr"],
    "gamma": SBGRADDIFF_INITIAL["gamma"],
    "beta": SBGRADDIFF_INITIAL["beta"],
    "scorer_lr": SBGRADDIFF_INITIAL["scorer_lr"],
}


def _suggest_sbgraddiff_inverted(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 5e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.1, 10.0, log=True)
    scorer_lr = trial.suggest_float("scorer_lr", 5e-3, 5e-1, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
        "trainer.method_args.scorer_trainer.optim_cfg.lr": scorer_lr,
    }


# From bayesian_summary.json Scorer best_params
SBGRADDIFF_INVERTED_INITIAL = {
    "lr": 2.067292963134508e-05,
    "gamma": 4.647522481492768,
    "alpha": 0.8977390904130474,
    "beta": 7.372496215407487,
    "scorer_lr": 0.05,
}


def _suggest_sbfundial(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 1.0, 30.0, log=True)
    scorer_lr = trial.suggest_float("scorer_lr", 1e-2, 5e-1, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
        "trainer.method_args.scorer_trainer.optim_cfg.lr": scorer_lr,
    }


# From bayesian_summary.json FUNDIAL best_params (top trial 43; drop mask_type, add scorer_lr default)
SBFUNDIAL_INITIAL = {
    "lr": 2.61e-5,
    "gamma": 0.1573,
    "alpha": 0.6257,
    "beta": 11.6131,
    "scorer_lr": 0.05,
}


METHODS: dict[str, MethodConfig] = {
    "SBDPO": MethodConfig(
        name="SBDPO",
        experiment="unlearn/tofu/idk",
        suggest_params=_suggest_sbdpo,
        initial_params=SBDPO_INITIAL,
        trainer_name="SBDPOLearned",
        extra_overrides=[f"model={MODEL}"] + SCORER_OVERRIDES,
    ),
    "SBNPO": MethodConfig(
        name="SBNPO",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbnpo,
        initial_params=SBNPO_INITIAL,
        trainer_name="SBNPOLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    "SBSimNPO": MethodConfig(
        name="SBSimNPO",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbsimnpo,
        initial_params=SBSIMNPO_INITIAL,
        trainer_name="SBSimNPOLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    "SBJensUn": MethodConfig(
        name="SBJensUn",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbjensun,
        initial_params=SBJENSUN_INITIAL,
        trainer_name="SBJensUnLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    "SBWGA": MethodConfig(
        name="SBWGA",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbwga,
        initial_params=SBWGA_INITIAL,
        trainer_name="SBWGALearned",
        # extra_overrides=SCORER_OVERRIDES + ["trainer.method_args.score_scale=1"],
        extra_overrides=SCORER_OVERRIDES,
    ),
    "SBWGAInverted": MethodConfig(
        name="SBWGAInverted",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbwga_inverted,
        initial_params=SBWGA_INVERTED_INITIAL,
        trainer_name="SBWGAInvertedLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    "SBGradDiff": MethodConfig(
        name="SBGradDiff",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbgraddiff,
        initial_params=SBGRADDIFF_INITIAL,
        trainer_name="SBGradDiffLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    "SBGradDiffInverted": MethodConfig(
        name="SBGradDiffInverted",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbgraddiff_inverted,
        initial_params=SBGRADDIFF_INVERTED_INITIAL,
        trainer_name="SBGradDiffInvertedLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    "SBGradDiffCorrect": MethodConfig(
        name="SBGradDiffCorrect",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbgraddiff_no_slr,
        initial_params=SBGRADDIFF_INITIAL,
        trainer_name="SBGradDiffCorrectLearned",
        extra_overrides=SCORER_OVERRIDES + ['trainer.method_args.scorer_trainer.optim_cfg.lr=0.05'],
    ),
    "SBGradDiffMatched": MethodConfig(
        name="SBGradDiffMatched",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbgraddiff_matched,
        initial_params=SBGRADDIFF_INITIAL,
        trainer_name="SBGradDiffMatchedLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    "SBGradDiffCorrectMatched": MethodConfig(
        name="SBGradDiffCorrectMatched",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbgraddiff_matched,
        initial_params=SBGRADDIFF_INITIAL,
        trainer_name="SBGradDiffCorrectMatchedLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    # Matched variants of NPO/SimNPO/WGA/DPO: swap to the matched scorer trainer
    # (ScorerTrainerSBNPO / SBSimNPO / SBWGA / SBDPO) so the scorer is trained on
    # the same forget objective as the model. Reuse base-variant suggest fn and
    # initial params — the only difference is the trainer config.
    "SBNPOMatched": MethodConfig(
        name="SBNPOMatched",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbnpo,
        initial_params=SBNPO_INITIAL,
        trainer_name="SBNPOMatchedLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    "SBSimNPOMatched": MethodConfig(
        name="SBSimNPOMatched",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbsimnpo,
        initial_params=SBSIMNPO_INITIAL,
        trainer_name="SBSimNPOMatchedLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    "SBWGAMatched": MethodConfig(
        name="SBWGAMatched",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbwga,
        initial_params=SBWGA_INITIAL,
        trainer_name="SBWGAMatchedLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    "SBDPOMatched": MethodConfig(
        name="SBDPOMatched",
        experiment="unlearn/tofu/idk",
        suggest_params=_suggest_sbdpo,
        initial_params=SBDPO_INITIAL,
        trainer_name="SBDPOMatchedLearned",
        extra_overrides=[f"model={MODEL}"] + SCORER_OVERRIDES,
    ),
    "SBGradDiffMatchedNoRetain": MethodConfig(
        name="SBGradDiffMatchedNoRetain",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbgraddiff_matched_no_alpha,
        initial_params=SBGRADDIFF_NO_ALPHA_INITIAL,
        trainer_name="SBGradDiffMatchedNoRetainLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    "SBGradDiffNoRetain": MethodConfig(
        name="SBGradDiffNoRetain",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbgraddiff_matched_no_alpha,
        initial_params=SBGRADDIFF_NO_ALPHA_INITIAL,
        trainer_name="SBGradDiffNoRetainLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    "SBFUNDIAL": MethodConfig(
        name="SBFUNDIAL",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbfundial,
        initial_params=SBFUNDIAL_INITIAL,
        trainer_name="SBFUNDIALLearned",
        extra_overrides=SCORER_OVERRIDES,
    ),
    # Joint variants: scorer params optimized via main loss (NoOp scorer trainer
    # in YAML), so the regularizer / update_every_n_steps overrides are flat
    # method_args, not nested under scorer_trainer.
    "SBGradDiffMatchedJoint": MethodConfig(
        name="SBGradDiffMatchedJoint",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbgraddiff_matched_joint,
        initial_params=SBGRADDIFF_INITIAL,
        trainer_name="SBGradDiffMatchedJointLearned",
        extra_overrides=[
            "trainer.method_args.scorer.cfg.input_dimension=2048",
            "trainer.method_args.lambda_entropy=1",
            "trainer.method_args.lambda_population=10",
            "trainer.method_args.budget=0.2",
            "trainer.method_args.lambda_l2=1",
        ],
    ),
    "SBGradDiffCorrectMatchedJoint": MethodConfig(
        name="SBGradDiffCorrectMatchedJoint",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_sbgraddiff_matched_joint,
        initial_params=SBGRADDIFF_INITIAL,
        trainer_name="SBGradDiffCorrectMatchedJointLearned",
        extra_overrides=[
            "trainer.method_args.scorer.cfg.input_dimension=2048",
            "trainer.method_args.lambda_entropy=1",
            "trainer.method_args.lambda_population=10",
            "trainer.method_args.budget=0.2",
            "trainer.method_args.lambda_l2=1",
        ],
    ),
    # ── Hard-scored methods (pre-computed masks, no learned scorer) ──
    "HardScoredGradDiff_GT": MethodConfig(
        name="HardScoredGradDiff_GT",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_hard_graddiff,
        initial_params=HARD_GRADDIFF_INITIAL,
        trainer_name="HardScoredGradDiff",
        extra_overrides=HARD_GT_OVERRIDES,
    ),
    "HardScoredGradDiff_SEUL": MethodConfig(
        name="HardScoredGradDiff_SEUL",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_hard_graddiff,
        initial_params=HARD_GRADDIFF_INITIAL,
        trainer_name="HardScoredGradDiff_SEUL",
    ),
    "HardScoredGradDiff_SU_LLM": MethodConfig(
        name="HardScoredGradDiff_SU_LLM",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_hard_graddiff,
        initial_params=HARD_GRADDIFF_INITIAL,
        trainer_name="HardScoredGradDiff_SU_LLM",
        extra_overrides=HARD_SU_LLM_OVERRIDES,
    ),
    "HardScoredGradDiff_SU_Ngram": MethodConfig(
        name="HardScoredGradDiff_SU_Ngram",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_hard_graddiff,
        initial_params=HARD_GRADDIFF_INITIAL,
        trainer_name="HardScoredGradDiff_SU_Ngram",
    ),
    "HardScoredSBGradDiff_GT": MethodConfig(
        name="HardScoredSBGradDiff_GT",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_hard_sbgraddiff,
        initial_params=HARD_SBGRADDIFF_INITIAL,
        trainer_name="HardScoredSBGradDiff",
        extra_overrides=HARD_GT_OVERRIDES,
    ),
    # ── Scored method (learned scorer, no saturation) ──
    "ScoredGradDiff": MethodConfig(
        name="ScoredGradDiff",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_scored_graddiff,
        initial_params=SCORED_GRADDIFF_INITIAL,
        trainer_name="ScoredGradDiff",
        extra_overrides=SCORER_OVERRIDES,
    ),
}

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def params_tag(params: dict[str, Any]) -> str:
    """Create a tag string from trial params for the output directory name."""
    parts = []
    for key in sorted(params.keys()):
        val = params[key]
        if isinstance(val, float):
            if val < 1e-3 or val > 1e4:
                parts.append(f"{key}{val:.2e}")
            else:
                parts.append(f"{key}{val:g}")
        else:
            parts.append(f"{key}{val}")
    return "_".join(parts)


def make_output_dir(method: str, trial_number: int, params: dict[str, Any]) -> str:
    """Build output path: hyperparam/<dataset>/<method>/trial_<N>_<param_values>"""
    tag = params_tag(params)
    # return str(
    #     Path(HYPERPARAM_DIR) / DATASET_NAME / method / f"trial_{trial_number}_{tag}"
    # )
    return str(
        Path(HYPERPARAM_DIR) / f'{DATASET_NAME}_sb' / method / f"trial_{trial_number}_{tag}"
    )


def build_train_command(
    method_cfg: MethodConfig,
    param_overrides: dict[str, Any],
    output_dir: str,
    num_epochs: int,
) -> list[str]:
    """Build the training CLI command."""
    cmd = [
        "python",
        "src/train.py",
        "--config-name=unlearn.yaml",
        f"experiment={method_cfg.experiment}",
        f"trainer={method_cfg.trainer_name or method_cfg.name}",
        f"model={MODEL}",
        "task_name=_unused_",
        f"paths.output_dir={output_dir}",
        f"trainer.args.num_train_epochs={num_epochs}",
        "trainer.args.eval_strategy=no",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
    ]
    for key, value in param_overrides.items():
        cmd.append(f"{key}={value}")
    for override in method_cfg.extra_overrides:
        cmd.append(override)
    return cmd


def build_eval_command(output_dir: str) -> list[str]:
    """Build the eval CLI command (extraction_strength + retain_extraction_strength only)."""
    cmd = [
        "python",
        "src/eval.py",
        "--config-name=eval.yaml",
        "experiment=eval/tofu/default",
        f"model={MODEL}",
        f"model.model_args.pretrained_model_name_or_path={output_dir}",
        "task_name=_unused_",
        f"paths.output_dir={output_dir}",
        "eval.tofu.overwrite=true",
    ]
    for override in EVAL_METRIC_OVERRIDES:
        cmd.append(override)
    return cmd


def run_subprocess(
    cmd: list[str], env: dict, stage_name: str, task_name: str,
    verbose: bool = False,
) -> bool:
    """Run a subprocess and return success status."""
    if verbose:
        result = subprocess.run(cmd, env=env)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        if not verbose:
            logger.error(
                f"  {stage_name} failed (exit {result.returncode}) for {task_name}. "
                f"stderr tail: {result.stderr[-500:] if result.stderr else 'empty'}"
            )
        else:
            logger.error(
                f"  {stage_name} failed (exit {result.returncode}) for {task_name}."
            )
        return False
    return True


def parse_results(output_dir: str) -> dict | None:
    """Parse TOFU_SUMMARY.json from the output directory."""
    summary_path = Path(output_dir) / "TOFU_SUMMARY.json"
    if not summary_path.exists():
        logger.warning(f"Summary not found: {summary_path}")
        return None
    with open(summary_path) as f:
        return json.load(f)


def compute_objective(results: dict) -> float:
    """Compute: retain_extraction_strength - extraction_strength (maximize)."""
    extraction = results.get("extraction_strength")
    retain_extraction = results.get("retain_extraction_strength")
    if extraction is None or retain_extraction is None:
        logger.warning(
            f"Missing metrics. extraction_strength={extraction}, "
            f"retain_extraction_strength={retain_extraction}"
        )
        return float("-inf")
    return retain_extraction - extraction


# ─────────────────────────────────────────────────────────────────────
# Trial runner
# ─────────────────────────────────────────────────────────────────────


def run_trial(
    method_cfg: MethodConfig,
    trial: optuna.Trial,
    num_epochs: int,
    dry_run: bool = False,
    gpus: str | None = None,
    verbose: bool = False,
) -> float | None:
    """Run a single trial: suggest params -> train -> eval -> objective."""
    param_overrides = method_cfg.suggest_params(trial)
    output_dir = make_output_dir(method_cfg.name, trial.number, trial.params)

    train_cmd = build_train_command(method_cfg, param_overrides, output_dir, num_epochs)
    eval_cmd = build_eval_command(output_dir)

    param_str = ", ".join(f"{k}={v}" for k, v in sorted(trial.params.items()))
    logger.info(f"[{method_cfg.name}|trial {trial.number}] {param_str}")

    if dry_run:
        logger.info(f"  DRY RUN train: {' '.join(train_cmd)}")
        logger.info(f"  DRY RUN eval:  {' '.join(eval_cmd)}")
        return 0.0

    env = os.environ.copy()
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus

    if not run_subprocess(train_cmd, env, "Training", output_dir, verbose=verbose):
        return None

    if not run_subprocess(eval_cmd, env, "Evaluation", output_dir, verbose=verbose):
        return None

    results = parse_results(output_dir)
    if results is None:
        return None

    obj = compute_objective(results)
    logger.info(
        f"  Results: extraction_strength={results.get('extraction_strength', 'N/A')}, "
        f"retain_extraction_strength={results.get('retain_extraction_strength', 'N/A')}, "
        f"objective={obj:.4f}"
    )
    return obj


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian HP optimization for scorer-adjusted (SB) unlearning methods"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        choices=list(METHODS.keys()),
        help="SB methods to optimize",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Total trials per method including the initial enqueued trial (default: 20)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing Optuna DB (safe to pass on first run too)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=OPTUNA_DB,
        help=f"Optuna storage URL (default: {OPTUNA_DB})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream subprocess stdout/stderr to terminal instead of capturing",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f"Training epochs per trial (default: {DEFAULT_NUM_EPOCHS})",
    )
    args = parser.parse_args()

    storage = args.db if not args.dry_run else None
    all_results = {}

    for method_name in args.methods:
        method_cfg = METHODS[method_name]
        study_name = f"sb_bayesian_{method_name}"

        logger.info(
            f"\n{'=' * 60}\n"
            f"Method: {method_name} | Trials: {args.n_trials}\n"
            f"{'=' * 60}"
        )

        if not args.resume and storage:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
                logger.info(f"  Deleted existing study '{study_name}' (fresh start)")
            except KeyError:
                pass

        sampler = optuna.samplers.TPESampler(seed=42)

        study = optuna.create_study(
            study_name=study_name,
            direction=OBJECTIVE_DIRECTION,
            sampler=sampler,
            storage=storage,
            load_if_exists=args.resume,
        )

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        n_completed = len(completed)

        # Advance sampler RNG past already-completed trials to avoid
        # duplicate suggestions when resuming a preempted job.
        if n_completed > 0:
            burn_study = optuna.create_study(
                direction=OBJECTIVE_DIRECTION,
                sampler=sampler,
            )
            for t in sorted(completed, key=lambda t: t.number):
                dummy = burn_study.ask()
                burn_study.tell(dummy, t.value if t.value is not None else 0.0)

        if n_completed == 0:
            study.enqueue_trial(method_cfg.initial_params)
            logger.info(f"  Enqueued initial trial: {method_cfg.initial_params}")
        else:
            logger.info(
                f"  Resuming study with {n_completed} completed trials"
            )

        remaining = args.n_trials - n_completed
        if remaining <= 0:
            logger.info(f"  Already have {n_completed}/{args.n_trials} trials, skipping")
            continue

        for _ in range(remaining):
            trial = study.ask()
            obj = run_trial(
                method_cfg,
                trial,
                num_epochs=args.num_epochs,
                dry_run=args.dry_run,
                gpus=args.gpus,
                verbose=args.verbose,
            )
            if obj is not None:
                study.tell(trial, obj)
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)

        if not args.dry_run and study.best_trial:
            best = study.best_trial
            logger.info(
                f"  BEST trial {best.number}: "
                f"params={best.params}, objective={best.value:.4f}"
            )
            all_results[method_name] = {
                "method": method_name,
                "best_params": best.params,
                "best_objective": best.value,
                "best_trial": best.number,
                "n_trials": len(study.trials),
            }

    # Save summary (atomic read-merge-write with file lock)
    if not args.dry_run and all_results:
        summary_path = Path(HYPERPARAM_DIR) / DATASET_NAME / "sb_bayesian_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = summary_path.with_suffix(".lock")
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                existing = {}
                if summary_path.exists():
                    with open(summary_path) as f:
                        existing = json.load(f)
                existing.update(all_results)
                with open(summary_path, "w") as f:
                    json.dump(existing, f, indent=2, default=str)
                all_results = existing
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
        logger.info(f"\nSummary saved to {summary_path}")

        # Print leaderboard
        logger.info("\n" + "=" * 70)
        logger.info(
            "LEADERBOARD (sorted by objective = retain_extraction - extraction)"
        )
        logger.info("=" * 70)
        sorted_results = sorted(
            all_results.items(),
            key=lambda x: x[1]["best_objective"],
            reverse=True,
        )
        for rank, (name, r) in enumerate(sorted_results, 1):
            logger.info(
                f"  {rank:2d}. {r['method']:12s} | trial {r['best_trial']:3d} | "
                f"obj={r['best_objective']:.4f} | {r['best_params']}"
            )


if __name__ == "__main__":
    main()
