#!/usr/bin/env python
"""
Full Bayesian hyperparameter optimization for unlearning methods on TOFU forget10.

Uses Optuna TPE sampler to search over ALL hyperparameters (lr, gamma, alpha,
and method-specific params). The first trial for each method uses pre-specified
initial values via study.enqueue_trial().

Objective: maximize (retain_extraction_strength - extraction_strength)

Usage:
    # Launch per-method jobs on dedicated GPUs (--resume is always safe)
    python scripts/hpsearch_bayesian.py --methods GradDiff --n-trials 20 --gpus 0 --resume
    python scripts/hpsearch_bayesian.py --methods NPO DPO --n-trials 20 --gpus 1 --resume

    # Dry run (print commands without executing)
    python scripts/hpsearch_bayesian.py --methods GradDiff --dry-run

    # More trials for specific methods
    python scripts/hpsearch_bayesian.py --methods GradDiff NPO --n-trials 30 --gpus 0 --resume
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
OPTUNA_DB = "sqlite:///scripts/hpsearch_bayesian.db"

# Only compute extraction_strength + retain_extraction_strength
EVAL_METRIC_OVERRIDES = [
    "~eval.tofu.metrics.model_utility",
    "~eval.tofu.metrics.privleak",
    "~eval.tofu.metrics.exact_memorization",
]

# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hpsearch_bayesian")

# ─────────────────────────────────────────────────────────────────────
# Method config dataclass
# ─────────────────────────────────────────────────────────────────────


@dataclass
class MethodConfig:
    """Configuration for a single unlearning method's Bayesian search."""

    name: str  # Display name / folder name (e.g., "Scorer")
    experiment: str  # Hydra experiment config path
    suggest_params: Callable  # fn(optuna.Trial) -> dict[str, Any] of Hydra overrides
    initial_params: dict[str, Any]  # Params for study.enqueue_trial()
    extra_overrides: list[str] = field(default_factory=list)
    trainer_name: str | None = None  # Hydra trainer config name; defaults to name if None


# ─────────────────────────────────────────────────────────────────────
# Per-method suggest functions and initial params
# ─────────────────────────────────────────────────────────────────────


def _suggest_graddiff(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    retain_loss_type = trial.suggest_categorical("retain_loss_type", ["NLL", "KL"])
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.retain_loss_type": retain_loss_type,
    }


GRADDIFF_INITIAL = {
    "lr": 1e-5,
    "gamma": 0.5,
    "alpha": 1.0,
    "retain_loss_type": "NLL",
}


def _suggest_npo(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.05, 0.2, log=True) # Same as SatImp
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
    }


NPO_INITIAL = {
    "lr": 1e-5,
    "gamma": 1.0,
    "alpha": 1.0,
    "beta": 0.1,
}


def _suggest_dpo(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.2, 0.5, log=True) # Same as SatImp
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
    }


DPO_INITIAL = {
    "lr": 1e-5,
    "gamma": 1.0,
    "alpha": 1.0,
    "beta": 0.4,
}


def _suggest_simnpo(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 2.0, 3.0) # Same as SatImp
    delta = trial.suggest_float("delta", 0.0, 2.0)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
        "trainer.method_args.delta": delta,
    }


SIMNPO_INITIAL = {
    "lr": 1e-5,
    "gamma": 0.125,
    "alpha": 1.0,
    "beta": 3.5,
    "delta": 0.0,
}


def _suggest_jensun(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 5e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
    }


JENSUN_INITIAL = {
    "lr": 1e-5,
    "gamma": 0.5, # Default from paper
    "alpha": 0.5, # Default from paper
}


def _suggest_wga(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.1, 5.0, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
    }


WGA_INITIAL = {
    "lr": 1e-5,
    "gamma": 1.0,
    "alpha": 1.0,
    "beta": 1.0, # Default from paper
}


def _suggest_satimp(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta1 = trial.suggest_float("beta1", 1.0, 10.0)
    beta2 = trial.suggest_float("beta2", 0.1, 5.0, log=True)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta1": beta1,
        "trainer.method_args.beta2": beta2,
    }


SATIMP_INITIAL = {
    "lr": 1e-5,
    "gamma": 1.0, # Default from paper
    "alpha": 0.1, # Default from paper
    "beta1": 5.0, # Default from paper
    "beta2": 1.0, # Default from paper
}


# def _suggest_scorer(trial: optuna.Trial) -> dict[str, Any]:
#     lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
#     gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
#     alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
#     lambda_entropy = trial.suggest_float("lambda_entropy", 0.1, 5.0, log=True)
#     lambda_population = trial.suggest_float("lambda_population", 1.0, 20.0, log=True)
#     budget = trial.suggest_float("budget", 0.05, 0.5)
#     lambda_l2 = trial.suggest_float("lambda_l2", 0.01, 5.0, log=True)
#     update_every_n_steps = trial.suggest_int("update_every_n_steps", 1, 20)
#     return {
#         "trainer.args.learning_rate": lr,
#         "trainer.method_args.gamma": gamma,
#         "trainer.method_args.alpha": alpha,
#         "trainer.method_args.scorer_trainer.lambda_entropy": lambda_entropy,
#         "trainer.method_args.scorer_trainer.lambda_population": lambda_population,
#         "trainer.method_args.scorer_trainer.budget": budget,
#         "trainer.method_args.scorer_trainer.lambda_l2": lambda_l2,
#         "trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps": update_every_n_steps,
#     }


# SCORER_INITIAL = {
#     "lr": 1e-5,
#     "gamma": 5.0,
#     "alpha": 0.5,
#     "lambda_entropy": 1.0,
#     "lambda_population": 10.0,
#     "budget": 0.2,
#     "lambda_l2": 1.0,
#     "update_every_n_steps": 5,
# }


def _suggest_scorer(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 5.0, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 1.0, 10.0)
    return {
        "trainer.args.learning_rate": lr,
        "trainer.method_args.gamma": gamma,
        "trainer.method_args.alpha": alpha,
        "trainer.method_args.beta": beta,
    }


SCORER_INITIAL = {
    "lr": 1e-5,
    "gamma": 5.0,
    "alpha": 0.5,
    "beta": 5.0,
}

# ─────────────────────────────────────────────────────────────────────
# Method registry
# ─────────────────────────────────────────────────────────────────────

METHODS: dict[str, MethodConfig] = {
    "GradDiff": MethodConfig(
        name="GradDiff",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_graddiff,
        initial_params=GRADDIFF_INITIAL,
    ),
    "NPO": MethodConfig(
        name="NPO",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_npo,
        initial_params=NPO_INITIAL,
    ),
    "DPO": MethodConfig(
        name="DPO",
        experiment="unlearn/tofu/idk",
        suggest_params=_suggest_dpo,
        initial_params=DPO_INITIAL,
        extra_overrides=["model=Llama-3.2-1B-Instruct"],
    ),
    "SimNPO": MethodConfig(
        name="SimNPO",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_simnpo,
        initial_params=SIMNPO_INITIAL,
    ),
    "JensUn": MethodConfig(
        name="JensUn",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_jensun,
        initial_params=JENSUN_INITIAL,
    ),
    "WGA": MethodConfig(
        name="WGA",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_wga,
        initial_params=WGA_INITIAL,
    ),
    "SatImp": MethodConfig(
        name="SatImp",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_satimp,
        initial_params=SATIMP_INITIAL,
    ),
    "Scorer": MethodConfig(
        name="Scorer",
        experiment="unlearn/tofu/default",
        suggest_params=_suggest_scorer,
        initial_params=SCORER_INITIAL,
        trainer_name="SBGradDiffLearned",
        extra_overrides=[
            "trainer.method_args.scorer.cfg.input_dimension=2048",
            "trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=5",
            "+trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear",
            "trainer.method_args.scorer_trainer.lambda_entropy=1",
            "trainer.method_args.scorer_trainer.lambda_population=10",
            "trainer.method_args.scorer_trainer.budget=0.2",
            "trainer.method_args.scorer_trainer.lambda_l2=1"
        ],
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
    return str(
        Path(HYPERPARAM_DIR) / DATASET_NAME / method / f"trial_{trial_number}_{tag}"
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
    """Run a single trial: suggest params -> train -> eval -> objective.

    Returns the objective value, or None if the trial failed.
    """
    # Step 1: Suggest all hyperparameters (binds params to the trial)
    param_overrides = method_cfg.suggest_params(trial)
    output_dir = make_output_dir(method_cfg.name, trial.number, trial.params)

    # Step 2: Build commands
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

    # Step 3: Train
    if not run_subprocess(train_cmd, env, "Training", output_dir, verbose=verbose):
        return None

    # Step 4: Eval
    if not run_subprocess(eval_cmd, env, "Evaluation", output_dir, verbose=verbose):
        return None

    # Step 5: Parse and compute objective
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
        description="Full Bayesian HP optimization for unlearning methods on TOFU forget10"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        choices=list(METHODS.keys()),
        help="Methods to optimize (launch separate jobs per GPU)",
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
        study_name = f"bayesian_{method_name}"

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
                pass  # Study doesn't exist yet

        study = optuna.create_study(
            study_name=study_name,
            direction=OBJECTIVE_DIRECTION,
            sampler=optuna.samplers.TPESampler(seed=42),
            storage=storage,
            load_if_exists=args.resume,
        )

        # Count only completed trials (RUNNING/WAITING from preempted runs don't count)
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        n_completed = len(completed)

        # Enqueue initial trial only if no completed trials exist
        if n_completed == 0:
            study.enqueue_trial(method_cfg.initial_params)
            logger.info(f"  Enqueued initial trial: {method_cfg.initial_params}")
        else:
            logger.info(
                f"  Resuming study with {n_completed} completed trials"
            )

        # Manual ask/tell loop: trials are only committed to DB on success.
        # If preempted mid-trial, nothing is written — safe for restarts.
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

    # Save summary (atomic read-merge-write with file lock to avoid race conditions)
    if not args.dry_run and all_results:
        summary_path = Path(HYPERPARAM_DIR) / DATASET_NAME / "bayesian_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = summary_path.with_suffix(".lock")
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                # Read existing summary from other jobs
                existing = {}
                if summary_path.exists():
                    with open(summary_path) as f:
                        existing = json.load(f)
                # Merge: our results overwrite only our methods
                existing.update(all_results)
                with open(summary_path, "w") as f:
                    json.dump(existing, f, indent=2, default=str)
                all_results = existing  # use merged results for leaderboard
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
                f"  {rank:2d}. {r['method']:10s} | trial {r['best_trial']:3d} | "
                f"obj={r['best_objective']:.4f} | {r['best_params']}"
            )


if __name__ == "__main__":
    main()
