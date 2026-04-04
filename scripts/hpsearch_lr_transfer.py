#!/usr/bin/env python
"""
LR-only Bayesian HP transfer for Llama-3.1-8B-Instruct on TOFU forget10.

Reads best params per method from a source summary (e.g., from the 1B search),
fixes all params except lr, and runs a small lr-only Bayesian search on the 8B model.

Objective: maximize (retain_extraction_strength - extraction_strength)

Usage:
    python scripts/hpsearch_lr_transfer.py --methods GradDiff --n-trials 5 --gpus 0 --resume --verbose
    python scripts/hpsearch_lr_transfer.py --methods GradDiff NPO DPO --n-trials 5 --gpus 1 --resume
    python scripts/hpsearch_lr_transfer.py --methods GradDiff --lr-min 1e-7 --lr-max 1e-5
"""

import argparse
import fcntl
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import optuna

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

MODEL = "Llama-3.1-8B-Instruct"
PRETRAINED_PATH = "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
FORGET_SPLIT = "forget10"

DEFAULT_NUM_EPOCHS = 10
DEFAULT_N_TRIALS = 5
DEFAULT_LR_MIN = 1e-7
DEFAULT_LR_MAX = 1e-5

OBJECTIVE_DIRECTION = "maximize"
DATASET_NAME = f"tofu_{FORGET_SPLIT}_8B"
HYPERPARAM_DIR = "hyperparam"
OPTUNA_DB = "sqlite:///scripts/hpsearch_lr_transfer.db"
DEFAULT_SOURCE = "hyperparam/tofu_forget10/bayesian_summary.json"

BATCH_SIZE = 8
GRAD_ACCUM = 4

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
logger = logging.getLogger("hpsearch_lr_transfer")

# ─────────────────────────────────────────────────────────────────────
# Per-method: param name → Hydra override path, plus static config
# ─────────────────────────────────────────────────────────────────────

# Maps param names (from bayesian_summary.json best_params) to Hydra override paths
PARAM_OVERRIDE_MAP: dict[str, dict[str, str]] = {
    "GradDiff": {
        "gamma": "trainer.method_args.gamma",
        "alpha": "trainer.method_args.alpha",
        "retain_loss_type": "trainer.method_args.retain_loss_type",
    },
    "NPO": {
        "gamma": "trainer.method_args.gamma",
        "alpha": "trainer.method_args.alpha",
        "beta": "trainer.method_args.beta",
    },
    "DPO": {
        "gamma": "trainer.method_args.gamma",
        "alpha": "trainer.method_args.alpha",
        "beta": "trainer.method_args.beta",
    },
    "SimNPO": {
        "gamma": "trainer.method_args.gamma",
        "alpha": "trainer.method_args.alpha",
        "beta": "trainer.method_args.beta",
        "delta": "trainer.method_args.delta",
    },
    "JensUn": {
        "gamma": "trainer.method_args.gamma",
        "alpha": "trainer.method_args.alpha",
    },
    "WGA": {
        "gamma": "trainer.method_args.gamma",
        "alpha": "trainer.method_args.alpha",
        "beta": "trainer.method_args.beta",
    },
    "SatImp": {
        "gamma": "trainer.method_args.gamma",
        "alpha": "trainer.method_args.alpha",
        "beta1": "trainer.method_args.beta1",
        "beta2": "trainer.method_args.beta2",
    },
    "Scorer": {
        "gamma": "trainer.method_args.gamma",
        "alpha": "trainer.method_args.alpha",
        "beta": "trainer.method_args.beta",
        "lambda_entropy": "trainer.method_args.scorer_trainer.lambda_entropy",
        "lambda_population": "trainer.method_args.scorer_trainer.lambda_population",
        "budget": "trainer.method_args.scorer_trainer.budget",
        "lambda_l2": "trainer.method_args.scorer_trainer.lambda_l2",
        "update_every_n_steps": "trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps",
    },
}

# Static per-method config: (experiment, trainer_name, extra_overrides)
METHOD_STATIC: dict[str, tuple[str, str, list[str]]] = {
    "GradDiff": ("unlearn/tofu/default", "GradDiff", []),
    "NPO": ("unlearn/tofu/default", "NPO", []),
    "DPO": ("unlearn/tofu/idk", "DPO", []),
    "SimNPO": ("unlearn/tofu/default", "SimNPO", []),
    "JensUn": ("unlearn/tofu/default", "JensUn", []),
    "WGA": ("unlearn/tofu/default", "WGA", []),
    "SatImp": ("unlearn/tofu/default", "SatImp", []),
    "Scorer": (
        "unlearn/tofu/default",
        "SBGradDiffLearned",
        [
            "trainer.method_args.scorer.cfg.input_dimension=4096",
            "trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=5",
            "+trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear",
            "trainer.method_args.scorer_trainer.lambda_entropy=1",
            "trainer.method_args.scorer_trainer.lambda_population=10",
            "trainer.method_args.scorer_trainer.budget=0.2",
            "trainer.method_args.scorer_trainer.lambda_l2=1",
        ],
    ),
}

ALL_METHODS = list(METHOD_STATIC.keys())

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def load_source_params(summary_path: str, method: str) -> dict[str, Any]:
    """Load best_params for a method from the source summary JSON."""
    with open(summary_path) as f:
        summary = json.load(f)
    if method not in summary:
        raise ValueError(
            f"Method '{method}' not found in {summary_path}. "
            f"Available: {list(summary.keys())}"
        )
    return summary[method]["best_params"]


def build_fixed_overrides(method: str, best_params: dict[str, Any]) -> list[str]:
    """Convert best_params (excluding lr) to Hydra CLI overrides."""
    param_map = PARAM_OVERRIDE_MAP[method]
    overrides = []
    for param_name, hydra_path in param_map.items():
        if param_name not in best_params:
            logger.warning(f"  Param '{param_name}' not in source best_params, skipping")
            continue
        overrides.append(f"{hydra_path}={best_params[param_name]}")
    return overrides


def make_output_dir(method: str, trial_number: int, lr: float) -> str:
    """Build output path with lr in the name."""
    if lr < 1e-3 or lr > 1e4:
        lr_str = f"lr{lr:.2e}"
    else:
        lr_str = f"lr{lr:g}"
    return str(
        Path(HYPERPARAM_DIR) / DATASET_NAME / method / f"trial_{trial_number}_{lr_str}"
    )


def build_train_command(
    method: str,
    experiment: str,
    trainer_name: str,
    fixed_overrides: list[str],
    extra_overrides: list[str],
    lr: float,
    output_dir: str,
    num_epochs: int,
) -> list[str]:
    """Build the training CLI command."""
    cmd = [
        "python",
        "src/train.py",
        "--config-name=unlearn.yaml",
        f"experiment={experiment}",
        f"trainer={trainer_name}",
        f"model={MODEL}",
        f"model.model_args.pretrained_model_name_or_path={PRETRAINED_PATH}",
        "++model.model_args.device_map='auto'",
        "task_name=_unused_",
        f"paths.output_dir={output_dir}",
        f"trainer.args.num_train_epochs={num_epochs}",
        f"trainer.args.per_device_train_batch_size={BATCH_SIZE}",
        f"trainer.args.gradient_accumulation_steps={GRAD_ACCUM}",
        "trainer.args.eval_strategy=no",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        "trainer.args.gradient_checkpointing=True",
        f"trainer.args.learning_rate={lr}",
    ]
    cmd.extend(fixed_overrides)
    cmd.extend(extra_overrides)
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
    cmd.extend(EVAL_METRIC_OVERRIDES)
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
    method: str,
    experiment: str,
    trainer_name: str,
    fixed_overrides: list[str],
    extra_overrides: list[str],
    trial: optuna.Trial,
    lr_min: float,
    lr_max: float,
    num_epochs: int,
    dry_run: bool = False,
    gpus: str | None = None,
    verbose: bool = False,
) -> float | None:
    """Run a single lr-only trial. Returns objective or None on failure."""
    lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
    output_dir = make_output_dir(method, trial.number, lr)

    train_cmd = build_train_command(
        method, experiment, trainer_name, fixed_overrides, extra_overrides,
        lr, output_dir, num_epochs,
    )
    eval_cmd = build_eval_command(output_dir)

    logger.info(f"[{method}|trial {trial.number}] lr={lr:.6e}")

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
        description="LR-only Bayesian HP transfer for 8B model on TOFU forget10"
    )
    parser.add_argument(
        "--methods", nargs="+", required=True, choices=ALL_METHODS,
        help="Methods to optimize",
    )
    parser.add_argument(
        "--n-trials", type=int, default=DEFAULT_N_TRIALS,
        help=f"Trials per method (default: {DEFAULT_N_TRIALS})",
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1')",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing Optuna DB (safe on first run)",
    )
    parser.add_argument("--verbose", action="store_true", help="Stream subprocess output")
    parser.add_argument(
        "--source-summary", type=str, default=DEFAULT_SOURCE,
        help=f"Path to source bayesian_summary.json (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--lr-min", type=float, default=None,
        help=f"Min lr for search (overrides --lr-relative-range)",
    )
    parser.add_argument(
        "--lr-max", type=float, default=None,
        help=f"Max lr for search (overrides --lr-relative-range)",
    )
    parser.add_argument(
        "--lr-relative-range", type=float, default=5.0,
        help="Set lr range to [source_lr/K, source_lr*K] per method (default: 5.0)",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS,
        help=f"Training epochs per trial (default: {DEFAULT_NUM_EPOCHS})",
    )
    parser.add_argument(
        "--db", type=str, default=OPTUNA_DB,
        help=f"Optuna storage URL (default: {OPTUNA_DB})",
    )
    args = parser.parse_args()

    storage = args.db if not args.dry_run else None
    all_results = {}

    for method in args.methods:
        # Load source params and build fixed overrides
        best_params = load_source_params(args.source_summary, method)
        source_lr = best_params.get("lr", DEFAULT_LR_MIN * 10)
        fixed_overrides = build_fixed_overrides(method, best_params)
        experiment, trainer_name, extra_overrides = METHOD_STATIC[method]

        # Compute lr range: explicit --lr-min/--lr-max override relative range
        if args.lr_min is not None or args.lr_max is not None:
            lr_min = args.lr_min if args.lr_min is not None else DEFAULT_LR_MIN
            lr_max = args.lr_max if args.lr_max is not None else DEFAULT_LR_MAX
        else:
            lr_min = source_lr / args.lr_relative_range
            lr_max = source_lr * args.lr_relative_range

        logger.info(
            f"\n{'=' * 60}\n"
            f"Method: {method} | Model: {MODEL} | Trials: {args.n_trials}\n"
            f"Source params: {best_params}\n"
            f"LR range: [{lr_min:.1e}, {lr_max:.1e}] (source lr={source_lr:.2e})\n"
            f"{'=' * 60}"
        )

        study_name = f"lr_transfer_{method}"

        if not args.resume and storage:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
                logger.info(f"  Deleted existing study '{study_name}' (fresh start)")
            except KeyError:
                pass

        study = optuna.create_study(
            study_name=study_name,
            direction=OBJECTIVE_DIRECTION,
            sampler=optuna.samplers.TPESampler(seed=42),
            storage=storage,
            load_if_exists=args.resume,
        )

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        n_completed = len(completed)

        if n_completed == 0:
            # Enqueue source lr as initial trial, clamped to search range
            init_lr = max(lr_min, min(lr_max, source_lr))
            study.enqueue_trial({"lr": init_lr})
            if init_lr != source_lr:
                logger.info(
                    f"  Source lr={source_lr:.6e} clamped to {init_lr:.6e} "
                    f"(range [{lr_min:.1e}, {lr_max:.1e}])"
                )
            logger.info(f"  Enqueued initial trial: lr={init_lr:.6e}")
        else:
            logger.info(f"  Resuming study with {n_completed} completed trials")

        remaining = args.n_trials - n_completed
        if remaining <= 0:
            logger.info(f"  Already have {n_completed}/{args.n_trials} trials, skipping")
            continue

        for _ in range(remaining):
            trial = study.ask()
            obj = run_trial(
                method, experiment, trainer_name, fixed_overrides, extra_overrides,
                trial, lr_min, lr_max, args.num_epochs,
                dry_run=args.dry_run, gpus=args.gpus, verbose=args.verbose,
            )
            if obj is not None:
                study.tell(trial, obj)
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)

        if not args.dry_run and study.best_trial:
            best = study.best_trial
            logger.info(
                f"  BEST trial {best.number}: lr={best.params['lr']:.6e}, "
                f"objective={best.value:.4f}"
            )
            all_results[method] = {
                "method": method,
                "best_lr": best.params["lr"],
                "best_objective": best.value,
                "best_trial": best.number,
                "n_trials": len(study.trials),
                "source_params": best_params,
            }

    # Save summary (atomic read-merge-write with file lock)
    if not args.dry_run and all_results:
        summary_path = Path(HYPERPARAM_DIR) / DATASET_NAME / "bayesian_summary.json"
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
        logger.info("LEADERBOARD (LR transfer, sorted by objective)")
        logger.info("=" * 70)
        sorted_results = sorted(
            all_results.items(),
            key=lambda x: x[1]["best_objective"],
            reverse=True,
        )
        for rank, (name, r) in enumerate(sorted_results, 1):
            logger.info(
                f"  {rank:2d}. {name:10s} | trial {r['best_trial']:3d} | "
                f"lr={r['best_lr']:.2e} | obj={r['best_objective']:.4f}"
            )


if __name__ == "__main__":
    main()
