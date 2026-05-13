#!/usr/bin/env python
"""
Transfer non-matched best params into the corresponding matched study.

For each --methods entry like SBNPOMatched, looks up the best trial from the
non-matched study (e.g. sb_bayesian_SBNPO), runs one train+eval iteration with
those params using the *Matched* trainer config, and records the result as a
new trial in the matched study (sb_bayesian_SBNPOMatched).

This is a diagnostic: it answers "how does the matched objective rate the same
operating point that the non-matched study converged on?" — and gives the
matched study a high-quality trial without spending TPE budget.

Usage:
    python scripts/transfer_to_matched.py --methods SBNPOMatched
    python scripts/transfer_to_matched.py --methods SBNPOMatched SBWGAMatched --gpus 0
    python scripts/transfer_to_matched.py --methods SBNPOMatched --dry-run --verbose
"""

import argparse
import fcntl
import json
import logging
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).parent))
from hpsearch_sb_methods import (
    DATASET_NAME,
    DEFAULT_NUM_EPOCHS,
    HYPERPARAM_DIR,
    METHODS,
    OPTUNA_DB,
    run_trial,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("transfer_to_matched")

# Fallback search order for source (non-matched) studies — they may live in
# older DB shards even when the active matched study lives in the latest one.
DEFAULT_SOURCE_STORAGES = [
    "sqlite:///scripts/hpsearch_sb_bayesian_3.db",
    "sqlite:///scripts/hpsearch_sb_bayesian_2.db",
    "sqlite:///scripts/hpsearch_sb_bayesian_1.db",
    "sqlite:///scripts/hpsearch_sb_bayesian.db",
]


def source_method(matched: str) -> str:
    """SBNPOMatched -> SBNPO (and so on for all *Matched suffix names)."""
    if not matched.endswith("Matched"):
        raise ValueError(f"{matched!r} does not end with 'Matched'")
    return matched[: -len("Matched")]


def find_source_study(name: str, storages: list[str]) -> tuple[optuna.Study, str] | None:
    """Look up an Optuna study by name across a list of storage URLs.

    Returns (study, storage_url) of the FIRST storage that has the study with
    at least one COMPLETE trial. Returns None if not found anywhere.
    """
    for storage in storages:
        try:
            s = optuna.load_study(study_name=name, storage=storage)
        except (KeyError, Exception):
            continue
        if any(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials):
            return s, storage
    return None


def update_summary_json(method_name: str, study: optuna.Study) -> Path:
    """Atomically update sb_bayesian_summary.json with best_trial of `study`.

    Mirrors the lock-protected merge logic in hpsearch_sb_methods.py:1029-1046.
    """
    summary_path = Path(HYPERPARAM_DIR) / DATASET_NAME / "sb_bayesian_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = summary_path.with_suffix(".lock")

    best = study.best_trial
    entry = {
        "method": method_name,
        "best_params": best.params,
        "best_objective": best.value,
        "best_trial": best.number,
        "n_trials": len(study.trials),
    }

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            existing = {}
            if summary_path.exists():
                with open(summary_path) as f:
                    existing = json.load(f)
            existing[method_name] = entry
            with open(summary_path, "w") as f:
                json.dump(existing, f, indent=2, default=str)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
    return summary_path


def main():
    matched_methods = sorted(m for m in METHODS if m.endswith("Matched"))

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        choices=matched_methods,
        help="One or more SB*Matched method names",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_NUM_EPOCHS})",
    )
    parser.add_argument("--gpus", type=str, default=None, help="CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1')")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and register the trial without running training")
    parser.add_argument("--verbose", action="store_true", help="Stream subprocess stdout/stderr")
    parser.add_argument("--storage", default=OPTUNA_DB, help=f"Optuna storage URL for the TARGET (matched) study (default: {OPTUNA_DB})")
    parser.add_argument(
        "--source-storage",
        action="append",
        default=None,
        help=f"Optuna storage URL(s) to search for the SOURCE (non-matched) study; can be repeated. "
             f"Default: search {DEFAULT_SOURCE_STORAGES} in order.",
    )
    args = parser.parse_args()

    source_storages = args.source_storage or DEFAULT_SOURCE_STORAGES

    for matched in args.methods:
        src = source_method(matched)
        logger.info(f"=== {matched} ← {src} ===")

        # Source: search for study across all candidate DBs
        found = find_source_study(f"sb_bayesian_{src}", source_storages)
        if found is None:
            logger.error(
                f"  source study sb_bayesian_{src} not found (or has no completed trials) "
                f"in any of {source_storages}; skipping"
            )
            continue
        src_study, src_storage = found
        logger.info(f"  source study found in {src_storage}")

        src_best = src_study.best_trial
        logger.info(
            f"  source best trial #{src_best.number}: value={src_best.value:.4f}"
        )
        for k, v in src_best.params.items():
            if isinstance(v, float):
                logger.info(f"    {k} = {v:.4g}")
            else:
                logger.info(f"    {k} = {v}")

        # Dry-run: don't touch the target study — just print what would happen.
        if args.dry_run:
            logger.info(
                f"  DRY RUN: would enqueue {src_best.params} into sb_bayesian_{matched} "
                f"in {args.storage}, then call run_trial with trainer={METHODS[matched].trainer_name}"
            )
            continue

        # Target: create-or-load study, enqueue source params, run one trial
        tgt_study = optuna.create_study(
            study_name=f"sb_bayesian_{matched}",
            storage=args.storage,
            direction="maximize",
            load_if_exists=True,
        )
        tgt_study.enqueue_trial(src_best.params, skip_if_exists=False)

        cfg = METHODS[matched]
        trial = tgt_study.ask()

        try:
            obj = run_trial(
                cfg, trial, args.num_epochs,
                dry_run=False, gpus=args.gpus, verbose=args.verbose,
            )
        except Exception:
            logger.exception(f"  run_trial raised for {matched}")
            tgt_study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue

        if obj is None:
            tgt_study.tell(trial, state=optuna.trial.TrialState.FAIL)
            logger.error(f"  {matched}: trial #{trial.number} failed (run_trial returned None)")
            continue

        tgt_study.tell(trial, obj)
        logger.info(f"  {matched}: recorded trial #{trial.number} value={obj:.4f}")

        # Refresh the summary JSON entry for this method (whether or not this
        # trial is now the best — re-reading study.best_trial is what matters).
        try:
            summary_path = update_summary_json(matched, tgt_study)
            best = tgt_study.best_trial
            logger.info(
                f"  {matched}: updated {summary_path} "
                f"(best={best.value:.4f} from trial #{best.number})"
            )
        except Exception:
            logger.exception(f"  {matched}: failed to update summary JSON (trial recorded in DB regardless)")


if __name__ == "__main__":
    main()
