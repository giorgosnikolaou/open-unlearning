#!/usr/bin/env python3
"""Consolidate RWKU experiment results into JSON archives.

Reads hyperparam trial directories (from LLaMA-Factory), LR sweep evals,
final runs, eval baselines, and Optuna databases, then writes self-contained
JSON files to ./results_archive/.

Usage:
    python scripts/consolidate_rwku_results.py
    python scripts/consolidate_rwku_results.py --output-dir my_archive
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

# External paths
LLAMA_FACTORY_DIR = Path("/tmlscratch/nikolaou/RWKU/LLaMA-Factory")
LLAMA_FACTORY_SCRIPTS = LLAMA_FACTORY_DIR / "scripts"
LLAMA_FACTORY_HP = LLAMA_FACTORY_DIR / "results_hp"

# ─────────────────────────────────────────────────────────────────────
# Utilities (shared with consolidate_results.py)
# ─────────────────────────────────────────────────────────────────────


def load_json_safe(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_yaml_safe(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def strip_secrets(obj):
    """Recursively strip keys containing 'token', 'secret', 'password' from dicts."""
    if isinstance(obj, dict):
        return {
            k: strip_secrets(v)
            for k, v in obj.items()
            if not any(s in k.lower() for s in ("token", "secret", "password"))
        }
    elif isinstance(obj, list):
        return [strip_secrets(v) for v in obj]
    return obj


_PARAM_PATTERN = re.compile(
    r"(retain_loss_type|scorer_lr|beta1(?=\d)|beta2(?=\d)|alpha|beta|gamma|delta|lr)"
    r"([\d.eE+\-]+|NLL|KL)"
)

# LR values with underscore-as-decimal: lr1_90e-06 = 1.90e-06
_LR_UNDERSCORE_PATTERN = re.compile(r"lr(\d+)_(\d+)(e[+\-]?\d+)")


def parse_hyperparams_from_dirname(dirname: str) -> tuple[int, dict]:
    """Parse trial number and hyperparams from a trial directory name."""
    m = re.match(r"trial_(\d+)(?:_(.*)|$)", dirname)
    if not m:
        return -1, {}
    trial_number = int(m.group(1))
    params_str = m.group(2) or ""

    params = {}

    # First try underscore-decimal LR pattern (e.g., lr1_90e-06 = 1.90e-06)
    lr_match = _LR_UNDERSCORE_PATTERN.search(params_str)
    if lr_match:
        integer_part = lr_match.group(1)
        decimal_part = lr_match.group(2)
        exponent = lr_match.group(3)
        params["lr"] = float(f"{integer_part}.{decimal_part}{exponent}")
        # Remove matched LR from params_str to avoid double-matching
        params_str = params_str[: lr_match.start()] + params_str[lr_match.end() :]

    for match in _PARAM_PATTERN.finditer(params_str):
        key = match.group(1)
        val_str = match.group(2)
        if key == "lr" and "lr" in params:
            continue  # Already parsed via underscore pattern
        if val_str in ("NLL", "KL"):
            params[key] = val_str
        else:
            params[key] = float(val_str)
    return trial_number, params


def _trial_sort_key(p: Path) -> int:
    m = re.match(r"trial_(\d+)", p.name)
    return int(m.group(1)) if m else -1


def extract_training_stats(trainer_state: dict) -> dict:
    """Extract summary training statistics from trainer_state.json."""
    log_history = trainer_state.get("log_history", [])

    summary_entry = None
    for entry in reversed(log_history):
        if "train_loss" in entry:
            summary_entry = entry
            break

    last_step = None
    for entry in reversed(log_history):
        if "loss" in entry:
            last_step = entry
            break

    stats = {
        "global_step": trainer_state.get("global_step"),
        "max_steps": trainer_state.get("max_steps"),
        "epoch": trainer_state.get("epoch"),
        "train_batch_size": trainer_state.get("train_batch_size"),
    }
    if summary_entry:
        stats["train_loss"] = summary_entry.get("train_loss")
        stats["train_runtime"] = summary_entry.get("train_runtime")
        stats["train_samples_per_second"] = summary_entry.get(
            "train_samples_per_second"
        )
    if last_step:
        stats["final_step_loss"] = last_step.get("loss")

    return stats


def extract_training_config(hydra_config: dict) -> dict:
    """Extract training config from .hydra/config.yaml."""
    trainer_cfg = hydra_config.get("trainer", {})
    args = trainer_cfg.get("args", {})

    config: dict = {
        "handler": trainer_cfg.get("handler"),
        "learning_rate": args.get("learning_rate"),
        "num_train_epochs": args.get("num_train_epochs"),
        "per_device_train_batch_size": args.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": args.get("gradient_accumulation_steps"),
        "lr_scheduler_type": args.get("lr_scheduler_type"),
        "warmup_ratio": args.get("warmup_ratio"),
    }

    method_args = trainer_cfg.get("method_args", {})
    regular_args = {}
    scorer_config = {}

    for k, v in method_args.items():
        if k == "scorer" and isinstance(v, dict):
            cfg = v.get("cfg", {})
            scorer_config["input_dimension"] = cfg.get("input_dimension")
        elif k == "scorer_trainer" and isinstance(v, dict):
            scorer_config["lambda_entropy"] = v.get("lambda_entropy")
            scorer_config["lambda_population"] = v.get("lambda_population")
            scorer_config["budget"] = v.get("budget")
            scorer_config["lambda_l2"] = v.get("lambda_l2")
            optim = v.get("optim_cfg", {})
            scorer_config["scorer_lr"] = optim.get("lr")
            scorer_config["update_every_n_steps"] = optim.get("update_every_n_steps")
            scorer_config["scheduler"] = optim.get("scheduler")
        elif k == "scorer_pretrain" and isinstance(v, dict):
            scorer_config["pretrain"] = v
        elif isinstance(v, (int, float, str, bool)):
            regular_args[k] = v

    config["method_args"] = regular_args
    if scorer_config:
        config["scorer_config"] = scorer_config

    return config


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    size_mb = path.stat().st_size / (1024 * 1024)
    log.info(f"Wrote {path} ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────
# Optuna utilities
# ─────────────────────────────────────────────────────────────────────


def _decode_param_value(param_value: float, distribution_json: str):
    """Decode a parameter value using its distribution metadata."""
    dist = json.loads(distribution_json)
    name = dist.get("name", "")
    if name == "CategoricalDistribution":
        choices = dist.get("attributes", {}).get("choices", [])
        idx = int(param_value)
        if 0 <= idx < len(choices):
            return choices[idx]
    return param_value


def _parse_distribution(distribution_json: str) -> dict:
    """Parse distribution JSON into a clean dict."""
    dist = json.loads(distribution_json)
    name = dist.get("name", "")
    attrs = dist.get("attributes", {})
    if name == "CategoricalDistribution":
        return {"type": name, "choices": attrs.get("choices", [])}
    result = {"type": name}
    for k in ("low", "high", "log", "step"):
        if k in attrs:
            result[k] = attrs[k]
    return result


def _params_match(dir_params: dict, db_params: dict) -> bool:
    """Check if directory-parsed params match Optuna DB params via :.2e rounding."""
    for key, dir_val in dir_params.items():
        if key not in db_params:
            return False
        db_val = db_params[key]
        if isinstance(dir_val, str):
            if dir_val != db_val:
                return False
        elif isinstance(dir_val, float):
            if not isinstance(db_val, (int, float)):
                return False
            if f"{dir_val:.2e}" != f"{float(db_val):.2e}":
                return False
    return True


def _merge_optuna_params(hp_data: dict, optuna_lookup: dict) -> None:
    """Replace truncated dir-parsed hyperparams with full-precision Optuna params.

    optuna_lookup: {method_name: {trial_number: [list of candidate param dicts]}}
    """
    merged = 0
    unmatched = 0
    for method_name, method_data in hp_data["methods"].items():
        candidates = optuna_lookup.get(method_name, {})
        if not candidates:
            base_name = re.sub(r"_v\d+$", "", method_name)
            candidates = optuna_lookup.get(base_name, {})
        for trial in method_data["trials"]:
            tn = trial["trial_number"]
            dir_params = trial["hyperparams"]
            if not dir_params or tn not in candidates:
                continue

            # Find all matching candidates, pick the one with the most keys
            best_match = None
            for db_params in candidates[tn]:
                if _params_match(dir_params, db_params):
                    if best_match is None or len(db_params) > len(best_match):
                        best_match = db_params

            if best_match is not None:
                trial["hyperparams"] = dict(best_match)
                merged += 1
                matched = True
            else:
                matched = False

            if not matched:
                unmatched += 1

    log.info(f"  Optuna merge: {merged} matched, {unmatched} unmatched")


def build_optuna_lookup(
    scripts_dir: Path,
    db_names: list[str],
    study_prefix: str,
) -> dict:
    """Build {method: {trial_number: [list of param dicts]}} from Optuna DBs."""
    lookup: dict[str, dict[int, list[dict]]] = {}

    for db_name in db_names:
        db_path = scripts_dir / db_name
        if not db_path.exists():
            continue

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        try:
            c.execute("SELECT study_id, study_name FROM studies")
            studies = c.fetchall()
        except sqlite3.OperationalError:
            conn.close()
            continue

        for study in studies:
            study_name = study["study_name"]
            if not study_name.startswith(study_prefix + "_"):
                continue
            method = study_name[len(study_prefix) + 1 :]
            study_id = study["study_id"]

            c.execute(
                "SELECT trial_id, number FROM trials WHERE study_id=? ORDER BY number",
                (study_id,),
            )
            trials = c.fetchall()

            if method not in lookup:
                lookup[method] = {}

            for trial_row in trials:
                trial_id = trial_row["trial_id"]
                trial_num = trial_row["number"]

                c.execute(
                    "SELECT param_name, param_value, distribution_json "
                    "FROM trial_params WHERE trial_id=?",
                    (trial_id,),
                )
                params = {}
                for p in c.fetchall():
                    params[p["param_name"]] = _decode_param_value(
                        p["param_value"], p["distribution_json"]
                    )

                lookup[method].setdefault(trial_num, []).append(params)

        conn.close()

    return lookup


def collect_optuna_databases(scripts_dir: Path, db_filenames: list[str]) -> dict:
    log.info("Collecting Optuna databases")

    result: dict = {
        "metadata": {
            "description": "Optuna study data from RWKU SQLite databases",
            "consolidated_at": datetime.now().isoformat(),
        },
        "databases": {},
    }

    for db_name in db_filenames:
        db_path = scripts_dir / db_name
        if not db_path.exists():
            log.warning(f"  DB not found: {db_path}")
            continue

        db_data: dict = {
            "file_size_bytes": db_path.stat().st_size,
            "studies": {},
        }

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        try:
            c.execute("SELECT study_id, study_name FROM studies")
            studies = c.fetchall()
        except sqlite3.OperationalError:
            log.warning(f"  Could not read studies from {db_name}")
            conn.close()
            continue

        for study in studies:
            study_id = study["study_id"]
            study_name = study["study_name"]

            try:
                c.execute(
                    "SELECT direction FROM study_directions WHERE study_id=?",
                    (study_id,),
                )
                direction_row = c.fetchone()
                direction_str = (
                    direction_row["direction"] if direction_row else "UNKNOWN"
                )
            except sqlite3.OperationalError:
                direction_str = "UNKNOWN"

            c.execute(
                "SELECT trial_id, number, state, datetime_start, datetime_complete "
                "FROM trials WHERE study_id=? ORDER BY number",
                (study_id,),
            )
            trials_rows = c.fetchall()

            trials_data = []
            for trial_row in trials_rows:
                trial_id = trial_row["trial_id"]

                c.execute(
                    "SELECT param_name, param_value, distribution_json "
                    "FROM trial_params WHERE trial_id=?",
                    (trial_id,),
                )
                param_rows = c.fetchall()

                params = {}
                distributions = {}
                for p in param_rows:
                    params[p["param_name"]] = _decode_param_value(
                        p["param_value"], p["distribution_json"]
                    )
                    distributions[p["param_name"]] = _parse_distribution(
                        p["distribution_json"]
                    )

                c.execute(
                    "SELECT value FROM trial_values WHERE trial_id=?", (trial_id,)
                )
                value_row = c.fetchone()
                objective_value = value_row["value"] if value_row else None

                duration = None
                dt_start = trial_row["datetime_start"]
                dt_complete = trial_row["datetime_complete"]
                if dt_start and dt_complete:
                    try:
                        start = datetime.fromisoformat(dt_start)
                        end = datetime.fromisoformat(dt_complete)
                        duration = (end - start).total_seconds()
                    except (ValueError, TypeError):
                        pass

                trial_data = {
                    "trial_id": trial_id,
                    "number": trial_row["number"],
                    "state": trial_row["state"],
                    "datetime_start": dt_start,
                    "datetime_complete": dt_complete,
                    "duration_seconds": duration,
                    "params": params,
                    "objective_value": objective_value,
                    "param_distributions": distributions,
                }
                trials_data.append(trial_data)

            complete = [t for t in trials_data if t["state"] == "COMPLETE"]
            best = (
                max(complete, key=lambda t: t["objective_value"])
                if complete
                else None
            )

            study_data = {
                "study_id": study_id,
                "direction": direction_str,
                "n_trials_complete": len(complete),
                "n_trials_running": sum(
                    1 for t in trials_data if t["state"] == "RUNNING"
                ),
                "n_trials_failed": sum(
                    1 for t in trials_data if t["state"] == "FAIL"
                ),
                "best_trial_number": best["number"] if best else None,
                "best_objective": best["objective_value"] if best else None,
                "trials": trials_data,
            }

            db_data["studies"][study_name] = study_data
            log.info(
                f"  {db_name}/{study_name}: {len(complete)} complete trials"
            )

        conn.close()
        result["databases"][db_name] = db_data

    return result


# ─────────────────────────────────────────────────────────────────────
# RWKU-specific trial processing
# ─────────────────────────────────────────────────────────────────────


def process_rwku_bayesian_trial(trial_dir: Path) -> dict:
    """Process a single RWKU bayesian HP search trial from LLaMA-Factory."""
    dirname = trial_dir.name
    trial_number, hyperparams = parse_hyperparams_from_dirname(dirname)

    aggregate = load_json_safe(trial_dir / "aggregate.json")
    status = "completed" if aggregate is not None else "incomplete"

    trial_data: dict = {
        "trial_number": trial_number,
        "dir_name": dirname,
        "status": status,
        "hyperparams": hyperparams,
    }

    if aggregate:
        # Store full aggregate (includes per-target breakdowns)
        trial_data["rwku_aggregate"] = aggregate

        # Compute objective: avg(neighbor) - avg(forget)
        forget = aggregate.get("forget", {})
        neighbor = aggregate.get("neighbor", {})

        forget_vals = [
            forget.get(f"level_{i}_rouge_l_r")
            for i in range(1, 4)
            if forget.get(f"level_{i}_rouge_l_r") is not None
        ]
        neighbor_vals = [
            neighbor.get(f"level_{i}_rouge_l_r")
            for i in range(1, 3)
            if neighbor.get(f"level_{i}_rouge_l_r") is not None
        ]

        if forget_vals and neighbor_vals:
            avg_forget = sum(forget_vals) / len(forget_vals)
            avg_neighbor = sum(neighbor_vals) / len(neighbor_vals)
            trial_data["objective_value"] = avg_neighbor - avg_forget
        else:
            trial_data["objective_value"] = None

        # Extract summary metrics for quick access
        trial_data["rwku_summary"] = {
            "forget_level1_rouge": forget.get("level_1_rouge_l_r"),
            "forget_level2_rouge": forget.get("level_2_rouge_l_r"),
            "forget_level3_rouge": forget.get("level_3_rouge_l_r"),
            "neighbor_level1_rouge": neighbor.get("level_1_rouge_l_r"),
            "neighbor_level2_rouge": neighbor.get("level_2_rouge_l_r"),
        }

        # MIA metrics
        forget_mia = aggregate.get("forget_mia", {})
        retain_mia = aggregate.get("retain_mia", {})
        trial_data["rwku_summary"]["forget_mia_loss"] = forget_mia.get("loss")
        trial_data["rwku_summary"]["retain_mia_loss"] = retain_mia.get("loss")
        trial_data["rwku_summary"]["forget_mia_zlib"] = forget_mia.get("zlib")
        trial_data["rwku_summary"]["retain_mia_zlib"] = retain_mia.get("zlib")
        trial_data["rwku_summary"]["forget_mia_mink20"] = forget_mia.get("mink20")
        trial_data["rwku_summary"]["retain_mia_mink20"] = retain_mia.get("mink20")

    return trial_data


def collect_rwku_hp_search(
    search_dir: Path,
    search_name: str,
    description: str,
    model: str,
    optuna_lookup: dict | None = None,
) -> dict:
    """Collect RWKU hyperparam search results from LLaMA-Factory output."""
    log.info(f"Collecting RWKU HP search: {search_name} from {search_dir}")

    result: dict = {
        "metadata": {
            "search_name": search_name,
            "description": description,
            "model": model,
            "objective": "maximize avg(neighbor_ROUGE) - avg(forget_ROUGE)",
            "source_dir": str(search_dir),
            "consolidated_at": datetime.now().isoformat(),
        },
        "methods": {},
    }

    for method_dir in sorted(search_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name

        trial_dirs = sorted(method_dir.glob("trial_*"), key=_trial_sort_key)
        if not trial_dirs:
            continue

        trials = [process_rwku_bayesian_trial(td) for td in trial_dirs]
        completed = sum(1 for t in trials if t["status"] == "completed")

        result["methods"][method_name] = {
            "total_trials": len(trials),
            "completed_trials": completed,
            "trials": trials,
        }
        log.info(f"  {method_name}: {completed}/{len(trials)} completed")

    total = sum(m["total_trials"] for m in result["methods"].values())
    completed = sum(m["completed_trials"] for m in result["methods"].values())
    result["metadata"]["total_trial_dirs"] = total
    result["metadata"]["completed_trials"] = completed
    result["metadata"]["incomplete_trials"] = total - completed

    if optuna_lookup:
        _merge_optuna_params(result, optuna_lookup)

    return result


# ─────────────────────────────────────────────────────────────────────
# LR sweep collection (open-unlearning evals)
# ─────────────────────────────────────────────────────────────────────


def collect_rwku_lr_sweep(sweep_dir: Path) -> dict:
    """Collect LR sweep eval results from open-unlearning SB_RWKU."""
    log.info(f"Collecting RWKU LR sweep from {sweep_dir}")

    result: dict = {
        "metadata": {
            "description": "RWKU LR sweep evaluation results",
            "source_dir": str(sweep_dir),
            "consolidated_at": datetime.now().isoformat(),
        },
        "methods": {},
    }

    for method_dir in sorted(sweep_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name

        trial_dirs = sorted(method_dir.glob("trial_*"), key=_trial_sort_key)
        if not trial_dirs:
            continue

        trials = []
        for trial_dir in trial_dirs:
            dirname = trial_dir.name
            trial_number, hyperparams = parse_hyperparams_from_dirname(dirname)

            rwku_summary = load_json_safe(trial_dir / "RWKU_SUMMARY.json")
            status = "completed" if rwku_summary is not None else "incomplete"

            trial_data: dict = {
                "trial_number": trial_number,
                "dir_name": dirname,
                "status": status,
                "hyperparams": hyperparams,
            }

            if rwku_summary:
                trial_data["rwku_summary"] = rwku_summary

            # Hydra config
            hydra_config = load_yaml_safe(trial_dir / ".hydra" / "config.yaml")
            if hydra_config:
                trial_data["training_config"] = strip_secrets(
                    extract_training_config(hydra_config)
                )

            trials.append(trial_data)

        completed = sum(1 for t in trials if t["status"] == "completed")
        result["methods"][method_name] = {
            "total_trials": len(trials),
            "completed_trials": completed,
            "trials": trials,
        }
        log.info(f"  {method_name}: {completed}/{len(trials)} completed")

    return result


# ─────────────────────────────────────────────────────────────────────
# Final runs collection
# ─────────────────────────────────────────────────────────────────────


def collect_rwku_final_runs(runs_dir: Path) -> dict:
    """Collect final RWKU optimal runs."""
    log.info(f"Collecting RWKU final runs from {runs_dir}")

    result: dict = {
        "metadata": {
            "description": "Final RWKU unlearning runs with best hyperparameters",
            "source_dir": str(runs_dir),
            "consolidated_at": datetime.now().isoformat(),
        },
        "runs": {},
    }

    for method_dir in sorted(runs_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name

        run_data: dict = {}

        # RWKU eval
        rwku_eval_dir = method_dir / "rwku_evals"
        run_data["rwku_summary"] = load_json_safe(
            rwku_eval_dir / "RWKU_SUMMARY.json"
        )

        # Paraphrase eval
        para_eval_dir = method_dir / "paraphrase_evals"
        run_data["paraphrase_summary"] = load_json_safe(
            para_eval_dir / "Paraphrase_SUMMARY.json"
        )
        run_data["paraphrase_eval"] = load_json_safe(
            para_eval_dir / "Paraphrase_EVAL.json"
        )

        # Training state
        trainer_state = load_json_safe(method_dir / "trainer_state.json")
        if trainer_state:
            run_data["training_stats"] = extract_training_stats(trainer_state)
            run_data["log_history"] = trainer_state.get("log_history", [])

        # Hydra training config
        hydra_config = load_yaml_safe(method_dir / ".hydra" / "config.yaml")
        if hydra_config:
            run_data["training_config"] = strip_secrets(
                extract_training_config(hydra_config)
            )

        result["runs"][method_name] = run_data
        log.info(f"  {method_name}")

    return result


# ─────────────────────────────────────────────────────────────────────
# Baselines collection
# ─────────────────────────────────────────────────────────────────────


def collect_rwku_baselines(eval_dir: Path, baseline_names: list[str]) -> dict:
    """Collect RWKU baseline evaluations."""
    log.info("Collecting RWKU eval baselines")

    result: dict = {
        "metadata": {
            "description": "RWKU baseline evaluations (original model)",
            "consolidated_at": datetime.now().isoformat(),
        },
        "baselines": {},
    }

    for name in baseline_names:
        baseline_dir = eval_dir / name
        if not baseline_dir.exists():
            log.warning(f"  Baseline not found: {baseline_dir}")
            continue

        baseline_data: dict = {}

        # RWKU evals
        baseline_data["rwku_summary"] = load_json_safe(
            baseline_dir / "rwku_evals" / "RWKU_SUMMARY.json"
        )

        # Paraphrase evals
        baseline_data["paraphrase_summary"] = load_json_safe(
            baseline_dir / "paraphrase_evals" / "Paraphrase_SUMMARY.json"
        )
        baseline_data["paraphrase_eval"] = load_json_safe(
            baseline_dir / "paraphrase_evals" / "Paraphrase_EVAL.json"
        )

        result["baselines"][name] = baseline_data
        log.info(f"  {name}")

    return result


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate RWKU experiment results"
    )
    parser.add_argument(
        "--output-dir",
        default="results_archive",
        help="Output directory for JSON files (default: results_archive)",
    )
    args = parser.parse_args()

    output_dir = ROOT_DIR / args.output_dir
    log.info(f"Output directory: {output_dir}")

    # 1. Bayesian HP search (step 1-2) with full-precision Optuna params
    bayesian_dir = LLAMA_FACTORY_HP / "rwku_bayesian"
    if bayesian_dir.exists():
        # Build Optuna lookup from LLaMA-Factory DBs
        optuna_lookup: dict = {}
        for db_names, study_prefix in [
            (["hpsearch_bayesian_transfer.db"], "bayesian_transfer"),
            (
                ["hpsearch_bayesian_transfer_scorer_lr.db"],
                "bayesian_transfer_scorer_lr",
            ),
        ]:
            partial = build_optuna_lookup(
                LLAMA_FACTORY_SCRIPTS, db_names, study_prefix
            )
            for method, trials_map in partial.items():
                merged_method = optuna_lookup.setdefault(method, {})
                for tn, candidates in trials_map.items():
                    merged_method.setdefault(tn, []).extend(candidates)

        data = collect_rwku_hp_search(
            bayesian_dir,
            "rwku_bayesian",
            "Bayesian HP search, Phi-3-mini-4k-instruct",
            "Phi-3-mini-4k-instruct",
            optuna_lookup,
        )
        write_json(output_dir / "rwku_hyperparam_bayesian.json", data)
    else:
        log.warning(f"Skipping bayesian HP: {bayesian_dir} not found")

    # 2. LR tuning (step 3) with full-precision Optuna params
    lr_dir = LLAMA_FACTORY_HP / "rwku_bayesian_lr"
    if lr_dir.exists():
        # Build Optuna lookup from multiple LR DBs
        optuna_lookup_lr: dict = {}
        for db_names, study_prefix in [
            (["hpsearch_lr_bayesian_transfer.db"], "lr_bayesian_transfer"),
            (["hpsearch_lr_transfer.db"], "lr_transfer"),
            (["hpsearch_lr_transfer_div10.db"], "lr_transfer"),
        ]:
            partial = build_optuna_lookup(
                LLAMA_FACTORY_SCRIPTS, db_names, study_prefix
            )
            for method, trials_map in partial.items():
                merged_method = optuna_lookup_lr.setdefault(method, {})
                for tn, candidates in trials_map.items():
                    merged_method.setdefault(tn, []).extend(candidates)

        data = collect_rwku_hp_search(
            lr_dir,
            "rwku_bayesian_lr",
            "LR tuning from best bayesian params, Phi-3-mini-4k-instruct",
            "Phi-3-mini-4k-instruct",
            optuna_lookup_lr,
        )
        write_json(output_dir / "rwku_hyperparam_lr.json", data)
    else:
        log.warning(f"Skipping LR tuning: {lr_dir} not found")

    # 3. LR sweep eval (step 4)
    sweep_dir = ROOT_DIR / "saves" / "unlearn" / "SB_RWKU" / "lr_sweep2"
    if sweep_dir.exists():
        data = collect_rwku_lr_sweep(sweep_dir)
        write_json(output_dir / "rwku_lr_sweep.json", data)
    else:
        log.warning(f"Skipping LR sweep: {sweep_dir} not found")

    # 4. Final optimal runs (step 5)
    runs_dir = ROOT_DIR / "saves" / "unlearn" / "RWKU" / "Optimal"
    if runs_dir.exists():
        data = collect_rwku_final_runs(runs_dir)
        if data["runs"]:
            write_json(output_dir / "rwku_final_runs.json", data)
        else:
            log.warning(f"Optimal directory exists but is empty: {runs_dir}")
    else:
        log.warning(f"Skipping final runs: {runs_dir} not found")

    # 5. Baselines
    eval_dir = ROOT_DIR / "saves" / "eval"
    baseline_names = [
        "RWKU_baseline_Phi-3-mini-4k-instruct",
    ]
    data = collect_rwku_baselines(eval_dir, baseline_names)
    if data["baselines"]:
        write_json(output_dir / "rwku_eval_baselines.json", data)

    # 6. Optuna databases
    db_files = [
        "hpsearch_bayesian_transfer.db",
        "hpsearch_bayesian_transfer_scorer_lr.db",
        "hpsearch_lr_bayesian_transfer.db",
        "hpsearch_lr_transfer.db",
        "hpsearch_lr_transfer_div10.db",
    ]
    data = collect_optuna_databases(LLAMA_FACTORY_SCRIPTS, db_files)
    write_json(output_dir / "rwku_optuna_databases.json", data)

    log.info("Done!")


if __name__ == "__main__":
    main()
