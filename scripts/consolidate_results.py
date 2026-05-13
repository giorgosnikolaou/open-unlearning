#!/usr/bin/env python3
"""Consolidate all experiment results into JSON archives.

Reads hyperparam trial directories, final runs, eval baselines, and Optuna
databases, then writes self-contained JSON files to ./results_archive/.

Usage:
    python scripts/consolidate_results.py
    python scripts/consolidate_results.py --output-dir my_archive
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

# ─────────────────────────────────────────────────────────────────────
# Utilities
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


# Regex pattern for parsing trial directory names.
# Keys are ordered longest-first so that e.g. "retain_loss_type" matches
# before "lr", and "scorer_lr" before "lr".
# beta1/beta2 (SatImp) use (?=\d) lookahead to avoid matching "beta2.786"
# which is actually key="beta" val="2.786" (e.g. in SBSimNPO).
_PARAM_PATTERN = re.compile(
    r"(retain_loss_type|scorer_lr|beta1(?=\d)|beta2(?=\d)|alpha|beta|gamma|delta|lr)"
    r"([\d.eE+\-]+|NLL|KL)"
)


def parse_hyperparams_from_dirname(dirname: str) -> tuple[int, dict]:
    """Parse trial number and hyperparams from a trial directory name.

    Returns (trial_number, params_dict).
    """
    m = re.match(r"trial_(\d+)(?:_(.*)|$)", dirname)
    if not m:
        return -1, {}
    trial_number = int(m.group(1))
    params_str = m.group(2) or ""

    params = {}
    for match in _PARAM_PATTERN.finditer(params_str):
        key = match.group(1)
        val_str = match.group(2)
        if val_str in ("NLL", "KL"):
            params[key] = val_str
        else:
            params[key] = float(val_str)
    return trial_number, params


def extract_training_stats(trainer_state: dict) -> dict:
    """Extract summary training statistics from trainer_state.json."""
    log_history = trainer_state.get("log_history", [])

    # The final entry with 'train_loss' is the summary entry
    summary_entry = None
    for entry in reversed(log_history):
        if "train_loss" in entry:
            summary_entry = entry
            break

    # Last step entry (with 'loss' key) for final loss
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
        stats["train_samples_per_second"] = summary_entry.get("train_samples_per_second")
    if last_step:
        stats["final_step_loss"] = last_step.get("loss")

    return stats


# ─────────────────────────────────────────────────────────────────────
# Hyperparam search collection
# ─────────────────────────────────────────────────────────────────────

def _trial_sort_key(p: Path) -> int:
    m = re.match(r"trial_(\d+)", p.name)
    return int(m.group(1)) if m else -1


def process_hyperparam_trial(trial_dir: Path) -> dict:
    dirname = trial_dir.name
    trial_number, hyperparams = parse_hyperparams_from_dirname(dirname)

    tofu_summary = load_json_safe(trial_dir / "TOFU_SUMMARY.json")
    trainer_state = load_json_safe(trial_dir / "trainer_state.json")

    status = "completed" if tofu_summary is not None else "incomplete"

    trial_data: dict = {
        "trial_number": trial_number,
        "dir_name": dirname,
        "status": status,
        "hyperparams": hyperparams,
    }

    if tofu_summary:
        trial_data["tofu_summary"] = tofu_summary
        es = tofu_summary.get("extraction_strength")
        res = tofu_summary.get("retain_extraction_strength")
        if es is not None and res is not None:
            trial_data["objective_value"] = res - es
        else:
            trial_data["objective_value"] = None

    if trainer_state:
        trial_data["training_stats"] = extract_training_stats(trainer_state)
        trial_data["log_history"] = trainer_state.get("log_history", [])

    return trial_data


def collect_hyperparam_search(
    search_dir: Path,
    search_name: str,
    description: str,
    model: str,
    forget_split: str,
    optuna_lookup: dict | None = None,
) -> dict:
    log.info(f"Collecting hyperparam search: {search_name} from {search_dir}")

    result: dict = {
        "metadata": {
            "search_name": search_name,
            "description": description,
            "model": model,
            "forget_split": forget_split,
            "objective": "maximize (retain_extraction_strength - extraction_strength)",
            "source_dir": str(search_dir.relative_to(ROOT_DIR)),
            "consolidated_at": datetime.now().isoformat(),
        },
        "methods": {},
    }

    # Embed existing summary files
    for summary_name in ("bayesian_summary.json", "sb_bayesian_summary.json"):
        summary_path = search_dir / summary_name
        if summary_path.exists():
            key = summary_name.replace(".json", "")
            result[key] = load_json_safe(summary_path)

    # Walk method directories
    for method_dir in sorted(search_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name

        trial_dirs = sorted(method_dir.glob("trial_*"), key=_trial_sort_key)
        if not trial_dirs:
            continue

        trials = [process_hyperparam_trial(td) for td in trial_dirs]
        completed = sum(1 for t in trials if t["status"] == "completed")

        result["methods"][method_name] = {
            "total_trials": len(trials),
            "completed_trials": completed,
            "trials": trials,
        }
        log.info(f"  {method_name}: {completed}/{len(trials)} completed")

    # Fill metadata counts
    total = sum(m["total_trials"] for m in result["methods"].values())
    completed = sum(m["completed_trials"] for m in result["methods"].values())
    result["metadata"]["total_trial_dirs"] = total
    result["metadata"]["completed_trials"] = completed
    result["metadata"]["incomplete_trials"] = total - completed

    # Merge full-precision params from Optuna if provided
    if optuna_lookup:
        _merge_optuna_params(result, optuna_lookup)

    return result


def _params_match(dir_params: dict, db_params: dict) -> bool:
    """Check if directory-parsed params match Optuna DB params via :.2e rounding."""
    for key, dir_val in dir_params.items():
        if key not in db_params:
            return False
        db_val = db_params[key]
        if isinstance(dir_val, str):
            # Categorical params (e.g. retain_loss_type)
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
        # Fallback: strip _v1, _v2 etc. suffixes (e.g. SBWGA_v1 → SBWGA)
        if not candidates:
            base_name = re.sub(r"_v\d+$", "", method_name)
            candidates = optuna_lookup.get(base_name, {})
        for trial in method_data["trials"]:
            tn = trial["trial_number"]
            dir_params = trial["hyperparams"]
            if not dir_params or tn not in candidates:
                continue

            matched = False
            for db_params in candidates[tn]:
                if _params_match(dir_params, db_params):
                    trial["hyperparams"] = {
                        k: db_params[k] for k in dir_params if k in db_params
                    }
                    merged += 1
                    matched = True
                    break

            if not matched:
                unmatched += 1

    log.info(f"  Optuna merge: {merged} matched, {unmatched} unmatched")


# ─────────────────────────────────────────────────────────────────────
# Final runs collection
# ─────────────────────────────────────────────────────────────────────

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


def _collect_one_run(run_dir: Path) -> dict:
    """Extract paraphrase / TOFU / trainer-state / hydra-config for one run dir."""
    run_data: dict = {}

    # Paraphrase eval - both summary and full breakdown
    para_eval_dir = run_dir / "paraphrase_evals"
    run_data["paraphrase_summary"] = load_json_safe(
        para_eval_dir / "Paraphrase_SUMMARY.json"
    )
    run_data["paraphrase_eval"] = load_json_safe(
        para_eval_dir / "Paraphrase_EVAL.json"
    )

    # TOFU eval - look in checkpoint-*/evals/ directories
    run_data["tofu_summary"] = None
    run_data["tofu_eval"] = None
    for ckpt_dir in sorted(run_dir.glob("checkpoint-*")):
        evals_dir = ckpt_dir / "evals"
        tofu_sum = load_json_safe(evals_dir / "TOFU_SUMMARY.json")
        if tofu_sum is not None:
            run_data["tofu_summary"] = tofu_sum
            run_data["tofu_eval"] = load_json_safe(evals_dir / "TOFU_EVAL.json")
            break

    # Training state
    trainer_state = load_json_safe(run_dir / "trainer_state.json")
    if trainer_state:
        run_data["training_stats"] = extract_training_stats(trainer_state)
        run_data["log_history"] = trainer_state.get("log_history", [])

    # Hydra training config
    hydra_config = load_yaml_safe(run_dir / ".hydra" / "config.yaml")
    if hydra_config:
        run_data["training_config"] = strip_secrets(
            extract_training_config(hydra_config)
        )

    return run_data


def collect_final_runs(runs_dir: Path) -> dict:
    log.info(f"Collecting final runs from {runs_dir}")

    result: dict = {
        "metadata": {
            "description": "Final unlearning runs with best hyperparameters",
            "source_dir": str(runs_dir.relative_to(ROOT_DIR)),
            "consolidated_at": datetime.now().isoformat(),
        },
        "runs": {},
    }

    for model_dir in sorted(runs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        result["runs"][model_name] = {}

        for split_dir in sorted(model_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split_name = split_dir.name
            result["runs"][model_name][split_name] = {}

            for method_dir in sorted(split_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method_name = method_dir.name

                result["runs"][model_name][split_name][method_name] = (
                    _collect_one_run(method_dir)
                )
                log.info(f"  {model_name}/{split_name}/{method_name}")

                # Regularizer-ablation subdirs: <method>/regs/<E#_P#_L#>/
                regs_dir = method_dir / "regs"
                if regs_dir.is_dir():
                    for combo_dir in sorted(regs_dir.iterdir()):
                        if not combo_dir.is_dir():
                            continue
                        key = f"{method_name}/regs/{combo_dir.name}"
                        result["runs"][model_name][split_name][key] = (
                            _collect_one_run(combo_dir)
                        )
                        log.info(f"  {model_name}/{split_name}/{key}")

    return result


# ─────────────────────────────────────────────────────────────────────
# Eval baselines collection
# ─────────────────────────────────────────────────────────────────────

def collect_eval_baselines(eval_dir: Path) -> dict:
    log.info(f"Collecting eval baselines from {eval_dir}")

    result: dict = {
        "metadata": {
            "description": "Evaluation baselines (full model and retain-only model)",
            "source_dir": str(eval_dir.relative_to(ROOT_DIR)),
            "consolidated_at": datetime.now().isoformat(),
        },
        "baselines": {},
    }

    for model_dir in sorted(eval_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        result["baselines"][model_name] = {}

        baselines_dir = model_dir / "baselines"
        if not baselines_dir.is_dir():
            continue

        for baseline_dir in sorted(baselines_dir.iterdir()):
            if not baseline_dir.is_dir():
                continue
            baseline_name = baseline_dir.name

            baseline_data: dict = {}

            # TOFU evals
            baseline_data["tofu_summary"] = load_json_safe(
                baseline_dir / "tofu_evals" / "TOFU_SUMMARY.json"
            )
            baseline_data["tofu_eval"] = load_json_safe(
                baseline_dir / "tofu_evals" / "TOFU_EVAL.json"
            )

            # Paraphrase evals
            baseline_data["paraphrase_summary"] = load_json_safe(
                baseline_dir / "paraphrase_evals" / "Paraphrase_SUMMARY.json"
            )
            baseline_data["paraphrase_eval"] = load_json_safe(
                baseline_dir / "paraphrase_evals" / "Paraphrase_EVAL.json"
            )

            result["baselines"][model_name][baseline_name] = baseline_data
            log.info(f"  {model_name}/{baseline_name}")

    return result


# ─────────────────────────────────────────────────────────────────────
# Optuna databases collection
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


def collect_optuna_databases(scripts_dir: Path, db_filenames: list[str]) -> dict:
    log.info("Collecting Optuna databases")

    result: dict = {
        "metadata": {
            "description": "Optuna study data from all SQLite databases",
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

            # Get direction
            try:
                c.execute(
                    "SELECT direction FROM study_directions WHERE study_id=?",
                    (study_id,),
                )
                direction_row = c.fetchone()
                direction_str = direction_row["direction"] if direction_row else "UNKNOWN"
            except sqlite3.OperationalError:
                direction_str = "UNKNOWN"

            # Get trials
            c.execute(
                "SELECT trial_id, number, state, datetime_start, datetime_complete "
                "FROM trials WHERE study_id=? ORDER BY number",
                (study_id,),
            )
            trials_rows = c.fetchall()

            trials_data = []
            for trial_row in trials_rows:
                trial_id = trial_row["trial_id"]

                # Get params
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

                # Get objective value
                c.execute(
                    "SELECT value FROM trial_values WHERE trial_id=?", (trial_id,)
                )
                value_row = c.fetchone()
                objective_value = value_row["value"] if value_row else None

                # Compute duration
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

            # Summary stats
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
# Main
# ─────────────────────────────────────────────────────────────────────

def build_optuna_lookup(
    scripts_dir: Path,
    db_names: list[str],
    study_prefix: str,
) -> dict:
    """Build {method: {trial_number: [list of param dicts]}} from Optuna DBs.

    Searches all dbs for studies matching `{study_prefix}_{METHOD}` and
    collects trial params indexed by trial number. Multiple DBs may contain
    the same study/trial, so each trial maps to a list of candidate params.
    """
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


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    size_mb = path.stat().st_size / (1024 * 1024)
    log.info(f"Wrote {path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Consolidate experiment results")
    parser.add_argument(
        "--output-dir",
        default="results_archive",
        help="Output directory for JSON files (default: results_archive)",
    )
    args = parser.parse_args()

    output_dir = ROOT_DIR / args.output_dir
    log.info(f"Output directory: {output_dir}")

    # 1. Hyperparam searches (with full-precision params from Optuna DBs)
    scripts_dir = ROOT_DIR / "scripts"

    # Build Optuna lookups: (search_name, desc, model, split, [(db_names, prefix), ...])
    hp_searches = [
        (
            "tofu_forget10",
            "Bayesian HP search, 1B model",
            "Llama-3.2-1B-Instruct",
            "forget10",
            [
                (["hpsearch_bayesian.db"], "bayesian"),
                (
                    [
                        "hpsearch_sb_bayesian.db",
                        "hpsearch_sb_bayesian_1.db",
                        "hpsearch_sb_bayesian_2.db",
                        "hpsearch_sb_bayesian_3.db",
                    ],
                    "sb_bayesian",
                ),
            ],
        ),
        (
            "tofu_forget10_8B",
            "LR transfer search, 8B model",
            "Llama-3.1-8B-Instruct",
            "forget10",
            [(["hpsearch_lr_transfer.db"], "lr_transfer")],
        ),
        (
            "tofu_forget10_sb",
            "SB Bayesian HP search, 1B model",
            "Llama-3.2-1B-Instruct",
            "forget10",
            [
                (
                    [
                        "hpsearch_sb_bayesian.db",
                        "hpsearch_sb_bayesian_1.db",
                        "hpsearch_sb_bayesian_2.db",
                        "hpsearch_sb_bayesian_3.db",
                    ],
                    "sb_bayesian",
                ),
            ],
        ),
    ]
    for search_name, desc, model, split, db_specs in hp_searches:
        search_dir = ROOT_DIR / "hyperparam" / search_name
        if search_dir.exists():
            # Merge lookups from all DB specs
            optuna_lookup: dict = {}
            for db_names, study_prefix in db_specs:
                partial = build_optuna_lookup(scripts_dir, db_names, study_prefix)
                for method, trials_map in partial.items():
                    merged_method = optuna_lookup.setdefault(method, {})
                    for tn, candidates in trials_map.items():
                        merged_method.setdefault(tn, []).extend(candidates)
            data = collect_hyperparam_search(
                search_dir, search_name, desc, model, split, optuna_lookup
            )
            write_json(output_dir / f"hyperparam_{search_name}.json", data)
        else:
            log.warning(f"Skipping {search_name}: {search_dir} not found")

    # 2. Final runs
    runs_dir = ROOT_DIR / "saves" / "unlearn" / "SB_TOFU"
    if runs_dir.exists():
        data = collect_final_runs(runs_dir)
        write_json(output_dir / "final_runs.json", data)
    else:
        log.warning(f"Skipping final runs: {runs_dir} not found")

    # 3. Eval baselines
    eval_dir = ROOT_DIR / "saves" / "eval" / "SB_TOFU"
    if eval_dir.exists():
        data = collect_eval_baselines(eval_dir)
        write_json(output_dir / "eval_baselines.json", data)
    else:
        log.warning(f"Skipping eval baselines: {eval_dir} not found")

    # 4. Optuna databases
    db_files = [
        "hpsearch_bayesian.db",
        "hpsearch_lr_transfer.db",
        "hpsearch_sb_bayesian.db",
        "hpsearch_sb_bayesian_1.db",
        "hpsearch_sb_bayesian_2.db",
        "hpsearch_sb_bayesian_3.db",
    ]
    data = collect_optuna_databases(ROOT_DIR / "scripts", db_files)
    write_json(output_dir / "optuna_databases.json", data)

    log.info("Done!")


if __name__ == "__main__":
    main()
