#!/usr/bin/env python3
"""Generate results_archive/final_runs_short.json.

Same per-run shape as final_runs.json (tofu_summary + paraphrase_summary +
paraphrase_eval + training_stats + log_history + training_config), but:

  - tofu_eval (full TOFU_EVAL.json) is dropped.
  - mia* keys are stripped from paraphrase_summary and paraphrase_eval.

Usage:
    python scripts/consolidate_final_runs_short.py
    python scripts/consolidate_final_runs_short.py --output my.json
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

from consolidate_results import (
    extract_training_config,
    extract_training_stats,
    load_json_safe,
    load_yaml_safe,
    strip_secrets,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent


def drop_mia(d):
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if not k.startswith("mia")}


def collect_one_run_short(run_dir: Path) -> dict:
    run_data: dict = {}

    para_eval_dir = run_dir / "paraphrase_evals"
    run_data["paraphrase_summary"] = drop_mia(
        load_json_safe(para_eval_dir / "Paraphrase_SUMMARY.json")
    )
    run_data["paraphrase_eval"] = drop_mia(
        load_json_safe(para_eval_dir / "Paraphrase_EVAL.json")
    )

    run_data["tofu_summary"] = None
    for ckpt_dir in sorted(run_dir.glob("checkpoint-*")):
        tofu_sum = load_json_safe(ckpt_dir / "evals" / "TOFU_SUMMARY.json")
        if tofu_sum is not None:
            run_data["tofu_summary"] = tofu_sum
            break

    trainer_state = load_json_safe(run_dir / "trainer_state.json")
    if trainer_state:
        run_data["training_stats"] = extract_training_stats(trainer_state)
        run_data["log_history"] = trainer_state.get("log_history", [])

    hydra_config = load_yaml_safe(run_dir / ".hydra" / "config.yaml")
    if hydra_config:
        run_data["training_config"] = strip_secrets(
            extract_training_config(hydra_config)
        )

    return run_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs-dir",
        default="saves/unlearn/SB_TOFU",
        help="Root dir to walk (model/split/method).",
    )
    p.add_argument(
        "--output",
        default="results_archive/final_runs_short.json",
    )
    args = p.parse_args()

    runs_dir = ROOT_DIR / args.runs_dir
    if not runs_dir.is_dir():
        raise SystemExit(f"runs dir not found: {runs_dir}")

    log.info(f"Collecting final runs from {runs_dir}")

    result: dict = {
        "metadata": {
            "description": "Final unlearning runs (short: no tofu_eval, no mia)",
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
                    collect_one_run_short(method_dir)
                )
                log.info(f"  {model_name}/{split_name}/{method_name}")

                regs_dir = method_dir / "regs"
                if regs_dir.is_dir():
                    for combo_dir in sorted(regs_dir.iterdir()):
                        if not combo_dir.is_dir():
                            continue
                        key = f"{method_name}/regs/{combo_dir.name}"
                        result["runs"][model_name][split_name][key] = (
                            collect_one_run_short(combo_dir)
                        )
                        log.info(f"  {model_name}/{split_name}/{key}")

    out_path = ROOT_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
