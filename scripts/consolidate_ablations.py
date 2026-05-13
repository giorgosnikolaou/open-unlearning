#!/usr/bin/env python3
"""Refresh results_archive/ablation_tofu_forget10.json from on-disk evals.

Reads the existing ablation JSON to get the curated `methods` list, then
re-reads `tofu_summary`, `paraphrase_summary`, and `paraphrase_eval` for each
method from `saves/unlearn/SB_TOFU/<model>/<split>/<dir_name>/`. Writes the
file back in place with a refreshed `consolidated_at` timestamp.

Usage:
    python scripts/consolidate_ablations.py
    python scripts/consolidate_ablations.py --file results_archive/ablation_tofu_forget10.json
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception as e:
        log.warning(f"  failed to read {path}: {e}")
        return None


def drop_mia(d):
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if not k.startswith("mia")}


def collect_method(saves_dir: Path, dir_name: str) -> dict:
    run_dir = saves_dir / dir_name
    out: dict = {
        "tofu_summary": None,
        "paraphrase_summary": None,
        "paraphrase_eval": None,
    }
    if not run_dir.is_dir():
        log.warning(f"  missing dir: {run_dir}")
        return out

    for ckpt_dir in sorted(run_dir.glob("checkpoint-*")):
        tofu_sum = load_json(ckpt_dir / "evals" / "TOFU_SUMMARY.json")
        if tofu_sum is not None:
            out["tofu_summary"] = tofu_sum
            break

    para_dir = run_dir / "paraphrase_evals"
    out["paraphrase_summary"] = drop_mia(load_json(para_dir / "Paraphrase_SUMMARY.json"))
    out["paraphrase_eval"] = drop_mia(load_json(para_dir / "Paraphrase_EVAL.json"))

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--file",
        default="results_archive/ablation_tofu_forget10.json",
        help="Ablation JSON to refresh in place.",
    )
    args = p.parse_args()

    json_path = ROOT_DIR / args.file
    data = load_json(json_path)
    if data is None:
        raise SystemExit(f"Cannot read {json_path}")

    meta = data["metadata"]
    model = meta["model"]
    split = meta["forget_split"]
    saves_dir = ROOT_DIR / meta["source_dir"]
    methods = meta["methods"]

    log.info(f"Refreshing {len(methods)} methods from {saves_dir}")

    runs: dict = {model: {split: {}}}
    for m in methods:
        dir_name = m["dir_name"]
        runs[model][split][dir_name] = collect_method(saves_dir, dir_name)
        log.info(f"  {dir_name}")

    data["runs"] = runs
    data["metadata"]["consolidated_at"] = datetime.now().isoformat()

    json_path.write_text(json.dumps(data, indent=2))
    log.info(f"Wrote {json_path}")


if __name__ == "__main__":
    main()