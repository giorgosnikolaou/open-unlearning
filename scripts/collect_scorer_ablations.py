#!/usr/bin/env python3
"""Collect Scorer-ablation TOFU and Paraphrase summaries into one ablations.json."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = "saves/unlearn/SB_TOFU/Llama-3.2-1B-Instruct/forget10/ScorerAblations"
DEFAULT_OUT = "results_archive/ablations.json"

ABLATIONS = [
    "trace",
    "frozen-trained",
    "pretrain-frozen",
    "pretrain-unfrozen",
    "no-retain",
    "joint",
    *[f"regularizers/E{e}_P{p}_L{l}" for e in (0, 1) for p in (0, 1) for l in (0, 1)],
    *[f"update-frequency/uev{u}" for u in (1, 5, 10)],
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  warn: failed to parse {path}: {e}")
        return None


def latest_tofu_summary(run_dir: Path) -> dict | None:
    candidates = list(run_dir.glob("checkpoint-*/evals/TOFU_SUMMARY.json"))
    if not candidates:
        return load_json(run_dir / "evals" / "TOFU_SUMMARY.json")

    def step(p: Path) -> int:
        m = re.search(r"checkpoint-(\d+)", str(p))
        return int(m.group(1)) if m else -1

    return load_json(max(candidates, key=step))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    counts = {"total": len(ABLATIONS), "complete": 0, "missing_tofu": 0, "missing_paraphrase": 0}

    for rel in ABLATIONS:
        run = root / rel
        tofu = latest_tofu_summary(run)
        para = load_json(run / "paraphrase_evals" / "Paraphrase_SUMMARY.json")
        results[rel] = {"tofu": tofu, "paraphrase": para}

        if tofu is not None and para is not None:
            tag = "[OK]"
            counts["complete"] += 1
        elif tofu is None and para is None:
            tag = "[missing]"
            counts["missing_tofu"] += 1
            counts["missing_paraphrase"] += 1
        else:
            tag = "[partial]"
            if tofu is None:
                counts["missing_tofu"] += 1
            if para is None:
                counts["missing_paraphrase"] += 1
        print(f"  {tag:<10} {rel}")

    payload = {
        "_meta": {
            "root_dir": str(root.resolve()),
            "collected_at": datetime.now(timezone.utc).isoformat(),
            **counts,
        },
        **{k: results[k] for k in sorted(results)},
    }

    with out.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)

    print(
        f"\nWrote {out} — {counts['complete']}/{counts['total']} complete "
        f"(missing tofu: {counts['missing_tofu']}, missing paraphrase: {counts['missing_paraphrase']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
