#!/usr/bin/env python3
"""Run the fixed-fold GPR comparison and save OOF predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemistory_gpr.handoff import (  # noqa: E402
    cross_validate_handoff,
    default_handoff_candidates,
    load_handoff_data,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "gpr_handoff")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--best-only", action="store_true", help="Run only the PCA8 candidate")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_handoff_data(args.data_dir)
    candidates = default_handoff_candidates()
    if args.best_only:
        candidates = candidates[-1:]

    metric_rows: list[dict[str, object]] = []
    for config in candidates:
        print(f"Running {config.name} ...", flush=True)
        prediction, metrics = cross_validate_handoff(data, config)
        prediction.to_csv(args.output_dir / f"gpr_handoff_oof_{config.name}.csv", index=False)
        kernels = metrics.pop("kernels")
        metrics["kernels_json"] = json.dumps(kernels, ensure_ascii=False)
        metric_rows.append(metrics)
        print(pd.Series(metrics).drop(labels="kernels_json").to_string())

    table = pd.DataFrame(metric_rows).sort_values("R2", ascending=False)
    table.to_csv(args.output_dir / "gpr_handoff_metrics.csv", index=False)
    print("\nComparison (higher R2, lower RMSE/MAE/NLPD):")
    print(table[["model", "R2", "RMSE", "MAE", "coverage_95", "NLPD"]].to_string(index=False))


if __name__ == "__main__":
    main()
