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
    handoff_kernel_candidates,
    load_handoff_data,
)
from chemistory_gpr.handoff_report import build_handoff_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "gpr_handoff")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--kernel-only", action="store_true", help="Skip the two feature-ablation baselines")
    parser.add_argument(
        "--candidate",
        action="append",
        help="Run only the named candidate; repeat this option to select several",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use zero optimizer restarts for RBF-ARD instead of the original five",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_handoff_data(args.data_dir)
    ard_restarts = 0 if args.quick else 5
    candidates = (
        handoff_kernel_candidates(rbf_ard_restarts=ard_restarts)
        if args.kernel_only
        else default_handoff_candidates(rbf_ard_restarts=ard_restarts)
    )
    if args.candidate:
        requested = set(args.candidate)
        candidates = [config for config in candidates if config.name in requested]
        missing = requested - {config.name for config in candidates}
        if missing:
            raise ValueError(f"Unknown candidate(s): {sorted(missing)}")

    metric_rows: list[dict[str, object]] = []
    for config in candidates:
        print(f"Running {config.name} ...", flush=True)
        prediction, metrics = cross_validate_handoff(data, config)
        prediction.to_csv(args.output_dir / f"gpr_handoff_oof_{config.name}.csv", index=False)
        kernels = metrics.pop("kernels")
        diagnostics = metrics.pop("kernel_diagnostics")
        metrics["kernels_json"] = json.dumps(kernels, ensure_ascii=False)
        metrics["kernel_diagnostics_json"] = json.dumps(diagnostics, ensure_ascii=False)
        metric_rows.append(metrics)
        print(pd.Series(metrics).drop(labels=["kernels_json", "kernel_diagnostics_json"]).to_string())

    table = pd.DataFrame(metric_rows).sort_values("R2", ascending=False)
    table.to_csv(args.output_dir / "gpr_handoff_metrics.csv", index=False)
    print("\nComparison (higher R2, lower RMSE/MAE/NLPD):")
    print(
        table[
            [
                "model",
                "kernel_family",
                "ard",
                "R2",
                "corr2",
                "RMSE",
                "MAE",
                "coverage_95",
                "NLPD",
            ]
        ].to_string(index=False)
    )
    paths = build_handoff_report(args.data_dir, args.output_dir)
    print("\nPrimary RF-versus-GPR report:")
    print(pd.read_csv(paths["comparison"])[["rank_R2", "source", "model", "R2", "RMSE", "MAE"]].to_string(index=False))


if __name__ == "__main__":
    main()
