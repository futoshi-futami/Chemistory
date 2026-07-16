#!/usr/bin/env python3
"""Build angle, token, and same-split diagnostics for next handoff models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemistory_gpr.handoff import load_handoff_data  # noqa: E402
from chemistory_gpr.next_model_report import build_next_model_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "gpr_handoff")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--skip-group10-refit", action="store_true")
    parser.add_argument("--skip-group-scheme-refit", action="store_true")
    parser.add_argument("--skip-fixed10-refit", action="store_true")
    args = parser.parse_args()

    data = load_handoff_data(args.data_dir)
    paths = build_next_model_report(
        data,
        args.results_dir,
        rerun_group10=not args.skip_group10_refit,
        rerun_group_schemes=not args.skip_group_scheme_refit,
        rerun_fixed10=not args.skip_fixed10_refit,
    )
    print("Same prefix-group10 comparison")
    print(pd.read_csv(paths["group10_metrics"]).to_string(index=False))
    print("\nSupplied fixed10 interpolation comparison")
    print(pd.read_csv(paths["fixed10_metrics"]).to_string(index=False))
    print("\nInteraction-kernel components by fold")
    print(
        pd.read_csv(paths["interaction_components"])[
            [
                "group_fold",
                "axis_additive_variance",
                "environment_additive_variance",
                "interaction_variance",
                "interaction_axis_length_scale",
                "interaction_environment_length_scale",
                "noise_variance",
            ]
        ].to_string(index=False)
    )
    print("\nSensitivity to the provisional physical grouping")
    print(pd.read_csv(paths["group_scheme_metrics"]).to_string(index=False))


if __name__ == "__main__":
    main()
