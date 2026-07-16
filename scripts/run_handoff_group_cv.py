#!/usr/bin/env python3
"""Run exploratory GPR validation with file-key prefix series held out."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemistory_gpr.group_validation import run_prefix_group_comparison  # noqa: E402
from chemistory_gpr.handoff import handoff_kernel_candidates, load_handoff_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "gpr_handoff")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--include-ard", action="store_true")
    args = parser.parse_args()

    data = load_handoff_data(args.data_dir)
    configs = handoff_kernel_candidates(rbf_ard_restarts=0)
    if not args.include_ard:
        configs = [config for config in configs if not config.ard]
    paths = run_prefix_group_comparison(data, configs, args.output_dir)
    table = pd.read_csv(paths["metrics"])
    print(
        table[
            [
                "model",
                "kernel_family",
                "matern_nu",
                "R2",
                "RMSE",
                "MAE",
                "coverage_95",
                "NLPD",
            ]
        ].to_string(index=False)
    )
    print(
        "\nCaution: the prefix is a candidate series id inferred from file_key. "
        "Confirm its physical meaning before treating this as the definitive external split."
    )


if __name__ == "__main__":
    main()
