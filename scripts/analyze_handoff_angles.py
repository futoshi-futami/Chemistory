#!/usr/bin/env python3
"""Build angle-stratified RF/GPR diagnostics for GPR_handoff."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemistory_gpr.angle_report import build_handoff_angle_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "gpr_handoff")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    paths = build_handoff_angle_report(args.data_dir, args.results_dir)
    print(pd.read_csv(paths["behavior_summary"]).to_string(index=False))
    print("\nAngle-bin winners (exploratory):")
    print(pd.read_csv(paths["winners"]).to_string(index=False))
    print("\nStrongest non-angle feature associations with the molecular axis:")
    print(pd.read_csv(paths["axis_feature_associations"]).head(10).to_string(index=False))
    print("\nHigh-angle low-response structural contrasts (exploratory):")
    print(pd.read_csv(paths["high_angle_structural_contrasts"]).head(10).to_string(index=False))
    print("\nCandidate file-key series containing the high-angle low-response branch:")
    series = pd.read_csv(paths["series_summary"])
    print(series.loc[series["n_high_angle_y_below_30"] > 0].to_string(index=False))


if __name__ == "__main__":
    main()
