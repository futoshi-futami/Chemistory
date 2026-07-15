#!/usr/bin/env python3
"""Run nested structure-group selection for the next handoff models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemistory_gpr.handoff import load_handoff_data  # noqa: E402
from chemistory_gpr.nested_group import (  # noqa: E402
    default_nested_candidates,
    run_nested_group_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "gpr_handoff")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--group-scheme",
        choices=["trajectory", "proximity_level", "orientation_family", "sweep_level"],
        default="trajectory",
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    data = load_handoff_data(args.data_dir)
    candidates = default_nested_candidates()
    paths = run_nested_group_comparison(
        data,
        candidates,
        args.output_dir,
        group_scheme=args.group_scheme,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        seed=args.seed,
    )
    print("\nCandidate-wide outer OOF diagnostics")
    print(pd.read_csv(paths["candidate_metrics"]).to_string(index=False))
    print("\nStrict nested estimate")
    print(pd.read_csv(paths["nested_metrics"]).to_string(index=False))
    print("\nInner-CV winner by outer fold")
    print(pd.read_csv(paths["selections"]).to_string(index=False))
    print(
        "\nThe default trajectory group is provisional: tokens 3–5 have strong but "
        "undocumented physical associations. Confirm the generator's token map before "
        "calling this the definitive external split."
    )


if __name__ == "__main__":
    main()
