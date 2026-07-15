#!/usr/bin/env python3
"""Generate the interactive handoff molecule/surface/uncertainty views."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemistory_gpr.handoff import load_handoff_data  # noqa: E402
from chemistory_gpr.visualization import build_handoff_visualizations  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "gpr_handoff")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    data = load_handoff_data(args.data_dir)
    paths = build_handoff_visualizations(
        data,
        args.results_dir,
        args.figures_dir,
        seed=args.seed,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
