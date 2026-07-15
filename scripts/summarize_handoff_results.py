#!/usr/bin/env python3
"""Build the RF-versus-GPR primary benchmark and behavior diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemistory_gpr.handoff_report import build_handoff_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "gpr_handoff")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    for name, path in build_handoff_report(args.data_dir, args.results_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
