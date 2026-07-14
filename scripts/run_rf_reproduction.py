#!/usr/bin/env python3
"""Call R from Python and compare both current and legacy RNG modes."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def find_rscript(explicit: str | None = None) -> str:
    candidates = [explicit, shutil.which("Rscript"), "/usr/bin/Rscript"]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError(
        "Rscript was not found. In Google Colab run the setup cell first; "
        "locally install R plus the randomForest and pls packages."
    )


def comparison_rows(
    reproduced: pd.DataFrame,
    supplied_current: pd.DataFrame,
    supplied_reported: pd.DataFrame,
) -> pd.DataFrame:
    aliases = {
        "summary_first_angle_raw": "summary_plus_first_angle_raw",
        "summary_first_angle_raw_residualPLS5_RF": "summary_plus_first_angle_raw_residualPLS5_RF",
    }
    rows: list[dict[str, object]] = []
    for _, row in reproduced.iterrows():
        model = row["model"]
        current = supplied_current.loc[supplied_current["model"] == model].iloc[0]
        reported = supplied_reported.loc[supplied_reported["model"] == aliases[model]].iloc[0]
        result: dict[str, object] = {
            "model": model,
            "rng_sample_kind": row["rng_sample_kind"],
            "R2_reproduced": row["R2"],
            "R2_supplied_current_run": current["R2"],
            "R2_supplied_reported": reported["R2"],
        }
        for metric in ("R2", "RMSE", "MAE"):
            result[f"abs_diff_current_{metric}"] = abs(float(row[metric]) - float(current[metric]))
            result[f"abs_diff_reported_{metric}"] = abs(float(row[metric]) - float(reported[metric]))
        rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "gpr_handoff")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--rscript", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rscript = find_rscript(args.rscript)

    base = args.data_dir / "01_base_summary_first_angle.csv"
    xproc = args.data_dir / "02_Xproc_matched.csv"
    folds = args.data_dir / "03_cv_folds_seed123.csv"
    all_metrics: list[pd.DataFrame] = []
    all_comparisons: list[pd.DataFrame] = []
    supplied_current = pd.read_csv(args.data_dir / "final_model_R_randomForest_from_python_metrics.csv")
    supplied_reported = pd.read_csv(args.data_dir / "04_reference_RF_results.csv")

    for rng_kind in ("Rejection", "Rounding"):
        suffix = rng_kind.lower()
        out_prediction = args.output_dir / f"rf_oof_{suffix}.csv"
        out_metrics = args.output_dir / f"rf_metrics_{suffix}.csv"
        command = [
            rscript,
            str(ROOT / "scripts" / "rf_reference.R"),
            str(base),
            str(xproc),
            str(folds),
            str(out_prediction),
            str(out_metrics),
            rng_kind,
        ]
        print("Running:", " ".join(command), flush=True)
        completed = subprocess.run(command, text=True, capture_output=True)
        print(completed.stdout)
        if completed.returncode != 0:
            print(completed.stderr)
            raise RuntimeError(f"R reproduction failed for sample.kind={rng_kind}")
        reproduced = pd.read_csv(out_metrics)
        all_metrics.append(reproduced)
        all_comparisons.append(comparison_rows(reproduced, supplied_current, supplied_reported))

    metrics = pd.concat(all_metrics, ignore_index=True)
    comparisons = pd.concat(all_comparisons, ignore_index=True)
    metrics.to_csv(args.output_dir / "rf_reproduction_all_metrics.csv", index=False)
    comparisons.to_csv(args.output_dir / "rf_reproduction_comparison.csv", index=False)
    print("\nReproduction comparison (zero is an exact match):")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
