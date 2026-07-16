#!/usr/bin/env python3
"""Evaluate the received two-stage R randomForest model by trajectory hold-out.

The received fixed 10-fold table splits rows.  This driver reuses the same R
model implementation, but replaces the fold table by deterministic GroupKFold
assignments in which the provisional ``file_key`` prefix (tokens 1--4) is kept
entirely on one side of every split.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from run_rf_reproduction import find_rscript


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {
    "summary_first_angle_raw": "RF_R_base",
    "summary_first_angle_raw_residualPLS5_RF": "RF_R_base_plus_residualPLS5",
}


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y, dtype=float) - np.asarray(prediction, dtype=float)
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    correlation = np.corrcoef(y, prediction)[0, 1]
    return {
        "R2": float(1.0 - np.sum(residual**2) / denominator),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "corr2": float(correlation**2),
        "n": int(len(y)),
    }


def load_group_fold_table(path: Path, fold_column: str, group_column: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"file_key", fold_column, group_column}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    table = table[["file_key", group_column, fold_column]].copy()
    table.columns = ["file_key", "trajectory_group_candidate", "fold"]
    leakage = table.groupby("trajectory_group_candidate")["fold"].nunique()
    if not leakage.eq(1).all():
        raise RuntimeError(f"Trajectory leakage detected in {path}")
    return table


def run_one_split(
    *,
    split_name: str,
    fold_table: pd.DataFrame,
    rscript: str,
    base_csv: Path,
    xproc_csv: Path,
    work_dir: Path,
    rng_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_csv = work_dir / f"{split_name}_folds.csv"
    prediction_csv = work_dir / f"{split_name}_predictions.csv"
    metrics_csv = work_dir / f"{split_name}_metrics.csv"
    fold_table[["file_key", "fold"]].rename(
        columns={"fold": "fold_seed123"}
    ).to_csv(fold_csv, index=False)
    command = [
        rscript,
        str(ROOT / "scripts" / "rf_reference.R"),
        str(base_csv),
        str(xproc_csv),
        str(fold_csv),
        str(prediction_csv),
        str(metrics_csv),
        rng_kind,
    ]
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    print(completed.stdout)
    if completed.returncode != 0:
        print(completed.stderr)
        raise RuntimeError(f"Trajectory RF failed for {split_name}")

    predictions = pd.read_csv(prediction_csv).merge(
        fold_table[["file_key", "trajectory_group_candidate"]],
        on="file_key",
        validate="one_to_one",
    )
    predictions.insert(0, "evaluation_split", split_name)
    metrics = pd.read_csv(metrics_csv)
    metrics.insert(0, "evaluation_split", split_name)
    metrics["model"] = metrics["model"].map(MODEL_NAMES)
    metrics["trajectory_group_count"] = int(
        fold_table["trajectory_group_candidate"].nunique()
    )
    metrics["fold_count"] = int(fold_table["fold"].nunique())
    return predictions, metrics


def summarize_by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prediction_columns = {
        "RF_R_base": "pred_base_angle",
        "RF_R_base_plus_residualPLS5": "pred_final",
    }
    for (split_name, fold), part in predictions.groupby(
        ["evaluation_split", "fold"], sort=True
    ):
        for model, column in prediction_columns.items():
            rows.append(
                {
                    "evaluation_split": split_name,
                    "fold": int(fold),
                    "model": model,
                    **regression_metrics(
                        part["y"].to_numpy(float), part[column].to_numpy(float)
                    ),
                    "trajectory_group_count": int(
                        part["trajectory_group_candidate"].nunique()
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_by_angle(predictions: pd.DataFrame, angle_csv: Path) -> pd.DataFrame:
    angles = pd.read_csv(angle_csv)[["file_key", "axis_angle_deg", "axis_angle_deg_bin"]]
    joined = predictions.merge(angles, on="file_key", validate="many_to_one")
    rows: list[dict[str, object]] = []
    prediction_columns = {
        "RF_R_base": "pred_base_angle",
        "RF_R_base_plus_residualPLS5": "pred_final",
    }
    for (split_name, angle_bin), part in joined.groupby(
        ["evaluation_split", "axis_angle_deg_bin"], sort=False, observed=True
    ):
        for model, column in prediction_columns.items():
            rows.append(
                {
                    "evaluation_split": split_name,
                    "angle_view": "molecular_axis",
                    "angle_bin": angle_bin,
                    "angle_min_deg": float(part["axis_angle_deg"].min()),
                    "angle_max_deg": float(part["axis_angle_deg"].max()),
                    "model": model,
                    **regression_metrics(
                        part["y"].to_numpy(float), part[column].to_numpy(float)
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_gp_comparison(rf_metrics: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in rf_metrics.loc[
        rf_metrics["model"].eq("RF_R_base_plus_residualPLS5")
    ].iterrows():
        rows.append(
            {
                "evaluation_split": row["evaluation_split"],
                "model": row["model"],
                "method": "received_two_stage_R_RF",
                "selection_status": "predefined_RF_model",
                **{key: row[key] for key in ("R2", "RMSE", "MAE", "n")},
            }
        )

    group5 = pd.read_csv(results_dir / "gpr_handoff_nested_group_candidate_metrics.csv")
    interaction5 = group5.loc[
        group5["candidate"].eq("axis_environment_interaction_matern32")
    ].iloc[0]
    rows.append(
        {
            "evaluation_split": "trajectory_group5",
            "model": interaction5["candidate"],
            "method": "GPR",
            "selection_status": "candidate_diagnostic_not_nested_selection",
            **{key: interaction5[key] for key in ("R2", "RMSE", "MAE", "n")},
        }
    )
    nested = pd.read_csv(results_dir / "gpr_handoff_nested_group_metrics.csv").iloc[0]
    rows.append(
        {
            "evaluation_split": "trajectory_group5",
            "model": nested["model"],
            "method": "GPR_or_MoE_selected_in_inner_group_CV",
            "selection_status": "strict_nested_estimate",
            **{key: nested[key] for key in ("R2", "RMSE", "MAE", "n")},
        }
    )
    group10 = pd.read_csv(results_dir / "gpr_handoff_group10_next_models_metrics.csv")
    interaction10 = group10.loc[
        group10["candidate"].eq("axis_environment_interaction_matern32")
    ].iloc[0]
    rows.append(
        {
            "evaluation_split": "trajectory_group10",
            "model": interaction10["candidate"],
            "method": "GPR",
            "selection_status": "predeclared_candidate_on_same_group10",
            **{key: interaction10[key] for key in ("R2", "RMSE", "MAE", "n")},
        }
    )
    return pd.DataFrame(rows).sort_values(
        ["evaluation_split", "R2"], ascending=[True, False]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "gpr_handoff")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--rscript", default=None)
    parser.add_argument("--rng-kind", default="Rejection", choices=("Rejection", "Rounding"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rscript = find_rscript(args.rscript)

    split_specs = (
        (
            "trajectory_group5",
            args.results_dir / "gpr_handoff_nested_group_outer_folds.csv",
            "outer_fold",
            "group",
        ),
        (
            "trajectory_group10",
            args.results_dir / "gpr_handoff_group10_prefix_folds.csv",
            "group_fold",
            "series_prefix_candidate",
        ),
    )
    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="chemistory_rf_trajectory_") as temporary:
        work_dir = Path(temporary)
        for split_name, fold_path, fold_column, group_column in split_specs:
            fold_table = load_group_fold_table(fold_path, fold_column, group_column)
            predictions, metrics = run_one_split(
                split_name=split_name,
                fold_table=fold_table,
                rscript=rscript,
                base_csv=args.data_dir / "01_base_summary_first_angle.csv",
                xproc_csv=args.data_dir / "02_Xproc_matched.csv",
                work_dir=work_dir,
                rng_kind=args.rng_kind,
            )
            all_predictions.append(predictions)
            all_metrics.append(metrics)

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    fold_metrics = summarize_by_fold(predictions)
    angle_metrics = summarize_by_angle(
        predictions, args.results_dir / "gpr_handoff_angle_features.csv"
    )
    comparison = build_gp_comparison(metrics, args.results_dir)

    predictions.to_csv(
        args.output_dir / "gpr_handoff_rf_trajectory_predictions.csv", index=False
    )
    metrics.to_csv(args.output_dir / "gpr_handoff_rf_trajectory_metrics.csv", index=False)
    fold_metrics.to_csv(
        args.output_dir / "gpr_handoff_rf_trajectory_fold_metrics.csv", index=False
    )
    angle_metrics.to_csv(
        args.output_dir / "gpr_handoff_rf_trajectory_angle_metrics.csv", index=False
    )
    comparison.to_csv(
        args.output_dir / "gpr_handoff_trajectory_model_comparison.csv", index=False
    )
    print("\nTrajectory-held-out RF metrics:")
    print(metrics.to_string(index=False))
    print("\nRF/GPR comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
