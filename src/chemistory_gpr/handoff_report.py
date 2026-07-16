"""RF-versus-GPR benchmark tables and behavior diagnostics for the handoff data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _metric_row(
    *,
    source: str,
    model: str,
    r2: float,
    rmse: float,
    mae: float,
    kernel_family: str = "",
    coverage_95: float = np.nan,
    nlpd: float = np.nan,
) -> dict[str, object]:
    return {
        "source": source,
        "model": model,
        "kernel_family": kernel_family,
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "coverage_95": float(coverage_95),
        "NLPD": float(nlpd),
    }


def build_handoff_report(data_dir: str | Path, results_dir: str | Path) -> dict[str, Path]:
    """Combine RF references with GPR OOF results and diagnose the best GPR.

    All comparisons use the supplied ``fold_seed123`` assignment. The reported RF
    row is the handoff value; the rerun row records the bundled Python-to-R output.
    """
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    gpr = pd.read_csv(results_dir / "gpr_handoff_metrics.csv").sort_values("R2", ascending=False)
    if gpr.empty:
        raise ValueError("gpr_handoff_metrics.csv contains no candidates")

    reported = pd.read_csv(data_dir / "04_reference_RF_results.csv")
    reported = reported.loc[reported["model"] == "summary_plus_first_angle_raw_residualPLS5_RF"].iloc[0]
    rerun = pd.read_csv(data_dir / "final_model_R_randomForest_from_python_metrics.csv")
    rerun = rerun.loc[rerun["model"] == "summary_first_angle_raw_residualPLS5_RF"].iloc[0]

    rows = [
        _metric_row(
            source="RF_reported_reference",
            model=str(reported["model"]),
            r2=reported["R2"],
            rmse=reported["RMSE"],
            mae=reported["MAE"],
        ),
        _metric_row(
            source="RF_current_R_rerun",
            model=str(rerun["model"]),
            r2=rerun["R2"],
            rmse=rerun["RMSE"],
            mae=rerun["MAE"],
        ),
    ]
    for item in gpr.itertuples(index=False):
        rows.append(
            _metric_row(
                source="GPR_fixed_10fold",
                model=str(item.model),
                kernel_family=str(item.kernel_family),
                r2=item.R2,
                rmse=item.RMSE,
                mae=item.MAE,
                coverage_95=item.coverage_95,
                nlpd=item.NLPD,
            )
        )
    comparison = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    comparison["rank_R2"] = np.arange(1, len(comparison) + 1)
    comparison["delta_R2_vs_reported_RF"] = comparison["R2"] - float(reported["R2"])
    comparison["RMSE_reduction_vs_reported_RF_pct"] = (
        100.0 * (float(reported["RMSE"]) - comparison["RMSE"]) / float(reported["RMSE"])
    )
    comparison["MAE_reduction_vs_reported_RF_pct"] = (
        100.0 * (float(reported["MAE"]) - comparison["MAE"]) / float(reported["MAE"])
    )

    all_kernel_fold_rows: list[dict[str, float | int | str]] = []
    for item in gpr.itertuples(index=False):
        candidate_prediction = pd.read_csv(results_dir / f"gpr_handoff_oof_{item.model}.csv")
        for fold, frame in candidate_prediction.groupby("fold", sort=True):
            all_kernel_fold_rows.append(
                {
                    "model": str(item.model),
                    "kernel_family": str(item.kernel_family),
                    "fold": int(fold),
                    "R2": float(r2_score(frame["y"], frame["pred_mean"])),
                    "RMSE": float(np.sqrt(mean_squared_error(frame["y"], frame["pred_mean"]))),
                    "MAE": float(mean_absolute_error(frame["y"], frame["pred_mean"])),
                }
            )
    all_kernel_folds = pd.DataFrame(all_kernel_fold_rows)
    all_kernel_folds["rank_RMSE_within_fold"] = all_kernel_folds.groupby("fold")["RMSE"].rank(
        method="min"
    )
    win_counts = (
        all_kernel_folds.loc[all_kernel_folds["rank_RMSE_within_fold"] == 1]
        .groupby("model")
        .size()
    )
    comparison["fold_RMSE_wins_out_of_10"] = comparison["model"].map(win_counts).fillna(0).astype(int)
    comparison_path = results_dir / "gpr_handoff_primary_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    all_kernel_fold_path = results_dir / "gpr_handoff_all_kernel_fold_metrics.csv"
    all_kernel_folds.to_csv(all_kernel_fold_path, index=False)

    best = gpr.iloc[0]
    prediction_path = results_dir / f"gpr_handoff_oof_{best['model']}.csv"
    prediction = pd.read_csv(prediction_path)
    fold_rows: list[dict[str, float | int]] = []
    for fold, frame in prediction.groupby("fold", sort=True):
        inside = (frame["y"] >= frame["lower_95"]) & (frame["y"] <= frame["upper_95"])
        fold_rows.append(
            {
                "fold": int(fold),
                "n": int(len(frame)),
                "R2": float(r2_score(frame["y"], frame["pred_mean"])),
                "RMSE": float(np.sqrt(mean_squared_error(frame["y"], frame["pred_mean"]))),
                "MAE": float(mean_absolute_error(frame["y"], frame["pred_mean"])),
                "bias_y_minus_pred": float(frame["residual"].mean()),
                "mean_pred_std": float(frame["pred_std"].mean()),
                "coverage_95": float(inside.mean()),
                "max_abs_error": float(frame["residual"].abs().max()),
            }
        )
    fold_table = pd.DataFrame(fold_rows)
    fold_path = results_dir / "gpr_handoff_best_fold_metrics.csv"
    fold_table.to_csv(fold_path, index=False)

    rf_prediction = pd.read_csv(data_dir / "final_model_R_randomForest_from_python_OOF_predictions.csv")
    paired = prediction.merge(
        rf_prediction[["file_key", "y", "pred_final"]].rename(
            columns={"y": "rf_y", "pred_final": "rf_pred"}
        ),
        on="file_key",
        how="left",
        validate="one_to_one",
    )
    if paired["rf_pred"].isna().any() or not np.allclose(paired["y"], paired["rf_y"]):
        raise ValueError("RF and GPR OOF predictions are not aligned")
    paired["gpr_abs_error"] = (paired["y"] - paired["pred_mean"]).abs()
    paired["rf_abs_error"] = (paired["y"] - paired["rf_pred"]).abs()
    paired["gpr_abs_error_gain_vs_rf"] = paired["rf_abs_error"] - paired["gpr_abs_error"]
    paired["gpr_has_lower_abs_error"] = paired["gpr_abs_error"] < paired["rf_abs_error"]
    paired_path = results_dir / "gpr_handoff_best_vs_rf_predictions.csv"
    paired.to_csv(paired_path, index=False)
    largest_path = results_dir / "gpr_handoff_largest_errors.csv"
    paired.nlargest(15, "gpr_abs_error").to_csv(largest_path, index=False)

    rf_fold_rows: list[dict[str, float | int]] = []
    for fold, frame in paired.groupby("fold", sort=True):
        rf_fold_rows.append(
            {
                "fold": int(fold),
                "RF_R2": float(r2_score(frame["y"], frame["rf_pred"])),
                "RF_RMSE": float(np.sqrt(mean_squared_error(frame["y"], frame["rf_pred"]))),
                "RF_MAE": float(mean_absolute_error(frame["y"], frame["rf_pred"])),
            }
        )
    best_vs_rf_folds = fold_table.merge(pd.DataFrame(rf_fold_rows), on="fold", validate="one_to_one")
    best_vs_rf_folds["GPR_RMSE_reduction_vs_RF_pct"] = (
        100.0 * (best_vs_rf_folds["RF_RMSE"] - best_vs_rf_folds["RMSE"]) / best_vs_rf_folds["RF_RMSE"]
    )
    best_vs_rf_folds["GPR_better_RMSE"] = best_vs_rf_folds["RMSE"] < best_vs_rf_folds["RF_RMSE"]
    best_vs_rf_fold_path = results_dir / "gpr_handoff_best_vs_rf_fold_metrics.csv"
    best_vs_rf_folds.to_csv(best_vs_rf_fold_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "best_gpr_model": best["model"],
                "best_gpr_R2": best["R2"],
                "reported_RF_R2": reported["R2"],
                "delta_R2_vs_reported_RF": best["R2"] - reported["R2"],
                "RMSE_reduction_vs_reported_RF_pct": 100.0
                * (reported["RMSE"] - best["RMSE"])
                / reported["RMSE"],
                "MAE_reduction_vs_reported_RF_pct": 100.0
                * (reported["MAE"] - best["MAE"])
                / reported["MAE"],
                "fraction_samples_gpr_lower_abs_error_than_rf": paired[
                    "gpr_has_lower_abs_error"
                ].mean(),
                "spearman_pred_std_vs_abs_error": paired["pred_std"].corr(
                    paired["gpr_abs_error"], method="spearman"
                ),
                "worst_fold_by_RMSE": int(fold_table.loc[fold_table["RMSE"].idxmax(), "fold"]),
                "worst_fold_RMSE": float(fold_table["RMSE"].max()),
                "best_fold_by_RMSE": int(fold_table.loc[fold_table["RMSE"].idxmin(), "fold"]),
                "best_fold_RMSE": float(fold_table["RMSE"].min()),
            }
        ]
    )
    summary_path = results_dir / "gpr_handoff_behavior_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {
        "comparison": comparison_path,
        "all_kernel_fold_metrics": all_kernel_fold_path,
        "fold_metrics": fold_path,
        "best_vs_rf_fold_metrics": best_vs_rf_fold_path,
        "paired_predictions": paired_path,
        "largest_errors": largest_path,
        "behavior_summary": summary_path,
    }
