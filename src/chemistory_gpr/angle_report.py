"""Angle-stratified diagnostics for the GPR handoff benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ANGLE_EDGES_DEG = np.array([-30.0, -10.0, 10.0, 30.0, 50.0, 70.0])
ANGLE_LABELS = ["[-30,-10)", "[-10,10)", "[10,30)", "[30,50)", "[50,70]"]
DEVIATION_EDGES_DEG = np.array([0.0, 0.25, 0.5, 3.1])
DEVIATION_LABELS = ["[0,0.25)", "[0.25,0.5)", "[0.5,3.1]"]
ANGLE_SOURCE_COLUMNS = {
    "C3H3_angle_xy",
    "C6H6_angle_xy",
    "angle_diff_C3_C6",
    "cos_angle_diff_C3_C6",
    "sin_angle_diff_C3_C6",
    "dot_C3H3_C6H6",
    "dot_xy_C3H3_C6H6",
}


def _wrap_radians(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def derive_angle_coordinates(base: pd.DataFrame) -> pd.DataFrame:
    """Replace two nearly antiparallel azimuths by axis and deviation coordinates.

    ``axis_angle_deg`` is the circular mean of C6->H6 and the reversed C3->H3
    direction. ``antiparallel_deviation_deg`` is zero for exactly opposite vectors.
    """
    theta3 = base["C3H3_angle_xy"].to_numpy(float)
    theta6 = base["C6H6_angle_xy"].to_numpy(float)
    theta3_reversed = _wrap_radians(theta3 + np.pi)
    axis = np.arctan2(
        np.sin(theta6) + np.sin(theta3_reversed),
        np.cos(theta6) + np.cos(theta3_reversed),
    )
    separation = np.abs(_wrap_radians(theta3 - theta6))
    deviation = np.abs(np.pi - separation)
    result = pd.DataFrame(
        {
            "file_key": base["file_key"].astype(str),
            "y": base["y"].to_numpy(float),
            "C3H3_angle_deg_raw": np.degrees(theta3),
            "C3H3_reversed_deg": np.degrees(theta3_reversed),
            "C6H6_angle_deg": np.degrees(theta6),
            "axis_angle_deg": np.degrees(axis),
            "antiparallel_deviation_deg": np.degrees(deviation),
            "input_angle_diff_deg": np.degrees(base["angle_diff_C3_C6"].to_numpy(float)),
        }
    )
    for column in ("C3H3_reversed_deg", "C6H6_angle_deg", "axis_angle_deg"):
        result[f"{column}_bin"] = pd.cut(
            result[column],
            ANGLE_EDGES_DEG,
            labels=ANGLE_LABELS,
            include_lowest=True,
            right=False,
        ).astype("string")
    result["antiparallel_deviation_deg_bin"] = pd.cut(
        result["antiparallel_deviation_deg"],
        DEVIATION_EDGES_DEG,
        labels=DEVIATION_LABELS,
        include_lowest=True,
        right=False,
    ).astype("string")
    if result.filter(like="_bin").isna().any().any():
        raise ValueError("At least one handoff angle falls outside the documented bins")
    return result


def _method_predictions(data_dir: Path, results_dir: Path, angles: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.read_csv(results_dir / "gpr_handoff_metrics.csv")
    rows: list[pd.DataFrame] = []
    for item in metrics.itertuples(index=False):
        prediction = pd.read_csv(results_dir / f"gpr_handoff_oof_{item.model}.csv")
        frame = angles.merge(prediction, on=["file_key", "y"], validate="one_to_one")
        frame["model"] = str(item.model)
        frame["method_type"] = "GPR"
        frame["kernel_family"] = str(item.kernel_family)
        rows.append(frame)

    rf = pd.read_csv(data_dir / "final_model_R_randomForest_from_python_OOF_predictions.csv")
    rf_frame = angles.merge(
        rf[["file_key", "y", "pred_final"]].rename(columns={"pred_final": "pred_mean"}),
        on=["file_key", "y"],
        validate="one_to_one",
    )
    folds = pd.read_csv(data_dir / "03_cv_folds_seed123.csv").rename(
        columns={"fold_seed123": "fold"}
    )
    rf_frame = rf_frame.merge(folds, on="file_key", validate="one_to_one")
    rf_frame["pred_std"] = np.nan
    rf_frame["lower_95"] = np.nan
    rf_frame["upper_95"] = np.nan
    rf_frame["model"] = "RF_current_residualPLS5"
    rf_frame["method_type"] = "RF"
    rf_frame["kernel_family"] = "randomForest+PLS"
    rows.append(rf_frame)
    return pd.concat(rows, ignore_index=True, sort=False)


def _stratified_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    views = {
        "molecular_axis": "axis_angle_deg_bin",
        "C6_to_H6_direction": "C6H6_angle_deg_bin",
        "reversed_C3_to_H3_direction": "C3H3_reversed_deg_bin",
        "deviation_from_antiparallel": "antiparallel_deviation_deg_bin",
    }
    rows: list[dict[str, object]] = []
    for view, bin_column in views.items():
        for (angle_bin, model), frame in predictions.groupby([bin_column, "model"], sort=False):
            residual = frame["y"] - frame["pred_mean"]
            inside = (
                (frame["y"] >= frame["lower_95"]) & (frame["y"] <= frame["upper_95"])
                if frame["method_type"].iloc[0] == "GPR"
                else pd.Series(False, index=frame.index)
            )
            rows.append(
                {
                    "angle_view": view,
                    "angle_bin": str(angle_bin),
                    "model": str(model),
                    "method_type": frame["method_type"].iloc[0],
                    "kernel_family": frame["kernel_family"].iloc[0],
                    "n": int(len(frame)),
                    "y_mean": float(frame["y"].mean()),
                    "y_std": float(frame["y"].std(ddof=1)),
                    "R2_within_bin": float(r2_score(frame["y"], frame["pred_mean"])),
                    "RMSE": float(np.sqrt(mean_squared_error(frame["y"], frame["pred_mean"]))),
                    "MAE": float(mean_absolute_error(frame["y"], frame["pred_mean"])),
                    "bias_y_minus_pred": float(residual.mean()),
                    "coverage_95": float(inside.mean())
                    if frame["method_type"].iloc[0] == "GPR"
                    else np.nan,
                    "mean_pred_std": float(frame["pred_std"].mean())
                    if frame["method_type"].iloc[0] == "GPR"
                    else np.nan,
                }
            )
    table = pd.DataFrame(rows)
    table["rank_RMSE_within_angle_bin"] = table.groupby(["angle_view", "angle_bin"])[
        "RMSE"
    ].rank(method="min")
    table["rank_MAE_within_angle_bin"] = table.groupby(["angle_view", "angle_bin"])[
        "MAE"
    ].rank(method="min")
    table["SSE"] = table["n"] * table["RMSE"] ** 2
    table["SSE_share_within_view_model"] = table["SSE"] / table.groupby(
        ["angle_view", "model"]
    )["SSE"].transform("sum")
    return table


def _structural_feature_diagnostics(
    base: pd.DataFrame, angles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe structure-angle association and the high-angle low-response branch.

    These are exploratory, univariate summaries. Angle-derived columns are excluded
    so that the first table ranks Mg/O environment descriptors rather than tautological
    angle features. The response threshold is descriptive and is not used to train a
    model.
    """
    numeric = base.drop(columns=["file_key", "y", *ANGLE_SOURCE_COLUMNS]).select_dtypes(
        include=[np.number]
    )
    association_rows: list[dict[str, object]] = []
    for feature in numeric.columns:
        values = numeric[feature]
        if values.nunique(dropna=True) < 2:
            continue
        association_rows.append(
            {
                "feature": feature,
                "n_unique": int(values.nunique(dropna=True)),
                "spearman_vs_axis_angle": values.corr(
                    angles["axis_angle_deg"], method="spearman"
                ),
                "spearman_vs_y_all_samples": values.corr(base["y"], method="spearman"),
            }
        )
    associations = pd.DataFrame(association_rows)
    associations["abs_spearman_vs_axis_angle"] = associations[
        "spearman_vs_axis_angle"
    ].abs()
    associations = associations.sort_values(
        ["abs_spearman_vs_axis_angle", "feature"], ascending=[False, True]
    )

    high_angle = angles["axis_angle_deg_bin"].eq("[50,70]")
    low_response = base["y"].lt(30.0)
    branch_rows: list[dict[str, object]] = []
    for feature in numeric.columns:
        all_values = numeric[feature]
        values = all_values.loc[high_angle]
        if values.nunique(dropna=True) < 2:
            continue
        global_sd = all_values.std(ddof=1)
        low_mean = values.loc[low_response].mean()
        other_mean = values.loc[~low_response].mean()
        branch_rows.append(
            {
                "feature": feature,
                "n_high_angle": int(high_angle.sum()),
                "n_high_angle_y_below_30": int((high_angle & low_response).sum()),
                "mean_y_below_30": low_mean,
                "mean_y_at_least_30": other_mean,
                "global_sd": global_sd,
                "standardized_mean_difference_low_minus_other": (
                    (low_mean - other_mean) / global_sd
                    if pd.notna(global_sd) and global_sd > 0
                    else np.nan
                ),
                "spearman_vs_y_within_high_angle": values.corr(
                    base.loc[high_angle, "y"], method="spearman"
                ),
            }
        )
    contrasts = pd.DataFrame(branch_rows)
    contrasts["abs_standardized_mean_difference"] = contrasts[
        "standardized_mean_difference_low_minus_other"
    ].abs()
    contrasts = contrasts.sort_values(
        ["abs_standardized_mean_difference", "feature"], ascending=[False, True]
    )
    return associations, contrasts


def _winner_table(metrics: pd.DataFrame, global_best_gpr: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (view, angle_bin), frame in metrics.groupby(["angle_view", "angle_bin"], sort=False):
        best = frame.loc[frame["RMSE"].idxmin()]
        best_gpr = frame.loc[frame["method_type"] == "GPR"].sort_values("RMSE").iloc[0]
        global_gpr = frame.loc[frame["model"] == global_best_gpr].iloc[0]
        rf = frame.loc[frame["method_type"] == "RF"].iloc[0]
        rows.append(
            {
                "angle_view": view,
                "angle_bin": angle_bin,
                "n": int(best["n"]),
                "y_mean": best["y_mean"],
                "y_std": best["y_std"],
                "best_overall_model": best["model"],
                "best_overall_RMSE": best["RMSE"],
                "best_GPR_model": best_gpr["model"],
                "best_GPR_RMSE": best_gpr["RMSE"],
                "global_best_GPR_RMSE": global_gpr["RMSE"],
                "RF_RMSE": rf["RMSE"],
                "global_best_GPR_RMSE_reduction_vs_RF_pct": 100.0
                * (rf["RMSE"] - global_gpr["RMSE"])
                / rf["RMSE"],
            }
        )
    return pd.DataFrame(rows)


def _training_angle_coverage(
    angles: pd.DataFrame,
    data_dir: Path,
    best_prediction: pd.DataFrame,
    rf_prediction: pd.DataFrame,
) -> pd.DataFrame:
    folds = pd.read_csv(data_dir / "03_cv_folds_seed123.csv")
    frame = angles.merge(folds, on="file_key", validate="one_to_one")
    axis_rad = np.radians(frame["axis_angle_deg"].to_numpy(float))
    deviation = frame["antiparallel_deviation_deg"].to_numpy(float)
    nearest_axis = np.full(len(frame), np.nan)
    within_five = np.zeros(len(frame), dtype=int)
    nearest_deviation = np.full(len(frame), np.nan)
    for i, fold in enumerate(frame["fold_seed123"].to_numpy(int)):
        train = frame["fold_seed123"].to_numpy(int) != fold
        circular = np.abs(_wrap_radians(axis_rad[i] - axis_rad[train]))
        circular_deg = np.degrees(circular)
        nearest_axis[i] = circular_deg.min()
        within_five[i] = int(np.sum(circular_deg <= 5.0))
        nearest_deviation[i] = np.min(np.abs(deviation[i] - deviation[train]))
    frame["nearest_train_axis_angle_deg"] = nearest_axis
    frame["train_count_within_5deg"] = within_five
    frame["nearest_train_antiparallel_deviation_deg"] = nearest_deviation
    frame = frame.merge(
        best_prediction[["file_key", "pred_mean", "pred_std"]].rename(
            columns={"pred_mean": "best_gpr_pred", "pred_std": "best_gpr_std"}
        ),
        on="file_key",
        validate="one_to_one",
    )
    frame = frame.merge(
        rf_prediction[["file_key", "pred_mean"]].rename(columns={"pred_mean": "rf_pred"}),
        on="file_key",
        validate="one_to_one",
    )
    frame["best_gpr_abs_error"] = (frame["y"] - frame["best_gpr_pred"]).abs()
    frame["rf_abs_error"] = (frame["y"] - frame["rf_pred"]).abs()
    return frame


def build_handoff_angle_report(data_dir: str | Path, results_dir: str | Path) -> dict[str, Path]:
    """Create angle-stratified RF/GPR metrics and training-coverage diagnostics."""
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    base = pd.read_csv(data_dir / "01_base_summary_first_angle.csv")
    angles = derive_angle_coordinates(base)
    angle_path = results_dir / "gpr_handoff_angle_features.csv"
    angles.to_csv(angle_path, index=False)

    predictions = _method_predictions(data_dir, results_dir, angles)
    metrics = _stratified_metrics(predictions)
    metrics_path = results_dir / "gpr_handoff_angle_method_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    global_best_gpr = (
        pd.read_csv(results_dir / "gpr_handoff_metrics.csv").sort_values("R2", ascending=False).iloc[0][
            "model"
        ]
    )
    winners = _winner_table(metrics, str(global_best_gpr))
    winners_path = results_dir / "gpr_handoff_angle_winners.csv"
    winners.to_csv(winners_path, index=False)

    axis_summary = (
        angles.groupby("axis_angle_deg_bin", sort=False)
        .agg(
            n=("y", "size"),
            y_mean=("y", "mean"),
            y_std=("y", "std"),
            y_min=("y", "min"),
            y_max=("y", "max"),
            fraction_y_below_30=("y", lambda x: float(np.mean(x < 30.0))),
            axis_mean_deg=("axis_angle_deg", "mean"),
        )
        .reset_index()
    )
    axis_summary_path = results_dir / "gpr_handoff_axis_angle_data_summary.csv"
    axis_summary.to_csv(axis_summary_path, index=False)

    associations, contrasts = _structural_feature_diagnostics(base, angles)
    associations_path = results_dir / "gpr_handoff_axis_feature_associations.csv"
    associations.to_csv(associations_path, index=False)
    contrasts_path = results_dir / "gpr_handoff_high_angle_structural_contrasts.csv"
    contrasts.to_csv(contrasts_path, index=False)

    best_prediction = predictions.loc[predictions["model"] == global_best_gpr]
    rf_prediction = predictions.loc[predictions["method_type"] == "RF"]
    coverage = _training_angle_coverage(angles, data_dir, best_prediction, rf_prediction)
    coverage_path = results_dir / "gpr_handoff_angle_coverage_diagnostics.csv"
    coverage.to_csv(coverage_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "axis_angle_min_deg": angles["axis_angle_deg"].min(),
                "axis_angle_max_deg": angles["axis_angle_deg"].max(),
                "antiparallel_deviation_mean_deg": angles[
                    "antiparallel_deviation_deg"
                ].mean(),
                "antiparallel_deviation_max_deg": angles[
                    "antiparallel_deviation_deg"
                ].max(),
                "spearman_axis_angle_vs_y": angles["axis_angle_deg"].corr(
                    angles["y"], method="spearman"
                ),
                "spearman_antiparallel_deviation_vs_y": angles[
                    "antiparallel_deviation_deg"
                ].corr(angles["y"], method="spearman"),
                "spearman_nearest_train_axis_vs_best_gpr_abs_error": coverage[
                    "nearest_train_axis_angle_deg"
                ].corr(coverage["best_gpr_abs_error"], method="spearman"),
                "spearman_train_count_within_5deg_vs_best_gpr_abs_error": coverage[
                    "train_count_within_5deg"
                ].corr(coverage["best_gpr_abs_error"], method="spearman"),
                "high_angle_n": int(
                    angles["axis_angle_deg_bin"].eq("[50,70]").sum()
                ),
                "high_angle_y_below_30_n": int(
                    (
                        angles["axis_angle_deg_bin"].eq("[50,70]")
                        & angles["y"].lt(30.0)
                    ).sum()
                ),
            }
        ]
    )
    summary_path = results_dir / "gpr_handoff_angle_behavior_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {
        "angle_features": angle_path,
        "method_metrics": metrics_path,
        "winners": winners_path,
        "axis_data_summary": axis_summary_path,
        "axis_feature_associations": associations_path,
        "high_angle_structural_contrasts": contrasts_path,
        "coverage_diagnostics": coverage_path,
        "behavior_summary": summary_path,
    }
