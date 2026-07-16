"""Reports and same-split comparisons for the next handoff models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .angle_report import derive_angle_coordinates
from .handoff import HandoffData
from .metrics import gaussian_regression_metrics, regression_metrics
from .nested_group import (
    NestedCandidate,
    build_nested_model,
    default_nested_candidates,
)
from .physical_features import candidate_group_labels, parse_file_key_tokens


def _metrics_with_optional_std(y, mean, std):
    if np.isfinite(std).all():
        return gaussian_regression_metrics(y, mean, std)
    result = regression_metrics(y, mean)
    result.update({"coverage_95": np.nan, "width_95": np.nan, "NLPD": np.nan})
    return result


def _interaction_components(model) -> dict[str, float | str]:
    fitted = model.gpr_.kernel_
    axis_component = fitted.k1.k1.k1
    environment_component = fitted.k1.k1.k2
    interaction_component = fitted.k1.k2
    return {
        "axis_additive_variance": float(axis_component.k1.constant_value),
        "axis_additive_length_scale": float(axis_component.k2.base_kernel.length_scale),
        "environment_additive_variance": float(environment_component.k1.constant_value),
        "environment_additive_length_scale": float(
            environment_component.k2.base_kernel.length_scale
        ),
        "interaction_variance": float(interaction_component.k1.k1.constant_value),
        "interaction_axis_length_scale": float(
            interaction_component.k1.k2.base_kernel.length_scale
        ),
        "interaction_environment_length_scale": float(
            interaction_component.k2.base_kernel.length_scale
        ),
        "noise_variance": float(fitted.k2.noise_level),
        "log_marginal_likelihood": float(model.gpr_.log_marginal_likelihood_value_),
        "optimized_kernel": str(fitted),
    }


def _same_group10_benchmark(
    data: HandoffData,
    results_dir: Path,
) -> tuple[Path, Path, Path]:
    folds = pd.read_csv(results_dir / "gpr_handoff_group10_prefix_folds.csv")
    fold_by_key = folds.set_index("file_key")["group_fold"]
    fold_id = pd.Series(data.file_key).map(fold_by_key).to_numpy(int)
    candidate_names = {
        "legacy_angles_global_matern32",
        "compact_axis_global_matern32",
        "compact_axis_global_rational_quadratic",
        "axis_plus_environment_matern32",
        "axis_environment_interaction_matern32",
        "moe_matern32_matern12_gate40",
        "moe_matern32_rf_gate40",
    }
    candidates = [
        item for item in default_nested_candidates() if item.name in candidate_names
    ]
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    component_rows: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(candidates):
        mean = np.full(len(data.y), np.nan)
        std = np.full(len(data.y), np.nan)
        for fold in sorted(np.unique(fold_id)):
            train = fold_id != fold
            test = ~train
            model = build_nested_model(candidate, 123 + 100 * candidate_index + int(fold)).fit(
                data.base.loc[train], data.xproc.loc[train], data.y[train]
            )
            mean[test], std[test] = model.predict(
                data.base.loc[test], data.xproc.loc[test]
            )
            if candidate.name == "axis_environment_interaction_matern32":
                component_rows.append(
                    {"candidate": candidate.name, "group_fold": int(fold), **_interaction_components(model)}
                )
        metric_rows.append(
            {
                "candidate": candidate.name,
                "evaluation_split": "same_prefix_group10_as_previous_report",
                **_metrics_with_optional_std(data.y, mean, std),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "candidate": candidate.name,
                    "file_key": data.file_key,
                    "group_fold": fold_id,
                    "y": data.y,
                    "pred_mean": mean,
                    "pred_std": std,
                    "residual": data.y - mean,
                }
            )
        )
    metrics_path = results_dir / "gpr_handoff_group10_next_models_metrics.csv"
    predictions_path = results_dir / "gpr_handoff_group10_next_models_predictions.csv"
    components_path = results_dir / "gpr_handoff_interaction_kernel_components.csv"
    pd.DataFrame(metric_rows).sort_values("RMSE").to_csv(metrics_path, index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(predictions_path, index=False)
    pd.DataFrame(component_rows).to_csv(components_path, index=False)
    return metrics_path, predictions_path, components_path


def _fixed10_benchmark(
    data: HandoffData,
    results_dir: Path,
) -> tuple[Path, Path]:
    """Compare new candidates on the supplied RF-corresponding interpolation folds."""
    candidate_names = {
        "compact_axis_global_matern32",
        "compact_axis_global_rational_quadratic",
        "axis_plus_environment_matern32",
        "axis_environment_interaction_matern32",
        "moe_matern32_matern12_gate40",
        "moe_matern32_rf_gate40",
    }
    candidates = [
        item for item in default_nested_candidates() if item.name in candidate_names
    ]
    fold_id = data.fold_id
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for candidate_index, candidate in enumerate(candidates):
        mean = np.full(len(data.y), np.nan)
        std = np.full(len(data.y), np.nan)
        for fold in sorted(np.unique(fold_id)):
            train = fold_id != fold
            test = ~train
            model = build_nested_model(
                candidate, 4123 + 100 * candidate_index + int(fold)
            ).fit(data.base.loc[train], data.xproc.loc[train], data.y[train])
            mean[test], std[test] = model.predict(
                data.base.loc[test], data.xproc.loc[test]
            )
        metric_rows.append(
            {
                "candidate": candidate.name,
                "evaluation_split": "supplied_fixed10_series_internal_interpolation",
                **_metrics_with_optional_std(data.y, mean, std),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "candidate": candidate.name,
                    "file_key": data.file_key,
                    "fold": fold_id,
                    "y": data.y,
                    "pred_mean": mean,
                    "pred_std": std,
                    "residual": data.y - mean,
                }
            )
        )
    metrics_path = results_dir / "gpr_handoff_fixed10_next_models_metrics.csv"
    predictions_path = results_dir / "gpr_handoff_fixed10_next_models_predictions.csv"
    pd.DataFrame(metric_rows).sort_values("RMSE").to_csv(metrics_path, index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(predictions_path, index=False)
    return metrics_path, predictions_path


def _group_scheme_benchmark(
    data: HandoffData,
    results_dir: Path,
) -> tuple[Path, Path]:
    """Show how the scientific question changes when a different token is unknown."""
    candidate_names = {
        "compact_axis_global_rational_quadratic",
        "axis_environment_interaction_matern32",
    }
    candidates = [
        item for item in default_nested_candidates() if item.name in candidate_names
    ]
    scheme_splits = {
        "trajectory": 5,
        "proximity_level": 4,
        "orientation_family": 6,
        "sweep_level": 5,
    }
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for scheme_index, (scheme, n_splits) in enumerate(scheme_splits.items()):
        groups = candidate_group_labels(data.file_key, scheme)
        splitter = GroupKFold(n_splits=n_splits)
        for candidate_index, candidate in enumerate(candidates):
            mean = np.full(len(data.y), np.nan)
            std = np.full(len(data.y), np.nan)
            fold_id = np.zeros(len(data.y), dtype=int)
            for fold, (train, test) in enumerate(
                splitter.split(data.base, groups=groups), start=1
            ):
                model = build_nested_model(
                    candidate,
                    9123 + 1000 * scheme_index + 100 * candidate_index + fold,
                ).fit(data.base.iloc[train], data.xproc.iloc[train], data.y[train])
                mean[test], std[test] = model.predict(
                    data.base.iloc[test], data.xproc.iloc[test]
                )
                fold_id[test] = fold
            metric_rows.append(
                {
                    "group_scheme": scheme,
                    "n_groups": int(groups.nunique()),
                    "n_splits": n_splits,
                    "candidate": candidate.name,
                    **_metrics_with_optional_std(data.y, mean, std),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "group_scheme": scheme,
                        "candidate": candidate.name,
                        "file_key": data.file_key,
                        "group": groups,
                        "fold": fold_id,
                        "y": data.y,
                        "pred_mean": mean,
                        "pred_std": std,
                    }
                )
            )
    metrics_path = results_dir / "gpr_handoff_group_scheme_model_metrics.csv"
    predictions_path = results_dir / "gpr_handoff_group_scheme_model_predictions.csv"
    pd.DataFrame(metric_rows).sort_values(["group_scheme", "RMSE"]).to_csv(
        metrics_path, index=False
    )
    pd.concat(prediction_rows, ignore_index=True).to_csv(predictions_path, index=False)
    return metrics_path, predictions_path


def _angle_metrics(predictions: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    angles = derive_angle_coordinates(base)[
        ["file_key", "axis_angle_deg", "axis_angle_deg_bin"]
    ]
    joined = predictions.merge(angles, on="file_key", validate="many_to_one")
    rows: list[dict[str, object]] = []
    for (candidate, angle_bin), frame in joined.groupby(
        ["candidate", "axis_angle_deg_bin"], sort=False
    ):
        rows.append(
            {
                "candidate": candidate,
                "angle_bin": str(angle_bin),
                **regression_metrics(frame["y"].to_numpy(), frame["pred_mean"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def _fold_metrics(predictions: pd.DataFrame, fold_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, fold), frame in predictions.groupby(["candidate", fold_column]):
        rows.append(
            {
                "candidate": candidate,
                "fold": int(fold),
                **regression_metrics(frame["y"].to_numpy(), frame["pred_mean"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def _token_level_summary(base: pd.DataFrame) -> pd.DataFrame:
    tokens = parse_file_key_tokens(base["file_key"])
    angles = derive_angle_coordinates(base)
    joined = pd.concat(
        [
            tokens,
            angles[
                [
                    "axis_angle_deg",
                    "axis_abs_elevation_deg_proxy",
                    "antiparallel_deviation_deg",
                ]
            ],
            base[
                [
                    "y",
                    "Mg_H3_count_d5",
                    "O_H3_count_d5",
                    "Mg_H6_count_d5",
                    "O_H6_count_d5",
                    "first_Mg_H3_d5",
                    "first_O_H3_d5",
                ]
            ],
        ],
        axis=1,
    )
    rows: list[dict[str, object]] = []
    summary_columns = [
        "y",
        "axis_angle_deg",
        "axis_abs_elevation_deg_proxy",
        "antiparallel_deviation_deg",
        "Mg_H3_count_d5",
        "O_H3_count_d5",
        "Mg_H6_count_d5",
        "O_H6_count_d5",
        "first_Mg_H3_d5",
        "first_O_H3_d5",
    ]
    for token in (3, 4, 5):
        column = f"token_{token}"
        for level, frame in joined.groupby(column, sort=True):
            row: dict[str, object] = {
                "token_position": token,
                "token_value": int(level),
                "n": int(len(frame)),
            }
            for summary_column in summary_columns:
                row[f"{summary_column}_mean"] = float(frame[summary_column].mean())
                row[f"{summary_column}_std"] = float(frame[summary_column].std(ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def build_next_model_report(
    data: HandoffData,
    results_dir: str | Path,
    *,
    rerun_group10: bool = True,
    rerun_group_schemes: bool = True,
    rerun_fixed10: bool = True,
) -> dict[str, Path]:
    """Build diagnostics for nested selection and a same-group10 comparison."""
    results_dir = Path(results_dir)
    if rerun_group10:
        group10_metrics, group10_predictions, components = _same_group10_benchmark(
            data, results_dir
        )
    else:
        group10_metrics = results_dir / "gpr_handoff_group10_next_models_metrics.csv"
        group10_predictions = results_dir / "gpr_handoff_group10_next_models_predictions.csv"
        components = results_dir / "gpr_handoff_interaction_kernel_components.csv"

    if rerun_fixed10:
        fixed10_metrics, fixed10_predictions = _fixed10_benchmark(data, results_dir)
    else:
        fixed10_metrics = results_dir / "gpr_handoff_fixed10_next_models_metrics.csv"
        fixed10_predictions = results_dir / "gpr_handoff_fixed10_next_models_predictions.csv"

    if rerun_group_schemes:
        group_scheme_metrics, group_scheme_predictions = _group_scheme_benchmark(
            data, results_dir
        )
    else:
        group_scheme_metrics = results_dir / "gpr_handoff_group_scheme_model_metrics.csv"
        group_scheme_predictions = (
            results_dir / "gpr_handoff_group_scheme_model_predictions.csv"
        )

    nested_candidate_predictions = pd.read_csv(
        results_dir / "gpr_handoff_nested_group_candidate_predictions.csv"
    )
    nested_predictions = pd.read_csv(
        results_dir / "gpr_handoff_nested_group_predictions.csv"
    ).rename(columns={"selected_candidate": "candidate"})
    group10_prediction_frame = pd.read_csv(group10_predictions)
    fixed10_prediction_frame = pd.read_csv(fixed10_predictions)

    paths = {
        "group10_metrics": group10_metrics,
        "group10_predictions": group10_predictions,
        "fixed10_metrics": fixed10_metrics,
        "fixed10_predictions": fixed10_predictions,
        "interaction_components": components,
        "group_scheme_metrics": group_scheme_metrics,
        "group_scheme_predictions": group_scheme_predictions,
        "group10_angle_metrics": results_dir
        / "gpr_handoff_group10_next_models_angle_metrics.csv",
        "fixed10_angle_metrics": results_dir
        / "gpr_handoff_fixed10_next_models_angle_metrics.csv",
        "nested_candidate_angle_metrics": results_dir
        / "gpr_handoff_nested_group_candidate_angle_metrics.csv",
        "nested_candidate_fold_metrics": results_dir
        / "gpr_handoff_nested_group_candidate_fold_metrics.csv",
        "nested_selected_fold_metrics": results_dir
        / "gpr_handoff_nested_group_selected_fold_metrics.csv",
        "nested_selection_frequency": results_dir
        / "gpr_handoff_nested_group_selection_frequency.csv",
        "token_level_summary": results_dir / "gpr_handoff_file_key_token_level_summary.csv",
    }
    _angle_metrics(group10_prediction_frame, data.base).to_csv(
        paths["group10_angle_metrics"], index=False
    )
    _angle_metrics(fixed10_prediction_frame, data.base).to_csv(
        paths["fixed10_angle_metrics"], index=False
    )
    _angle_metrics(nested_candidate_predictions, data.base).to_csv(
        paths["nested_candidate_angle_metrics"], index=False
    )
    _fold_metrics(nested_candidate_predictions, "outer_fold").to_csv(
        paths["nested_candidate_fold_metrics"], index=False
    )
    _fold_metrics(nested_predictions, "outer_fold").to_csv(
        paths["nested_selected_fold_metrics"], index=False
    )
    selections = pd.read_csv(results_dir / "gpr_handoff_nested_group_selections.csv")
    (
        selections["selected_candidate"]
        .value_counts()
        .rename_axis("selected_candidate")
        .reset_index(name="outer_fold_count")
        .to_csv(paths["nested_selection_frequency"], index=False)
    )
    _token_level_summary(data.base).to_csv(paths["token_level_summary"], index=False)
    return paths
