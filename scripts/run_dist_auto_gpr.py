#!/usr/bin/env python3
"""Fit GPR to the dist_auto tag split and create a predictive surface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemistory_gpr.dist_auto import (  # noqa: E402
    default_dist_auto_candidates,
    fit_held_out_tag,
    leave_one_tag_out,
    load_dist_auto_data,
    make_grid,
    predict_grid,
    standardized_tag_centroid_distances,
    summarize_dist_auto_metrics,
)


def save_surface_html(surface: pd.DataFrame, path: Path, title: str) -> bool:
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        print("plotly is not installed; the surface CSV was saved but HTML rendering was skipped.")
        return False

    mean_pivot = surface.pivot(index="y", columns="x", values="pred_mean")
    std_pivot = surface.pivot(index="y", columns="x", values="pred_std")
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=mean_pivot.columns.to_numpy(),
            y=mean_pivot.index.to_numpy(),
            z=mean_pivot.to_numpy(),
            surfacecolor=std_pivot.to_numpy(),
            colorbar={"title": "predictive std"},
        )
    )
    fig.update_layout(title=title, scene={"xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "prediction"})
    fig.write_html(path, include_plotlyjs="cdn")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "dist_auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--test-tag", default="10")
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-group-cv", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel held-out tags (use 1 in small Colab runtimes)")
    parser.add_argument(
        "--candidate",
        action="append",
        help="Run only the named candidate; repeat this option to select several",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use zero optimizer restarts for RBF-ARD instead of the original five",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dist_auto_data(args.data_dir)
    centroid_distances = standardized_tag_centroid_distances(data)
    centroid_distances.rename_axis("tag").reset_index().to_csv(
        args.output_dir / "dist_auto_tag_centroid_distances.csv",
        index=False,
    )
    candidates = default_dist_auto_candidates(rbf_ard_restarts=0 if args.quick else 5)
    if args.candidate:
        requested = set(args.candidate)
        candidates = [config for config in candidates if config.name in requested]
        missing = requested - {config.name for config in candidates}
        if missing:
            raise ValueError(f"Unknown candidate(s): {sorted(missing)}")
    prediction_tables: list[pd.DataFrame] = []
    metric_tables: list[pd.DataFrame] = []
    fitted_test_models = {}
    for config in candidates:
        print(f"Running {config.name} ...", flush=True)
        if args.skip_group_cv:
            fitted, prediction, metrics = fit_held_out_tag(data, args.test_tag, config)
            fitted_test_models[config.name] = fitted
            metric_table = pd.DataFrame([metrics])
        else:
            prediction, metric_table = leave_one_tag_out(data, config, n_jobs=args.n_jobs)
        prediction.insert(0, "model", config.name)
        prediction.to_csv(args.output_dir / f"dist_auto_oof_{config.name}.csv", index=False)
        prediction_tables.append(prediction)
        metric_tables.append(metric_table)

    comparison_predictions = pd.concat(prediction_tables, ignore_index=True)
    comparison_metrics = pd.concat(metric_tables, ignore_index=True)
    comparison_predictions.to_csv(
        args.output_dir / "dist_auto_kernel_comparison_predictions.csv",
        index=False,
    )
    comparison_metrics.to_csv(
        args.output_dir / "dist_auto_kernel_comparison_metrics.csv",
        index=False,
    )
    if not args.skip_group_cv:
        summarize_dist_auto_metrics(comparison_metrics).to_csv(
            args.output_dir / "dist_auto_kernel_comparison_summary.csv",
            index=False,
        )
    test_metrics = comparison_metrics.loc[comparison_metrics["test_tag"].astype(str) == str(args.test_tag)]
    print("\nHeld-out-tag comparison:")
    print(
        test_metrics[
            ["model", "kernel_family", "ard", "R2", "corr2", "RMSE", "MAE", "coverage_95"]
        ].sort_values("R2", ascending=False).to_string(index=False)
    )

    best_name = test_metrics.sort_values("R2", ascending=False).iloc[0]["model"]
    best_config = next(config for config in candidates if config.name == best_name)
    model = None
    if best_name in fitted_test_models:
        model = fitted_test_models[best_name]
        prediction = comparison_predictions.loc[
            comparison_predictions["model"] == best_name
        ].drop(columns="model").reset_index(drop=True)
        metrics = test_metrics.loc[test_metrics["model"] == best_name].iloc[0].to_dict()
    elif args.skip_grid:
        prediction = comparison_predictions.loc[
            (comparison_predictions["model"] == best_name)
            & (comparison_predictions["tag"].astype(str) == str(args.test_tag))
        ].drop(columns="model").reset_index(drop=True)
        metrics = test_metrics.loc[test_metrics["model"] == best_name].iloc[0].to_dict()
    else:
        model, prediction, metrics = fit_held_out_tag(data, args.test_tag, best_config)
    prediction.to_csv(args.output_dir / f"dist_auto_test_{args.test_tag}_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(args.output_dir / f"dist_auto_test_{args.test_tag}_metrics.csv", index=False)

    if not args.skip_grid:
        if model is None:
            raise RuntimeError("A fitted model is required to create the prediction grid")
        grid, grid_features = make_grid(args.data_dir, args.test_tag, data.feature_columns, args.grid_size)
        surface = predict_grid(model, grid, grid_features)
        surface.insert(0, "model", best_name)
        surface.to_csv(args.output_dir / f"dist_auto_surface_{args.test_tag}.csv", index=False)
        save_surface_html(
            surface,
            args.output_dir / f"dist_auto_surface_{args.test_tag}.html",
            f"dist_auto GPR surface: {best_name}, held-out tag {args.test_tag}",
        )
        best_mean = surface.loc[surface["pred_mean"].idxmax(), ["x", "y", "pred_mean", "pred_std"]]
        best_lcb = surface.loc[
            surface["lower_confidence_bound"].idxmax(),
            ["x", "y", "pred_mean", "pred_std", "lower_confidence_bound"],
        ]
        print("\nMaximum predictive mean:\n", best_mean.to_string())
        print("\nMaximum 95% lower confidence bound:\n", best_lcb.to_string())


if __name__ == "__main__":
    main()
