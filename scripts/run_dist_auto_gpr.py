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
    DistAutoGPRConfig,
    fit_held_out_tag,
    leave_one_tag_out,
    load_dist_auto_data,
    make_grid,
    predict_grid,
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
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dist_auto_data(args.data_dir)
    config = DistAutoGPRConfig()
    model, prediction, metrics = fit_held_out_tag(data, args.test_tag, config)
    prediction.to_csv(args.output_dir / f"dist_auto_test_{args.test_tag}_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(args.output_dir / f"dist_auto_test_{args.test_tag}_metrics.csv", index=False)
    print(pd.Series(metrics).to_string())

    if not args.skip_group_cv:
        oof, group_metrics = leave_one_tag_out(data, config)
        oof.to_csv(args.output_dir / "dist_auto_leave_one_tag_out_predictions.csv", index=False)
        group_metrics.to_csv(args.output_dir / "dist_auto_leave_one_tag_out_metrics.csv", index=False)
        print("\nLeave-one-tag-out diagnostic:")
        print(group_metrics[["test_tag", "R2", "RMSE", "MAE", "coverage_95"]].to_string(index=False))

    if not args.skip_grid:
        grid, grid_features = make_grid(args.data_dir, args.test_tag, data.feature_columns, args.grid_size)
        surface = predict_grid(model, grid, grid_features)
        surface.to_csv(args.output_dir / f"dist_auto_surface_{args.test_tag}.csv", index=False)
        save_surface_html(
            surface,
            args.output_dir / f"dist_auto_surface_{args.test_tag}.html",
            f"dist_auto GPR surface: held-out tag {args.test_tag}",
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
