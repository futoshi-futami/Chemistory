#!/usr/bin/env python3
"""Build a GitHub-renderable preview of the molecular-axis and GP surfaces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemistory_gpr.handoff import load_handoff_data  # noqa: E402
from chemistory_gpr.visualization import _canonical_axis_vectors  # noqa: E402


def _surface_arrays(table: pd.DataFrame, value: str):
    pivot = table.pivot(
        index="surface_feature_value", columns="axis_angle_deg", values=value
    ).sort_index().sort_index(axis=1)
    x, y = np.meshgrid(
        pivot.columns.to_numpy(float), pivot.index.to_numpy(float)
    )
    return x, y, pivot.to_numpy(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "gpr_handoff")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--file-key", default="0-0-3-18-10")
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_handoff_data(args.data_dir)
    row = data.base.set_index("file_key").loc[args.file_key]
    c3h3, c6h6, axis = _canonical_axis_vectors(row)
    surface = pd.read_csv(
        args.results_dir / "gpr_handoff_interaction_surface_axis_tilt.csv"
    )
    x_mean, y_mean, z_mean = _surface_arrays(surface, "pred_mean")
    x_std, y_std, z_std = _surface_arrays(surface, "pred_std")

    figure = plt.figure(figsize=(16, 5.1), constrained_layout=True)
    molecular = figure.add_subplot(1, 3, 1, projection="3d")
    grid = np.linspace(-1.05, 1.05, 2)
    plane_x, plane_y = np.meshgrid(grid, grid)
    molecular.plot_surface(
        plane_x, plane_y, np.zeros_like(plane_x), alpha=0.12, color="#9ecae1"
    )
    for vector, color, label in (
        (c3h3, "#d62728", "C3→H3"),
        (c6h6, "#1f77b4", "C6→H6"),
    ):
        molecular.quiver(
            0,
            0,
            0,
            *vector,
            color=color,
            linewidth=2.8,
            arrow_length_ratio=0.12,
            label=label,
        )
    molecular.plot(
        [-axis[0], axis[0]],
        [-axis[1], axis[1]],
        [-axis[2], axis[2]],
        linestyle="--",
        linewidth=2,
        color="#2ca02c",
        label="compact molecular axis",
    )
    molecular.set(
        xlim=(-1.1, 1.1),
        ylim=(-1.1, 1.1),
        zlim=(-1.1, 1.1),
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title=(
            f"Direction-vector schematic — {args.file_key}\n"
            "Raw atom coordinates unavailable; Mg/O not drawn"
        ),
    )
    molecular.legend(loc="upper left", fontsize=8)

    mean_axis = figure.add_subplot(1, 3, 2, projection="3d")
    mean_surface = mean_axis.plot_surface(
        x_mean, y_mean, z_mean, cmap="viridis", linewidth=0, antialiased=True
    )
    mean_axis.set(
        xlabel="Axis azimuth (deg)",
        ylabel="Absolute elevation proxy (deg)",
        zlabel="Predicted mean",
        title="Product GP conditional mean",
    )
    figure.colorbar(mean_surface, ax=mean_axis, shrink=0.58, pad=0.08)

    std_axis = figure.add_subplot(1, 3, 3, projection="3d")
    std_surface = std_axis.plot_surface(
        x_std, y_std, z_std, cmap="magma", linewidth=0, antialiased=True
    )
    std_axis.set(
        xlabel="Axis azimuth (deg)",
        ylabel="Absolute elevation proxy (deg)",
        zlabel="Predictive std",
        title="Product GP uncertainty",
    )
    figure.colorbar(std_surface, ax=std_axis, shrink=0.58, pad=0.08)
    figure.suptitle(
        "GPR_handoff: molecular-axis schematic and conditional Product-GP surfaces",
        fontsize=14,
    )

    output = args.figures_dir / "gpr_handoff_static_overview.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
