"""Interactive Plotly views for the handoff interaction-GP analysis.

The received handoff data contain molecular-axis directions and radial summary
features, but not the original atomic coordinates.  The direction animation in
this module is therefore explicitly a vector schematic.  A separate raw-
coordinate viewer is provided for the matching structures when they become
available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .angle_report import derive_angle_coordinates
from .handoff import HandoffData
from .nested_group import NestedCandidate, build_nested_model
from .physical_features import COMPACT_AXIS_COLUMNS, compact_axis_base


INTERACTION_CANDIDATE = "axis_environment_interaction_matern32"
DEFAULT_TRAJECTORY = "0-0-3-18"
DEFAULT_REFERENCE_FILE_KEY = "0-0-3-18-10"


def fit_full_interaction_gp(data: HandoffData, *, seed: int = 123):
    """Fit the structured interaction GP to all 170 rows for visualization.

    This fit is descriptive only.  Performance numbers must continue to come
    from out-of-fold predictions or a held-out outer group.
    """
    candidate = NestedCandidate(
        name="full_data_axis_environment_interaction_matern32",
        kind="structured_gp",
        include_interaction=True,
    )
    return build_nested_model(candidate, seed).fit(data.base, data.xproc, data.y)


def _prediction_rows(
    predictions: pd.DataFrame,
    *,
    candidate: str = INTERACTION_CANDIDATE,
) -> pd.DataFrame:
    required = {"file_key", "y", "pred_mean", "pred_std"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    selected = predictions.copy()
    if "candidate" in selected.columns:
        selected = selected.loc[selected["candidate"].eq(candidate)].copy()
    if selected.empty:
        raise ValueError(f"No prediction rows found for candidate={candidate!r}")
    if selected["file_key"].duplicated().any():
        raise ValueError("Prediction table contains duplicated file_key rows")
    return selected


def _canonical_axis_vectors(base_row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a mirror-equivalent 3-D reconstruction of the two C-H directions.

    Only the product of the two z components is available.  We choose the
    canonical sign ``z(C6->H6) >= 0``; reflecting the diagram through z=0 is
    observationally equivalent for the supplied summary features.
    """
    dot_3d = float(base_row["dot_C3H3_C6H6"])
    dot_xy = float(base_row["dot_xy_C3H3_C6H6"])
    z_abs = float(np.sqrt(np.clip(-(dot_3d - dot_xy), 0.0, 1.0)))
    xy_norm = float(np.sqrt(max(0.0, 1.0 - z_abs**2)))
    theta3 = float(base_row["C3H3_angle_xy"])
    theta6 = float(base_row["C6H6_angle_xy"])
    c3_to_h3 = np.array(
        [xy_norm * np.cos(theta3), xy_norm * np.sin(theta3), -z_abs]
    )
    c6_to_h6 = np.array(
        [xy_norm * np.cos(theta6), xy_norm * np.sin(theta6), z_abs]
    )
    axis = c6_to_h6 - c3_to_h3
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError("Could not construct a molecular-axis direction")
    return c3_to_h3, c6_to_h6, axis / norm


def _vector_trace(vector: np.ndarray, *, name: str, color: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=[0.0, float(vector[0])],
        y=[0.0, float(vector[1])],
        z=[0.0, float(vector[2])],
        mode="lines+markers+text",
        line={"color": color, "width": 8},
        marker={"color": color, "size": [4, 8]},
        text=["", name.split("→")[-1]],
        textposition="top center",
        name=name,
        hovertemplate=(
            f"{name}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>"
        ),
    )


def _axis_trace(vector: np.ndarray) -> go.Scatter3d:
    return go.Scatter3d(
        x=[-float(vector[0]), float(vector[0])],
        y=[-float(vector[1]), float(vector[1])],
        z=[-float(vector[2]), float(vector[2])],
        mode="lines",
        line={"color": "#2ca02c", "width": 5, "dash": "dash"},
        name="整理した分子軸",
        hovertemplate="整理した分子軸<extra></extra>",
    )


def molecular_axis_uncertainty_animation(
    base: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    trajectory: str | None = DEFAULT_TRAJECTORY,
    candidate: str = INTERACTION_CANDIDATE,
) -> go.Figure:
    """Animate the axis schematic and the corresponding OOF uncertainty.

    The left panel is not an atomistic structure.  It reconstructs only the two
    C-H directions, with an arbitrary but documented sign for z.  The right
    panel keeps all OOF points visible and highlights the current sample, in the
    same play/slider style used by the earlier search-path visualization.
    """
    pred = _prediction_rows(predictions, candidate=candidate)
    angles = derive_angle_coordinates(base)
    environment_columns = [
        "file_key",
        "Mg_H3_count_d5",
        "O_H3_count_d5",
        "Mg_H6_count_d5",
        "O_H6_count_d5",
        "sum_invd_LH3_d5",
        "sum_invd_LH6_d5",
    ]
    merged = (
        pred.merge(angles, on=["file_key", "y"], validate="one_to_one")
        .merge(base[environment_columns], on="file_key", validate="one_to_one")
    )
    if trajectory is None:
        selected = merged.copy()
    else:
        selected = merged.loc[merged["series_prefix_candidate"].eq(trajectory)].copy()
    if selected.empty:
        available = sorted(merged["series_prefix_candidate"].unique())
        raise ValueError(f"Unknown trajectory {trajectory!r}; examples: {available[:5]}")
    selected = selected.sort_values(
        ["series_prefix_candidate", "series_step_candidate", "file_key"]
    ).reset_index(drop=True)
    base_by_key = base.set_index("file_key")

    first = selected.iloc[0]
    row = base_by_key.loc[first["file_key"]]
    v3, v6, axis = _canonical_axis_vectors(row)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.48, 0.52],
        horizontal_spacing=0.06,
    )
    plane = np.linspace(-1.1, 1.1, 7)
    plane_x, plane_y = np.meshgrid(plane, plane)
    fig.add_trace(
        go.Surface(
            x=plane_x,
            y=plane_y,
            z=np.zeros_like(plane_x),
            showscale=False,
            opacity=0.13,
            colorscale=[[0, "#9ecae1"], [1, "#9ecae1"]],
            name="xy plane",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(_vector_trace(v3, name="C3→H3", color="#d62728"), row=1, col=1)
    fig.add_trace(_vector_trace(v6, name="C6→H6", color="#1f77b4"), row=1, col=1)
    fig.add_trace(_axis_trace(axis), row=1, col=1)

    custom = np.column_stack(
        [merged["file_key"], merged["pred_std"], np.abs(merged["y"] - merged["pred_mean"])]
    )
    fig.add_trace(
        go.Scatter(
            x=merged["y"],
            y=merged["pred_mean"],
            mode="markers",
            marker={
                "size": 7,
                "color": merged["axis_angle_deg"],
                "colorscale": "Turbo",
                "opacity": 0.45,
                "colorbar": {"title": "axis<br>azimuth (deg)", "x": 1.02},
            },
            customdata=custom,
            name="全OOF点",
            hovertemplate=(
                "%{customdata[0]}<br>observed=%{x:.3f}<br>OOF mean=%{y:.3f}"
                "<br>std=%{customdata[1]:.3f}<br>|error|=%{customdata[2]:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=selected["y"],
            y=selected["pred_mean"],
            mode="lines+markers",
            line={"color": "#7f7f7f", "width": 2},
            marker={"size": 7, "color": selected["series_step_candidate"], "colorscale": "Viridis"},
            customdata=selected[["file_key", "series_step_candidate"]],
            name="選択trajectory",
            hovertemplate=(
                "%{customdata[0]}<br>step=%{customdata[1]}<br>observed=%{x:.3f}"
                "<br>OOF mean=%{y:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    low = float(min(merged["y"].min(), merged["pred_mean"].min()))
    high = float(max(merged["y"].max(), merged["pred_mean"].max()))
    padding = 0.04 * (high - low)
    limits = [low - padding, high + padding]
    fig.add_trace(
        go.Scatter(
            x=limits,
            y=limits,
            mode="lines",
            line={"color": "#444", "dash": "dash"},
            name="1:1",
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[float(first["y"])],
            y=[float(first["pred_mean"])],
            mode="markers",
            marker={"size": 15, "color": "#ff7f0e", "symbol": "diamond", "line": {"width": 2, "color": "#333"}},
            error_y={"type": "data", "array": [1.959963984540054 * float(first["pred_std"])], "visible": True, "thickness": 3},
            name="現在点と95%区間",
            hovertemplate="observed=%{x:.3f}<br>OOF mean=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    frames: list[go.Frame] = []
    slider_steps: list[dict[str, Any]] = []
    for index, item in selected.iterrows():
        base_row = base_by_key.loc[item["file_key"]]
        item_v3, item_v6, item_axis = _canonical_axis_vectors(base_row)
        title = (
            f"{item['file_key']} | observed={item['y']:.2f}, "
            f"OOF mean={item['pred_mean']:.2f} ± {1.959963984540054 * item['pred_std']:.2f} (95%)"
            f"<br>azimuth={item['axis_angle_deg']:.2f}°, |tilt| proxy={item['axis_abs_elevation_deg_proxy']:.2f}°, "
            f"H3 Mg/O count(d5)={int(item['Mg_H3_count_d5'])}/{int(item['O_H3_count_d5'])}"
        )
        frame_name = f"sample-{index:03d}"
        frames.append(
            go.Frame(
                name=frame_name,
                traces=[1, 2, 3, 7],
                data=[
                    _vector_trace(item_v3, name="C3→H3", color="#d62728"),
                    _vector_trace(item_v6, name="C6→H6", color="#1f77b4"),
                    _axis_trace(item_axis),
                    go.Scatter(
                        x=[float(item["y"])],
                        y=[float(item["pred_mean"])],
                        mode="markers",
                        marker={"size": 15, "color": "#ff7f0e", "symbol": "diamond", "line": {"width": 2, "color": "#333"}},
                        error_y={"type": "data", "array": [1.959963984540054 * float(item["pred_std"])], "visible": True, "thickness": 3},
                    ),
                ],
                layout=go.Layout(title={"text": title, "x": 0.5}),
            )
        )
        slider_steps.append(
            {
                "label": str(item["file_key"]),
                "method": "animate",
                "args": [[frame_name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
            }
        )

    fig.frames = frames
    initial_title = frames[0].layout.title.text
    fig.update_layout(
        title={"text": initial_title, "x": 0.5},
        height=670,
        margin={"l": 25, "r": 40, "t": 105, "b": 95},
        legend={"orientation": "h", "y": -0.13, "x": 0.5, "xanchor": "center"},
        scene={
            "xaxis": {"title": "x", "range": [-1.15, 1.15]},
            "yaxis": {"title": "y", "range": [-1.15, 1.15]},
            "zaxis": {"title": "z (sign is canonical)", "range": [-1.15, 1.15]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 0.95}},
        },
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "current: "},
                "pad": {"t": 40},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.08,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 850, "redraw": True}, "fromcurrent": True, "transition": {"duration": 250}}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
    )
    fig.update_xaxes(title="Observed y", range=limits, row=1, col=2)
    fig.update_yaxes(title="OOF predictive mean", range=limits, row=1, col=2)
    return fig


def interaction_surface_table(
    data: HandoffData,
    *,
    model=None,
    reference_file_key: str = DEFAULT_REFERENCE_FILE_KEY,
    surface_feature: str = "sum_invd_LH3_d5",
    angle_points: int = 51,
    feature_points: int = 35,
    seed: int = 123,
) -> pd.DataFrame:
    """Evaluate a 2-D conditional slice of the full-data interaction GP.

    ``surface_feature`` may be an environment column, ``axis_tilt_deg`` or
    ``antiparallel_deviation_deg``.  Every unplotted coordinate is held at the
    selected observed reference row, so this is a model slice rather than a
    literal two-dimensional potential-energy surface.
    """
    if angle_points < 3 or feature_points < 3:
        raise ValueError("angle_points and feature_points must both be at least three")
    matches = np.flatnonzero(data.base["file_key"].astype(str).eq(reference_file_key))
    if len(matches) != 1:
        raise ValueError(f"reference_file_key must identify one row: {reference_file_key}")
    reference_index = int(matches[0])
    if model is None:
        model = fit_full_interaction_gp(data, seed=seed)
    if not hasattr(model, "preprocessor_") or not hasattr(model, "gpr_"):
        raise TypeError("model must be a fitted structured interaction GP")

    compact = compact_axis_base(data.base)
    reference = compact.iloc[[reference_index]].copy()
    reference_xproc = data.xproc.iloc[[reference_index]].copy()
    observed_angles = derive_angle_coordinates(data.base)
    angle_values = np.linspace(
        float(observed_angles["axis_angle_deg"].min()),
        float(observed_angles["axis_angle_deg"].max()),
        angle_points,
    )

    if surface_feature == "axis_tilt_deg":
        feature_values = np.linspace(
            float(observed_angles["axis_abs_elevation_deg_proxy"].min()),
            float(observed_angles["axis_abs_elevation_deg_proxy"].max()),
            feature_points,
        )
        compact_column = "axis_abs_elevation_rad_proxy"
        feature_label = "absolute out-of-plane tilt proxy (degree)"
        grid_kind = "axis_tilt"
    elif surface_feature == "antiparallel_deviation_deg":
        feature_values = np.linspace(
            float(observed_angles["antiparallel_deviation_deg"].min()),
            float(observed_angles["antiparallel_deviation_deg"].max()),
            feature_points,
        )
        compact_column = "antiparallel_deviation_rad"
        feature_label = "deviation from antiparallel (degree)"
        grid_kind = "axis_deviation"
    else:
        if surface_feature not in compact.columns:
            raise ValueError(f"Unknown surface feature: {surface_feature}")
        if surface_feature in COMPACT_AXIS_COLUMNS:
            raise ValueError("Use axis_tilt_deg or antiparallel_deviation_deg for axis coordinates")
        values = compact[surface_feature].to_numpy(float)
        feature_values = np.linspace(float(np.min(values)), float(np.max(values)), feature_points)
        compact_column = surface_feature
        feature_label = surface_feature
        grid_kind = "axis_environment"

    angle_grid, feature_grid = np.meshgrid(angle_values, feature_values)
    n_grid = int(angle_grid.size)
    compact_grid = pd.concat([reference] * n_grid, ignore_index=True)
    xproc_grid = pd.concat([reference_xproc] * n_grid, ignore_index=True)
    radians = np.radians(angle_grid.ravel())
    compact_grid["axis_azimuth_sin"] = np.sin(radians)
    compact_grid["axis_azimuth_cos"] = np.cos(radians)
    if compact_column in {"axis_abs_elevation_rad_proxy", "antiparallel_deviation_rad"}:
        compact_grid[compact_column] = np.radians(feature_grid.ravel())
    else:
        compact_grid[compact_column] = feature_grid.ravel()
    compact_grid["file_key"] = [f"surface-{index}" for index in range(n_grid)]
    xproc_grid["file_key"] = compact_grid["file_key"]
    design = model.preprocessor_.transform(compact_grid, xproc_grid)
    mean, std = model.gpr_.predict(design, return_std=True)
    z95 = 1.959963984540054
    reference_angles = observed_angles.iloc[reference_index]
    return pd.DataFrame(
        {
            "axis_angle_deg": angle_grid.ravel(),
            "surface_feature_value": feature_grid.ravel(),
            "pred_mean": mean,
            "pred_std": std,
            "interval_width_95": 2.0 * z95 * std,
            "lower_95": mean - z95 * std,
            "upper_95": mean + z95 * std,
            "surface_feature": surface_feature,
            "surface_feature_label": feature_label,
            "grid_kind": grid_kind,
            "reference_file_key": reference_file_key,
            "reference_tilt_deg_proxy": float(reference_angles["axis_abs_elevation_deg_proxy"]),
            "reference_deviation_deg": float(reference_angles["antiparallel_deviation_deg"]),
        }
    )


def _surface_matrix(table: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = table.pivot(
        index="surface_feature_value", columns="axis_angle_deg", values=column
    ).sort_index(axis=0).sort_index(axis=1)
    return (
        pivot.columns.to_numpy(float),
        pivot.index.to_numpy(float),
        pivot.to_numpy(float),
    )


def interaction_surface_figure(surface: pd.DataFrame, base: pd.DataFrame) -> go.Figure:
    """Plot a conditional GP slice without a misleading projected data cloud.

    The surface holds every unplotted input at ``reference_file_key``.  Other
    observations generally have different values for those hidden inputs, so
    they are not observations of this conditional slice.  Only the reference
    observation is therefore overlaid.  Goodness of fit belongs in the OOF
    observed-versus-predicted figure, not in this descriptive surface.
    """
    required = {
        "axis_angle_deg",
        "surface_feature_value",
        "pred_mean",
        "pred_std",
        "interval_width_95",
        "lower_95",
        "upper_95",
        "surface_feature",
        "surface_feature_label",
        "grid_kind",
        "reference_file_key",
    }
    missing = sorted(required.difference(surface.columns))
    if missing:
        raise ValueError(f"Surface table is missing columns: {missing}")
    metadata = surface.iloc[0]
    feature = str(metadata["surface_feature"])
    label = str(metadata["surface_feature_label"])
    grid_kind = str(metadata["grid_kind"])
    reference = str(metadata["reference_file_key"])
    reference_rows = base.loc[base["file_key"].astype(str).eq(reference)]
    if len(reference_rows) != 1:
        raise ValueError(f"reference_file_key must identify one base row: {reference}")
    reference_row = reference_rows.iloc[0]
    angles = derive_angle_coordinates(base)
    if grid_kind not in {"axis_tilt", "axis_deviation"} and feature not in base.columns:
        raise ValueError(f"Base data do not contain {feature}")

    columns = [
        ("pred_mean", "予測平均", "Viridis"),
        ("pred_std", "予測標準偏差", "Inferno"),
        ("interval_width_95", "95%予測区間幅", "Magma"),
        ("lower_95", "95%下限", "Viridis"),
        ("upper_95", "95%上限", "Viridis"),
    ]
    fig = go.Figure()
    for index, (column, title, colorscale) in enumerate(columns):
        x, y, z = _surface_matrix(surface, column)
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale=colorscale,
                colorbar={"title": title},
                visible=index == 0,
                name=title,
                hovertemplate=(
                    "azimuth=%{x:.2f}°<br>slice coordinate=%{y:.3f}"
                    f"<br>{title}=%{{z:.3f}}<extra></extra>"
                ),
            )
        )
    reference_angle_row = angles.loc[
        angles["file_key"].astype(str).eq(reference)
    ].iloc[0]
    if grid_kind == "axis_tilt":
        reference_feature = float(reference_angle_row["axis_abs_elevation_deg_proxy"])
    elif grid_kind == "axis_deviation":
        reference_feature = float(reference_angle_row["antiparallel_deviation_deg"])
    else:
        reference_feature = float(reference_row[feature])
    fig.add_trace(
        go.Scatter3d(
            x=[float(reference_angle_row["axis_angle_deg"])],
            y=[reference_feature],
            z=[float(reference_row["y"])],
            mode="markers",
            marker={
                "size": 7,
                "color": "#ff7f0e",
                "symbol": "diamond",
                "line": {"width": 1.5, "color": "#333"},
            },
            customdata=[[reference, float(reference_row["y"])]],
            name="基準試料の観測値（この面と比較可能）",
            visible=True,
            hovertemplate=(
                "%{customdata[0]}<br>azimuth=%{x:.2f}°<br>slice coordinate=%{y:.3f}"
                "<br>observed y=%{customdata[1]:.3f}<extra></extra>"
            ),
        )
    )
    buttons = []
    for index, (_, title, _) in enumerate(columns):
        visible = [False] * len(columns) + [index == 0]
        visible[index] = True
        buttons.append(
            {
                "label": title,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title.text": (
                            f"Interaction GP conditional slice — {title}<br>"
                            f"reference={reference}; unshown inputs are fixed to this sample"
                        ),
                        "scene.zaxis.title.text": title,
                        "scene.zaxis.autorange": True,
                    },
                ],
            }
        )
    fig.update_layout(
        title={
            "text": (
                f"Interaction GP conditional slice — 予測平均<br>reference={reference}; "
                "only the orange reference observation is comparable to this slice"
            ),
            "x": 0.5,
        },
        height=710,
        margin={"l": 15, "r": 35, "t": 100, "b": 20},
        scene={
            "xaxis": {"title": "molecular-axis azimuth (degree)"},
            "yaxis": {"title": label},
            "zaxis": {"title": "予測平均"},
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 0.9}},
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.0,
                "y": 1.08,
                "buttons": buttons,
            }
        ],
        legend={"x": 0.0, "y": 0.98},
    )
    return fig


def oof_uncertainty_figure(
    base: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    candidate: str = INTERACTION_CANDIDATE,
) -> go.Figure:
    """Show OOF prediction intervals and whether GP std ranks errors."""
    pred = _prediction_rows(predictions, candidate=candidate)
    angles = derive_angle_coordinates(base)
    frame = pred.merge(
        angles[["file_key", "axis_angle_deg", "axis_abs_elevation_deg_proxy"]],
        on="file_key",
        validate="one_to_one",
    )
    frame["abs_error"] = np.abs(frame["y"] - frame["pred_mean"])
    frame["covered_95"] = frame["abs_error"] <= 1.959963984540054 * frame["pred_std"]
    custom = frame[
        ["file_key", "pred_std", "abs_error", "axis_angle_deg", "covered_95"]
    ].to_numpy()
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1)
    fig.add_trace(
        go.Scatter(
            x=frame["y"],
            y=frame["pred_mean"],
            mode="markers",
            marker={
                "size": 8,
                "color": frame["axis_angle_deg"],
                "colorscale": "Turbo",
                "colorbar": {"title": "axis<br>azimuth", "x": 0.47},
            },
            error_y={
                "type": "data",
                "array": 1.959963984540054 * frame["pred_std"],
                "visible": True,
                "thickness": 1,
                "width": 0,
            },
            customdata=custom,
            name="OOF mean ±95%",
            hovertemplate=(
                "%{customdata[0]}<br>observed=%{x:.3f}<br>OOF mean=%{y:.3f}"
                "<br>std=%{customdata[1]:.3f}<br>|error|=%{customdata[2]:.3f}"
                "<br>azimuth=%{customdata[3]:.2f}°<br>covered=%{customdata[4]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    low = float(min(frame["y"].min(), frame["pred_mean"].min()))
    high = float(max(frame["y"].max(), frame["pred_mean"].max()))
    fig.add_trace(
        go.Scatter(x=[low, high], y=[low, high], mode="lines", line={"dash": "dash", "color": "#444"}, name="1:1"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["pred_std"],
            y=frame["abs_error"],
            mode="markers",
            marker={"size": 8, "color": frame["axis_angle_deg"], "colorscale": "Turbo"},
            customdata=custom,
            name="uncertainty vs error",
            hovertemplate=(
                "%{customdata[0]}<br>std=%{x:.3f}<br>|error|=%{y:.3f}"
                "<br>azimuth=%{customdata[3]:.2f}°<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    rho = float(frame["pred_std"].corr(frame["abs_error"], method="spearman"))
    coverage = float(frame["covered_95"].mean())
    fig.update_xaxes(title="Observed y", row=1, col=1)
    fig.update_yaxes(title="OOF predictive mean", row=1, col=1)
    fig.update_xaxes(title="OOF predictive std", row=1, col=2)
    fig.update_yaxes(title="absolute OOF error", row=1, col=2)
    fig.update_layout(
        title={
            "text": f"Interaction GP OOF uncertainty — 95% coverage={coverage:.3f}, Spearman(std, |error|)={rho:.3f}",
            "x": 0.5,
        },
        height=560,
        margin={"l": 45, "r": 25, "t": 80, "b": 40},
        legend={"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center"},
    )
    return fig


def raw_structure_figure(coordinates: pd.DataFrame, file_key: str) -> go.Figure:
    """Display actual atomic coordinates when a matching long table is supplied."""
    required = {"file_key", "atom_label", "element", "x", "y", "z"}
    missing = sorted(required.difference(coordinates.columns))
    if missing:
        raise ValueError(f"Coordinate table is missing columns: {missing}")
    frame = coordinates.loc[coordinates["file_key"].astype(str).eq(str(file_key))].copy()
    if frame.empty:
        raise ValueError(f"No coordinates found for file_key={file_key!r}")
    colors = {"C": "#333333", "H": "#d9d9d9", "Mg": "#2ca25f", "O": "#de2d26"}
    sizes = {"C": 9, "H": 6, "Mg": 10, "O": 9}
    fig = go.Figure()
    for element, atoms in frame.groupby("element", sort=False):
        fig.add_trace(
            go.Scatter3d(
                x=atoms["x"],
                y=atoms["y"],
                z=atoms["z"],
                mode="markers+text",
                marker={"size": sizes.get(str(element), 7), "color": colors.get(str(element), "#756bb1"), "opacity": 0.78},
                text=atoms["atom_label"],
                textposition="top center",
                customdata=atoms[["atom_label", "element"]],
                name=str(element),
                hovertemplate=(
                    "%{customdata[0]} (%{customdata[1]})<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                ),
            )
        )
    indexed = frame.set_index("atom_label")
    for start, end, color in (("C3", "H3", "#d62728"), ("C6", "H6", "#1f77b4")):
        if start in indexed.index and end in indexed.index:
            segment = indexed.loc[[start, end]]
            fig.add_trace(
                go.Scatter3d(
                    x=segment["x"],
                    y=segment["y"],
                    z=segment["z"],
                    mode="lines",
                    line={"width": 7, "color": color},
                    name=f"{start}–{end}",
                    hoverinfo="skip",
                )
            )
    fig.update_layout(
        title={"text": f"Raw-coordinate structure: {file_key}", "x": 0.5},
        height=690,
        scene={
            "xaxis": {"title": "x"},
            "yaxis": {"title": "y"},
            "zaxis": {"title": "z"},
            "aspectmode": "data",
        },
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
    )
    return fig


def build_handoff_visualizations(
    data: HandoffData,
    results_dir: str | Path,
    figures_dir: str | Path,
    *,
    seed: int = 123,
) -> dict[str, Path]:
    """Build saved surface tables and compact standalone interactive HTML files."""
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = results_dir / "gpr_handoff_fixed10_next_models_predictions.csv"
    predictions = pd.read_csv(prediction_path)
    model = fit_full_interaction_gp(data, seed=seed)
    surface_specs = {
        "axis_tilt": "axis_tilt_deg",
        "h3_environment": "sum_invd_LH3_d5",
    }
    paths: dict[str, Path] = {}
    for label, feature in surface_specs.items():
        table = interaction_surface_table(
            data,
            model=model,
            reference_file_key=DEFAULT_REFERENCE_FILE_KEY,
            surface_feature=feature,
        )
        csv_path = results_dir / f"gpr_handoff_interaction_surface_{label}.csv"
        html_path = figures_dir / f"gpr_handoff_interaction_surface_{label}.html"
        table.to_csv(csv_path, index=False)
        interaction_surface_figure(table, data.base).write_html(
            html_path, include_plotlyjs="cdn", full_html=True
        )
        paths[f"surface_{label}_csv"] = csv_path
        paths[f"surface_{label}_html"] = html_path

    animation_path = figures_dir / "gpr_handoff_molecular_axis_uncertainty_animation.html"
    molecular_axis_uncertainty_animation(data.base, predictions).write_html(
        animation_path, include_plotlyjs="cdn", full_html=True
    )
    uncertainty_path = figures_dir / "gpr_handoff_oof_uncertainty.html"
    oof_uncertainty_figure(data.base, predictions).write_html(
        uncertainty_path, include_plotlyjs="cdn", full_html=True
    )
    paths["molecular_axis_animation_html"] = animation_path
    paths["oof_uncertainty_html"] = uncertainty_path
    return paths
