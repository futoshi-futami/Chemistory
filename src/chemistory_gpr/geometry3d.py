"""Rotation-invariant 3-D features for future raw handoff structures."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COORDINATE_COLUMNS = {"file_key", "atom_label", "element", "x", "y", "z"}
ANCHOR_LABELS = ("C3", "H3", "C6", "H6")


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("A C-H anchor vector has zero length")
    return vector / norm


def _anchor(group: pd.DataFrame, label: str) -> np.ndarray:
    rows = group.loc[group["atom_label"].eq(label), ["x", "y", "z"]]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {label} row per structure; found {len(rows)}")
    return rows.iloc[0].to_numpy(float)


def _summarize_element(
    element: str,
    atoms: np.ndarray,
    center: np.ndarray,
    h3: np.ndarray,
    h6: np.ndarray,
    axis: np.ndarray,
    cutoff: float,
) -> dict[str, float | int]:
    prefix = element
    relative_center = atoms - center
    projection = relative_center @ axis
    perpendicular = np.linalg.norm(relative_center - np.outer(projection, axis), axis=1)
    min_perpendicular_index = int(np.argmin(perpendicular))
    result: dict[str, float | int] = {
        f"{prefix}_atom_count": int(len(atoms)),
        f"{prefix}_axis_perpendicular_min": float(perpendicular[min_perpendicular_index]),
        f"{prefix}_axis_signed_projection_at_min_perpendicular": float(
            projection[min_perpendicular_index]
        ),
        f"{prefix}_axis_projection_min": float(projection.min()),
        f"{prefix}_axis_projection_max": float(projection.max()),
    }
    anchor_summaries: dict[str, dict[str, float]] = {}
    for label, anchor in (("H3", h3), ("H6", h6)):
        relative = atoms - anchor
        distance = np.linalg.norm(relative, axis=1)
        signed_projection = relative @ axis
        anchor_perpendicular = np.linalg.norm(
            relative - np.outer(signed_projection, axis), axis=1
        )
        nearest = int(np.argmin(distance))
        safe_distance = np.maximum(distance, 1e-12)
        summary = {
            "distance_min": float(distance[nearest]),
            "perpendicular_min": float(anchor_perpendicular.min()),
            "signed_projection_at_nearest": float(signed_projection[nearest]),
            "inverse_distance_sum": float(np.sum(1.0 / safe_distance[distance <= cutoff])),
            "count_within_cutoff": float(np.sum(distance <= cutoff)),
        }
        anchor_summaries[label] = summary
        for statistic, value in summary.items():
            suffix = f"_d{cutoff:g}" if statistic in {
                "inverse_distance_sum",
                "count_within_cutoff",
            } else ""
            result[f"{prefix}_{label}_{statistic}{suffix}"] = value

    for statistic in (
        "distance_min",
        "perpendicular_min",
        "signed_projection_at_nearest",
        "inverse_distance_sum",
        "count_within_cutoff",
    ):
        suffix = f"_d{cutoff:g}" if statistic in {
            "inverse_distance_sum",
            "count_within_cutoff",
        } else ""
        result[f"{prefix}_{statistic}_H3_minus_H6{suffix}"] = (
            anchor_summaries["H3"][statistic] - anchor_summaries["H6"][statistic]
        )
    return result


def derive_rotation_invariant_features(
    coordinates: pd.DataFrame,
    *,
    environment_elements: tuple[str, ...] = ("Mg", "O"),
    cutoff: float = 5.0,
) -> pd.DataFrame:
    """Create molecular-axis projection, perpendicular, and asymmetry features.

    Input is a long table with one row per atom.  Coordinates should already
    contain the intended periodic images (for example the same 9-cell expansion
    used to create the handoff summaries).  This function does not infer or wrap
    periodic boundary conditions.
    """
    missing = sorted(REQUIRED_COORDINATE_COLUMNS.difference(coordinates.columns))
    if missing:
        raise ValueError(f"Missing coordinate columns: {missing}")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    rows: list[dict[str, float | int | str]] = []
    for file_key, group in coordinates.groupby("file_key", sort=False):
        c3 = _anchor(group, "C3")
        h3 = _anchor(group, "H3")
        c6 = _anchor(group, "C6")
        h6 = _anchor(group, "H6")
        u3 = _unit(h3 - c3)
        u6 = _unit(h6 - c6)
        axis = _unit(u6 - u3)
        separation = float(np.arccos(np.clip(np.dot(u3, u6), -1.0, 1.0)))
        center = 0.5 * (c3 + c6)
        row: dict[str, float | int | str] = {
            "file_key": str(file_key),
            "axis_unit_x": float(axis[0]),
            "axis_unit_y": float(axis[1]),
            "axis_unit_z": float(axis[2]),
            "axis_azimuth_sin": float(axis[1] / max(np.hypot(axis[0], axis[1]), 1e-12)),
            "axis_azimuth_cos": float(axis[0] / max(np.hypot(axis[0], axis[1]), 1e-12)),
            "axis_abs_elevation_rad": float(np.arcsin(np.clip(abs(axis[2]), 0.0, 1.0))),
            "antiparallel_deviation_rad": float(abs(np.pi - separation)),
        }
        for element in environment_elements:
            atoms = group.loc[
                group["element"].astype(str).str.casefold().eq(element.casefold()),
                ["x", "y", "z"],
            ].to_numpy(float)
            if len(atoms) == 0:
                raise ValueError(f"No {element} atoms found for {file_key}")
            row.update(_summarize_element(element, atoms, center, h3, h6, axis, cutoff))
        rows.append(row)
    return pd.DataFrame(rows)
