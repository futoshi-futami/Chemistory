"""Physically compact angle coordinates and file-key diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .angle_report import ANGLE_SOURCE_COLUMNS, derive_angle_coordinates


COMPACT_AXIS_COLUMNS = (
    "axis_azimuth_sin",
    "axis_azimuth_cos",
    "axis_abs_elevation_rad_proxy",
    "antiparallel_deviation_rad",
)


def compact_axis_base(base: pd.DataFrame) -> pd.DataFrame:
    """Replace seven redundant angle columns by four physical coordinates.

    The azimuth is represented by sine/cosine.  Elevation is an absolute-value
    proxy because the handoff CSV contains only the product of the two C-H z
    components, not their individual signs.  The deviation is zero for exactly
    antiparallel C3->H3 and C6->H6 vectors.
    """
    required = {"file_key", "y", *ANGLE_SOURCE_COLUMNS}
    missing = sorted(required.difference(base.columns))
    if missing:
        raise ValueError(f"Missing columns needed for compact axis features: {missing}")
    angles = derive_angle_coordinates(base)
    axis_rad = np.radians(angles["axis_angle_deg"].to_numpy(float))
    compact = pd.DataFrame(
        {
            "file_key": base["file_key"].astype(str).to_numpy(),
            "y": base["y"].to_numpy(float),
            "axis_azimuth_sin": np.sin(axis_rad),
            "axis_azimuth_cos": np.cos(axis_rad),
            "axis_abs_elevation_rad_proxy": np.radians(
                angles["axis_abs_elevation_deg_proxy"].to_numpy(float)
            ),
            "antiparallel_deviation_rad": np.radians(
                angles["antiparallel_deviation_deg"].to_numpy(float)
            ),
        }
    )
    environment = base.drop(columns=[*ANGLE_SOURCE_COLUMNS, "file_key", "y"])
    return pd.concat([compact, environment.reset_index(drop=True)], axis=1)


def parse_file_key_tokens(file_keys: Iterable[str]) -> pd.DataFrame:
    """Parse the five integer tokens present in the received handoff keys."""
    values = pd.Series(file_keys, dtype="string", name="file_key")
    tokens = values.str.split("-", expand=True)
    if tokens.shape[1] != 5 or tokens.isna().any().any():
        raise ValueError("Every handoff file_key must contain exactly five hyphen tokens")
    numeric = tokens.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Every handoff file_key token must be an integer")
    numeric = numeric.astype(int)
    numeric.columns = [f"token_{index}" for index in range(1, 6)]
    result = pd.concat([values, numeric], axis=1)
    result["trajectory_group_candidate"] = (
        result[["token_1", "token_2", "token_3", "token_4"]]
        .astype(str)
        .agg("-".join, axis=1)
    )
    return result


def candidate_group_labels(file_keys: Iterable[str], scheme: str) -> pd.Series:
    """Return a selectable provisional grouping derived from file-key tokens."""
    tokens = parse_file_key_tokens(file_keys)
    if scheme in {"trajectory", "prefix4", "token3_token4"}:
        labels = tokens["trajectory_group_candidate"]
    elif scheme in {"proximity_level", "token3"}:
        labels = "token3=" + tokens["token_3"].astype(str)
    elif scheme in {"orientation_family", "token4"}:
        labels = "token4=" + tokens["token_4"].astype(str)
    elif scheme in {"sweep_level", "token5"}:
        labels = "token5=" + tokens["token_5"].astype(str)
    else:
        raise ValueError(
            "scheme must be trajectory, proximity_level, orientation_family, or sweep_level"
        )
    return pd.Series(labels.to_numpy(), index=tokens.index, dtype="string", name="group")


def file_key_token_diagnostics(base: pd.DataFrame) -> pd.DataFrame:
    """Quantify what each token tracks without assigning undocumented semantics."""
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
    targets = [
        "axis_angle_deg",
        "axis_abs_elevation_deg_proxy",
        "antiparallel_deviation_deg",
        "y",
        "Mg_H3_count_d5",
        "O_H3_count_d5",
        "Mg_H6_count_d5",
        "O_H6_count_d5",
        "first_Mg_H3_d5",
        "first_O_H3_d5",
    ]
    rows: list[dict[str, object]] = []
    for index in range(1, 6):
        column = f"token_{index}"
        values = joined[column]
        row: dict[str, object] = {
            "token_position": index,
            "column": column,
            "n_unique": int(values.nunique()),
            "observed_values": "|".join(map(str, sorted(values.unique()))),
            "is_constant": bool(values.nunique() == 1),
        }
        for target in targets:
            row[f"spearman_vs_{target}"] = (
                np.nan
                if values.nunique() == 1
                else values.corr(joined[target], method="spearman")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def group_scheme_summary(file_keys: Iterable[str]) -> pd.DataFrame:
    """Summarize the provisional outer-group choices and their sample sizes."""
    rows: list[dict[str, object]] = []
    for scheme in ("trajectory", "proximity_level", "orientation_family", "sweep_level"):
        groups = candidate_group_labels(file_keys, scheme)
        counts = groups.value_counts()
        rows.append(
            {
                "scheme": scheme,
                "n_groups": int(counts.size),
                "min_group_n": int(counts.min()),
                "median_group_n": float(counts.median()),
                "max_group_n": int(counts.max()),
            }
        )
    return pd.DataFrame(rows)
