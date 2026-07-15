"""Exploratory structure-series holdout validation for the handoff data."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import GroupKFold

from .handoff import HandoffData, HandoffGPRConfig, cross_validate_handoff


def file_key_series(file_keys: Iterable[str]) -> pd.Series:
    """Treat all but the final hyphen-delimited token as a candidate series id."""
    values = pd.Series(file_keys, dtype="string")
    series = values.str.replace(r"-[^-]+$", "", regex=True)
    if series.isna().any() or series.eq(values).any():
        raise ValueError("Every file_key must contain a final hyphen-delimited step token")
    return series


def make_prefix_group_folds(data: HandoffData, n_splits: int = 10) -> pd.DataFrame:
    """Hold candidate file-key series together using deterministic GroupKFold."""
    series = file_key_series(data.file_key)
    if series.nunique() < n_splits:
        raise ValueError(f"Need at least {n_splits} candidate series")
    fold = pd.Series(index=series.index, dtype="int64")
    splitter = GroupKFold(n_splits=n_splits)
    for fold_number, (_, test_index) in enumerate(
        splitter.split(data.base, data.y, groups=series), start=1
    ):
        fold.iloc[test_index] = fold_number
    result = pd.DataFrame(
        {
            "file_key": data.file_key,
            "series_prefix_candidate": series,
            "group_fold": fold.astype(int),
        }
    )
    leakage = result.groupby("series_prefix_candidate")["group_fold"].nunique()
    if not leakage.eq(1).all():
        raise RuntimeError("A candidate series was split across group folds")
    return result


def run_prefix_group_comparison(
    data: HandoffData,
    configs: Iterable[HandoffGPRConfig],
    results_dir: str | Path,
    *,
    n_splits: int = 10,
) -> dict[str, Path]:
    """Run GPR candidates while holding each candidate file-key series intact."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    fold_table = make_prefix_group_folds(data, n_splits=n_splits)
    grouped_data = HandoffData(
        base=data.base,
        xproc=data.xproc,
        folds=fold_table[["file_key", "group_fold"]].rename(
            columns={"group_fold": "fold_seed123"}
        ),
    )
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for config in configs:
        grouped_config = replace(config, name=f"prefix_group{n_splits}_{config.name}")
        prediction, metrics = cross_validate_handoff(grouped_data, grouped_config)
        metrics.pop("kernels")
        metrics.pop("kernel_diagnostics")
        metrics.update(
            {
                "evaluation_split": f"file_key_prefix_group{n_splits}",
                "candidate_series_count": int(
                    fold_table["series_prefix_candidate"].nunique()
                ),
            }
        )
        metric_rows.append(metrics)
        prediction = prediction.merge(
            fold_table[["file_key", "series_prefix_candidate"]],
            on="file_key",
            validate="one_to_one",
        )
        prediction.insert(0, "model", grouped_config.name)
        prediction_rows.append(prediction)

    metrics = pd.DataFrame(metric_rows).sort_values("R2", ascending=False)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics_path = results_dir / "gpr_handoff_group10_prefix_metrics.csv"
    predictions_path = results_dir / "gpr_handoff_group10_prefix_predictions.csv"
    folds_path = results_dir / "gpr_handoff_group10_prefix_folds.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    fold_table.to_csv(folds_path, index=False)
    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "folds": folds_path,
    }
