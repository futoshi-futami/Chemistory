"""Nested group validation for compact-axis, interaction, and expert models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GroupKFold

from .angle_report import derive_angle_coordinates
from .handoff import HandoffData, HandoffFeatureTransformer, HandoffGPR, HandoffGPRConfig
from .kernels import build_axis_environment_kernel
from .metrics import gaussian_regression_metrics, regression_metrics
from .physical_features import (
    COMPACT_AXIS_COLUMNS,
    candidate_group_labels,
    compact_axis_base,
    file_key_token_diagnostics,
    group_scheme_summary,
)


@dataclass(frozen=True)
class NestedCandidate:
    """One predeclared model/boundary candidate for inner group-CV selection."""

    name: str
    kind: str
    kernel_family: str = "matern"
    matern_nu: float = 1.5
    include_interaction: bool = False
    threshold_deg: float | None = None
    high_expert: str | None = None
    xproc_components: int = 8
    rf_trees: int = 300


def default_nested_candidates() -> list[NestedCandidate]:
    """Small, hypothesis-driven candidate set; no outcome-defined y gate is used."""
    candidates = [
        NestedCandidate("legacy_angles_global_matern32", "legacy_gp"),
        NestedCandidate("compact_axis_global_matern32", "compact_gp"),
        NestedCandidate(
            "compact_axis_global_matern12", "compact_gp", matern_nu=0.5
        ),
        NestedCandidate(
            "compact_axis_global_rational_quadratic",
            "compact_gp",
            kernel_family="rational_quadratic",
        ),
        NestedCandidate("axis_plus_environment_matern32", "structured_gp"),
        NestedCandidate(
            "axis_environment_interaction_matern32",
            "structured_gp",
            include_interaction=True,
        ),
    ]
    for threshold in (40.0, 45.0, 50.0):
        label = int(threshold)
        candidates.append(
            NestedCandidate(
                f"moe_matern32_matern12_gate{label}",
                "mixture",
                threshold_deg=threshold,
                high_expert="matern12",
            )
        )
        candidates.append(
            NestedCandidate(
                f"moe_matern32_rf_gate{label}",
                "mixture",
                threshold_deg=threshold,
                high_expert="rf",
            )
        )
    return candidates


def _config_for_candidate(candidate: NestedCandidate, seed: int) -> HandoffGPRConfig:
    return HandoffGPRConfig(
        name=candidate.name,
        cyclic_angles=candidate.kind == "legacy_gp",
        use_xproc=True,
        xproc_components=candidate.xproc_components,
        kernel_family=candidate.kernel_family,
        matern_nu=candidate.matern_nu,
        seed=seed,
        optimizer_restarts=0,
    )


class _CompactGP:
    def __init__(self, candidate: NestedCandidate, seed: int):
        self.candidate = candidate
        self.seed = seed

    def fit(self, base: pd.DataFrame, xproc: pd.DataFrame, y: np.ndarray):
        config = _config_for_candidate(self.candidate, self.seed)
        self.model_ = HandoffGPR(config).fit(compact_axis_base(base), xproc, y)
        return self

    def predict(self, base: pd.DataFrame, xproc: pd.DataFrame):
        return self.model_.predict(compact_axis_base(base), xproc, return_std=True)


class _LegacyGP:
    def __init__(self, candidate: NestedCandidate, seed: int):
        self.candidate = candidate
        self.seed = seed

    def fit(self, base: pd.DataFrame, xproc: pd.DataFrame, y: np.ndarray):
        config = _config_for_candidate(self.candidate, self.seed)
        self.model_ = HandoffGPR(config).fit(base, xproc, y)
        return self

    def predict(self, base: pd.DataFrame, xproc: pd.DataFrame):
        return self.model_.predict(base, xproc, return_std=True)


class _StructuredGP:
    def __init__(self, candidate: NestedCandidate, seed: int):
        self.candidate = candidate
        self.seed = seed

    def fit(self, base: pd.DataFrame, xproc: pd.DataFrame, y: np.ndarray):
        compact = compact_axis_base(base)
        config = _config_for_candidate(self.candidate, self.seed)
        self.preprocessor_ = HandoffFeatureTransformer(config)
        design = self.preprocessor_.fit_transform(compact, xproc)
        selected = self.preprocessor_.base_selected_columns_
        axis_dims = tuple(
            index for index, column in enumerate(selected) if column in COMPACT_AXIS_COLUMNS
        )
        environment_dims = tuple(
            index
            for index in range(design.shape[1])
            if index not in set(axis_dims)
        )
        if len(axis_dims) != len(COMPACT_AXIS_COLUMNS):
            raise RuntimeError("A compact axis column was removed unexpectedly")
        self.axis_dims_ = axis_dims
        self.environment_dims_ = environment_dims
        kernel = build_axis_environment_kernel(
            axis_dims=axis_dims,
            environment_dims=environment_dims,
            include_interaction=self.candidate.include_interaction,
            axis_nu=1.5,
            environment_nu=1.5,
            signal_bounds=config.signal_bounds,
            length_scale_bounds=config.length_scale_bounds,
            noise_bounds=config.noise_bounds,
        )
        self.gpr_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=config.alpha,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=self.seed,
        )
        self.gpr_.fit(design, np.asarray(y, dtype=float))
        return self

    def predict(self, base: pd.DataFrame, xproc: pd.DataFrame):
        design = self.preprocessor_.transform(compact_axis_base(base), xproc)
        return self.gpr_.predict(design, return_std=True)


class _CompactRF:
    """Python RF expert on the same compact base + fold-local PCA8 design."""

    def __init__(self, candidate: NestedCandidate, seed: int):
        self.candidate = candidate
        self.seed = seed

    def fit(self, base: pd.DataFrame, xproc: pd.DataFrame, y: np.ndarray):
        config = _config_for_candidate(self.candidate, self.seed)
        self.preprocessor_ = HandoffFeatureTransformer(config)
        design = self.preprocessor_.fit_transform(compact_axis_base(base), xproc)
        self.model_ = RandomForestRegressor(
            n_estimators=self.candidate.rf_trees,
            max_features=1.0 / 3.0,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=self.seed,
            n_jobs=-1,
        ).fit(design, np.asarray(y, dtype=float))
        return self

    def predict(self, base: pd.DataFrame, xproc: pd.DataFrame):
        design = self.preprocessor_.transform(compact_axis_base(base), xproc)
        mean = self.model_.predict(design)
        # Tree dispersion is not a calibrated GP predictive standard deviation.
        return mean, np.full(len(mean), np.nan)


class _MixtureOfExperts:
    """Hard azimuth gate selected only through inner group CV."""

    def __init__(self, candidate: NestedCandidate, seed: int):
        if candidate.threshold_deg is None or candidate.high_expert is None:
            raise ValueError("Mixture candidates require threshold_deg and high_expert")
        self.candidate = candidate
        self.seed = seed

    def _high_angle(self, base: pd.DataFrame) -> np.ndarray:
        angles = derive_angle_coordinates(base)
        return angles["axis_angle_deg"].to_numpy(float) >= float(
            self.candidate.threshold_deg
        )

    def fit(self, base: pd.DataFrame, xproc: pd.DataFrame, y: np.ndarray):
        high = self._high_angle(base)
        if min(int(high.sum()), int((~high).sum())) < 8:
            raise ValueError("A mixture expert has fewer than eight training samples")
        low_candidate = replace(
            self.candidate,
            name=f"{self.candidate.name}__middle_matern32",
            kind="compact_gp",
            kernel_family="matern",
            matern_nu=1.5,
        )
        self.middle_ = _CompactGP(low_candidate, self.seed + 1).fit(
            base.loc[~high], xproc.loc[~high], np.asarray(y)[~high]
        )
        if self.candidate.high_expert == "matern12":
            high_candidate = replace(
                self.candidate,
                name=f"{self.candidate.name}__high_matern12",
                kind="compact_gp",
                kernel_family="matern",
                matern_nu=0.5,
            )
            self.high_ = _CompactGP(high_candidate, self.seed + 2)
        elif self.candidate.high_expert == "rf":
            high_candidate = replace(
                self.candidate,
                name=f"{self.candidate.name}__high_rf",
                kind="compact_gp",
                kernel_family="matern",
                matern_nu=1.5,
            )
            self.high_ = _CompactRF(high_candidate, self.seed + 2)
        else:
            raise ValueError("high_expert must be matern12 or rf")
        self.high_.fit(base.loc[high], xproc.loc[high], np.asarray(y)[high])
        return self

    def predict(self, base: pd.DataFrame, xproc: pd.DataFrame):
        high = self._high_angle(base)
        mean = np.empty(len(base), dtype=float)
        std = np.full(len(base), np.nan, dtype=float)
        if (~high).any():
            mean[~high], std[~high] = self.middle_.predict(
                base.loc[~high], xproc.loc[~high]
            )
        if high.any():
            mean[high], std[high] = self.high_.predict(base.loc[high], xproc.loc[high])
        return mean, std


def build_nested_model(candidate: NestedCandidate, seed: int):
    if candidate.kind == "legacy_gp":
        return _LegacyGP(candidate, seed)
    if candidate.kind == "compact_gp":
        return _CompactGP(candidate, seed)
    if candidate.kind == "structured_gp":
        return _StructuredGP(candidate, seed)
    if candidate.kind == "mixture":
        return _MixtureOfExperts(candidate, seed)
    raise ValueError(f"Unknown candidate kind: {candidate.kind}")


def _group_splits(
    indices: np.ndarray,
    groups: pd.Series,
    n_splits: int,
):
    local_groups = groups.iloc[indices].reset_index(drop=True)
    splitter = GroupKFold(n_splits=n_splits)
    for train_local, test_local in splitter.split(indices, groups=local_groups):
        yield indices[train_local], indices[test_local]


def _metrics_with_optional_std(
    y: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> dict[str, float]:
    if np.isfinite(std).all():
        return gaussian_regression_metrics(y, mean, std)
    metrics = regression_metrics(y, mean)
    metrics.update({"coverage_95": np.nan, "width_95": np.nan, "NLPD": np.nan})
    return metrics


def run_nested_group_comparison(
    data: HandoffData,
    candidates: list[NestedCandidate],
    results_dir: str | Path,
    *,
    group_scheme: str = "trajectory",
    outer_splits: int = 5,
    inner_splits: int = 4,
    seed: int = 123,
) -> dict[str, Path]:
    """Select model and azimuth gate inside each outer group fold.

    Candidate-wide outer OOF rows are diagnostic comparisons.  The strictly
    nested estimate is the row assembled from each outer fold's inner-CV winner.
    """
    if len(candidates) < 2:
        raise ValueError("Nested selection requires at least two candidates")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    n = len(data.y)
    all_indices = np.arange(n)
    groups = candidate_group_labels(data.file_key, group_scheme)
    if groups.nunique() < outer_splits:
        raise ValueError("Not enough groups for the requested outer_splits")

    outer_fold = np.zeros(n, dtype=int)
    candidate_mean = {candidate.name: np.full(n, np.nan) for candidate in candidates}
    candidate_std = {candidate.name: np.full(n, np.nan) for candidate in candidates}
    nested_mean = np.full(n, np.nan)
    nested_std = np.full(n, np.nan)
    nested_model = np.full(n, "", dtype=object)
    inner_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    outer_iterator = _group_splits(all_indices, groups, outer_splits)
    for outer_number, (outer_train, outer_test) in enumerate(outer_iterator, start=1):
        outer_fold[outer_test] = outer_number
        train_groups = groups.iloc[outer_train]
        actual_inner_splits = min(inner_splits, int(train_groups.nunique()))
        scores: list[dict[str, object]] = []
        for candidate_index, candidate in enumerate(candidates):
            inner_mean = np.full(len(outer_train), np.nan)
            inner_position = {global_index: pos for pos, global_index in enumerate(outer_train)}
            inner_iterator = _group_splits(outer_train, groups, actual_inner_splits)
            for inner_number, (inner_train, inner_valid) in enumerate(
                inner_iterator, start=1
            ):
                model_seed = seed + 10000 * outer_number + 100 * candidate_index + inner_number
                model = build_nested_model(candidate, model_seed).fit(
                    data.base.iloc[inner_train],
                    data.xproc.iloc[inner_train],
                    data.y[inner_train],
                )
                prediction, _ = model.predict(
                    data.base.iloc[inner_valid], data.xproc.iloc[inner_valid]
                )
                positions = [inner_position[index] for index in inner_valid]
                inner_mean[positions] = prediction
            if np.isnan(inner_mean).any():
                raise RuntimeError("Inner OOF predictions were not filled")
            score = regression_metrics(data.y[outer_train], inner_mean)
            row = {
                "outer_fold": outer_number,
                "candidate": candidate.name,
                "kind": candidate.kind,
                "threshold_deg": candidate.threshold_deg,
                "high_expert": candidate.high_expert,
                "inner_group_splits": actual_inner_splits,
                **score,
            }
            scores.append(row)
            inner_rows.append(row)
        score_table = pd.DataFrame(scores).sort_values(
            ["RMSE", "MAE", "candidate"], ascending=[True, True, True]
        )
        winner_name = str(score_table.iloc[0]["candidate"])
        selection_rows.append(
            {
                "outer_fold": outer_number,
                "selected_candidate": winner_name,
                "inner_RMSE": float(score_table.iloc[0]["RMSE"]),
                "inner_MAE": float(score_table.iloc[0]["MAE"]),
                "n_train": int(len(outer_train)),
                "n_test": int(len(outer_test)),
                "n_train_groups": int(groups.iloc[outer_train].nunique()),
                "n_test_groups": int(groups.iloc[outer_test].nunique()),
            }
        )

        for candidate_index, candidate in enumerate(candidates):
            model_seed = seed + 100000 * outer_number + candidate_index
            model = build_nested_model(candidate, model_seed).fit(
                data.base.iloc[outer_train],
                data.xproc.iloc[outer_train],
                data.y[outer_train],
            )
            mean, std = model.predict(
                data.base.iloc[outer_test], data.xproc.iloc[outer_test]
            )
            candidate_mean[candidate.name][outer_test] = mean
            candidate_std[candidate.name][outer_test] = std
            if candidate.name == winner_name:
                nested_mean[outer_test] = mean
                nested_std[outer_test] = std
                nested_model[outer_test] = winner_name

    if (outer_fold == 0).any() or np.isnan(nested_mean).any() or (nested_model == "").any():
        raise RuntimeError("Outer predictions or folds were not filled")

    candidate_metric_rows: list[dict[str, object]] = []
    candidate_prediction_rows: list[pd.DataFrame] = []
    for candidate in candidates:
        mean = candidate_mean[candidate.name]
        std = candidate_std[candidate.name]
        if np.isnan(mean).any():
            raise RuntimeError(f"Candidate OOF predictions missing for {candidate.name}")
        candidate_metric_rows.append(
            {
                "candidate": candidate.name,
                "kind": candidate.kind,
                "threshold_deg": candidate.threshold_deg,
                "high_expert": candidate.high_expert,
                "group_scheme": group_scheme,
                "outer_splits": outer_splits,
                **_metrics_with_optional_std(data.y, mean, std),
            }
        )
        candidate_prediction_rows.append(
            pd.DataFrame(
                {
                    "candidate": candidate.name,
                    "file_key": data.file_key,
                    "group": groups,
                    "outer_fold": outer_fold,
                    "y": data.y,
                    "pred_mean": mean,
                    "pred_std": std,
                    "residual": data.y - mean,
                }
            )
        )

    nested_metrics = pd.DataFrame(
        [
            {
                "model": "nested_inner_group_selected",
                "group_scheme": group_scheme,
                "outer_splits": outer_splits,
                "inner_splits": inner_splits,
                "candidate_count": len(candidates),
                **_metrics_with_optional_std(data.y, nested_mean, nested_std),
            }
        ]
    )
    nested_predictions = pd.DataFrame(
        {
            "file_key": data.file_key,
            "group": groups,
            "outer_fold": outer_fold,
            "selected_candidate": nested_model,
            "y": data.y,
            "pred_mean": nested_mean,
            "pred_std": nested_std,
            "residual": data.y - nested_mean,
        }
    )
    outer_folds = pd.DataFrame(
        {
            "file_key": data.file_key,
            "group": groups,
            "outer_fold": outer_fold,
        }
    )

    paths = {
        "candidate_metrics": results_dir / "gpr_handoff_nested_group_candidate_metrics.csv",
        "candidate_predictions": results_dir
        / "gpr_handoff_nested_group_candidate_predictions.csv",
        "inner_scores": results_dir / "gpr_handoff_nested_group_inner_scores.csv",
        "selections": results_dir / "gpr_handoff_nested_group_selections.csv",
        "nested_metrics": results_dir / "gpr_handoff_nested_group_metrics.csv",
        "nested_predictions": results_dir / "gpr_handoff_nested_group_predictions.csv",
        "outer_folds": results_dir / "gpr_handoff_nested_group_outer_folds.csv",
        "token_diagnostics": results_dir / "gpr_handoff_file_key_token_diagnostics.csv",
        "group_schemes": results_dir / "gpr_handoff_group_scheme_summary.csv",
    }
    pd.DataFrame(candidate_metric_rows).sort_values("RMSE").to_csv(
        paths["candidate_metrics"], index=False
    )
    pd.concat(candidate_prediction_rows, ignore_index=True).to_csv(
        paths["candidate_predictions"], index=False
    )
    pd.DataFrame(inner_rows).to_csv(paths["inner_scores"], index=False)
    pd.DataFrame(selection_rows).to_csv(paths["selections"], index=False)
    nested_metrics.to_csv(paths["nested_metrics"], index=False)
    nested_predictions.to_csv(paths["nested_predictions"], index=False)
    outer_folds.to_csv(paths["outer_folds"], index=False)
    file_key_token_diagnostics(data.base).to_csv(paths["token_diagnostics"], index=False)
    group_scheme_summary(data.file_key).to_csv(paths["group_schemes"], index=False)
    return paths
