"""Gaussian-process adaptation of the original dist_auto workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from .kernels import build_signal_plus_white_kernel, fitted_kernel_diagnostics
from .metrics import gaussian_regression_metrics
from .xmat import build_Xmat


DEFAULT_TAGS = ("a", "b", "10", "15", "20", "25")


@dataclass(frozen=True)
class DistAutoData:
    X: pd.DataFrame
    y: np.ndarray
    tags: np.ndarray
    xy: pd.DataFrame
    feature_columns: list[str]


@dataclass(frozen=True)
class DistAutoGPRConfig:
    name: str = "dist_auto_full_matern32"
    representation: str = "full"  # full or pca
    pca_components: int = 10
    include_xy: bool = True
    drop_constant_features: bool = True
    kernel_family: str = "matern"
    ard: bool = False
    matern_nu: float = 1.5
    signal_bounds: tuple[float, float] = (1e-2, 1e3)
    length_scale_bounds: tuple[float, float] = (1e-2, 1e3)
    noise_bounds: tuple[float, float] = (1e-6, 1e1)
    alpha: float = 1e-10
    seed: int = 123
    optimizer_restarts: int = 0


def load_dist_auto_data(
    data_dir: str | Path,
    tags: tuple[str, ...] = DEFAULT_TAGS,
) -> DistAutoData:
    """Load precomputed Xmat tables and align them explicitly to response.csv."""
    data_dir = Path(data_dir)
    frames = {tag: pd.read_csv(data_dir / f"Xmat_{tag}.csv") for tag in tags}
    feature_columns = [
        column for column in frames[tags[0]].columns if all(column in frames[tag].columns for tag in tags)
    ]
    if not feature_columns:
        raise ValueError("No common Xmat feature columns were found")

    response = pd.read_csv(data_dir / "response.csv", encoding="utf-8-sig")
    if "tag" not in response:
        raise ValueError("response.csv is missing tag")
    target_column = response.columns[6]
    xy_reference = pd.read_csv(data_dir / "xy_shift.csv")[["x", "y"]]

    x_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []
    tag_parts: list[np.ndarray] = []
    xy_parts: list[pd.DataFrame] = []
    for tag in tags:
        response_tag = response.loc[response["tag"].astype(str) == tag].reset_index(drop=True)
        feature_tag = frames[tag].loc[:, feature_columns].apply(pd.to_numeric, errors="raise").reset_index(drop=True)
        if len(feature_tag) != len(response_tag):
            raise ValueError(f"Row mismatch for tag={tag}: X={len(feature_tag)}, y={len(response_tag)}")
        if len(response_tag) != len(xy_reference):
            raise ValueError(f"tag={tag} does not have the expected xy_shift row count")
        if not np.allclose(response_tag[["x", "y"]].to_numpy(float), xy_reference.to_numpy(float)):
            raise ValueError(f"response xy order does not match xy_shift for tag={tag}")
        x_parts.append(feature_tag)
        y_parts.append(response_tag[target_column].to_numpy(dtype=float))
        tag_parts.append(np.repeat(tag, len(response_tag)))
        xy_parts.append(response_tag[["x", "y"]].astype(float))

    X = pd.concat(x_parts, ignore_index=True)
    if X.isna().any().any():
        raise ValueError("dist_auto common features contain missing values")
    return DistAutoData(
        X=X,
        y=np.concatenate(y_parts),
        tags=np.concatenate(tag_parts),
        xy=pd.concat(xy_parts, ignore_index=True),
        feature_columns=feature_columns,
    )


def standardized_tag_centroid_distances(data: DistAutoData) -> pd.DataFrame:
    """Exploratory Euclidean distances between tag centroids in standardized Xmat space.

    This global descriptive transform must not be reused as a predictive preprocessing
    step; fold-local transforms are used by :class:`DistAutoGPR`.
    """
    raw = VarianceThreshold().fit_transform(data.X.to_numpy(dtype=float))
    standardized = StandardScaler().fit_transform(raw)
    ordered_tags = list(dict.fromkeys(data.tags.tolist()))
    centroids = np.vstack([standardized[data.tags == tag].mean(axis=0) for tag in ordered_tags])
    distances = np.sqrt(np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=2))
    return pd.DataFrame(distances, index=ordered_tags, columns=ordered_tags)


class DistAutoGPR:
    """A fitted dist_auto feature transform and configurable GPR."""

    def __init__(self, config: DistAutoGPRConfig):
        if config.representation not in {"full", "pca"}:
            raise ValueError("representation must be 'full' or 'pca'")
        self.config = config

    def fit(self, X: pd.DataFrame, xy: pd.DataFrame, y: np.ndarray) -> "DistAutoGPR":
        self.feature_columns_ = X.columns.tolist()
        raw = X.loc[:, self.feature_columns_].to_numpy(dtype=float)
        self.variance_: VarianceThreshold | None = None
        if self.config.drop_constant_features:
            self.variance_ = VarianceThreshold()
            raw = self.variance_.fit_transform(raw)
        self.scaler_ = StandardScaler()
        design = self.scaler_.fit_transform(raw)

        if self.config.representation == "pca":
            n_components = min(self.config.pca_components, design.shape[0] - 1, design.shape[1])
            self.pca_ = PCA(
                n_components=n_components,
                svd_solver="randomized",
                iterated_power=7,
                random_state=self.config.seed,
            )
            design = self.pca_.fit_transform(design)

        if self.config.include_xy:
            self.xy_scaler_ = StandardScaler()
            design = np.column_stack([design, self.xy_scaler_.fit_transform(xy[["x", "y"]].to_numpy(float))])

        kernel = build_signal_plus_white_kernel(
            kernel_family=self.config.kernel_family,
            n_features=design.shape[1],
            ard=self.config.ard,
            matern_nu=self.config.matern_nu,
            signal_bounds=self.config.signal_bounds,
            length_scale_bounds=self.config.length_scale_bounds,
            noise_bounds=self.config.noise_bounds,
        )
        self.gpr_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.config.alpha,
            normalize_y=True,
            n_restarts_optimizer=self.config.optimizer_restarts,
            random_state=self.config.seed,
        )
        self.gpr_.fit(design, np.asarray(y, dtype=float))
        return self

    def _design(self, X: pd.DataFrame, xy: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.feature_columns_ if column not in X]
        if missing:
            raise ValueError(f"Prediction Xmat is missing {len(missing)} features, e.g. {missing[:5]}")
        raw = X.loc[:, self.feature_columns_].to_numpy(dtype=float)
        if self.variance_ is not None:
            raw = self.variance_.transform(raw)
        design = self.scaler_.transform(raw)
        if self.config.representation == "pca":
            design = self.pca_.transform(design)
        if self.config.include_xy:
            design = np.column_stack([design, self.xy_scaler_.transform(xy[["x", "y"]].to_numpy(float))])
        return design

    def predict(
        self,
        X: pd.DataFrame,
        xy: pd.DataFrame,
        return_std: bool = True,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        return self.gpr_.predict(self._design(X, xy), return_std=return_std)


def fit_held_out_tag(
    data: DistAutoData,
    test_tag: str,
    config: DistAutoGPRConfig,
) -> tuple[DistAutoGPR, pd.DataFrame, dict[str, float]]:
    """Match the original dist_auto experiment: train on five tags, test on one."""
    train = data.tags != str(test_tag)
    test = ~train
    if not test.any():
        raise ValueError(f"Unknown test_tag={test_tag}")
    model = DistAutoGPR(config).fit(data.X.loc[train], data.xy.loc[train], data.y[train])
    mean, std = model.predict(data.X.loc[test], data.xy.loc[test], return_std=True)
    prediction = data.xy.loc[test].reset_index(drop=True).copy()
    prediction.insert(0, "tag", str(test_tag))
    prediction["y"] = data.y[test]
    prediction["pred_mean"] = mean
    prediction["pred_std"] = std
    prediction["lower_95"] = mean - 1.959963984540054 * std
    prediction["upper_95"] = mean + 1.959963984540054 * std
    metrics = gaussian_regression_metrics(data.y[test], mean, std)
    diagnostics = fitted_kernel_diagnostics(
        model.gpr_,
        length_scale_bounds=config.length_scale_bounds,
    )
    metrics.update(
        {
            "model": config.name,
            "test_tag": str(test_tag),
            "representation": config.representation,
            "include_xy": config.include_xy,
            "drop_constant_features": config.drop_constant_features,
            "kernel_family": config.kernel_family,
            "ard": config.ard,
            "matern_nu": config.matern_nu if config.kernel_family == "matern" else np.nan,
            "optimizer_restarts": config.optimizer_restarts,
            **diagnostics,
        }
    )
    return model, prediction, metrics


def leave_one_tag_out(
    data: DistAutoData,
    config: DistAutoGPRConfig,
    *,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnostic group CV; it reveals tags that are genuine out-of-domain cases."""
    ordered_tags = list(dict.fromkeys(data.tags.tolist()))
    if n_jobs == 1:
        fitted = [fit_held_out_tag(data, tag, config) for tag in ordered_tags]
    else:
        fitted = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(fit_held_out_tag)(data, tag, config) for tag in ordered_tags
        )
    outputs = [item[1] for item in fitted]
    metric_rows = [item[2] for item in fitted]
    all_predictions = pd.concat(outputs, ignore_index=True)
    overall = gaussian_regression_metrics(
        all_predictions["y"].to_numpy(float),
        all_predictions["pred_mean"].to_numpy(float),
        all_predictions["pred_std"].to_numpy(float),
    )
    overall.update(
        {
            "model": config.name,
            "test_tag": "ALL_OOF",
            "representation": config.representation,
            "include_xy": config.include_xy,
            "drop_constant_features": config.drop_constant_features,
            "kernel_family": config.kernel_family,
            "ard": config.ard,
            "matern_nu": config.matern_nu if config.kernel_family == "matern" else np.nan,
            "optimizer_restarts": config.optimizer_restarts,
            "optimized_kernel": "per-fold",
        }
    )
    metric_rows.append(overall)
    return all_predictions, pd.DataFrame(metric_rows)


def summarize_dist_auto_metrics(metric_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-tag and all-OOF metrics without awarding tied tag wins."""
    metrics = metric_table.copy()
    metrics["test_tag"] = metrics["test_tag"].astype(str)
    per_tag = metrics.loc[metrics["test_tag"] != "ALL_OOF"].copy()
    maxima = per_tag.groupby("test_tag")["R2"].transform("max")
    per_tag["is_best"] = np.isclose(per_tag["R2"], maxima, rtol=0.0, atol=1e-12)
    best_per_tag_count = per_tag.groupby("test_tag")["is_best"].transform("sum")
    unique_best_counts = (
        per_tag.loc[per_tag["is_best"] & (best_per_tag_count == 1)]
        .groupby("model")
        .size()
        .rename("tags_unique_best_R2")
    )
    summary = (
        per_tag.groupby("model", as_index=False)
        .agg(
            kernel_family=("kernel_family", "first"),
            ard=("ard", "first"),
            mean_tag_R2=("R2", "mean"),
            median_tag_R2=("R2", "median"),
            worst_tag_R2=("R2", "min"),
            mean_tag_RMSE=("RMSE", "mean"),
            mean_coverage_95=("coverage_95", "mean"),
        )
        .merge(unique_best_counts, on="model", how="left")
    )
    overall = metrics.loc[
        metrics["test_tag"] == "ALL_OOF", ["model", "R2", "RMSE", "MAE"]
    ].rename(columns={"R2": "all_oof_R2", "RMSE": "all_oof_RMSE", "MAE": "all_oof_MAE"})
    summary = summary.merge(overall, on="model", how="left")
    summary["tags_unique_best_R2"] = summary["tags_unique_best_R2"].fillna(0).astype(int)
    return summary.sort_values("mean_tag_R2", ascending=False).reset_index(drop=True)


def dist_auto_kernel_candidates(
    seed: int = 0,
    *,
    rbf_ard_restarts: int = 5,
) -> list[DistAutoGPRConfig]:
    """Controlled comparison using the original Xmat-only preprocessing."""
    shared = {
        "representation": "full",
        "include_xy": False,
        "drop_constant_features": False,
        "alpha": 0.0,
        "seed": seed,
    }
    return [
        DistAutoGPRConfig(
            name="dist_auto_full_xmat_matern12",
            kernel_family="matern",
            matern_nu=0.5,
            **shared,
        ),
        DistAutoGPRConfig(
            name="dist_auto_full_xmat_matern32",
            kernel_family="matern",
            matern_nu=1.5,
            **shared,
        ),
        DistAutoGPRConfig(
            name="dist_auto_full_xmat_rbf_iso",
            kernel_family="rbf",
            ard=False,
            **shared,
        ),
        DistAutoGPRConfig(
            name="dist_auto_full_xmat_rbf_ard_original",
            kernel_family="rbf",
            ard=True,
            optimizer_restarts=rbf_ard_restarts,
            **shared,
        ),
    ]


def default_dist_auto_candidates(
    seed: int = 0,
    *,
    rbf_ard_restarts: int = 5,
) -> list[DistAutoGPRConfig]:
    """Previous Matérn+xy baseline followed by the controlled kernel set."""
    previous = DistAutoGPRConfig(
        name="dist_auto_full_xy_matern32_previous",
        representation="full",
        include_xy=True,
        drop_constant_features=True,
        kernel_family="matern",
        matern_nu=1.5,
        signal_bounds=(1e-3, 1e3),
        length_scale_bounds=(1e-3, 1e3),
        noise_bounds=(1e-8, 1e1),
        alpha=1e-10,
        seed=123,
    )
    return [previous] + dist_auto_kernel_candidates(seed, rbf_ard_restarts=rbf_ard_restarts)


def make_grid(
    data_dir: str | Path,
    test_tag: str,
    feature_columns: list[str],
    grid_size: int = 30,
    xy_min: float = 0.0,
    xy_max: float = 1.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a square prediction grid and the matching Xmat features in pure Python."""
    data_dir = Path(data_dir)
    axis = np.linspace(xy_min, xy_max, grid_size)
    grid = pd.DataFrame([(x, y) for x in axis for y in axis], columns=["x", "y"])
    out_path = data_dir / f"Xmat_{test_tag}_grid.csv"
    built = build_Xmat(
        xydata=grid,
        tag=f"_{test_tag}",
        csvfile=str(data_dir / f"xyz_9cell_all_{test_tag}.csv"),
        out=str(out_path),
        filter_adist_first=10,
        verbose=False,
    )
    missing = [column for column in feature_columns if column not in built]
    if missing:
        raise ValueError(f"Generated grid lacks {len(missing)} training features, e.g. {missing[:5]}")
    return grid, built.loc[:, feature_columns]


def predict_grid(
    model: DistAutoGPR,
    grid: pd.DataFrame,
    grid_features: pd.DataFrame,
) -> pd.DataFrame:
    mean, std = model.predict(grid_features, grid, return_std=True)
    result = grid.copy()
    result["pred_mean"] = mean
    result["pred_std"] = std
    result["lower_95"] = mean - 1.959963984540054 * std
    result["upper_95"] = mean + 1.959963984540054 * std
    result["lower_confidence_bound"] = result["lower_95"]
    return result
