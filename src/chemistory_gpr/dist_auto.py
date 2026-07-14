"""Gaussian-process adaptation of the original dist_auto workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

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
    matern_nu: float = 1.5
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


class DistAutoGPR:
    """A fitted dist_auto feature transform and Matérn GPR."""

    def __init__(self, config: DistAutoGPRConfig):
        if config.representation not in {"full", "pca"}:
            raise ValueError("representation must be 'full' or 'pca'")
        self.config = config

    def fit(self, X: pd.DataFrame, xy: pd.DataFrame, y: np.ndarray) -> "DistAutoGPR":
        self.feature_columns_ = X.columns.tolist()
        raw = X.loc[:, self.feature_columns_].to_numpy(dtype=float)
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

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=self.config.matern_nu)
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
        )
        self.gpr_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
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
        design = self.scaler_.transform(self.variance_.transform(raw))
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
    metrics.update({"model": config.name, "test_tag": str(test_tag), "kernel": str(model.gpr_.kernel_)})
    return model, prediction, metrics


def leave_one_tag_out(
    data: DistAutoData,
    config: DistAutoGPRConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnostic group CV; it reveals tags that are genuine out-of-domain cases."""
    outputs: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float]] = []
    ordered_tags = list(dict.fromkeys(data.tags.tolist()))
    for index, tag in enumerate(ordered_tags):
        fold_config = replace(config, seed=config.seed + index)
        _, prediction, metrics = fit_held_out_tag(data, tag, fold_config)
        outputs.append(prediction)
        metric_rows.append(metrics)
    all_predictions = pd.concat(outputs, ignore_index=True)
    overall = gaussian_regression_metrics(
        all_predictions["y"].to_numpy(float),
        all_predictions["pred_mean"].to_numpy(float),
        all_predictions["pred_std"].to_numpy(float),
    )
    overall.update({"model": config.name, "test_tag": "ALL_OOF", "kernel": "per-fold"})
    metric_rows.append(overall)
    return all_predictions, pd.DataFrame(metric_rows)


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
