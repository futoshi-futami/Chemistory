"""Leakage-safe Gaussian-process models for the three handoff CSV files."""

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


RAW_ANGLE_COLUMNS = ("C3H3_angle_xy", "C6H6_angle_xy", "angle_diff_C3_C6")


@dataclass(frozen=True)
class HandoffGPRConfig:
    """Configuration for one GPR candidate."""

    name: str = "cyclic_xproc_pca8_matern32"
    cyclic_angles: bool = True
    use_xproc: bool = True
    xproc_components: int = 8
    matern_nu: float = 1.5
    seed: int = 123
    optimizer_restarts: int = 0


@dataclass(frozen=True)
class HandoffData:
    base: pd.DataFrame
    xproc: pd.DataFrame
    folds: pd.DataFrame

    @property
    def y(self) -> np.ndarray:
        return self.base["y"].to_numpy(dtype=float)

    @property
    def file_key(self) -> np.ndarray:
        return self.base["file_key"].astype(str).to_numpy()

    @property
    def fold_id(self) -> np.ndarray:
        return self.folds["fold_seed123"].to_numpy(dtype=int)


def load_handoff_data(data_dir: str | Path) -> HandoffData:
    """Load the three CSVs and fail early if their row identities differ."""
    data_dir = Path(data_dir)
    base = pd.read_csv(data_dir / "01_base_summary_first_angle.csv")
    xproc = pd.read_csv(data_dir / "02_Xproc_matched.csv")
    folds = pd.read_csv(data_dir / "03_cv_folds_seed123.csv")

    for frame, name in ((base, "base"), (xproc, "xproc"), (folds, "folds")):
        if "file_key" not in frame:
            raise ValueError(f"{name} is missing file_key")
        if frame["file_key"].duplicated().any():
            raise ValueError(f"{name} contains duplicate file_key values")
    if not base["file_key"].equals(xproc["file_key"]):
        raise ValueError("base and X_proc file_key/order do not match")
    if not base["file_key"].equals(folds["file_key"]):
        raise ValueError("base and fold file_key/order do not match")
    if base.isna().any().any() or xproc.isna().any().any() or folds.isna().any().any():
        raise ValueError("The handoff inputs contain missing values")
    if sorted(folds["fold_seed123"].unique().tolist()) != list(range(1, 11)):
        raise ValueError("Expected fold_seed123 to contain folds 1,...,10")
    return HandoffData(base=base, xproc=xproc, folds=folds)


def transform_angles(base_features: pd.DataFrame, cyclic: bool) -> pd.DataFrame:
    """Replace raw radian angles by continuous sin/cos coordinates when requested."""
    out = base_features.copy()
    if not cyclic:
        return out
    for column in RAW_ANGLE_COLUMNS:
        if column not in out:
            continue
        out[f"{column}__sin"] = np.sin(out[column].to_numpy(dtype=float))
        out[f"{column}__cos"] = np.cos(out[column].to_numpy(dtype=float))
        out = out.drop(columns=column)
    return out


class HandoffGPR:
    """Fold-local preprocessing followed by an isotropic Matérn GPR."""

    def __init__(self, config: HandoffGPRConfig):
        self.config = config

    def fit(self, base: pd.DataFrame, xproc: pd.DataFrame, y: np.ndarray) -> "HandoffGPR":
        base_x = transform_angles(base.drop(columns=["file_key", "y"], errors="ignore"), self.config.cyclic_angles)
        self.base_columns_ = base_x.columns.tolist()
        base_array = base_x.to_numpy(dtype=float)
        self.base_variance_ = VarianceThreshold()
        base_array = self.base_variance_.fit_transform(base_array)
        self.base_scaler_ = StandardScaler()
        design = self.base_scaler_.fit_transform(base_array)

        if self.config.use_xproc:
            xp = xproc.drop(columns="file_key", errors="ignore").to_numpy(dtype=float)
            self.xproc_variance_ = VarianceThreshold()
            xp = self.xproc_variance_.fit_transform(xp)
            self.xproc_scaler_ = StandardScaler()
            xp = self.xproc_scaler_.fit_transform(xp)
            n_components = min(self.config.xproc_components, xp.shape[0] - 1, xp.shape[1])
            self.xproc_pca_ = PCA(
                n_components=n_components,
                svd_solver="randomized",
                iterated_power=7,
                random_state=self.config.seed,
            )
            scores = self.xproc_pca_.fit_transform(xp)
            # Do not re-standardize PCA scores: their variances encode component importance.
            design = np.column_stack([design, scores])

        kernel = (
            ConstantKernel(1.0, (1e-2, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=self.config.matern_nu)
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e2))
        )
        self.gpr_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            normalize_y=True,
            n_restarts_optimizer=self.config.optimizer_restarts,
            random_state=self.config.seed,
        )
        self.gpr_.fit(design, np.asarray(y, dtype=float))
        return self

    def _design(self, base: pd.DataFrame, xproc: pd.DataFrame) -> np.ndarray:
        base_x = transform_angles(base.drop(columns=["file_key", "y"], errors="ignore"), self.config.cyclic_angles)
        missing = [column for column in self.base_columns_ if column not in base_x]
        if missing:
            raise ValueError(f"Missing base features at prediction time: {missing[:5]}")
        base_array = base_x.loc[:, self.base_columns_].to_numpy(dtype=float)
        design = self.base_scaler_.transform(self.base_variance_.transform(base_array))
        if self.config.use_xproc:
            xp = xproc.drop(columns="file_key", errors="ignore").to_numpy(dtype=float)
            xp = self.xproc_scaler_.transform(self.xproc_variance_.transform(xp))
            design = np.column_stack([design, self.xproc_pca_.transform(xp)])
        return design

    def predict(
        self,
        base: pd.DataFrame,
        xproc: pd.DataFrame,
        return_std: bool = True,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        return self.gpr_.predict(self._design(base, xproc), return_std=return_std)


def cross_validate_handoff(
    data: HandoffData,
    config: HandoffGPRConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run the supplied fixed 10 folds with every transform fitted inside its fold."""
    n = len(data.base)
    pred = np.full(n, np.nan)
    std = np.full(n, np.nan)
    kernels: dict[int, str] = {}
    for fold in sorted(np.unique(data.fold_id)):
        train = data.fold_id != fold
        test = ~train
        fold_config = replace(config, seed=config.seed + int(fold))
        model = HandoffGPR(fold_config).fit(data.base.loc[train], data.xproc.loc[train], data.y[train])
        pred[test], std[test] = model.predict(data.base.loc[test], data.xproc.loc[test], return_std=True)
        kernels[int(fold)] = str(model.gpr_.kernel_)

    if np.isnan(pred).any() or np.isnan(std).any():
        raise RuntimeError("OOF predictions were not filled for every row")
    output = pd.DataFrame(
        {
            "file_key": data.file_key,
            "fold": data.fold_id,
            "y": data.y,
            "pred_mean": pred,
            "pred_std": std,
            "residual": data.y - pred,
            "lower_95": pred - 1.959963984540054 * std,
            "upper_95": pred + 1.959963984540054 * std,
        }
    )
    metrics = gaussian_regression_metrics(data.y, pred, std)
    metrics.update(
        {
            "model": config.name,
            "cyclic_angles": config.cyclic_angles,
            "use_xproc": config.use_xproc,
            "xproc_components": config.xproc_components if config.use_xproc else 0,
            "matern_nu": config.matern_nu,
            "kernels": kernels,
        }
    )
    return output, metrics


def default_handoff_candidates(seed: int = 123) -> list[HandoffGPRConfig]:
    """Small, interpretable candidate set used by the Colab notebook."""
    return [
        HandoffGPRConfig(
            name="base_raw_matern12",
            cyclic_angles=False,
            use_xproc=False,
            matern_nu=0.5,
            seed=seed,
        ),
        HandoffGPRConfig(
            name="base_cyclic_matern32",
            cyclic_angles=True,
            use_xproc=False,
            matern_nu=1.5,
            seed=seed,
        ),
        HandoffGPRConfig(
            name="base_cyclic_xproc_pca8_matern32",
            cyclic_angles=True,
            use_xproc=True,
            xproc_components=8,
            matern_nu=1.5,
            seed=seed,
        ),
    ]
