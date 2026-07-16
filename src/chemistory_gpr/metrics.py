"""Metrics for the handoff RF/GPR experiments."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the exact R2/RMSE/MAE definitions used in the handoff README."""
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    residual = y - pred
    denominator = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if denominator == 0 else 1.0 - np.sum(residual**2) / denominator
    corr = np.corrcoef(y, pred)[0, 1] if len(y) > 1 else np.nan
    return {
        "R2": float(r2),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "corr2": float(corr**2),
        "n": int(len(y)),
    }


def gaussian_regression_metrics(
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
) -> dict[str, float]:
    """Point metrics plus Gaussian predictive-interval coverage and NLPD."""
    y = np.asarray(y_true, dtype=float)
    mean = np.asarray(pred_mean, dtype=float)
    std = np.maximum(np.asarray(pred_std, dtype=float), 1e-12)
    result = regression_metrics(y, mean)
    for level in levels:
        z = NormalDist().inv_cdf((1.0 + level) / 2.0)
        result[f"coverage_{int(level * 100)}"] = float(np.mean(np.abs(y - mean) <= z * std))
        result[f"width_{int(level * 100)}"] = float(np.mean(2.0 * z * std))
    result["NLPD"] = float(np.mean(0.5 * np.log(2.0 * np.pi * std**2) + 0.5 * ((y - mean) / std) ** 2))
    return result
