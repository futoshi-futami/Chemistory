"""Kernel builders and diagnostics shared by the GPR workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel


def build_signal_plus_white_kernel(
    *,
    kernel_family: str,
    n_features: int,
    ard: bool,
    matern_nu: float,
    signal_bounds: tuple[float, float],
    length_scale_bounds: tuple[float, float],
    noise_bounds: tuple[float, float],
):
    """Build ``signal variance × spatial kernel + white noise``.

    ``ard=True`` assigns one length scale to every transformed input feature.
    This is the RBF-ARD form used in the original ``dist_auto`` notebook.
    """
    if n_features < 1:
        raise ValueError("n_features must be positive")
    family = kernel_family.lower()
    length_scale: float | np.ndarray
    length_scale = np.ones(n_features, dtype=float) if ard else 1.0
    if family == "rbf":
        spatial = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
    elif family == "matern":
        spatial = Matern(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            nu=matern_nu,
        )
    else:
        raise ValueError("kernel_family must be 'matern' or 'rbf'")
    return (
        ConstantKernel(1.0, signal_bounds) * spatial
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=noise_bounds)
    )


def fitted_kernel_diagnostics(
    gpr: GaussianProcessRegressor,
    *,
    length_scale_bounds: tuple[float, float],
) -> dict[str, Any]:
    """Return compact diagnostics for an optimized signal-plus-white kernel."""
    fitted = gpr.kernel_
    spatial = fitted.k1.k2
    scales = np.atleast_1d(np.asarray(spatial.length_scale, dtype=float))
    lower, upper = length_scale_bounds
    # Optimizer values at a box constraint can differ by a few ulps after exp().
    lower_hits = int(np.sum(scales <= lower * (1.0 + 1e-6)))
    upper_hits = int(np.sum(scales >= upper * (1.0 - 1e-6)))
    return {
        "optimized_kernel": str(fitted),
        "signal_variance": float(fitted.k1.k1.constant_value),
        "noise_variance": float(fitted.k2.noise_level),
        "length_scale_count": int(scales.size),
        "length_scale_min": float(np.min(scales)),
        "length_scale_median": float(np.median(scales)),
        "length_scale_max": float(np.max(scales)),
        "length_scales_at_lower_bound": lower_hits,
        "length_scales_at_upper_bound": upper_hits,
        "length_scales_at_upper_fraction": float(upper_hits / scales.size),
        "log_marginal_likelihood": float(gpr.log_marginal_likelihood_value_),
    }
