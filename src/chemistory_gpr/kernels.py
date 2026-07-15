"""Kernel builders and diagnostics shared by the GPR workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)


def build_signal_plus_white_kernel(
    *,
    kernel_family: str,
    n_features: int,
    ard: bool,
    matern_nu: float,
    signal_bounds: tuple[float, float],
    length_scale_bounds: tuple[float, float],
    noise_bounds: tuple[float, float],
    rq_alpha_bounds: tuple[float, float] = (1e-2, 1e3),
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
    elif family == "rational_quadratic":
        if ard:
            raise ValueError("RationalQuadratic in scikit-learn is isotropic; ard must be False")
        spatial = RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
            length_scale_bounds=length_scale_bounds,
            alpha_bounds=rq_alpha_bounds,
        )
    elif family == "linear":
        if ard:
            raise ValueError("The linear kernel does not use ARD length scales")
        spatial = DotProduct(sigma_0=1.0, sigma_0_bounds=length_scale_bounds)
    else:
        raise ValueError(
            "kernel_family must be 'matern', 'rbf', 'rational_quadratic', or 'linear'"
        )
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
    lower, upper = length_scale_bounds
    if hasattr(spatial, "length_scale"):
        scales = np.atleast_1d(np.asarray(spatial.length_scale, dtype=float))
        # Optimizer values at a box constraint can differ by a few ulps after exp().
        lower_hits = int(np.sum(scales <= lower * (1.0 + 1e-6)))
        upper_hits = int(np.sum(scales >= upper * (1.0 - 1e-6)))
        scale_summary = {
            "length_scale_count": int(scales.size),
            "length_scale_min": float(np.min(scales)),
            "length_scale_median": float(np.median(scales)),
            "length_scale_max": float(np.max(scales)),
            "length_scales_at_lower_bound": lower_hits,
            "length_scales_at_upper_bound": upper_hits,
            "length_scales_at_upper_fraction": float(upper_hits / scales.size),
        }
    else:
        scale_summary = {
            "length_scale_count": 0,
            "length_scale_min": np.nan,
            "length_scale_median": np.nan,
            "length_scale_max": np.nan,
            "length_scales_at_lower_bound": 0,
            "length_scales_at_upper_bound": 0,
            "length_scales_at_upper_fraction": np.nan,
        }
    diagnostics = {
        "optimized_kernel": str(fitted),
        "signal_variance": float(fitted.k1.k1.constant_value),
        "noise_variance": float(fitted.k2.noise_level),
        "log_marginal_likelihood": float(gpr.log_marginal_likelihood_value_),
    }
    diagnostics.update(scale_summary)
    if isinstance(spatial, RationalQuadratic):
        diagnostics["rq_alpha"] = float(spatial.alpha)
    if isinstance(spatial, DotProduct):
        diagnostics["dot_sigma_0"] = float(spatial.sigma_0)
    return diagnostics
