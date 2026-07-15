from pathlib import Path

import numpy as np

from chemistory_gpr.dist_auto import (
    DistAutoGPRConfig,
    dist_auto_kernel_candidates,
    fit_held_out_tag,
    load_dist_auto_data,
)
from chemistory_gpr.handoff import (
    HandoffGPR,
    HandoffGPRConfig,
    handoff_kernel_candidates,
    load_handoff_data,
)
from chemistory_gpr.kernels import build_signal_plus_white_kernel


ROOT = Path(__file__).resolve().parents[1]


def test_handoff_one_fold_predicts_finite_values():
    data = load_handoff_data(ROOT / "data" / "gpr_handoff")
    train = data.fold_id != 1
    test = ~train
    model = HandoffGPR(HandoffGPRConfig()).fit(data.base.loc[train], data.xproc.loc[train], data.y[train])
    mean, std = model.predict(data.base.loc[test], data.xproc.loc[test])
    assert len(mean) == len(std) == 17
    assert (std > 0).all()


def test_dist_auto_tag10_reproduces_high_predictive_accuracy():
    data = load_dist_auto_data(ROOT / "data" / "dist_auto")
    _, _, metrics = fit_held_out_tag(data, "10", DistAutoGPRConfig())
    assert metrics["R2"] > 0.95
    assert metrics["coverage_95"] > 0.85


def test_rbf_ard_candidate_matches_original_kernel_shape():
    config = dist_auto_kernel_candidates(rbf_ard_restarts=5)[-1]
    kernel = build_signal_plus_white_kernel(
        kernel_family=config.kernel_family,
        n_features=309,
        ard=config.ard,
        matern_nu=config.matern_nu,
        signal_bounds=config.signal_bounds,
        length_scale_bounds=config.length_scale_bounds,
        noise_bounds=config.noise_bounds,
    )
    assert config.optimizer_restarts == 5
    assert config.include_xy is False
    assert config.drop_constant_features is False
    assert np.asarray(kernel.k1.k2.length_scale).shape == (309,)


def test_handoff_kernel_candidates_include_isotropic_and_ard_rbf():
    candidates = handoff_kernel_candidates(rbf_ard_restarts=0)
    signatures = {(item.kernel_family, item.ard) for item in candidates}
    assert ("rbf", False) in signatures
    assert ("rbf", True) in signatures
