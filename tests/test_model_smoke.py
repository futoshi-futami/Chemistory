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
from chemistory_gpr.geometry3d import derive_rotation_invariant_features
from chemistory_gpr.kernels import build_axis_environment_kernel
from chemistory_gpr.nested_group import build_nested_model, default_nested_candidates
from chemistory_gpr.visualization import (
    fit_full_interaction_gp,
    interaction_surface_figure,
    interaction_surface_table,
    molecular_axis_uncertainty_animation,
    raw_structure_figure,
)


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
    assert ("matern", False) in signatures
    assert ("rational_quadratic", False) in signatures
    assert ("linear", False) in signatures


def test_additional_handoff_kernels_build_with_expected_parameters():
    common = {
        "n_features": 12,
        "ard": False,
        "matern_nu": 1.5,
        "signal_bounds": (1e-2, 1e3),
        "length_scale_bounds": (1e-2, 1e3),
        "noise_bounds": (1e-6, 1e1),
    }
    rq = build_signal_plus_white_kernel(kernel_family="rational_quadratic", **common)
    linear = build_signal_plus_white_kernel(kernel_family="linear", **common)
    assert rq.k1.k2.alpha == 1.0
    assert linear.k1.k2.sigma_0 == 1.0


def test_axis_environment_interaction_kernel_and_model_predict():
    kernel = build_axis_environment_kernel(
        axis_dims=(0, 1, 2, 3),
        environment_dims=(4, 5),
        include_interaction=True,
    )
    assert len(kernel.theta) == 8

    data = load_handoff_data(ROOT / "data" / "gpr_handoff")
    train = data.fold_id != 1
    test = ~train
    candidate = next(
        item
        for item in default_nested_candidates()
        if item.name == "axis_environment_interaction_matern32"
    )
    model = build_nested_model(candidate, 123).fit(
        data.base.loc[train], data.xproc.loc[train], data.y[train]
    )
    mean, std = model.predict(data.base.loc[test], data.xproc.loc[test])
    assert np.isfinite(mean).all()
    assert (std > 0).all()


def test_rotation_invariant_geometry_features_on_synthetic_axis():
    rows = [
        ("s1", "C3", "C", -1.0, 0.0, 0.0),
        ("s1", "H3", "H", -2.0, 0.0, 0.0),
        ("s1", "C6", "C", 1.0, 0.0, 0.0),
        ("s1", "H6", "H", 2.0, 0.0, 0.0),
        ("s1", "Mg1", "Mg", -3.0, 1.0, 0.0),
        ("s1", "Mg2", "Mg", 3.0, 2.0, 0.0),
        ("s1", "O1", "O", -2.0, 2.0, 0.0),
        ("s1", "O2", "O", 2.0, 1.0, 0.0),
    ]
    coordinates = np.array(rows, dtype=object)
    import pandas as pd

    table = pd.DataFrame(
        coordinates, columns=["file_key", "atom_label", "element", "x", "y", "z"]
    )
    table[["x", "y", "z"]] = table[["x", "y", "z"]].astype(float)
    features = derive_rotation_invariant_features(table).iloc[0]
    assert np.isclose(features["axis_azimuth_cos"], 1.0)
    assert np.isclose(features["axis_abs_elevation_rad"], 0.0)
    assert np.isclose(features["antiparallel_deviation_rad"], 0.0)
    assert features["Mg_distance_min_H3_minus_H6"] < 0


def test_interactive_axis_animation_tracks_high_angle_trajectory():
    import pandas as pd

    data = load_handoff_data(ROOT / "data" / "gpr_handoff")
    predictions = pd.read_csv(
        ROOT / "results" / "gpr_handoff_fixed10_next_models_predictions.csv"
    )
    figure = molecular_axis_uncertainty_animation(data.base, predictions)
    assert len(figure.frames) == 7
    assert len(figure.data) == 8
    assert "0-0-3-18" in figure.frames[0].layout.title.text


def test_interaction_gp_surface_contains_mean_and_uncertainty_views():
    data = load_handoff_data(ROOT / "data" / "gpr_handoff")
    model = fit_full_interaction_gp(data)
    surface = interaction_surface_table(
        data,
        model=model,
        surface_feature="axis_tilt_deg",
        angle_points=5,
        feature_points=4,
    )
    assert len(surface) == 20
    assert np.isfinite(surface["pred_mean"]).all()
    assert (surface["pred_std"] > 0).all()
    figure = interaction_surface_figure(surface, data.base)
    assert len(figure.data) == 6
    assert len(figure.layout.updatemenus[0].buttons) == 5


def test_raw_coordinate_viewer_uses_actual_atoms_without_bond_inference():
    import pandas as pd

    coordinates = pd.DataFrame(
        [
            ("s1", "C3", "C", 0.0, 0.0, 0.0),
            ("s1", "H3", "H", 1.0, 0.0, 0.0),
            ("s1", "C6", "C", 0.0, 1.0, 0.0),
            ("s1", "H6", "H", -1.0, 1.0, 0.0),
            ("s1", "Mg1", "Mg", 2.0, 0.0, 0.0),
            ("s1", "O1", "O", -2.0, 0.0, 0.0),
        ],
        columns=["file_key", "atom_label", "element", "x", "y", "z"],
    )
    figure = raw_structure_figure(coordinates, "s1")
    assert {trace.name for trace in figure.data} >= {"C", "H", "Mg", "O", "C3–H3", "C6–H6"}
