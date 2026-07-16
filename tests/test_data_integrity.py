from pathlib import Path

import numpy as np
import pandas as pd

from chemistory_gpr.angle_report import derive_angle_coordinates
from chemistory_gpr.group_validation import make_prefix_group_folds
from chemistory_gpr.handoff import (
    HandoffFeatureTransformer,
    HandoffGPRConfig,
    load_handoff_data,
)
from chemistory_gpr.physical_features import (
    COMPACT_AXIS_COLUMNS,
    candidate_group_labels,
    compact_axis_base,
    file_key_token_diagnostics,
    parse_file_key_tokens,
)


ROOT = Path(__file__).resolve().parents[1]


def test_handoff_alignment_and_shape():
    data = load_handoff_data(ROOT / "data" / "gpr_handoff")
    assert data.base.shape == (170, 113)
    assert data.xproc.shape == (170, 3103)
    assert np.bincount(data.fold_id)[1:].tolist() == [17] * 10


def test_handoff_primary_report_prioritizes_rf_comparison_and_behavior():
    comparison = pd.read_csv(ROOT / "results" / "gpr_handoff_primary_comparison.csv")
    fold_metrics = pd.read_csv(ROOT / "results" / "gpr_handoff_all_kernel_fold_metrics.csv")
    behavior = pd.read_csv(ROOT / "results" / "gpr_handoff_behavior_summary.csv").iloc[0]
    assert comparison.iloc[0]["model"] == "base_cyclic_xproc_pca8_matern32"
    assert comparison.iloc[0]["R2"] > comparison.loc[
        comparison["source"] == "RF_reported_reference", "R2"
    ].iloc[0]
    assert len(fold_metrics) == 7 * 10
    assert behavior["fraction_samples_gpr_lower_abs_error_than_rf"] > 0.5


def test_handoff_angles_form_one_nearly_antiparallel_molecular_axis():
    base = pd.read_csv(ROOT / "data" / "gpr_handoff" / "01_base_summary_first_angle.csv")
    angles = derive_angle_coordinates(base)
    assert angles["axis_angle_deg"].between(-30, 70).all()
    assert angles["antiparallel_deviation_deg"].max() < 3.1
    assert angles["antiparallel_deviation_deg"].mean() < 0.5
    assert np.allclose(angles["dot_C3H3_C6H6"], -1.0, atol=1e-4)

    high_angle = angles["axis_angle_deg_bin"].eq("[50,70]")
    low_response = angles["y"].lt(30.0)
    assert angles.loc[
        high_angle & low_response, "axis_abs_elevation_deg_proxy"
    ].mean() < (
        angles.loc[
            high_angle & ~low_response, "axis_abs_elevation_deg_proxy"
        ].mean()
        - 10.0
    )


def test_handoff_high_angle_regime_favors_rf_and_rougher_gpr():
    winners = pd.read_csv(ROOT / "results" / "gpr_handoff_angle_winners.csv")
    row = winners.loc[
        (winners["angle_view"] == "molecular_axis")
        & (winners["angle_bin"] == "[50,70]")
    ].iloc[0]
    assert row["best_overall_model"] == "RF_current_residualPLS5"
    assert row["best_GPR_model"] == "base_cyclic_xproc_pca8_matern12"

    metrics = pd.read_csv(ROOT / "results" / "gpr_handoff_angle_method_metrics.csv")
    matern32 = metrics.loc[
        (metrics["angle_view"] == "molecular_axis")
        & (metrics["angle_bin"] == "[50,70]")
        & (metrics["model"] == "base_cyclic_xproc_pca8_matern32")
    ].iloc[0]
    assert matern32["SSE_share_within_view_model"] > 0.5


def test_candidate_file_key_series_are_not_split_in_group_folds():
    data = load_handoff_data(ROOT / "data" / "gpr_handoff")
    folds = make_prefix_group_folds(data)
    assert folds["series_prefix_candidate"].nunique() == 30
    assert folds.groupby("series_prefix_candidate")["group_fold"].nunique().eq(1).all()
    assert folds["group_fold"].value_counts().between(16, 18).all()


def test_group_holdout_exposes_structure_series_extrapolation_gap():
    metrics = pd.read_csv(ROOT / "results" / "gpr_handoff_group10_prefix_metrics.csv")
    assert metrics.iloc[0]["kernel_family"] == "rational_quadratic"
    matern32 = metrics.loc[metrics["model"].str.endswith("matern32")].iloc[0]
    assert matern32["R2"] < 0

    series = pd.read_csv(ROOT / "results" / "gpr_handoff_series_summary.csv")
    low_branch = set(
        series.loc[
            series["n_high_angle_y_below_30"] > 0, "series_prefix_candidate"
        ]
    )
    assert low_branch == {"0-0-2-16", "0-0-3-18"}


def test_file_key_tokens_support_only_provisional_physical_interpretation():
    data = load_handoff_data(ROOT / "data" / "gpr_handoff")
    tokens = parse_file_key_tokens(data.file_key)
    diagnostics = file_key_token_diagnostics(data.base).set_index("token_position")
    assert tokens[["token_1", "token_2"]].nunique().eq(1).all()
    assert diagnostics.loc[3, "spearman_vs_y"] > 0.65
    assert diagnostics.loc[4, "spearman_vs_axis_angle_deg"] > 0.8
    assert diagnostics.loc[5, "spearman_vs_antiparallel_deviation_deg"] > 0.9
    groups = candidate_group_labels(data.file_key, "trajectory")
    assert groups.nunique() == 30


def test_compact_axis_replaces_redundant_angles_with_four_coordinates():
    data = load_handoff_data(ROOT / "data" / "gpr_handoff")
    compact = compact_axis_base(data.base)
    assert set(COMPACT_AXIS_COLUMNS).issubset(compact.columns)
    assert not {
        "C3H3_angle_xy",
        "C6H6_angle_xy",
        "angle_diff_C3_C6",
        "dot_C3H3_C6H6",
    }.intersection(compact.columns)
    assert np.allclose(
        compact["axis_azimuth_sin"] ** 2 + compact["axis_azimuth_cos"] ** 2,
        1.0,
    )


def test_product_gp_partition_is_four_axis_plus_110_environment_dimensions():
    data = load_handoff_data(ROOT / "data" / "gpr_handoff")
    compact = compact_axis_base(data.base)
    transformer = HandoffFeatureTransformer(
        HandoffGPRConfig(cyclic_angles=False, use_xproc=True, xproc_components=8)
    )
    design = transformer.fit_transform(compact, data.xproc)
    axis = [
        column
        for column in transformer.base_selected_columns_
        if column in COMPACT_AXIS_COLUMNS
    ]
    environment_base = [
        column
        for column in transformer.base_selected_columns_
        if column not in COMPACT_AXIS_COLUMNS
    ]
    assert axis == list(COMPACT_AXIS_COLUMNS)
    assert len(environment_base) == 102
    assert design.shape == (170, 114)
    assert set(transformer.base_columns_) - set(transformer.base_selected_columns_) == {
        "first_invd_LH3_atomOther",
        "first_invd_LH6_atomOther",
    }


def test_received_rf_trajectory_results_use_group_holdout_and_match_gp_comparison():
    metrics = pd.read_csv(ROOT / "results" / "gpr_handoff_rf_trajectory_metrics.csv")
    final = metrics.loc[metrics["model"].eq("RF_R_base_plus_residualPLS5")].set_index(
        "evaluation_split"
    )
    assert np.isclose(final.loc["trajectory_group5", "R2"], 0.0259956, atol=1e-6)
    assert np.isclose(final.loc["trajectory_group10", "R2"], -0.212622, atol=1e-6)

    predictions = pd.read_csv(
        ROOT / "results" / "gpr_handoff_rf_trajectory_predictions.csv"
    )
    assert predictions.groupby("evaluation_split")["file_key"].nunique().eq(170).all()
    assert predictions.groupby(
        ["evaluation_split", "trajectory_group_candidate"]
    )["fold"].nunique().eq(1).all()

    comparison = pd.read_csv(
        ROOT / "results" / "gpr_handoff_trajectory_model_comparison.csv"
    )
    for split_name in ("trajectory_group5", "trajectory_group10"):
        part = comparison.loc[comparison["evaluation_split"].eq(split_name)]
        gp_r2 = part.loc[
            part["model"].eq("axis_environment_interaction_matern32"), "R2"
        ].iloc[0]
        rf_r2 = part.loc[
            part["model"].eq("RF_R_base_plus_residualPLS5"), "R2"
        ].iloc[0]
        assert gp_r2 > rf_r2

    angle = pd.read_csv(
        ROOT / "results" / "gpr_handoff_trajectory_angle_model_comparison.csv"
    )
    gp_wins = angle.loc[
        angle["model"].eq("axis_environment_interaction_matern32")
        & angle["is_RMSE_winner"]
    ].groupby("evaluation_split").size()
    assert gp_wins.to_dict() == {"trajectory_group10": 5, "trajectory_group5": 4}


def test_interaction_gp_improves_fixed_and_grouped_predictions():
    fixed = pd.read_csv(ROOT / "results" / "gpr_handoff_fixed10_next_models_metrics.csv")
    group = pd.read_csv(ROOT / "results" / "gpr_handoff_group10_next_models_metrics.csv")
    fixed_interaction = fixed.loc[
        fixed["candidate"].eq("axis_environment_interaction_matern32")
    ].iloc[0]
    group_interaction = group.loc[
        group["candidate"].eq("axis_environment_interaction_matern32")
    ].iloc[0]
    group_legacy = group.loc[
        group["candidate"].eq("legacy_angles_global_matern32")
    ].iloc[0]
    assert fixed_interaction["R2"] > 0.97
    assert group_interaction["R2"] > 0.35
    assert group_interaction["RMSE"] < group_legacy["RMSE"]


def test_nested_group_selection_never_splits_a_trajectory():
    folds = pd.read_csv(ROOT / "results" / "gpr_handoff_nested_group_outer_folds.csv")
    assert folds["group"].nunique() == 30
    assert folds.groupby("group")["outer_fold"].nunique().eq(1).all()
    nested = pd.read_csv(ROOT / "results" / "gpr_handoff_nested_group_metrics.csv").iloc[0]
    assert nested["R2"] > 0.3
    assert nested["R2"] < 0.5


def test_saved_interaction_surfaces_include_mean_variance_and_reference_metadata():
    for name in ("axis_tilt", "h3_environment"):
        surface = pd.read_csv(
            ROOT / "results" / f"gpr_handoff_interaction_surface_{name}.csv"
        )
        assert len(surface) == 51 * 35
        assert np.isfinite(surface[["pred_mean", "pred_std", "lower_95", "upper_95"]]).all().all()
        assert (surface["pred_std"] > 0).all()
        assert surface["reference_file_key"].eq("0-0-3-18-10").all()
        assert (
            ROOT / "figures" / f"gpr_handoff_interaction_surface_{name}.html"
        ).stat().st_size > 10_000

    assert (
        ROOT / "figures" / "gpr_handoff_molecular_axis_uncertainty_animation.html"
    ).stat().st_size > 10_000
    assert (
        ROOT / "figures" / "gpr_handoff_static_overview.png"
    ).stat().st_size > 100_000
