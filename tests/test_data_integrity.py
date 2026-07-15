from pathlib import Path

import numpy as np
import pandas as pd

from chemistory_gpr.dist_auto import load_dist_auto_data, standardized_tag_centroid_distances
from chemistory_gpr.angle_report import derive_angle_coordinates
from chemistory_gpr.handoff import load_handoff_data


ROOT = Path(__file__).resolve().parents[1]


def test_handoff_alignment_and_shape():
    data = load_handoff_data(ROOT / "data" / "gpr_handoff")
    assert data.base.shape == (170, 113)
    assert data.xproc.shape == (170, 3103)
    assert np.bincount(data.fold_id)[1:].tolist() == [17] * 10


def test_dist_auto_alignment_and_shape():
    data = load_dist_auto_data(ROOT / "data" / "dist_auto")
    assert data.X.shape == (330, 309)
    assert len(data.feature_columns) == 309
    assert set(data.tags) == {"a", "b", "10", "15", "20", "25"}
    assert not data.X.isna().any().any()


def test_dist_auto_tag_b_is_geometrically_outlying_in_xmat_space():
    data = load_dist_auto_data(ROOT / "data" / "dist_auto")
    distances = standardized_tag_centroid_distances(data)
    mean_other_distance = distances.mask(np.eye(len(distances), dtype=bool)).mean(axis=1)
    assert mean_other_distance.idxmax() == "b"
    assert mean_other_distance["b"] > 2 * mean_other_distance.drop("b").max()


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
