from pathlib import Path

from chemistory_gpr.dist_auto import DistAutoGPRConfig, fit_held_out_tag, load_dist_auto_data
from chemistory_gpr.handoff import HandoffGPR, HandoffGPRConfig, load_handoff_data


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
