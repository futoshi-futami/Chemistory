from pathlib import Path

import numpy as np

from chemistory_gpr.dist_auto import load_dist_auto_data
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
