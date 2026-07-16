import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_colab_notebooks_install_package_before_first_model_import():
    for name in ("01_RF_and_GPR_handoff_Colab.ipynb",):
        notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        setup = code_cells[0]
        assert "pip', 'install', '-q', '-e'" in setup
        assert "sys.path.insert(0, src_dir)" in setup
        assert "import chemistory_gpr" in setup
        joined = "\n".join(code_cells)
        assert "ARD_RESTARTS" in joined
        assert "rbf" in joined.lower()
        assert "RUN_NESTED_GROUP_CV" in joined
        assert "axis_environment_interaction_matern32" in joined
        assert "derive_rotation_invariant_features" in joined
        assert "molecular_axis_uncertainty_animation" in joined
        assert "interaction_surface_figure" in joined
        assert "oof_uncertainty_figure" in joined
