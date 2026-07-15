#!/usr/bin/env python3
"""Generate the two source notebooks from reviewed Python cell strings."""

from __future__ import annotations

import json
import re
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:
    # Keep artifact generation dependency-free in minimal Python environments.
    class _V4:
        @staticmethod
        def new_notebook():
            return {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}

        @staticmethod
        def new_markdown_cell(source):
            return {"cell_type": "markdown", "metadata": {}, "source": source}

        @staticmethod
        def new_code_cell(source):
            return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

    class _NBFallback:
        NotebookNode = dict
        v4 = _V4()

        @staticmethod
        def write(notebook, path):
            Path(path).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    nbf = _NBFallback()


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)


def notebook_one() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"].update(
        {
            "colab": {"name": "01_RF_and_GPR_handoff_Colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# 01 — RF再現とGaussian Process Regression\n\n"
            "受領した3つのCSVを使い、(1) R `randomForest` + `pls` の再現、(2) 同じ固定10-foldでGPRを比較します。\n\n"
            "重要: 標準化、角度変換、分散ゼロ列除去、`X_proc` のPCAは各訓練foldだけでfitします。"""
        ),
        nbf.v4.new_code_cell(
            """# GitHubから開いたColabはファイルを自動取得しないため、最初にcloneします。\n"
            "from pathlib import Path\n"
            "import os, subprocess, sys\n\n"
            "REPO_URL = 'https://github.com/futoshi-futami/Chemistory.git'\n"
            "REPO_REF = os.environ.get('CHEMISTORY_REF', 'agent/rbf-kernel-comparison')\n"
            "cwd = Path.cwd()\n"
            "if (cwd / 'pyproject.toml').exists():\n"
            "    PROJECT_ROOT = cwd\n"
            "elif (Path('/content/Chemistory') / 'pyproject.toml').exists():\n"
            "    PROJECT_ROOT = Path('/content/Chemistory')\n"
            "elif 'google.colab' in sys.modules:\n"
            "    PROJECT_ROOT = Path('/content/Chemistory')\n"
            "    subprocess.run(['git', 'clone', '--depth', '1', '--branch', REPO_REF, REPO_URL, str(PROJECT_ROOT)], check=True)\n"
            "else:\n"
            "    raise FileNotFoundError('Chemistory project rootでノートブックを実行してください。')\n"
            "os.chdir(PROJECT_ROOT)\n"
            "subprocess.run([sys.executable, 'scripts/prepare_data.py'], check=True)\n"
            "# このセルだけで、後続セルのchemistory_gpr importまで準備します。\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT_ROOT)], check=True)\n"
            "src_dir = str(PROJECT_ROOT / 'src')\n"
            "if src_dir not in sys.path:\n"
            "    sys.path.insert(0, src_dir)\n"
            "import importlib\n"
            "importlib.invalidate_caches()\n"
            "import chemistory_gpr\n"
            "print('PROJECT_ROOT =', PROJECT_ROOT)\n"
            "print('chemistory_gpr =', chemistory_gpr.__file__)"""
        ),
        nbf.v4.new_code_cell(
            """# Python環境\n"
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import display\n"
            "RESULTS = PROJECT_ROOT / 'results'\n"
            "RESULTS.mkdir(exist_ok=True)"""
        ),
        nbf.v4.new_markdown_cell("## A. 入力データの整合性検査"),
        nbf.v4.new_code_cell(
            """from chemistory_gpr.handoff import load_handoff_data\n\n"
            "DATA_DIR = PROJECT_ROOT / 'data' / 'gpr_handoff'\n"
            "data = load_handoff_data(DATA_DIR)\n"
            "print('n =', len(data.y))\n"
            "print('base feature count =', data.base.shape[1] - 2)\n"
            "print('X_proc feature count =', data.xproc.shape[1] - 1)\n"
            "print('fold counts =', pd.Series(data.fold_id).value_counts().sort_index().to_dict())\n"
            "display(pd.read_csv(DATA_DIR / '04_reference_RF_results.csv'))"""
        ),
        nbf.v4.new_markdown_cell(
            """## B. R randomForestの再現\n\n"
            "RFはRの `randomForest` と `pls::plsr(method='simpls')` をそのまま使います。"
            "R 3.6で `sample()` の方式が変わったため、現在の `Rejection` と旧 `Rounding` を両方試します。"""
        ),
        nbf.v4.new_code_cell(
            """import shutil\n\n"
            "IN_COLAB = 'google.colab' in sys.modules\n"
            "if shutil.which('Rscript') is None and IN_COLAB:\n"
            "    subprocess.run(['apt-get', 'update', '-qq'], check=True)\n"
            "    subprocess.run(['apt-get', 'install', '-y', '-qq', 'r-base', 'r-cran-randomforest', 'r-cran-pls'], check=True)\n"
            "elif shutil.which('Rscript') is None:\n"
            "    print('この環境にはRscriptがないためRFセルだけをスキップします。Colabでは自動導入されます。')\n"
            "else:\n"
            "    package_check = 'missing <- c(\"randomForest\",\"pls\")[!sapply(c(\"randomForest\",\"pls\"), requireNamespace, quietly=TRUE)]; if(length(missing)) install.packages(missing, repos=\"https://cloud.r-project.org\")'\n"
            "    subprocess.run(['Rscript', '-e', package_check], check=True)\n"
            "print('Rscript =', shutil.which('Rscript'))"""
        ),
        nbf.v4.new_code_cell(
            """if shutil.which('Rscript'):\n"
            "    subprocess.run([sys.executable, 'scripts/run_rf_reproduction.py'], check=True)\n"
            "    rf_metrics = pd.read_csv(RESULTS / 'rf_reproduction_all_metrics.csv')\n"
            "    rf_comparison = pd.read_csv(RESULTS / 'rf_reproduction_comparison.csv')\n"
            "    display(rf_metrics)\n"
            "    display(rf_comparison)\n"
            "else:\n"
            "    print('RF再現は未実行です。')"""
        ),
        nbf.v4.new_markdown_cell(
            """## C. GPR候補の固定10-fold比較\n\n"
            "周期角度 + `X_proc` fold内PCA8を共通にし、Matérn 1/2、Matérn 3/2、等方RBF、"
            "元の `dist_auto` と同型のRBF-ARDを比較します。すべて `signal × kernel + WhiteKernel` です。\n\n"
            "`ARD_RESTARTS=0` でも帯域幅は第二種最尤法で最適化されます。元コードどおり5回の追加初期化を"
            "行うには5へ変更してください（計算時間は大きく増えます）。`R2` は相関係数の二乗ではなく、"
            "`1-SSE/SST` です。"""
        ),
        nbf.v4.new_code_cell(
            """from chemistory_gpr.handoff import cross_validate_handoff, handoff_kernel_candidates\n\n"
            "ARD_RESTARTS = 0  # 元のdist_autoと同じ追加初期化回数は5（かなり時間がかかります）\n"
            "candidate_metrics = []\n"
            "candidate_predictions = {}\n"
            "for config in handoff_kernel_candidates(rbf_ard_restarts=ARD_RESTARTS):\n"
            "    print('Running', config.name)\n"
            "    prediction, metrics = cross_validate_handoff(data, config)\n"
            "    candidate_predictions[config.name] = prediction\n"
            "    metrics = {k: v for k, v in metrics.items() if k not in {'kernels','kernel_diagnostics'}}\n"
            "    candidate_metrics.append(metrics)\n"
            "    prediction.to_csv(RESULTS / f'gpr_handoff_oof_{config.name}.csv', index=False)\n"
            "metric_table = pd.DataFrame(candidate_metrics).sort_values('R2', ascending=False)\n"
            "metric_table.to_csv(RESULTS / 'gpr_handoff_metrics.csv', index=False)\n"
            "metric_table['upper_bound_fraction'] = (\n"
            "    metric_table['length_scales_at_upper_bound_total'] / metric_table['length_scales_total']\n"
            ")\n"
            "display(metric_table[['model','kernel_family','ard','optimizer_restarts','R2','corr2','RMSE','MAE','coverage_95','NLPD','upper_bound_fraction']])"""
        ),
        nbf.v4.new_code_cell(
            """# 最良候補のOOF予測と不確実性\n"
            "best_name = metric_table.iloc[0]['model']\n"
            "best = candidate_predictions[best_name]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].errorbar(best['y'], best['pred_mean'], yerr=1.96*best['pred_std'], fmt='o', ms=4, alpha=.65)\n"
            "limits = [min(best['y'].min(), best['lower_95'].min()), max(best['y'].max(), best['upper_95'].max())]\n"
            "axes[0].plot(limits, limits, '--', color='black')\n"
            "axes[0].set(xlabel='Observed y', ylabel='OOF predicted y', title=f'{best_name}: mean ± 95%')\n"
            "ordered = best.sort_values('pred_std').reset_index(drop=True)\n"
            "axes[1].fill_between(np.arange(len(ordered)), ordered['lower_95'], ordered['upper_95'], alpha=.25, label='95% interval')\n"
            "axes[1].plot(ordered['y'].to_numpy(), '.', ms=3, label='observed')\n"
            "axes[1].plot(ordered['pred_mean'].to_numpy(), '-', lw=1, label='mean')\n"
            "axes[1].set(title='OOF predictive intervals (sorted by uncertainty)', xlabel='sample', ylabel='y')\n"
            "axes[1].legend()\n"
            "plt.tight_layout(); plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """### 読み方\n\n"
            "既定データではMatérn 3/2がわずかに最良で、等方RBFはほぼ同等です。"
            "RBF-ARDは多数の長さ尺度が上限に達し、OOF性能も低下します。"
            "170件に対して特徴別帯域幅を約120個推定するため、過剰パラメータ化になっています。"
            "95% coverageが名目値に近いかも同時に確認してください。"
            "候補選択自体も同じCV結果を見ているため、最終的な外部評価には新しい独立データが必要です。"""
        ),
    ]
    return nb


def notebook_two() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"].update(
        {
            "colab": {"name": "02_dist_auto_GPR_Colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# 02 — dist_autoへのGPR適用\n\n"
            "元コードと同じく、1つのtagを丸ごとテストにして、残りtagから予測します。"
            "既定は `TEST_TAG='10'` です。予測平均に加え、不確実性と95%区間を作ります。"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path\n"
            "import os, subprocess, sys\n\n"
            "REPO_URL = 'https://github.com/futoshi-futami/Chemistory.git'\n"
            "REPO_REF = os.environ.get('CHEMISTORY_REF', 'agent/rbf-kernel-comparison')\n"
            "cwd = Path.cwd()\n"
            "if (cwd / 'pyproject.toml').exists():\n"
            "    PROJECT_ROOT = cwd\n"
            "elif (Path('/content/Chemistory') / 'pyproject.toml').exists():\n"
            "    PROJECT_ROOT = Path('/content/Chemistory')\n"
            "elif 'google.colab' in sys.modules:\n"
            "    PROJECT_ROOT = Path('/content/Chemistory')\n"
            "    subprocess.run(['git', 'clone', '--depth', '1', '--branch', REPO_REF, REPO_URL, str(PROJECT_ROOT)], check=True)\n"
            "else:\n"
            "    raise FileNotFoundError('Chemistory project rootでノートブックを実行してください。')\n"
            "os.chdir(PROJECT_ROOT)\n"
            "subprocess.run([sys.executable, 'scripts/prepare_data.py'], check=True)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT_ROOT)], check=True)\n"
            "src_dir = str(PROJECT_ROOT / 'src')\n"
            "if src_dir not in sys.path:\n"
            "    sys.path.insert(0, src_dir)\n"
            "import importlib\n"
            "importlib.invalidate_caches()\n"
            "import chemistory_gpr\n"
            "print('PROJECT_ROOT =', PROJECT_ROOT)\n"
            "print('chemistory_gpr =', chemistory_gpr.__file__)"""
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import plotly.graph_objects as go\n"
            "from IPython.display import display\n"
            "from chemistory_gpr.dist_auto import (\n"
            "    default_dist_auto_candidates, fit_held_out_tag, leave_one_tag_out,\n"
            "    load_dist_auto_data, make_grid, predict_grid, standardized_tag_centroid_distances,\n"
            "    summarize_dist_auto_metrics,\n"
            ")\n\n"
            "DATA_DIR = PROJECT_ROOT / 'data' / 'dist_auto'\n"
            "RESULTS = PROJECT_ROOT / 'results'\n"
            "RESULTS.mkdir(exist_ok=True)\n"
            "TEST_TAG = '10'\n"
            "GRID_SIZE = 30\n"
            "ARD_RESTARTS = 0  # 元コードどおりは5。0でも第二種最尤法による最適化は実施\n"
            "RUN_FULL_LOTO = False  # Trueで全tag×全kernelを再計算（Colabでは時間を要します）"""
        ),
        nbf.v4.new_markdown_cell("## A. データとtag外挿設定"),
        nbf.v4.new_code_cell(
            """data = load_dist_auto_data(DATA_DIR)\n"
            "summary = pd.DataFrame({'tag': data.tags, 'y': data.y}).groupby('tag', sort=False)['y'].agg(['count','mean','std','min','max'])\n"
            "print('common Xmat features =', len(data.feature_columns))\n"
            "display(summary)\n"
            "centroid_distances = standardized_tag_centroid_distances(data)\n"
            "display(centroid_distances.style.format('{:.3f}'))"""
        ),
        nbf.v4.new_markdown_cell(
            """`tag=b` は目的変数の平均・分散に加え、標準化Xmat空間の重心も他tagから大きく離れます。"
            "これは物理条件名を特定するものではありませんが、構造領域外挿であることを支持します。"
            "したがって、tag 10だけでなく全tagのleave-one-tag-out診断も後で確認します。"""
        ),
        nbf.v4.new_markdown_cell(
            """## B. 指定tagを完全にhold outしてカーネルを比較\n\n"
            "前回モデルに加え、Xmatだけを訓練foldで標準化する元の前処理に揃えて、"
            "Matérn 1/2、Matérn 3/2、等方RBF、RBF-ARDを比較します。"""
        ),
        nbf.v4.new_code_cell(
            """candidates = default_dist_auto_candidates(rbf_ard_restarts=ARD_RESTARTS)\n"
            "test_rows, candidate_models, candidate_heldout = [], {}, {}\n"
            "for config in candidates:\n"
            "    print('Running', config.name)\n"
            "    fitted, prediction, metrics = fit_held_out_tag(data, TEST_TAG, config)\n"
            "    candidate_models[config.name] = fitted\n"
            "    candidate_heldout[config.name] = prediction\n"
            "    test_rows.append(metrics)\n"
            "comparison = pd.DataFrame(test_rows).sort_values('R2', ascending=False)\n"
            "comparison.to_csv(RESULTS / f'dist_auto_test_{TEST_TAG}_kernel_comparison.csv', index=False)\n"
            "display(comparison[['model','kernel_family','ard','optimizer_restarts','R2','corr2','RMSE','MAE','coverage_95','length_scales_at_upper_bound','length_scale_count']])\n"
            "best_name = comparison.iloc[0]['model']\n"
            "best_config = next(c for c in candidates if c.name == best_name)\n"
            "model = candidate_models[best_name]\n"
            "heldout = candidate_heldout[best_name]\n"
            "heldout.to_csv(RESULTS / f'dist_auto_test_{TEST_TAG}_predictions.csv', index=False)\n"
            "print('Selected for the surface:', best_name)\n"
            "print('optimized kernel:', comparison.iloc[0]['optimized_kernel'])"""
        ),
        nbf.v4.new_code_cell(
            """fig, ax = plt.subplots(figsize=(5.5, 5))\n"
            "ax.errorbar(heldout['y'], heldout['pred_mean'], yerr=1.96*heldout['pred_std'], fmt='o', ms=4, alpha=.7)\n"
            "lims = [min(heldout['y'].min(), heldout['lower_95'].min()), max(heldout['y'].max(), heldout['upper_95'].max())]\n"
            "ax.plot(lims, lims, '--', color='black')\n"
            "ax.set(xlabel='Observed y', ylabel='Predicted y', title=f'Held-out tag {TEST_TAG}: mean ± 95%')\n"
            "plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """この比較ではhold-out tagの真値を見てカーネルを選んでいます。したがって最良値は候補比較値であり、"
            "選択後の独立テスト値ではありません。次の新規tagまたは構造では、選んだカーネルを事前固定して評価してください。"""
        ),
        nbf.v4.new_markdown_cell(
            """## C. leave-one-tag-out診断\n\n"
            "全tag比較は高次元RBF-ARDのため時間がかかります。`RUN_FULL_LOTO=True` なら再計算し、"
            "FalseならGitHubに保存済みの結果表を読みます。"""
        ),
        nbf.v4.new_code_cell(
            """if RUN_FULL_LOTO:\n"
            "    all_predictions, all_metrics = [], []\n"
            "    for config in candidates:\n"
            "        print('Running all held-out tags:', config.name)\n"
            "        group_oof, group_metrics = leave_one_tag_out(data, config, n_jobs=1)\n"
            "        group_oof.insert(0, 'model', config.name)\n"
            "        all_predictions.append(group_oof)\n"
            "        all_metrics.append(group_metrics)\n"
            "    pd.concat(all_predictions, ignore_index=True).to_csv(RESULTS / 'dist_auto_kernel_comparison_predictions.csv', index=False)\n"
            "    group_metrics = pd.concat(all_metrics, ignore_index=True)\n"
            "    group_metrics.to_csv(RESULTS / 'dist_auto_kernel_comparison_metrics.csv', index=False)\n"
            "else:\n"
            "    group_metrics = pd.read_csv(RESULTS / 'dist_auto_kernel_comparison_metrics.csv')\n"
            "display(group_metrics[['model','test_tag','kernel_family','ard','R2','corr2','RMSE','MAE','coverage_95']])\n"
            "group_summary = summarize_dist_auto_metrics(group_metrics)\n"
            "group_summary.to_csv(RESULTS / 'dist_auto_kernel_comparison_summary.csv', index=False)\n"
            "display(group_summary)"""
        ),
        nbf.v4.new_markdown_cell(
            """`corr2` は元コードとの比較用です。主指標は `R2=1-SSE/SST` です。"
            "相関が高くても平均がずれると `corr2` は高いまま `R2` が悪化するので、両者を混同しないでください。\n\n"
            "保存済み結果ではRBF-ARDがtag 10–25の4条件でR²首位、等方RBFがtag別平均R²と全OOF R²で僅差の首位、"
            "Matérn 3/2が95% coverageで首位です。tag bの最良R²は負のままで、カーネル変更だけでは解決していません。"""
        ),
        nbf.v4.new_markdown_cell("## D. 新しいxyグリッドの予測面と不確実性"),
        nbf.v4.new_code_cell(
            """grid, grid_features = make_grid(DATA_DIR, TEST_TAG, data.feature_columns, grid_size=GRID_SIZE)\n"
            "surface = predict_grid(model, grid, grid_features)\n"
            "surface.insert(0, 'model', best_name)\n"
            "surface.to_csv(RESULTS / f'dist_auto_surface_{TEST_TAG}.csv', index=False)\n"
            "mean_pivot = surface.pivot(index='y', columns='x', values='pred_mean')\n"
            "std_pivot = surface.pivot(index='y', columns='x', values='pred_std')\n"
            "fig = go.Figure(go.Surface(\n"
            "    x=mean_pivot.columns, y=mean_pivot.index, z=mean_pivot.to_numpy(),\n"
            "    surfacecolor=std_pivot.to_numpy(), colorbar={'title':'predictive std'},\n"
            "))\n"
            "fig.update_layout(title=f'GPR mean surface — held-out tag {TEST_TAG}', scene={'xaxis_title':'x','yaxis_title':'y','zaxis_title':'prediction'})\n"
            "fig.show()"""
        ),
        nbf.v4.new_code_cell(
            """fig = go.Figure(go.Heatmap(x=std_pivot.columns, y=std_pivot.index, z=std_pivot.to_numpy(), colorbar={'title':'std'}))\n"
            "fig.update_layout(title='Predictive uncertainty', xaxis_title='x', yaxis_title='y')\n"
            "fig.show()\n\n"
            "best_mean = surface.loc[surface['pred_mean'].idxmax(), ['x','y','pred_mean','pred_std','lower_95']]\n"
            "best_lcb = surface.loc[surface['lower_confidence_bound'].idxmax(), ['x','y','pred_mean','pred_std','lower_confidence_bound']]\n"
            "print('Maximum predictive mean:\\n', best_mean.to_string())\n"
            "print('\\nMaximum 95% lower confidence bound:\\n', best_lcb.to_string())"""
        ),
        nbf.v4.new_markdown_cell(
            """### 回転角について\n\n"
            "このColab版は検証可能な `angle=0` を対象にします。受領ZIPの `rotate_xyz.exe` はWindows専用で、"
            "同梱の `source.xyz` と `*-altered.xyz` はNULのみでした。"
            "非ゼロ回転を厳密に追加するには、回転プログラムのCソースまたは正常な回転前後XYZが必要です。"""
        ),
    ]
    return nb


def main() -> None:
    notebooks = [
        (notebook_one(), OUT / "01_RF_and_GPR_handoff_Colab.ipynb"),
        (notebook_two(), OUT / "02_dist_auto_GPR_Colab.ipynb"),
    ]
    for notebook, path in notebooks:
        # The long cells above use visually aligned adjacent string fragments.
        # Remove only their source-code quote boundaries from the emitted text.
        for cell in notebook["cells"]:
            cell["source"] = re.sub(r'"\n\s*"', "", cell["source"])
        nbf.write(notebook, path)


if __name__ == "__main__":
    main()
