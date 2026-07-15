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
            """# 01 — 主解析: RF基準とGaussian Processカーネル比較\n\n"
            "このノートブックが本プロジェクトの中心です。受領した3つのCSVを使い、"
            "(1) R `randomForest` + residual PLS5の再現、(2) 同じ固定10-foldで複数のGPRカーネルを比較、"
            "(3) 最良GPRがRFをどこで改善し、どこで失敗するかを確認します。\n\n"
            "重要: 標準化、角度変換、分散ゼロ列除去、`X_proc` のPCAは各訓練foldだけでfitします。"""
        ),
        nbf.v4.new_code_cell(
            """# GitHubから開いたColabはファイルを自動取得しないため、最初にcloneします。\n"
            "from pathlib import Path\n"
            "import os, subprocess, sys\n\n"
            "REPO_URL = 'https://github.com/futoshi-futami/Chemistory.git'\n"
            "REPO_REF = os.environ.get('CHEMISTORY_REF', 'main')\n"
            "FALLBACK_REF = 'agent/rbf-kernel-comparison'  # PR #2。mainへmerge後はfallback不要\n"
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
            "# PRのレビュー中も実行可能にし、merge後は自動的にmainを使います。\n"
            "if 'google.colab' in sys.modules and not (PROJECT_ROOT / 'src/chemistory_gpr/kernels.py').exists():\n"
            "    subprocess.run(['git', '-C', str(PROJECT_ROOT), 'fetch', '--depth', '1', 'origin', FALLBACK_REF], check=True)\n"
            "    subprocess.run(['git', '-C', str(PROJECT_ROOT), 'checkout', '--detach', 'FETCH_HEAD'], check=True)\n"
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
            "周期角度 + `X_proc` fold内PCA8を共通にし、Matérn 1/2・3/2・5/2、等方RBF、"
            "Rational Quadratic、RBF-ARD、Linearを比較します。すべてWhiteKernel付きです。\n\n"
            "既定では計算の重い120次元RBF-ARDだけコミット済み結果を読み、他6候補を再fitします。"
            "`INCLUDE_ARD=True` でARDも再計算できます。`ARD_RESTARTS=0` でも帯域幅は第二種最尤法で最適化されます。"
            "`R2` は相関係数の二乗ではなく、"
            "`1-SSE/SST` です。"""
        ),
        nbf.v4.new_code_cell(
            """import warnings\n"
            "from sklearn.exceptions import ConvergenceWarning\n"
            "from chemistory_gpr.handoff import cross_validate_handoff, handoff_kernel_candidates\n\n"
            "INCLUDE_ARD = False  # TrueではColabで数分以上かかることがあります\n"
            "ARD_RESTARTS = 0\n"
            "committed_metrics = pd.read_csv(RESULTS / 'gpr_handoff_metrics.csv')\n"
            "candidate_metrics = []\n"
            "candidate_predictions = {}\n"
            "configs = handoff_kernel_candidates(rbf_ard_restarts=ARD_RESTARTS)\n"
            "if not INCLUDE_ARD:\n"
            "    configs = [config for config in configs if not config.ard]\n"
            "warnings.filterwarnings('ignore', category=ConvergenceWarning)\n"
            "for config in configs:\n"
            "    print('Running', config.name)\n"
            "    prediction, metrics = cross_validate_handoff(data, config)\n"
            "    candidate_predictions[config.name] = prediction\n"
            "    metrics = {k: v for k, v in metrics.items() if k not in {'kernels','kernel_diagnostics'}}\n"
            "    candidate_metrics.append(metrics)\n"
            "    prediction.to_csv(RESULTS / f'gpr_handoff_oof_{config.name}.csv', index=False)\n"
            "metric_table = pd.DataFrame(candidate_metrics)\n"
            "if not INCLUDE_ARD:\n"
            "    ard_reference = committed_metrics.loc[committed_metrics['ard'].astype(bool)].copy()\n"
            "    metric_table = pd.concat([metric_table, ard_reference], ignore_index=True)\n"
            "    for name in ard_reference['model']:\n"
            "        candidate_predictions[name] = pd.read_csv(RESULTS / f'gpr_handoff_oof_{name}.csv')\n"
            "metric_table = metric_table.sort_values('R2', ascending=False)\n"
            "metric_table.to_csv(RESULTS / 'gpr_handoff_metrics.csv', index=False)\n"
            "metric_table['upper_bound_fraction'] = np.where(\n"
            "    metric_table['length_scales_total'] > 0,\n"
            "    metric_table['length_scales_at_upper_bound_total'] / metric_table['length_scales_total'],\n"
            "    np.nan,\n"
            ")\n"
            "display(metric_table[['model','kernel_family','ard','optimizer_restarts','R2','corr2','RMSE','MAE','coverage_95','NLPD','upper_bound_fraction']])"""
        ),
        nbf.v4.new_code_cell(
            """# RFを主基準に加え、fold別・試料別の挙動診断を作成\n"
            "from chemistory_gpr.handoff_report import build_handoff_report\n\n"
            "report_paths = build_handoff_report(DATA_DIR, RESULTS)\n"
            "primary = pd.read_csv(report_paths['comparison'])\n"
            "behavior = pd.read_csv(report_paths['behavior_summary'])\n"
            "fold_vs_rf = pd.read_csv(report_paths['best_vs_rf_fold_metrics'])\n"
            "largest_errors = pd.read_csv(report_paths['largest_errors'])\n"
            "display(primary[['rank_R2','source','model','kernel_family','R2','RMSE','MAE','coverage_95','NLPD','fold_RMSE_wins_out_of_10']])\n"
            "display(behavior)\n"
            "display(largest_errors[['file_key','fold','y','pred_mean','pred_std','rf_pred','gpr_abs_error','rf_abs_error']].head(10))"""
        ),
        nbf.v4.new_code_cell(
            """# 最良GPRとRFのOOF予測・fold別RMSE・不確実性の挙動\n"
            "best_name = metric_table.iloc[0]['model']\n"
            "best = candidate_predictions[best_name]\n"
            "paired = pd.read_csv(report_paths['paired_predictions'])\n"
            "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))\n"
            "axes[0].scatter(paired['y'], paired['rf_pred'], s=18, alpha=.55, label='RF')\n"
            "axes[0].scatter(paired['y'], paired['pred_mean'], s=18, alpha=.55, label='best GPR')\n"
            "limits = [paired['y'].min(), paired['y'].max()]\n"
            "axes[0].plot(limits, limits, '--', color='black')\n"
            "axes[0].set(xlabel='Observed y', ylabel='OOF prediction', title='RF vs best GPR')\n"
            "axes[0].legend()\n"
            "x = np.arange(len(fold_vs_rf))\n"
            "axes[1].bar(x-.2, fold_vs_rf['RF_RMSE'], width=.4, label='RF')\n"
            "axes[1].bar(x+.2, fold_vs_rf['RMSE'], width=.4, label='best GPR')\n"
            "axes[1].set(xticks=x, xticklabels=fold_vs_rf['fold'], xlabel='fold', ylabel='RMSE', title='Performance is fold-dependent')\n"
            "axes[1].legend()\n"
            "axes[2].scatter(paired['pred_std'], paired['gpr_abs_error'], s=20, alpha=.65)\n"
            "rho = paired['pred_std'].corr(paired['gpr_abs_error'], method='spearman')\n"
            "axes[2].set(xlabel='GPR predictive std', ylabel='absolute error', title=f'Uncertainty ranking: Spearman={rho:.3f}')\n"
            "plt.tight_layout(); plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## D. 分子軸の角度帯ごとの挙動\n\n"
            "C3→H3とC6→H6は全例でほぼ反対向きなので、2本を別々にbin分けせず、"
            "C3→H3を反転してC6→H6と円周平均した `axis_angle_deg` を主解析に使います。"
            "これは化学結合角ではなく、共通xy座標系における分子軸の方位です。"
            "角度帯は結果を見た後の探索的層別であり、各帯の勝者を確定モデルとするにはnested CVが必要です。"""
        ),
        nbf.v4.new_code_cell(
            """from chemistory_gpr.angle_report import build_handoff_angle_report\n\n"
            "angle_paths = build_handoff_angle_report(DATA_DIR, RESULTS)\n"
            "angle_summary = pd.read_csv(angle_paths['behavior_summary'])\n"
            "axis_data_summary = pd.read_csv(angle_paths['axis_data_summary'])\n"
            "angle_winners = pd.read_csv(angle_paths['winners'])\n"
            "angle_metrics = pd.read_csv(angle_paths['method_metrics'])\n"
            "axis_associations = pd.read_csv(angle_paths['axis_feature_associations'])\n"
            "high_angle_contrasts = pd.read_csv(angle_paths['high_angle_structural_contrasts'])\n"
            "axis_winners = angle_winners.loc[angle_winners['angle_view'].eq('molecular_axis')].copy()\n"
            "bin_order = ['[-30,-10)', '[-10,10)', '[10,30)', '[30,50)', '[50,70]']\n"
            "axis_winners['angle_bin'] = pd.Categorical(axis_winners['angle_bin'], bin_order, ordered=True)\n"
            "axis_winners = axis_winners.sort_values('angle_bin')\n"
            "display(angle_summary)\n"
            "display(axis_data_summary)\n"
            "display(axis_winners[['angle_bin','n','y_mean','y_std','best_overall_model','best_overall_RMSE','best_GPR_model','best_GPR_RMSE','RF_RMSE']])\n"
            "print('分子軸と最も強く対応する非角度特徴（探索的）')\n"
            "display(axis_associations.head(10))\n"
            "print('50–70°の低応答9例と残り32例の構造差（探索的）')\n"
            "display(high_angle_contrasts.head(10))"""
        ),
        nbf.v4.new_code_cell(
            """# 応答の角度依存をH3近傍環境・面外傾きで色分けし、右に角度帯別RMSEを表示\n"
            "angle_features = pd.read_csv(angle_paths['angle_features'])\n"
            "base_for_plot = pd.read_csv(DATA_DIR / '01_base_summary_first_angle.csv')\n"
            "angle_plot = angle_features.merge(base_for_plot[['file_key','O_H3_count_d5']], on='file_key', validate='one_to_one')\n"
            "fig, axes = plt.subplots(1, 3, figsize=(19, 5))\n"
            "scatter = axes[0].scatter(angle_plot['axis_angle_deg'], angle_plot['y'], c=angle_plot['O_H3_count_d5'], cmap='viridis', s=38, alpha=.8)\n"
            "axes[0].axvspan(50, 70, color='tab:red', alpha=.08)\n"
            "axes[0].set(xlabel='Molecular-axis azimuth (degree)', ylabel='Observed y', title='High-angle branch and local H3 environment')\n"
            "fig.colorbar(scatter, ax=axes[0], label='O_H3_count_d5')\n"
            "tilt = axes[1].scatter(angle_plot['axis_angle_deg'], angle_plot['y'], c=angle_plot['axis_abs_elevation_deg_proxy'], cmap='plasma', s=38, alpha=.8)\n"
            "axes[1].axvspan(50, 70, color='tab:red', alpha=.08)\n"
            "axes[1].set(xlabel='Molecular-axis azimuth (degree)', ylabel='Observed y', title='Absolute out-of-plane tilt proxy')\n"
            "fig.colorbar(tilt, ax=axes[1], label='absolute elevation proxy (degree)')\n"
            "model_labels = {\n"
            "    'RF_current_residualPLS5': 'RF',\n"
            "    'base_cyclic_xproc_pca8_matern12': 'Matérn 1/2',\n"
            "    'base_cyclic_xproc_pca8_matern32': 'Matérn 3/2',\n"
            "    'base_cyclic_xproc_pca8_matern52': 'Matérn 5/2',\n"
            "    'base_cyclic_xproc_pca8_rbf_iso': 'RBF',\n"
            "}\n"
            "plot_metrics = angle_metrics.loc[\n"
            "    angle_metrics['angle_view'].eq('molecular_axis') & angle_metrics['model'].isin(model_labels)\n"
            "].copy()\n"
            "plot_metrics['angle_bin'] = pd.Categorical(plot_metrics['angle_bin'], bin_order, ordered=True)\n"
            "for model, label in model_labels.items():\n"
            "    line = plot_metrics.loc[plot_metrics['model'].eq(model)].sort_values('angle_bin')\n"
            "    axes[2].plot(bin_order, line['RMSE'], marker='o', label=label)\n"
            "axes[2].set(xlabel='Molecular-axis azimuth bin (degree)', ylabel='OOF RMSE', title='Best method changes in the 50–70° regime')\n"
            "axes[2].tick_params(axis='x', rotation=25)\n"
            "axes[2].legend()\n"
            "plt.tight_layout(); plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """### 読み方\n\n"
            "Matérn 3/2は受領RF報告値よりR²が約0.026高く、RMSEを約15%減らします。"
            "一方、等方RBF・Rational Quadratic・Matérn 5/2との差は非常に小さく、foldごとの首位も入れ替わります。"
            "Linearの大幅低下は非線形性の必要性を、RBF-ARDの低下と長さ尺度上限到達は過剰パラメータ化を示します。"
            "最良GPRでもRFに負けるfoldと大誤差例があり、予測標準偏差と絶対誤差の対応も強くありません。"
            "分子軸10–50°ではMatérn 3/2が特に強い一方、50–70°ではRFが最良、GPR内では粗いMatérn 1/2が最良です。"
            "この帯の低応答枝はほぼ面内で、H3近傍のMg/O密度・距離特徴とも対応します。単なる角度データ不足ではなく、"
            "xy方位・面外傾き・局所配位環境の相互作用としきい値構造を示唆します。"
            "同じCVで候補と角度帯を選んでいるため、最終確定にはnested CVまたは独立データが必要です。"""
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
            """# 02 — 参考付録: dist_autoへのGPR適用\n\n"
            "このノートブックはGPR実装を確認する参考資料であり、主解析は01の `GPR_handoff` です。"
            "元コードと同じく、1つのtagを丸ごとテストにして、残りtagから予測します。"
            "既定は `TEST_TAG='10'` です。予測平均に加え、不確実性と95%区間を作ります。"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path\n"
            "import os, subprocess, sys\n\n"
            "REPO_URL = 'https://github.com/futoshi-futami/Chemistory.git'\n"
            "REPO_REF = os.environ.get('CHEMISTORY_REF', 'main')\n"
            "FALLBACK_REF = 'agent/rbf-kernel-comparison'  # PR #2。mainへmerge後はfallback不要\n"
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
            "# PRのレビュー中も実行可能にし、merge後は自動的にmainを使います。\n"
            "if 'google.colab' in sys.modules and not (PROJECT_ROOT / 'src/chemistory_gpr/kernels.py').exists():\n"
            "    subprocess.run(['git', '-C', str(PROJECT_ROOT), 'fetch', '--depth', '1', 'origin', FALLBACK_REF], check=True)\n"
            "    subprocess.run(['git', '-C', str(PROJECT_ROOT), 'checkout', '--detach', 'FETCH_HEAD'], check=True)\n"
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
