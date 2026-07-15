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
            "series_summary = pd.read_csv(angle_paths['series_summary'])\n"
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
            "display(high_angle_contrasts.head(10))\n"
            "print('高角度・低応答枝を含むfile_key prefix候補系列')\n"
            "display(series_summary.loc[series_summary['n_high_angle_y_below_30'].gt(0)])"""
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
            """## E. 候補構造系列を丸ごとhold outする補助評価\n\n"
            "file_keyの最後のtokenを除いたprefixを候補series IDとみなし、同一prefixを訓練・テストへ分けない"
            "GroupKFoldも確認します。これは命名規則から推定したgroupなので、物理的意味の確認が必要です。"
            "指定固定foldは受領RFとの対応比較、こちらは未知系列外挿という別の問いです。"
            "RのRFはこのgroup splitで未実行なので、ここではGPR同士だけを比較します。"""
        ),
        nbf.v4.new_code_cell(
            """from chemistory_gpr.group_validation import run_prefix_group_comparison\n\n"
            "RUN_PREFIX_GROUP_CV = False  # Trueで非ARD 6カーネルを再fit（数十秒程度）\n"
            "if RUN_PREFIX_GROUP_CV:\n"
            "    group_configs = [config for config in handoff_kernel_candidates(rbf_ard_restarts=0) if not config.ard]\n"
            "    group_paths = run_prefix_group_comparison(data, group_configs, RESULTS)\n"
            "    group_metrics = pd.read_csv(group_paths['metrics'])\n"
            "else:\n"
            "    group_metrics = pd.read_csv(RESULTS / 'gpr_handoff_group10_prefix_metrics.csv')\n"
            "display(group_metrics[['model','kernel_family','matern_nu','R2','RMSE','MAE','coverage_95','NLPD']])\n"
            "print('固定fold Matérn 3/2 R² = 0.933874: 既知系列内の補間')\n"
            "print(f\"prefix-group最良R² = {group_metrics.iloc[0]['R2']:.6f}: 候補系列の外挿\")"""
        ),
        nbf.v4.new_markdown_cell(
            """## F. 物理角度・相互作用GP・mixtureをnested group CVで比較\n\n"
            "生角度7列を `sin(分子軸方位), cos(分子軸方位), 面外傾きproxy, 反平行ずれ` の4変数へ整理します。"
            "相互作用GPは `k_axis + k_environment + k_axis×k_environment + White` です。"
            "この積項は、全体Matérn 3/2の誤差が高角度へ集中し、同じ角度帯でもH3近傍Mg/Oと面外傾きによって応答が枝分かれしたため導入しました。"
            "加法項だけなら軸効果と環境効果は独立ですが、積項は軸も環境も似た試料にだけ強い共分散を与えます。"
            "これはfunctional ANOVA / tensor-product GPとして、`配向が環境効果を修飾する` 仮説を表します。"
            "mixtureでは40/45/50°を候補とし、高角度専門家をMatérn 1/2またはPython RFにします。"
            "モデルと境界は外側trajectory groupを見ず、内側group CVのRMSEだけで選びます。\n\n"
            "既定は保存済み結果を表示します。再fitはColabで時間を要するため、必要なtoggleだけTrueにしてください。"""
        ),
        nbf.v4.new_code_cell(
            """from chemistory_gpr.nested_group import default_nested_candidates, run_nested_group_comparison\n"
            "from chemistory_gpr.next_model_report import build_next_model_report\n\n"
            "RUN_NESTED_GROUP_CV = False  # True: 外側5-fold×内側4-fold×12候補\n"
            "RUN_NEXT_MODEL_REPORT_REFIT = False  # True: fixed10/group10/group感度を再fit\n"
            "if RUN_NESTED_GROUP_CV:\n"
            "    nested_paths = run_nested_group_comparison(\n"
            "        data, default_nested_candidates(), RESULTS,\n"
            "        group_scheme='trajectory', outer_splits=5, inner_splits=4, seed=123,\n"
            "    )\n"
            "if RUN_NEXT_MODEL_REPORT_REFIT:\n"
            "    next_paths = build_next_model_report(data, RESULTS)\n"
            "\n"
            "token_diagnostics = pd.read_csv(RESULTS / 'gpr_handoff_file_key_token_diagnostics.csv')\n"
            "fixed_next = pd.read_csv(RESULTS / 'gpr_handoff_fixed10_next_models_metrics.csv')\n"
            "group10_next = pd.read_csv(RESULTS / 'gpr_handoff_group10_next_models_metrics.csv')\n"
            "nested_metrics = pd.read_csv(RESULTS / 'gpr_handoff_nested_group_metrics.csv')\n"
            "nested_selections = pd.read_csv(RESULTS / 'gpr_handoff_nested_group_selections.csv')\n"
            "group_sensitivity = pd.read_csv(RESULTS / 'gpr_handoff_group_scheme_model_metrics.csv')\n"
            "components = pd.read_csv(RESULTS / 'gpr_handoff_interaction_kernel_components.csv')\n"
            "print('file_key tokenと既知物理特徴の対応（名前はまだ暫定）')\n"
            "display(token_diagnostics)\n"
            "print('受領RF対応 fixed10: 既知trajectory内補間')\n"
            "display(fixed_next[['candidate','R2','RMSE','MAE','coverage_95','NLPD']])\n"
            "print('同じprefix-group10: trajectory外挿候補')\n"
            "display(group10_next[['candidate','R2','RMSE','MAE','coverage_95','NLPD']])\n"
            "print('strict nested group estimate')\n"
            "display(nested_metrics)\n"
            "display(nested_selections)"""
        ),
        nbf.v4.new_code_cell(
            """fixed_angle = pd.read_csv(RESULTS / 'gpr_handoff_fixed10_next_models_angle_metrics.csv')\n"
            "selected = {\n"
            "    'compact_axis_global_matern32': 'global M3/2',\n"
            "    'axis_plus_environment_matern32': 'axis + env',\n"
            "    'axis_environment_interaction_matern32': 'axis + env + interaction',\n"
            "    'moe_matern32_rf_gate40': '40° gate + RF',\n"
            "}\n"
            "fig, axes = plt.subplots(1, 3, figsize=(20, 5))\n"
            "for candidate, label in selected.items():\n"
            "    line = fixed_angle.loc[fixed_angle['candidate'].eq(candidate)].copy()\n"
            "    line['angle_bin'] = pd.Categorical(line['angle_bin'], bin_order, ordered=True)\n"
            "    line = line.sort_values('angle_bin')\n"
            "    axes[0].plot(bin_order, line['RMSE'], marker='o', label=label)\n"
            "axes[0].set(xlabel='axis azimuth bin', ylabel='fixed10 RMSE', title='Interaction improves every angle band')\n"
            "axes[0].tick_params(axis='x', rotation=25); axes[0].legend()\n"
            "pivot = group_sensitivity.pivot(index='group_scheme', columns='candidate', values='R2')\n"
            "pivot.plot(kind='bar', ax=axes[1])\n"
            "axes[1].set(ylabel='OOF R²', title='Result depends on what is held unknown')\n"
            "axes[1].tick_params(axis='x', rotation=25); axes[1].legend(fontsize=8)\n"
            "variance_cols = ['axis_additive_variance','environment_additive_variance','interaction_variance']\n"
            "axes[2].boxplot([components[c] for c in variance_cols], labels=['axis','environment','axis×environment'])\n"
            "axes[2].set(ylabel='optimized signal variance', title='Product term carries the fitted signal')\n"
            "plt.tight_layout(); plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## G. 元3D構造へ戻れる場合\n\n"
            "`geometry3d.py` は `file_key, atom_label, element, x, y, z` のlong表から、"
            "Mg/Oの分子軸方向への符号付き射影、軸からの垂直距離、H3/H6最短距離・近傍逆距離和、"
            "H3−H6非対称性を生成します。少なくともC3/H3/C6/H6と同じ9-cellのMg/O周期像が必要です。\n\n"
            "現在のhandoff ZIPには170構造の元座標がなく、dist_autoの座標は別の330例なので、"
            "3D新特徴は今回の性能表へ混ぜていません。"""
        ),
        nbf.v4.new_code_cell(
            """from chemistory_gpr.geometry3d import (\n"
            "    REQUIRED_COORDINATE_COLUMNS, derive_rotation_invariant_features,\n"
            ")\n"
            "print('required raw-coordinate columns =', sorted(REQUIRED_COORDINATE_COLUMNS))\n"
            "print('raw coordinates are not included; function is ready for a matching 170-file_key table')"""
        ),
        nbf.v4.new_markdown_cell(
            """## H. 動かせる分子軸模式図・3D fitting面・GP分散\n\n"
            "### H.1 trajectoryとOOF予測区間を同期再生\n\n"
            "左側は受領summaryから復元できるC3→H3・C6→H6の**方向ベクトル模式図**です。"
            "元原子座標ではありません。z符号は不明なのでC6→H6側を正とする鏡映同値な表示を使い、Mg/O位置は捏造しません。"
            "右側は全170例のOOF予測、選択trajectory、現在試料の予測平均と95%区間です。"
            "Playまたはsliderでtoken 5の進行と予測挙動を同期して確認できます。"""
        ),
        nbf.v4.new_code_cell(
            """from chemistory_gpr.visualization import (\n"
            "    fit_full_interaction_gp, interaction_surface_figure, interaction_surface_table,\n"
            "    molecular_axis_uncertainty_animation, oof_uncertainty_figure, raw_structure_figure,\n"
            ")\n\n"
            "interaction_predictions = pd.read_csv(RESULTS / 'gpr_handoff_fixed10_next_models_predictions.csv')\n"
            "TRAJECTORY_TO_VIEW = '0-0-3-18'  # 例: '0-0-2-16'; Noneなら全170例\n"
            "trajectory_figure = molecular_axis_uncertainty_animation(\n"
            "    data.base, interaction_predictions, trajectory=TRAJECTORY_TO_VIEW,\n"
            ")\n"
            "trajectory_figure.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """### H.2 予測平均・標準偏差・95%区間幅の3D slice\n\n"
            "高次元の相互作用GPを、(a) 分子軸方位×面外傾き、(b) 分子軸方位×H3近傍逆距離和の2面で切ります。"
            "表示しないbaseとX_procは観測試料 `0-0-3-18-10` に固定しています。"
            "面はマウスで回転・拡大でき、上のボタンで平均、標準偏差、95%幅、上下限を切り替えられます。"
            "これは全データfitの説明用条件付き断面であり、元3D原子構造でもOOF評価面でもありません。"
            "分散が大きい領域の平均値は、物理傾向として強く解釈しないでください。"""
        ),
        nbf.v4.new_code_cell(
            """axis_tilt_surface = pd.read_csv(RESULTS / 'gpr_handoff_interaction_surface_axis_tilt.csv')\n"
            "h3_environment_surface = pd.read_csv(RESULTS / 'gpr_handoff_interaction_surface_h3_environment.csv')\n"
            "print('方位 × 面外傾きproxy')\n"
            "interaction_surface_figure(axis_tilt_surface, data.base).show()\n"
            "print('方位 × H3近傍逆距離和')\n"
            "interaction_surface_figure(h3_environment_surface, data.base).show()"""
        ),
        nbf.v4.new_code_cell(
            """# 別の基準試料・環境特徴で条件付き面を再fitする場合だけTrue\n"
            "RUN_CUSTOM_SURFACE = False\n"
            "CUSTOM_REFERENCE_FILE_KEY = '0-0-3-18-10'\n"
            "CUSTOM_SURFACE_FEATURE = 'sum_invd_LH3_d5'  # axis_tilt_deg, antiparallel_deviation_deg, またはbase列\n"
            "if RUN_CUSTOM_SURFACE:\n"
            "    full_interaction_model = fit_full_interaction_gp(data, seed=123)\n"
            "    custom_surface = interaction_surface_table(\n"
            "        data, model=full_interaction_model,\n"
            "        reference_file_key=CUSTOM_REFERENCE_FILE_KEY,\n"
            "        surface_feature=CUSTOM_SURFACE_FEATURE,\n"
            "    )\n"
            "    interaction_surface_figure(custom_surface, data.base).show()"""
        ),
        nbf.v4.new_markdown_cell(
            """### H.3 OOF不確実性と実座標viewer\n\n"
            "左図の縦線は各試料の95%予測区間、右図は予測標準偏差と絶対誤差です。"
            "95% coverageは0.976ですが、標準偏差と絶対誤差の順位相関は約0.199なので、分散は完全な誤差検出器ではありません。"
            "対応するlong形式の元座標CSVが得られた後は、下の `RAW_COORDINATE_CSV` を指定するとC/H/Mg/Oを実座標で回転表示できます。"""
        ),
        nbf.v4.new_code_cell(
            """oof_uncertainty_figure(data.base, interaction_predictions).show()\n\n"
            "RAW_COORDINATE_CSV = None  # 例: PROJECT_ROOT / 'data' / 'gpr_handoff' / 'raw_coordinates.csv'\n"
            "RAW_FILE_KEY = '0-0-3-18-10'\n"
            "if RAW_COORDINATE_CSV is not None:\n"
            "    raw_coordinates = pd.read_csv(RAW_COORDINATE_CSV)\n"
            "    raw_structure_figure(raw_coordinates, RAW_FILE_KEY).show()\n"
            "else:\n"
            "    print('元座標は未受領です。現在のtrajectory図は方向ベクトル模式図です。')"""
        ),
        nbf.v4.new_markdown_cell(
            """### 読み方\n\n"
            "相互作用GPは固定foldでR²=0.9738、RMSE=1.639となり、全体Matérn 3/2とRFを上回ります。"
            "改善は全角度帯で見られ、最適化でも加法項より軸×環境の積項が主に使われます。"
            "一方、trajectoryを丸ごと未知にする同じgroup10ではR²=0.365、strict nestedでは0.334です。"
            "したがって、既知trajectory内補間では相互作用GPが新しい主候補ですが、未知系列外挿は未解決です。"
            "mixtureは高角度帯だけなら有利でも、境界・専門家の内側選択が外側foldで安定しません。"
            "file_keyの正式なtoken定義を確認して外側groupを固定し、次の独立系列では相互作用GPを事前指定して評価してください。"""
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
