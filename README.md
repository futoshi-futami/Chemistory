# GPR_handoff: RF基準とのGaussian Process比較（Google Colab対応）

このリポジトリの主目的は、`GPR_handoff.zip` の3つのCSVと指定済み固定10-foldを使い、受領したRF結果に対応するGaussian Process Regression（GPR）を複数カーネルで比較することです。`dist_auto.zip` はGPR実装とRBF-ARD設定の参考例であり、主解析・主モデル選択の対象ではありません。

## 主結果

170例の同じ固定10-foldで、GPRはbase特徴の角度をsin–cos化し、`X_proc` 3,102列を各訓練fold内でPCA 8成分へ圧縮してから学習しました。

| モデル | R² | RMSE | MAE | 95% coverage | NLPD |
| --- | ---: | ---: | ---: | ---: | ---: |
| **GPR: Matérn 3/2** | **0.933874** | **2.604119** | 1.191774 | 0.952941 | 2.470919 |
| GPR: 等方RBF | 0.933505 | 2.611376 | 1.217731 | 0.935294 | **2.425369** |
| GPR: Rational Quadratic | 0.932962 | 2.622012 | 1.192167 | 0.941176 | 2.488398 |
| GPR: Matérn 5/2 | 0.932828 | 2.624628 | 1.232611 | 0.947059 | 2.470277 |
| GPR: Matérn 1/2（Exponential） | 0.926884 | 2.738293 | **1.176260** | 0.952941 | 2.484521 |
| 受領RF報告値: RF + residual PLS5 | 0.908223 | 3.067897 | 1.261954 | — | — |
| 現環境でのRF再実行値 | 0.900572 | 3.193211 | 1.339298 | — | — |
| GPR: RBF-ARD | 0.872676 | 3.613510 | 1.264649 | 0.876471 | 5.221269 |
| GPR: 線形 | 0.662463 | 5.883489 | 3.698088 | 0.935294 | 3.173079 |

受領RF報告値と比べ、Matérn 3/2 GPRはR²が0.025651増加し、RMSEは15.1%、MAEは5.6%減少しました。したがって、現在の主候補はMatérn 3/2です。ただし上位4カーネルのR²差は0.0011以内で、等方RBFはNLPDが最良です。「Matérn 3/2だけが明確に優れる」という結果ではありません。

挙動診断では、Matérn 3/2 GPRはRFより7/10 foldでRMSEが小さく、170例中57.1%で絶対誤差が小さくなりました。一方、fold 1・6・7ではRFに負け、最悪fold 7のRMSEは4.234です。予測標準偏差と絶対誤差のSpearman相関も0.207に留まるため、全体の95% coverageが0.953でも、個々の大誤差を強く識別できているわけではありません。

## Colab

- [主解析 — 01: RF再現とGPRカーネル比較](https://colab.research.google.com/github/futoshi-futami/Chemistory/blob/main/notebooks/01_RF_and_GPR_handoff_Colab.ipynb)
- [参考付録 — 02: dist_autoへのGPR適用](https://colab.research.google.com/github/futoshi-futami/Chemistory/blob/main/notebooks/02_dist_auto_GPR_Colab.ipynb)

1冊目が本研究の中心です。必要ならColabへR本体と `randomForest`, `pls` を導入し、Pythonから `Rscript` を呼びます。ローカルWindows用の `Rscript.exe` パス指定は不要です。

最初の初期化セルがclone、データ展開、editable install、import確認を行います。`ModuleNotFoundError: chemistory_gpr` が出る場合はランタイムを再起動し、最初から「すべてのセルを実行」してください。

## 主解析のデータと前処理

入力は次の3ファイルです。

- `01_base_summary_first_angle.csv`: `file_key`, 目的変数 `y`, 111個のsummary・カテゴリ・角度特徴
- `02_Xproc_matched.csv`: `file_key` と3,102個の高次元特徴
- `03_cv_folds_seed123.csv`: RF比較にも使う固定10-fold番号

各foldで、テストfoldを一切使わず次をfitします。

1. baseの分散ゼロ列除去
2. 3つの角度列をsin–cosへ変換
3. baseと`X_proc`の標準化
4. `X_proc`のPCA 8成分
5. `Constant × spatial kernel + WhiteKernel` の第二種最尤推定

GPRは `normalize_y=True` で、予測平均・標準偏差・50/80/90/95%区間・coverage・NLPDを保存します。RFの最終参照モデルはbase RFに`X_proc`のresidual PLS5補正を加えたものなので、入力情報源とfoldは共通ですが、圧縮法と回帰器は同一ではありません。

## 比較したカーネル

- Matérn 1/2（Exponential）: 粗い関数
- Matérn 3/2: 中程度に滑らか
- Matérn 5/2: さらに滑らか
- RBF / Squared Exponential: 非常に滑らか
- Rational Quadratic: 複数の長さ尺度を混ぜたRBF型
- RBF-ARD: 特徴ごとに長さ尺度を持つ高自由度版
- Linear / DotProduct: 非線形性の必要性を確認する対照

関数形、White noise、第二種最尤法、最適化後の長さ尺度は[カーネルの関数形とhandoff結果](docs/KERNELS_JA.md)にまとめています。データ、RF再現、fold別挙動、大誤差例は[主実験の詳細](docs/EXPERIMENTS_JA.md)を参照してください。

## コマンドライン実行

```bash
python -m pip install -e .
python scripts/prepare_data.py
python scripts/run_rf_reproduction.py       # RscriptとR packagesが必要
python scripts/run_gpr_handoff.py --kernel-only --quick
python scripts/summarize_handoff_results.py
pytest -q
```

`--quick` は高次元RBF-ARDの追加ランダム再始動を0回にしますが、第二種最尤最適化そのものは実行します。ARDは120個の長さ尺度の74.1%が上限へ達し、今回の170例では過剰パラメータ化しています。

## 主要な出力

- `results/gpr_handoff_primary_comparison.csv`: RF報告値・RF再実行値・全GPRの主比較
- `results/gpr_handoff_metrics.csv`: GPRの全指標と最適化済みカーネル
- `results/gpr_handoff_all_kernel_fold_metrics.csv`: カーネル×fold別の性能
- `results/gpr_handoff_best_vs_rf_fold_metrics.csv`: 最良GPRとRFのfold別比較
- `results/gpr_handoff_best_vs_rf_predictions.csv`: 同一試料上のGPR/RF予測と誤差
- `results/gpr_handoff_largest_errors.csv`: GPRの絶対誤差上位15例
- `results/gpr_handoff_behavior_summary.csv`: 改善量と不確実性診断の要約

## RF再現値について

現環境でのRF再実行値は、同梱されたPython→R出力と表示精度内で一致しました。一方、受領報告表の最終値 `R²=0.908223` と再実行値 `R²=0.900572` には差があります。Rの現行 `sample.kind="Rejection"` と旧 `"Rounding"` の双方で同じ結果だったため、乱数方式だけでは説明できません。報告表作成時のコード・パッケージ・入力のいずれかが現在の同梱物と異なった可能性があります。

## 評価上の注意

7つのGPRカーネルを同じ固定CV結果で比較しているため、Matérn 3/2の `R²=0.933874` は候補選択を含む探索的比較値です。最終性能として確定するには、Matérn 3/2を事前固定して新しい独立データで評価するか、内側でカーネルを選ぶnested CVが必要です。

`dist_auto` のコードと結果は削除していませんが、`GPR_handoff` の最高モデルを決める根拠には使用しません。`dist_auto` は、RBF-ARDの再現方法、tag全体hold-out、予測面作成を確認するための参考付録です。

## ディレクトリ

- `data/gpr_handoff`: 主解析の3入力CSV、RF参照値、RF OOF予測
- `notebooks/01_RF_and_GPR_handoff_Colab.ipynb`: 主Colab
- `src/chemistory_gpr/handoff.py`: leakage-safeなGPR pipeline
- `src/chemistory_gpr/kernels.py`: カーネル実装
- `results/gpr_handoff_*`: 主解析結果
- `data/dist_auto`, `notebooks/02_*`, `results/dist_auto_*`: 参考付録
