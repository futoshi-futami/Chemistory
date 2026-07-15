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

## 分子軸の角度帯ごとの結果

`C3H3_angle_xy` と `C6H6_angle_xy` は化学結合角ではなく、共通xy座標系におけるC3→H3・C6→H6方向の方位角です。2本は全170例で3Dでもほぼ反対向き（内積−0.999960）で、xy方位の反平行からのずれは平均0.374°、最大3.012°でした。そのため、C3→H3を反転してC6→H6と円周平均した一本の「分子軸方位」へ整理しました。分子軸は−22.95°から60.94°に分布します。元PowerPointでは目的変数 `y` は「ディラジカル性」と説明されていますが、単位と量子化学的な定義は同梱CSVにはありません。

| 分子軸方位 | n | y 平均 ± SD | 最良GPR（RMSE） | RF RMSE | 読み取り |
| --- | ---: | ---: | --- | ---: | --- |
| −30–−10° | 24 | 46.10 ± 7.65 | Matérn 5/2（2.956） | 4.338 | 滑らかなGPR群がRFより良い |
| −10–10° | 17 | 45.79 ± 4.38 | RBF-ARD（1.068） | 2.132 | ARDが局所首位だがnが小さい |
| 10–30° | 36 | 47.94 ± 2.72 | Matérn 3/2（1.026） | 1.934 | Matérn 3/2がRFのRMSEを46.9%低減 |
| 30–50° | 52 | 42.58 ± 8.96 | Matérn 3/2（1.758） | 3.585 | Matérn 3/2がRFのRMSEを51.0%低減 |
| 50–70° | 41 | 38.26 ± 15.25 | Matérn 1/2（3.525） | **3.129** | RFが全体首位。粗いGPRほど相対的に有利 |

全体首位のMatérn 3/2が強いのは、主に10–50°の88例を非常によく表せるためです。一方、50–70°だけでMatérn 3/2の全二乗誤差の59.4%を占め、95% coverageも0.829へ下がります。この帯には `y<30` の低ディラジカル性枝が9/41例あり、残り32例との単変量比較では、H3から入力上の距離5以内にあるO数が7.33対0.97、Mg数が6.67対0.91、逆距離和が3.492対0.422でした。最初のO/Mg距離も約3.03/3.14対5.30/5.39と短く、低応答枝はH3近傍の高密度・近距離Mg/O環境と対応します。

3D内積とxy内積の差は `u3_z × u6_z` です。2本がほぼ完全に反平行なので、ここから分子軸の面外傾きの絶対値を近似できます。50–70°の低応答9例では平均3.15°、他32例では19.80°で、低応答枝9例はすべて9.13°以下でした。これは元PowerPointの「外れ値はC3H3/C6H6の向きと関係し、z≈0も重要」という観察を定量化したものです。ただし面外傾き10°未満でも高応答例は存在するため、面内配置だけで決まるのではなく、面内方位・傾き・H3近傍Mg/O環境の組み合わせが必要です。

最寄り訓練分子軸角とMatérn 3/2絶対誤差のSpearman相関は−0.009で、高角度帯にも41例あります。したがって失敗の主因は「その角度の訓練例がない」ことより、同じような方位でもMg/O局所環境によって応答が枝分かれすることだと考えるのが妥当です。滑らかな定常GPRは枝の間を平均化しやすく、粗いMatérn 1/2やしきい値分割を行うRFが相対的に有利になります。

ただし、角度帯と `y<30` は結果を見た後に定義した探索的診断で、特徴量同士も相関しています。「Mg/Oが近いから応答が低下する」という因果の確定ではありません。角度帯別の勝者を実運用するには、内側foldでregimeとモデルを選ぶnested CV、または新規構造での検証が必要です。詳細と次の物理仮説は[主実験の詳細](docs/EXPERIMENTS_JA.md#6-分子軸角度別の挙動と構造仮説)にまとめています。

### 未知の構造系列への外挿は別問題

ファイル名の最後のtokenだけを除いたprefixを「候補構造系列」とみなすと30系列あります。低ディラジカル性枝9例は `0-0-2-16` の3例と `0-0-3-18` の6例だけに集中し、指定固定foldは前者を3 fold、後者を5 foldへ分けています。同じ系列の近傍構造が訓練とテストの双方へ入るため、指定foldは主に系列内補間を測っています。これは受領RFとの対応比較には正しい分割ですが、未知系列への外挿評価ではありません。

30 prefixを一切分割しない探索的10-fold GroupKFoldでは、GPRは次のように低下しました。

| GPR | prefix-group R² | RMSE | MAE |
| --- | ---: | ---: | ---: |
| Rational Quadratic | **0.204954** | **9.029633** | **4.472770** |
| Matérn 1/2 | 0.144438 | 9.366987 | 4.726188 |
| Matérn 3/2 | −0.053689 | 10.395140 | 5.276053 |
| Matérn 5/2 | −0.128628 | 10.758446 | 5.512620 |
| RBF | −0.227962 | 11.221902 | 6.059563 |
| Linear | −1.140907 | 14.817432 | 9.188284 |

未知系列では複数尺度を混ぜるRational Quadraticと粗いMatérn 1/2が相対的に有利ですが、最良でもR² 0.205です。したがって現モデルは「既知系列内の補間」には高精度でも、「未知の分子構造系列への外挿」は未解決です。prefixの物理的意味を確認したうえで、実運用単位に対応するGroupKFoldを主外部評価にすべきです。このgroup評価ではRのRFをまだ再実行していないため、RF対GPRの結論は指定固定foldに限定します。

## Colab

- [主解析 — 01: RF再現とGPRカーネル比較](https://colab.research.google.com/github/futoshi-futami/Chemistory/blob/main/notebooks/01_RF_and_GPR_handoff_Colab.ipynb)
- [参考付録 — 02: dist_autoへのGPR適用](https://colab.research.google.com/github/futoshi-futami/Chemistory/blob/main/notebooks/02_dist_auto_GPR_Colab.ipynb)

1冊目が本研究の中心です。必要ならColabへR本体と `randomForest`, `pls` を導入し、Pythonから `Rscript` を呼びます。ローカルWindows用の `Rscript.exe` パス指定は不要です。

最初の初期化セルがclone、データ展開、editable install、import確認を行います。`ModuleNotFoundError: chemistory_gpr` が出る場合はランタイムを再起動し、最初から「すべてのセルを実行」してください。

## 主解析のデータと前処理

入力は次の3ファイルです。

- `01_base_summary_first_angle.csv`: `file_key`, 目的変数 `y`（PowerPoint上のディラジカル性）, 111個のsummary・カテゴリ・角度特徴
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
python scripts/analyze_handoff_angles.py
python scripts/run_handoff_group_cv.py   # file_key prefixを候補系列とする補助評価
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
- `results/gpr_handoff_angle_winners.csv`: 分子軸・各方向・反平行ずれ別の局所首位
- `results/gpr_handoff_angle_method_metrics.csv`: 角度帯×全モデルのRMSE、MAE、coverage、誤差寄与
- `results/gpr_handoff_high_angle_structural_contrasts.csv`: 50–70°の低応答枝と他試料のMg/O特徴差
- `results/gpr_handoff_series_summary.csv`: file_key prefix候補系列と低応答枝の集中
- `results/gpr_handoff_group10_prefix_metrics.csv`: 候補系列を分割しないGPR補助評価

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
