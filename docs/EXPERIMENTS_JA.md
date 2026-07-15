# GPR_handoff主解析: RF基準、GPR前処理、カーネル比較、挙動診断

この文書の主対象は `GPR_handoff.zip` です。受領RFと同じ170例・固定10-foldを基準に、どの入力をどのfold内前処理へ通し、どのGPカーネルで予測したか、RFよりどこが改善し、どこで失敗するかを再現可能な形でまとめます。`dist_auto.zip` はRBF-ARD実装を確認する参考資料であり、後半の付録へ分離しています。

## 1. 先に要点

- `GPR_handoff` は170例の回帰データです。111個のbase特徴量と3,102個の `X_proc` 特徴量を使用します。
- 最良のGPRは、角度を周期表現にし、`X_proc` を各訓練fold内でPCA 8成分に圧縮してbase特徴量へ結合したMatérn 3/2 GPRで、固定10-fold OOFの `R²=0.933874` です。
- 受領RF報告値 `R²=0.908223` に対してR²は0.025651増え、RMSEは15.1%、MAEは5.6%減りました。現環境で再実行したRF `R²=0.900572` に対しても改善しています。
- 等方RBF、Rational Quadratic、Matérn 5/2も `R²=0.9328–0.9335` でほぼ同等です。最良カーネルを強く一意に識別できるほどの差ではありません。
- 線形カーネルは `R²=0.662463` まで低下するため、非線形性は必要です。120次元RBF-ARDも `R²=0.872676` へ低下し、74.1%の長さ尺度が上限へ達しました。
- Matérn 3/2はRFより7/10 foldでRMSEが小さい一方、fold 1・6・7ではRFより悪く、最悪foldのRMSEは4.234です。全体指標だけではなく、この不均一性も主結果です。
- `dist_auto` の結果は付録に残しますが、`GPR_handoff` のカーネル選択には使用しません。

## 2. `GPR_handoff` のデータ

### 2.1 観測単位と目的変数

- 標本数: 170
- 目的変数: `01_base_summary_first_angle.csv` の `y`
- `y` の範囲: 1.89–52.24（平均43.489、標準偏差10.157）
- 識別子: `file_key`
- 評価fold: 10-fold、各fold 17例

3ファイルは `file_key` と行順が完全に一致している必要があります。読込時に、一意性、行対応、欠損、fold番号1–10を検査し、不一致があれば停止します。

### 2.2 入力CSV

| ファイル | 大きさ | 役割 |
| --- | ---: | --- |
| `01_base_summary_first_angle.csv` | 170 × 113 | `file_key`、`y`、111個のbase特徴量 |
| `02_Xproc_matched.csv` | 170 × 3,103 | `file_key`、3,102個のwideな `X_proc` 特徴量 |
| `03_cv_folds_seed123.csv` | 170 × 2 | `file_key`、固定fold番号 |

base特徴量には、距離しきい値 `d=3,5,7,10` ごとのMg/O数、逆距離の和・最大・上位3個、H3/H6間の差と和、最初に非ゼロとなる逆距離、原子種dummy、および方向特徴量が含まれます。

### 2.3 「角度」の意味

ここでの角度は化学結合角ではなく、共通座標系のxy平面上で見たC–Hベクトルの方位角です。

\[
\theta_3=\operatorname{atan2}(y_{H3}-y_{C3},x_{H3}-x_{C3}),\qquad
\theta_6=\operatorname{atan2}(y_{H6}-y_{C6},x_{H6}-x_{C6}).
\]

- `C3H3_angle_xy`: C3→H3の方位角
- `C6H6_angle_xy`: C6→H6の方位角
- `angle_diff_C3_C6`: 上記2方向の差を `[-pi, pi]` に丸めたもの
- `cos_angle_diff_C3_C6`, `sin_angle_diff_C3_C6`: 相対方向の周期表現
- `dot_C3H3_C6H6`, `dot_xy_C3H3_C6H6`: 2方向の平行性に関する内積

これらは座標系依存であり、全構造が同じ9-cell座標系で整列していることを前提に意味を持ちます。

## 3. `GPR_handoff` の前処理とモデル

### 3.1 情報リークを防ぐ単位

外側foldを1つテストとし、以下の学習を残り9 foldだけで行います。

1. 分散ゼロ列の選択
2. 平均0・分散1への標準化
3. `X_proc` のPCA
4. GPRのカーネルハイパーパラメータ推定

テストfoldには、訓練foldで決めた列選択、平均・標準偏差、PCA射影を適用するだけです。全170例でPCAや標準化を先に学習する処理は行いません。

### 3.2 角度の周期変換

周期版では、生の角度 `theta` を次の2変数へ置き換えます。

\[
\theta\longmapsto(\sin\theta,\cos\theta).
\]

これにより、数値上は遠い `-pi` と `+pi` が、方向として近いことを表現できます。現在の入力には角度差のsin/cosが既に含まれ、実装でも生の `angle_diff_C3_C6` からsin/cosを生成するため、角度差の情報が重複します。したがって、今後の厳密なablationでは重複を除き、同じカーネルで生角度と周期角度だけを比較する必要があります。

### 3.3 `X_proc` のPCA

3,102次元の `X_proc` について、訓練fold内で分散ゼロ列を除去し、標準化してからrandomized PCAを学習します。既定では上位8 scoreをbase特徴量へ結合します。PCA scoreは、成分ごとの分散差を残すため、結合前に再標準化していません。

これは受領RFの「PLSで残差を圧縮して第2のRFで補正する」二段階モデルとは異なります。現在のGPRは、base特徴量と教師なしPCA scoreを同時に入力する一段階GPRです。

### 3.4 GPRとカーネル

全候補でscikit-learnの `GaussianProcessRegressor` を使用し、目的変数を訓練fold内で正規化します。カーネルは共通して

\[
k(x,x')=\sigma_f^2 k_{\mathrm{space}}(x,x';\ell)
+\sigma_n^2\mathbf{1}[x=x']
\]

すなわち `ConstantKernel × spatial kernel + WhiteKernel` です。信号分散、長さ尺度、観測noiseは対数周辺尤度最大化、すなわち第二種最尤法で学習します。

公平なカーネル比較では、角度のsin/cos周期表現とfold内 `X_proc` PCA8を共通にし、空間カーネルだけを変更しました。

| 候補 | spatial kernel | 長さ尺度 |
| --- | --- | --- |
| `base_cyclic_xproc_pca8_matern12` | Matérn 1/2 = Exponential | 全次元共通 |
| `base_cyclic_xproc_pca8_matern32` | Matérn 3/2 | 全次元共通 |
| `base_cyclic_xproc_pca8_matern52` | Matérn 5/2 | 全次元共通 |
| `base_cyclic_xproc_pca8_rbf_iso` | RBF / Squared Exponential | 全次元共通 |
| `base_cyclic_xproc_pca8_rational_quadratic` | Rational Quadratic | 全次元共通、混合尺度alphaも最適化 |
| `base_cyclic_xproc_pca8_rbf_ard` | RBF / Squared Exponential | 特徴ごとのARD |
| `base_cyclic_xproc_pca8_linear` | Linear / DotProduct | 長さ尺度なし |

各関数形、isotropicとARDの違い、White noise、第二種最尤法、元 `dist_auto` の境界値は[GPRカーネルの関数形と比較結果](KERNELS_JA.md)に数式付きで整理しています。

## 4. 受領RFの再現方法

RFはRの `randomForest` と `pls::plsr(method="simpls")` を使用します。

1. 各外側foldの訓練データで、111個のbase特徴量から1,000本の回帰木RFを学習します。
2. 訓練データに対するRFのout-of-bag予測から残差を作ります。
3. `X_proc` の分散ゼロ列を訓練fold内で除き、標準化して、その残差を目的とするPLS 5成分を学習します。
4. PLS scoreから残差を予測する1,000本の第2RFを学習します。
5. テストfoldでは「base RF予測 + residual RF予測」を最終予測とします。

R 3.6以降の `sample()` 変更の影響を確認するため、`sample.kind="Rejection"` と旧 `"Rounding"` の両方を実行しましたが、今回の結果は同一でした。

## 5. `GPR_handoff` の評価と結果

`03_cv_folds_seed123.csv` の固定10-foldから全170例のout-of-fold予測を作り、次を計算します。

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2},\quad
\mathrm{RMSE}=\sqrt{\frac1n\sum_i(y_i-\hat y_i)^2},\quad
\mathrm{MAE}=\frac1n\sum_i|y_i-\hat y_i|.
\]

GPRでは予測標準偏差、50/80/90/95%予測区間、coverage、区間幅、negative log predictive density（NLPD）も保存します。

| モデル | R² | RMSE | MAE | 95% coverage | NLPD |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPR: Matérn 3/2 | **0.933874** | **2.604119** | 1.191774 | 0.952941 | 2.470919 |
| GPR: 等方RBF | 0.933505 | 2.611376 | 1.217731 | 0.935294 | **2.425369** |
| GPR: Rational Quadratic | 0.932962 | 2.622012 | 1.192167 | 0.941176 | 2.488398 |
| GPR: Matérn 5/2 | 0.932828 | 2.624628 | 1.232611 | 0.947059 | 2.470277 |
| GPR: Matérn 1/2 | 0.926884 | 2.738293 | **1.176260** | 0.952941 | 2.484521 |
| RF + residual PLS5（受領報告値） | 0.908223 | 3.067897 | 1.261954 | – | – |
| RF + residual PLS5（現環境再実行） | 0.900572 | 3.193211 | 1.339298 | – | – |
| GPR: RBF-ARD | 0.872676 | 3.613510 | 1.264649 | 0.876471 | 5.221269 |
| GPR: Linear | 0.662463 | 5.883489 | 3.698088 | 0.935294 | 3.173079 |

### 5.1 読み取れること

1. **方向・角度ブロックは有用と考えられます。** 元のRFの段階的比較では、summaryのみの `R2≈0.751` から角度ブロック追加後に `R2≈0.854` まで改善しています。ただし複数特徴量を同時に加えた比較なので、個々の角度の因果的寄与とは断定できません。
2. **現在の主候補はMatérn 3/2ですが、上位は実質的に僅差です。** 等方RBFとの差は0.000369、上位4候補の幅も0.001046だけです。fold別RMSE首位回数は等方RBFが3回、Matérn 3/2とMatérn 1/2が各2回、Matérn 5/2とRational Quadraticが各1回で、特定カーネルが全foldを支配していません。
3. **RBF-ARDはこのデータでは不利です。** 120個の長さ尺度をfoldごとに推定し、全1,200個中889個が上限1,000へ達しました。170例に対して自由度が大きく、過剰パラメータ化になっています。
4. **点予測と確率予測の評価軸は一致しません。** R²はMatérn 3/2、NLPDは等方RBF、MAEはMatérn 1/2が最良です。不確実性の評価はR²だけでなくcoverage、区間幅、NLPDを併記すべきです。
5. **RFより有望ですが、改善は全試料で一様ではありません。** Matérn 3/2はRFより7/10 foldでRMSEが小さく、57.1%の試料で絶対誤差が小さくなりました。逆にfold 1ではRMSE 3.346対RF 1.130、fold 7では4.234対2.676でRFに負けています。
6. **予測区間の全体coverageだけでは十分ではありません。** 95% coverageは0.953ですが、予測標準偏差と絶対誤差のSpearman相関は0.207です。大誤差例を高不確実性として順位付けする能力は弱く、個別予測での不確実性解釈には注意が必要です。
7. **候補選択後の確定性能ではありません。** 同じ10-foldで7カーネルを順位付けしているため、独立テスト、nested CV、または新規構造でMatérn 3/2を事前固定した評価が必要です。

同梱された報告表のRF最終値 `R2=0.908223` と、同梱コードを現在の入力で再実行した値 `R2=0.900572` は一致しません。乱数サンプリング方式では説明できず、報告表作成時のコード、入力、または設定のいずれかが現在のZIPと異なった可能性があります。

## 参考付録 A. `dist_auto` のデータ

### A.1 データ構造

- tag: `a`, `b`, `10`, `15`, `20`, `25` の6種類
- 各tag: 55個のxy位置
- 全標本数: 330
- xy範囲: 0–1.8
- 目的変数: `response.csv` の7列目 `y.1`（`x`, `y` は座標であり、目的変数ではありません）

各tagのXmatには396–492列があります。全6 tagに共通する309列だけを同じ順序で使用します。各列は主として、Mg/O原子からC3→H3・C6→H6半直線への垂線距離の逆数、および分子中心C0までの距離です。

tag bの目的変数は平均0.4615、標準偏差0.0133で、他tagの平均0.4694–0.4722、標準偏差0.0020–0.0026から大きく外れます。この違いはhold-out結果を読むうえで重要です。

### A.2 tagの物理的意味について分かる範囲

元PowerPoint `打ち合わせ2026_5月.pptx` の全20枚は、H3/H6埋め込み、距離要約、C3→H3・C6→H6ベクトル、角度、RF/PLSの特徴量検討を扱っていますが、`dist_auto` の `tag=a,b,10,15,20,25` がどの実験条件に対応するかは記載していません。`response.csv` とファイル名にもtagの説明表はありません。

一方、Xmatの分散ゼロ列を除き、全330点で各列を標準化した探索的診断では、tag重心から他5 tag重心までの平均距離が `b=31.984`、その他は `9.459–12.937` でした。したがってtag bがXmatで表される幾何領域から外れること自体は確認できます。この全データ標準化は可視化・診断専用で、予測モデルの前処理には使用していません。

現在の資料から確実に言えるのは「tag bでは応答分布とXmat幾何が他tagから外れ、leave-one-tag-outが難しい」という統計的事実までです。tag bが距離、回転、材料組成、境界条件などの何を表すかを断定することはできません。regime別・階層・multi-task GPの設計前に、tagと実験条件の対応表を入手する必要があります。

## 参考付録 B. `dist_auto` の前処理・GPR・評価

1つのtag全体55例をテストとし、残る5 tagの275例だけで標準化とGPR fittingを行います。公平な4カーネル比較では、元ノートブックに揃えて309個の共通Xmat特徴量をPCAせず使用し、xy座標2列を別途追加しません。完全に一定の列も元コードどおり残します。previous baselineだけは、これまでの「分散ゼロ列除去 + xy追加 + Matérn 3/2」です。

比較するspatial kernelはMatérn 1/2、Matérn 3/2、等方RBF、元設定のRBF-ARDです。すべてWhiteKernelを加え、目的変数は訓練tag内で正規化し、カーネルハイパーパラメータを第二種最尤法で推定します。

`TEST_TAG=10` の予測面では、tag 10の目的変数を学習に使わず、tag 10の構造から作ったXmatと新しい30 × 30のxy gridをモデルへ入力します。予測平均の最大点に加え、`mean - 1.96 × std` を最大化する保守的な点も出力します。

`tag=10` の比較は次のとおりです。

| kernel / 前処理 | R2 | RMSE | MAE | 95% coverage | NLPD |
| --- | ---: | ---: | ---: | ---: | ---: |
| RBF-ARD、元条件 | **0.993259** | **0.000203** | **0.000157** | 0.745455 | **-6.659100** |
| 等方RBF、Xmatのみ | 0.982271 | 0.000330 | 0.000268 | 0.600000 | -5.153477 |
| previous Matérn 3/2 + xy | 0.978301 | 0.000365 | 0.000296 | **0.927273** | -6.426258 |
| Matérn 3/2、Xmatのみ | 0.978186 | 0.000366 | 0.000297 | **0.927273** | -6.430683 |
| Matérn 1/2、Xmatのみ | 0.941178 | 0.000601 | 0.000465 | 1.000000 | -5.484667 |

元ノートブックの相関二乗 `corr²=0.994183` に対し、再実装RBF-ARDは `0.994186`（差約 `2.7×10⁻⁶`）で実質的に一致しました。previous Matérn 3/2に対し、RBF-ARDはRMSEを約44%、MAEを約47%減らしています。

全tag leave-one-tag-outでは、RBF-ARDがtag 10, 15, 20, 25の4条件でR²首位でした。tag別R²中央値も0.993314で最良です。一方、tag別R²平均と全OOF R²は等方RBFがそれぞれ0.637880、0.044527で僅差の首位、95% coverage平均はMatérn 3/2が0.945455で最良でした。詳細表は[カーネル比較文書](KERNELS_JA.md#6-dist_auto-全tag-leave-one-tag-out)と `results/dist_auto_kernel_comparison_summary.csv` にあります。

tag 10の30×30グリッドをRBF-ARDで再計算したところ、予測平均最大点と95%下側予測限界最大点はいずれも `(x,y)=(1.8,0.0)` でした。そこでの予測平均は0.479287、予測標準偏差は0.000105、95%下側は0.479082です。全グリッドは `results/dist_auto_surface_10.csv` にモデル名付きで保存しています。

### B.1 読み取れること

- tag 10では、元のRBF-ARDが点予測とNLPDの最高性能モデルへ変わりました。したがって予測面の第一候補もRBF-ARDに変更します。
- ただしtag 10の真値を見てカーネルを選んだため、この数値は候補比較値です。選択後の独立テスト性能としては扱わず、次の新規tag・構造ではRBF-ARDを事前固定して評価する必要があります。
- 一方、RBF-ARDの95% coverageは0.745で、区間が過度に狭いです。点予測の改善と不確実性の較正は別問題です。
- 309個のARD長さ尺度のうち79個が上限へ達し、全体で完全に一定の列も81個あります。長さ尺度を物理的重要度として直接解釈するのは危険です。
- tag a・bを含む領域外挿が、カーネル変更だけで解決するとは限りません。tag識別に対応する幾何記述子の追加、regime別モデル、階層／multi-task GPが引き続き候補です。
- 実際、tag bの最良R²も `-0.512663` で負のままです。RBFへの変更は数値tag 10–25を改善しましたが、tag b問題を解消していません。

## 参考付録 C. 二種類の「角度」を混同しないための注意

- `GPR_handoff` の角度: C3→H3およびC6→H6のxy方位角で、モデルへ入力する特徴量です。
- `dist_auto` の回転角: 元プログラムが構造を回転して新しいXmatを作るために意図した操作です。

受領した `rotate_xyz.exe` はWindows専用で、同梱の `source.xyz` と `*-altered.xyz` はNULバイトのみでした。そのため、現在のColab版で検証できる `dist_auto` は `angle=0` と同梱済み座標CSVに限られます。非ゼロ回転を再現するには、正常な回転前後XYZまたは回転プログラムのソースが必要です。

## 主解析で次に行うべき検証

1. Matérn 3/2と固定foldを共通にして、生角度対sin/cos角度を比較する。
2. 角度差sin/cosの重複を除いたうえで、角度特徴量ごとのablationを行う。
3. PCA成分数を事前に限定したnested CVで選ぶ。
4. Matérn 3/2を事前固定した外部データ評価、または内側foldでカーネルを選ぶnested CVを行う。
5. fold 1・7と絶対誤差上位例の構造・カテゴリ・目的値領域を調べ、訓練分布からの距離と誤差を対応させる。
6. RBF-ARDは全120特徴のまま使わず、base特徴選択やfold内PLS/PCAで次元を十分に下げてから再評価する。
7. 95% coverageに加え、標準化残差、coverage curve、必要ならconformal calibrationを比較する。

主解析の実装は `src/chemistory_gpr/kernels.py`、`handoff.py`、`handoff_report.py`、実行結果は `results/gpr_handoff_primary_comparison.csv`、`gpr_handoff_metrics.csv`、`gpr_handoff_best_vs_rf_fold_metrics.csv`、`gpr_handoff_largest_errors.csv` にあります。`dist_auto.py` と `results/dist_auto_*` は参考付録です。
