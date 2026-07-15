# GPRカーネルの関数形と比較結果

この文書は、`GPR_handoff` と `dist_auto` で比較したカーネルの関数形、ハイパーパラメータの最適化方法、および固定分割での結果をまとめたものです。

## 1. 共通のモデル

すべての候補で、共分散関数を

\[
k_{\mathrm{total}}(x,x')=\sigma_f^2 k_{\mathrm{space}}(x,x')
+\sigma_n^2\mathbf{1}[x=x']
\]

としています。コード上は `ConstantKernel × spatial kernel + WhiteKernel` です。

- `ConstantKernel` の \(\sigma_f^2\): 信号の振幅
- `spatial kernel`: 入力がどれだけ離れると応答の相関が弱まるか
- `WhiteKernel` の \(\sigma_n^2\): 観測ノイズ、または入力特徴量だけでは説明できない独立変動

標準化後の二点間距離を

\[
r(x,x')=\sqrt{\sum_j\left(\frac{x_j-x'_j}{\ell_j}\right)^2}
\]

とします。等方（isotropic）版では全特徴に共通の \(\ell\) を1個だけ使い、ARD版では特徴ごとに \(\ell_j\) を推定します。長さ尺度が大きい特徴は、予測に対する変化が小さいと解釈されます。

## 2. 比較した関数系

| 名称 | 関数形 \(k_{\mathrm{space}}\) | 想定する関数の滑らかさ | この実装での役割 |
| --- | --- | --- | --- |
| Matérn 1/2（Exponential） | \(\exp(-r)\) | 粗く、局所的に折れ曲がりやすい | しきい値的・急な変化を許す |
| Matérn 3/2 | \((1+\sqrt{3}r)\exp(-\sqrt{3}r)\) | Exponentialより滑らか、RBFより粗い | 現在の主baseline |
| RBF / Squared Exponential | \(\exp(-r^2/2)\) | 非常に滑らか | 元の `dist_auto` と同じ関数族 |
| White noise | \(\sigma_n^2\mathbf{1}[x=x']\) | 点ごとに独立 | 全候補へ加算するノイズ項 |

「Exponential + White noise」は Matérn \(\nu=1/2\) + WhiteKernel を意味します。「RBF + White noise」は別の関数族で、距離に対してより速く滑らかに相関が変化します。

## 3. 第二種最尤法による帯域幅最適化

信号分散 \(\sigma_f^2\)、長さ尺度 \(\ell\) または \(\ell_j\)、White noise \(\sigma_n^2\) は、訓練データの対数周辺尤度を最大化して推定します。これは第二種最尤法、empirical Bayes、またはevidence maximizationと呼ばれる方法です。

`n_restarts_optimizer=0` でも、初期値からL-BFGS-Bによる最適化を1回行います。値を5にすると、その1回に加えてランダム初期値から5回再最適化し、最良の対数周辺尤度を採用します。

元の `dist_auto/Untitled.ipynb` の設定は次のとおりです。

```python
ConstantKernel(1.0, (1e-2, 1e3))
* RBF(length_scale=np.ones(n_features),
      length_scale_bounds=(1e-2, 1e3))
+ WhiteKernel(noise_level=1e-2,
              noise_level_bounds=(1e-6, 1e1))
```

さらに `alpha=0.0`, `normalize_y=True`, `n_restarts_optimizer=5`, `random_state=0` です。今回追加した `dist_auto_full_xmat_rbf_ard_original` は、このXmatのみの標準化、RBF-ARD、境界値、WhiteKernelを再現します。

## 4. `GPR_handoff` の同一特徴量比較

比較条件は、170例、指定済み固定10-fold、角度のsin/cos変換、base + `X_proc` のfold内PCA8です。標準化、分散ゼロ列除去、PCA、カーネル最適化は訓練fold内だけで行いました。下表のRBF-ARDは計算量を考慮して追加restart 0回ですが、長さ尺度を含む第二種最尤最適化自体は実施しています。

| spatial kernel | ARD | R² | RMSE | MAE | 95% coverage | NLPD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matérn 3/2 | なし | **0.933874** | **2.604119** | 1.191774 | 0.952941 | 2.470919 |
| RBF | なし | 0.933505 | 2.611376 | 1.217731 | 0.935294 | **2.425369** |
| Matérn 1/2（Exponential） | なし | 0.926884 | 2.738293 | **1.176260** | 0.952941 | 2.484521 |
| RBF | あり | 0.872676 | 3.613510 | 1.264649 | 0.876471 | 5.221269 |

結論は、点予測R²を主指標にすると最高性能はMatérn 3/2のままです。等方RBFとの差はR²で0.000369しかなく、実質的には僅差です。一方、RBF-ARDは120個の長さ尺度をfoldごとに推定し、全1,200個のうち889個（74.1%）が上限1,000へ達しました。標本数170に対してARDの自由度が大きく、長さ尺度を安定に識別できないことが性能低下の主因と考えられます。

## 5. `dist_auto` の元条件での `tag=10` 比較

元ノートブックと同じく、tag 10の55点を完全にhold outし、残る5 tag、275点だけを訓練に使いました。公平なカーネル比較の4候補では、共通Xmat 309列だけを訓練データで標準化し、xy座標を別入力として追加していません。`previous` 行だけは、これまで本リポジトリで使用していた「分散ゼロ列除去 + xy追加」のMatérn 3/2です。

| モデル | R² | corr² | RMSE | MAE | 95% coverage | NLPD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RBF-ARD、元条件 | **0.993259** | **0.994186** | **0.000203** | **0.000157** | 0.745455 | **-6.659100** |
| RBF、等方 | 0.982271 | 0.983811 | 0.000330 | 0.000268 | 0.600000 | -5.153477 |
| previous: Matérn 3/2 + xy | 0.978301 | 0.980110 | 0.000365 | 0.000296 | **0.927273** | -6.426258 |
| Matérn 3/2、Xmatのみ | 0.978186 | 0.979948 | 0.000366 | 0.000297 | **0.927273** | -6.430683 |
| Matérn 1/2、Xmatのみ | 0.941178 | 0.961789 | 0.000601 | 0.000465 | 1.000000 | -5.484667 |

ここでは最高の点予測モデルがMatérn 3/2から元のRBF-ARDへ変わります。previousモデルに対してRMSEは約44%、MAEは約47%減りました。`corr²=0.994186` は元ノートブックの出力 `0.994183` と実質的に一致し、差は約 `2.7×10⁻⁶` です。追加restart 0回の初期解が、元の追加restart 5回で採用された解と同等の予測性能へ収束したと判断できます。

ただし、RBF-ARDの95% coverageは0.745で、予測区間が過度に狭いです。点予測とNLPDは最良でも、95%区間の較正はprevious Matérn 3/2のほうが良好です。また309個の長さ尺度のうち79個が上限へ達し、全330例で完全に一定のXmat列も81個あります。ARDの各長さ尺度を物理的重要度としてそのまま解釈するのは危険です。

また、この比較ではtag 10の真値を見てカーネルを選んでいます。そのため `R²=0.993259` は候補比較値であり、選択後の完全に独立した外部評価値ではありません。次の新規tagまたは新規構造に対してRBF-ARDを事前固定して評価して初めて、改善量を確証できます。

## 6. `dist_auto` 全tag leave-one-tag-out

6 tagを1回ずつ完全にhold outした結果を、tag別R²の平均・中央値と、全330点をまとめたOOF R²で要約します。すべてのtagは55点なので、tag別R²平均では各tagの重みは同じです。

| モデル | tag別R²平均 | tag別R²中央値 | 全OOF R² | 95% coverage平均 | 単独R²首位tag数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 等方RBF、Xmatのみ | **0.637880** | 0.969480 | **0.044527** | 0.660606 | 1 |
| previous Matérn 3/2 + xy | 0.636626 | 0.981234 | -0.040446 | **0.945455** | 1 |
| Matérn 3/2、Xmatのみ | 0.636201 | 0.981034 | -0.037421 | **0.945455** | 0 |
| RBF-ARD、元条件 | 0.629032 | **0.993314** | 0.042735 | 0.766667 | **4** |
| Matérn 1/2、Xmatのみ | 0.548817 | 0.898308 | -0.025075 | 0.915152 | 0 |

RBF-ARDはtag 10, 15, 20, 25の4条件で首位となり、R²はそれぞれ0.993259、0.999087、0.998595、0.993369でした。一方、tag aではprevious Matérn 3/2が0.522413で首位です。tag bは等方RBFとRBF-ARDが表示精度でともに `R²=-0.512663` で、改善後も平均予測より悪いままです。

したがって「最高性能」は評価目的で変わります。

- 数値tag 10–25の点予測: RBF-ARDが明確に最良。
- 6 tagの平均R²・全OOF R²: 等方RBFが僅差で最良。
- 95%区間の較正: previous / Xmat-only Matérn 3/2が最良。
- tag bの領域外挿: どのカーネルも実用的に解決していない。

RBF-ARDのtag別R²平均が首位でないのは、tag aの低下が大きいためです。単一の「総合首位」だけを採用せず、予測対象のregimeと点予測・確率予測のどちらを重視するかを事前に定める必要があります。

## 7. モデル選択への反映

- `GPR_handoff`: 主モデルは引き続きMatérn 3/2。等方RBFを感度分析として併記する。
- `dist_auto`, tag 10–25の点予測面: 元のRBF-ARDを第一候補に変更する。
- 未知tagを一律に扱う単一モデル: 等方RBFとMatérn 3/2も残し、独立tagで選択する。
- `dist_auto` の不確実性: RBF-ARDの区間を無条件には採用せず、tag別coverage、標準化残差、必要ならconformal calibrationで再較正する。
- tag bを含む構造領域外挿: カーネル変更だけで解決したとみなさず、tag差を表す幾何特徴、regime別GPR、階層・multi-task GPを別途比較する。

数値の機械可読版は `results/gpr_handoff_metrics.csv`、`results/dist_auto_kernel_comparison_metrics.csv`、元ノートブックとの照合は `results/dist_auto_original_rbf_reference_comparison.csv`、探索的なtag重心距離は `results/dist_auto_tag_centroid_distances.csv` にあります。実装は `src/chemistory_gpr/kernels.py`、`handoff.py`、`dist_auto.py` です。
