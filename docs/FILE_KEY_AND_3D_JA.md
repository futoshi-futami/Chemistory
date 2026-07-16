# file_key、外側group、3D構造特徴の扱い

## 1. 結論

`GPR_handoff.zip` には `file_key` を生成したプログラム、各tokenの定義表、元の170構造の原子座標が含まれていません。PowerPointとGitHub内にもtoken対応表は見つかりませんでした。したがって、各tokenの物理名を断定することはできません。

一方、170例でtokenと既知の幾何特徴の対応を調べると、次の役割はかなり明瞭です。

| token | 値 | データ上の対応 | 現時点の呼び方 |
| --- | --- | --- | --- |
| 1 | 全例0 | 変化なし | 未同定・固定条件 |
| 2 | 全例0 | 変化なし | 未同定・固定条件 |
| 3 | 2, 3, 4, 5 | `y` とSpearman 0.681。値が増えるほどH3/H6近傍Mg/O数が減り、最初のMg/O距離が増える | 表面近接度／高さ候補 |
| 4 | 1–18 | 分子軸xy方位とSpearman 0.859、面外傾きproxyと−0.578 | 方位family候補 |
| 5 | 0–18 | 反平行ずれとSpearman 0.943、面外傾きproxyと0.555。多くのprefix内で単調な走査 | 系列内sweep候補 |

token 3を「高さ」、token 4を「方位角」、token 5を「傾斜角」と正式に命名するには、構造生成側の対応表が必要です。本文とCSVでは、断定を避けて `proximity_level`, `orientation_family`, `sweep_level` と呼びます。

## 2. 暫定groupの選び方

どのgroupが正しいかは、将来何を未知として予測するかで決まります。性能が良くなる分割を選んではいけません。

| 実運用で未知になるもの | 暫定group | group数 | 問う性能 |
| --- | --- | ---: | --- |
| token 3・4を共有し、token 5を走査する系列全体 | `trajectory = token3×token4` | 30 | 未知trajectoryへの外挿 |
| token 3のレベル全体 | `proximity_level = token3` | 4 | 未知の近接度／高さ候補への外挿 |
| token 4のレベル全体 | `orientation_family = token4` | 18 | 未知の方位familyへの外挿 |
| token 5の同じ値だけ | `sweep_level = token5` | 19 | 既知trajectory間での未知sweep値の補間に近い |

相互作用GPとcompact Rational Quadraticを同じgroupごとに比較した感度分析は次のとおりです。split数が異なるため、group間で小数点以下の順位を競う表ではありません。

| group | 相互作用GP R² / RMSE | compact RQ R² / RMSE |
| --- | ---: | ---: |
| trajectory, 5-fold | 0.465 / 7.406 | 0.179 / 9.176 |
| proximity level, leave-one-level-out | 0.250 / 8.771 | 0.099 / 9.610 |
| orientation family, 6-fold | 0.285 / 8.562 | 0.027 / 9.989 |
| sweep level, 5-fold | 0.927 / 2.740 | 0.854 / 3.865 |

`sweep_level` が高いのは、同じtrajectoryの隣接sweep値が訓練に残るためです。未知の分子構造系列を主眼にするなら、現在は `trajectory` が最も保守的で実用的な暫定外側groupです。ただし、構造生成者が「独立な候補構造」を別tokenで定義している場合は、その定義へ差し替えます。

## 3. nested group CV

暫定 `trajectory` を外側5-fold（各fold 6系列）、内側4-foldとし、次を内側RMSEだけで選びました。

- 従来角度の全体Matérn 3/2
- 4変数へ整理した全体Matérn 3/2、Matérn 1/2、Rational Quadratic
- `k_axis + k_environment + White`
- `k_axis + k_environment + k_axis×k_environment + White`
- gate 40°, 45°, 50° × 高角度Matérn 1/2またはPython RF

外側foldの内側勝者は、RF gate 40°が3回、RF gate 50°が1回、相互作用GPが1回でした。厳密なnested OOFは `R²=0.334`, `RMSE=8.261`, `MAE=4.536` です。相互作用GPを事前固定した候補別外側OOFは `R²=0.466`, `RMSE=7.403` でした。後者を見てから選ぶと楽観性が入るため、次の独立構造で相互作用GPを固定して評価するのが次段階です。

受領したbase RF＋残差PLS5/RFも同じ暫定trajectoryで評価しました。group5は `R²=0.026, RMSE=9.994`、group10は `R²=-0.213, RMSE=11.152` で、同じ分割の相互作用GP（0.466 / 7.403、0.365 / 8.068）より低性能でした。したがってtrajectory Held-outの低下はGPだけの問題ではなく、現在の特徴量から未知系列へ外挿する課題そのものです。

## 4. 元3D構造から作る特徴

`src/chemistory_gpr/geometry3d.py` に、将来の座標入力から次を生成する実装を追加しています。

- C6→H6と反転C3→H3の平均から3D分子軸を再計算
- Mg/O位置の分子軸方向への符号付き射影
- Mg/O位置の分子軸からの垂直距離
- H3/H6それぞれからの最短距離、垂直距離、近傍数、逆距離和
- 上記の `H3 − H6` 非対称性
- 軸方位のsin/cos、符号付きz成分、面外傾き、反平行ずれ

必要なlong形式は1原子1行で、次の列を持ちます。

| 列 | 内容 |
| --- | --- |
| `file_key` | 170例と一致する構造ID |
| `atom_label` | 少なくとも `C3`, `H3`, `C6`, `H6` を各1行 |
| `element` | `C`, `H`, `Mg`, `O` など |
| `x`, `y`, `z` | 同一単位・同一座標系の原子座標 |

Mg/Oは、既存特徴を作ったものと同じ9-cell周期像を含めてください。実装は周期境界を推定せず、入力された原子像をそのまま使います。

現在の `GPR_handoff.zip` は要約距離と2つの内積だけを含むため、z符号、個々のMg/O座標、厳密な射影距離は復元できません。170個の `file_key` に対応する元座標表が必要です。

## 5. 現在表示できる3D模式図と、元座標取得後の表示

元座標がない現段階でも、`C3H3_angle_xy`, `C6H6_angle_xy`, 3D内積とxy内積から、2本のC–H**方向ベクトル**は鏡映を除いて表示できます。主Colabでは同一trajectoryをtoken 5順にPlay/slider再生し、OOF予測平均・95%区間と同期させています。

この模式図については次を区別してください。

- 正確に表示するもの: 2本のxy方位、面外傾き絶対値proxy、反平行ずれ
- 任意に固定するもの: zの符号。C6→H6側を正とする鏡映同値な規約
- 表示しないもの: C3/C6の実座標、Mg/Oの方向・位置、原子間結合網

したがって、これは「分子の元3D構造」ではありません。対応するlong座標が得られた後は `raw_structure_figure` を使い、C/H/Mg/Oの実座標を回転・拡大・hover表示します。既存の要約特徴と同じ周期像を渡せば、表示と `derive_rotation_invariant_features` の計算対象を一致させられます。
