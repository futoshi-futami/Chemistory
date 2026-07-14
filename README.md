# Chemistory: RF再現とGPR検証（Google Colab対応）

受領した `GPR_handoff.zip` と `dist_auto.zip` を、固定パスやWindows専用実行ファイルに依存せず実行できるよう整理したものです。

## Colabで開く

- [01: RF再現とGPR比較](https://colab.research.google.com/github/futoshi-futami/Chemistory/blob/main/notebooks/01_RF_and_GPR_handoff_Colab.ipynb)
- [02: dist_autoへのGPR適用](https://colab.research.google.com/github/futoshi-futami/Chemistory/blob/main/notebooks/02_dist_auto_GPR_Colab.ipynb)

上から順にセルを実行してください。1冊目は、必要ならColabへR本体と `randomForest`, `pls` を導入し、Pythonから `Rscript` を呼びます。ローカルWindowsの `Rscript.exe` パス指定は不要です。

## 1冊目で行うこと

1. 3つのCSVの `file_key`、行順、欠損、固定10-foldを検査します。
2. 元の R `randomForest` + `pls::plsr(method="simpls")` を再実行します。
3. R 3.6で変更された `sample()` の仕様差を調べるため、`sample.kind="Rejection"` と旧仕様の `"Rounding"` の双方を実行します。
4. 同梱された2種類のRF参照値（報告表と、Python→R実行出力）との差を表示します。
5. GPRを同じ固定foldで比較します。標準化、分散ゼロ列除去、角度変換、X_procのPCAはすべて各訓練fold内だけでfitし、情報リークを防ぎます。

既定GPRは次の構成です。

- baseの角度をsin–cos表現へ変換
- `X_proc`（3,102次元）を訓練fold内PCAで8次元へ圧縮
- base特徴量とPCA scoreを結合
- Matérn 3/2 + White noise kernel
- 予測平均だけでなく標準偏差、95%区間、coverage、NLPDも保存

## 2冊目で行うこと

元の `dist_auto` と同じく、既定では `tag=10` をテスト、残り5 tagを訓練にします。共通Xmat特徴量の抽出順を固定し、Matérn GPRでテスト予測と新しいxyグリッドの予測面を作ります。

- 正しい決定係数 `1-SSE/SST` と、旧コードで使われていた相関係数の二乗を区別して表示
- 予測平均、予測標準偏差、95%予測区間を出力
- 平均最大点と95%下側信頼限界最大点を別々に表示
- 6通りのleave-one-tag-out診断を実行し、外挿が難しいtagを検出

回転用 `rotate_xyz.exe` はWindows専用で、ZIP内の `source.xyz` と `*-altered.xyz` は全バイトがNULでした。このためColab版の再現対象は、検証可能な `angle=0` と同梱済み座標CSVです。回転角付き予測を厳密に復元するには、元のCソースまたは正常な回転後XYZが別途必要です。

## コマンドライン実行

```bash
python -m pip install -e .
python scripts/run_gpr_handoff.py
python scripts/run_dist_auto_gpr.py --test-tag 10
python scripts/run_rf_reproduction.py  # Rscript + R packagesが必要
```

テスト:

```bash
pytest -q
```

## ディレクトリ

- `data/gpr_handoff`: 3入力CSVとRF参照値
- `data/dist_auto`: 応答、xy、座標、事前計算Xmat
- `data_archives`: GitHub用の圧縮データ（clone後に `scripts/prepare_data.py` が自動展開）
- `src/chemistory_gpr`: 再利用可能なモデル・特徴量コード
- `scripts`: Colab以外からの一括実行
- `notebooks`: Colab用ノートブック
- `results`: 検証済み出力
- `source_reference`: 受領した元コードと説明（比較用）

## 評価上の注意

`dist_auto` の `tag=b` は目的変数の平均と分散が他tagから大きく外れます。tag 10, 15, 20, 25への外挿が良好でも、全tagを一括したOOF指標は `tag=b` に強く左右されます。ノートブックでは総合値だけでなくtag別指標を必ず表示します。
