GPR validation dataset
======================

目的:
  Gaussian Process Regression により，以下の2つのモデルを検証する。

Model 1:
  summary + first_invd + angle_raw
  使用ファイル:
    01_base_summary_first_angle.csv

Model 2:
  summary + first_invd + angle_raw + residual XprocPLS5 correction
  使用ファイル:
    01_base_summary_first_angle.csv
    02_Xproc_matched.csv
    03_cv_folds_seed123.csv

評価方法:
  10-fold CV.
  fold は 03_cv_folds_seed123.csv の fold_seed123 を使用する。
  全例の out-of-fold prediction を作り，以下で評価する。

  R2   = 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)
  RMSE = sqrt(mean((y - yhat)^2))
  MAE  = mean(abs(y - yhat))

重要な注意:
  residual PLS5 は，全データで作ってはいけない。
  必ず各 outer fold の訓練データだけで PLS 軸を作る。
  テストデータは，訓練データで作った PLS 軸へ射影する。
  全データでPLSを作ると，テストfoldの情報がPLS軸に入るためリークになる。

推奨するGPRでのModel 2手順:
  1. 03_cv_folds_seed123.csv に従って outer fold を1つ取り出す。
  2. 01_base_summary_first_angle.csv の訓練データで base GPR を学習する。
  3. テストデータに対して base GPR の予測を行う。
  4. 訓練データ内で cross-fitting または内部CVを行い，base GPR の訓練残差を作る。
  5. 02_Xproc_matched.csv の訓練データだけを用いて，訓練残差を目的変数とする PLS5 を作る。
  6. 訓練データとテストデータを，訓練データで作った PLS 軸へ射影する。
  7. PLS1〜PLS5を説明変数，訓練残差を目的変数として residual GPR を学習する。
  8. 最終予測を base GPR prediction + residual GPR prediction とする。

参考RF結果:
  summary_only:
      R2=0.7506, RMSE=5.057, MAE=2.468
  summary + first_invd:
      R2=0.7524, RMSE=5.040, MAE=2.262
  summary + first_invd + angle_raw:
      R2=0.8542, RMSE=3.867, MAE=1.658
  summary + first_invd + angle_raw + residual XprocPLS5 + RF correction:
      R2=0.9082, RMSE=3.068, MAE=1.262

特徴量の解釈:
  first_invd_LH3 / first_invd_LH6:
      invd_LH3 / invd_LH6 列で最初に現れる非ゼロ値。
  first_invd_LH3_d / first_invd_LH6_d:
      その非ゼロ値が最初に現れた距離 d。
  angle_raw:
      C3->H3, C6->H6 ベクトルを xy 平面に投影したときの方位角。
      化学結合角ではない。
      C3H3_angle_xy, C6H6_angle_xy, angle_diff_C3_C6,
      cos_angle_diff_C3_C6, sin_angle_diff_C3_C6 などを含む。

注意:
  angle_raw は座標系に依存する。
  すべての構造が同じ9cell座標系でそろっていることを前提に意味をもつ。

ファイル一覧:
  01_base_summary_first_angle.csv
      y, file_key, summary + first_invd + angle_raw の数値化済み特徴量。
  02_Xproc_matched.csv
      file_key順を01に合わせた X_proc wide 特徴量。
  03_cv_folds_seed123.csv
      10-fold CVのfold番号。
  04_reference_RF_results.csv
      こちらで得たRF系モデルの参照結果。
  05_feature_dictionary.csv
      追加特徴量の簡単な説明。
