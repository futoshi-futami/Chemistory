# ============================================================
# Final model:
# summary + first_invd + angle_raw
# + residual XprocPLS5 + RF correction
#
# Input:
#   GPR_handoff/01_base_summary_first_angle.csv
#   GPR_handoff/02_Xproc_matched.csv
#   GPR_handoff/03_cv_folds_seed123.csv
#
# Output:
#   GPR_handoff/final_model_RF_residualPLS5_OOF_predictions.csv
# ============================================================

setwd("D:/Users/TSdell1/Dropbox/myfile/Dr.Tada/run_all")

library(randomForest)
library(pls)

# ============================================================
# 1. データ読み込み
# ============================================================

dat <- read.csv(
  "GPR_handoff/01_base_summary_first_angle.csv",
  check.names = FALSE
)

Xproc <- read.csv(
  "GPR_handoff/02_Xproc_matched.csv",
  check.names = FALSE
)

folds <- read.csv(
  "GPR_handoff/03_cv_folds_seed123.csv",
  check.names = FALSE
)

# file_key の整合性確認
stopifnot(all(dat$file_key == Xproc$file_key))
stopifnot(all(dat$file_key == folds$file_key))

y <- dat$y

base_cols <- setdiff(names(dat), c("file_key", "y"))
xproc_cols <- setdiff(names(Xproc), "file_key")

X_base <- dat[, base_cols, drop = FALSE]
X_proc <- Xproc[, xproc_cols, drop = FALSE]

# 数値化
for (v in names(X_base)) {
  X_base[[v]] <- as.numeric(X_base[[v]])
}

for (v in names(X_proc)) {
  X_proc[[v]] <- as.numeric(X_proc[[v]])
}

fold_id <- folds$fold_seed123

cat("\n=== data check ===\n")
cat("n =", nrow(dat), "\n")
cat("base feature p =", ncol(X_base), "\n")
cat("X_proc feature p =", ncol(X_proc), "\n")
print(table(fold_id))


# ============================================================
# 2. 評価関数
# ============================================================

calc_metrics <- function(y, pred) {
  resid <- y - pred
  
  data.frame(
    R2 = 1 - sum(resid^2) / sum((y - mean(y))^2),
    RMSE = sqrt(mean(resid^2)),
    MAE = mean(abs(resid)),
    n = length(y)
  )
}


# ============================================================
# 3. 最終モデルの OOF CV
# ============================================================

fit_final_oof <- function(y,
                          X_base,
                          X_proc,
                          fold_id,
                          ncomp = 5,
                          ntree_base = 1000,
                          ntree_resid = 1000,
                          seed = 123) {
  
  n <- length(y)
  
  pred_base <- rep(NA_real_, n)
  pred_final <- rep(NA_real_, n)
  
  unique_folds <- sort(unique(fold_id))
  
  for (k in unique_folds) {
    
    cat("\nfold", k, "\n")
    
    train_id <- which(fold_id != k)
    test_id  <- which(fold_id == k)
    
    Xb_train <- X_base[train_id, , drop = FALSE]
    Xb_test  <- X_base[test_id, , drop = FALSE]
    
    Xp_train0 <- X_proc[train_id, , drop = FALSE]
    Xp_test0  <- X_proc[test_id, , drop = FALSE]
    
    y_train <- y[train_id]
    
    # --------------------------------------------------------
    # 3.1 base RF:
    #     summary + first_invd + angle_raw
    # --------------------------------------------------------
    
    set.seed(seed + k)
    
    fit_base <- randomForest(
      x = Xb_train,
      y = y_train,
      ntree = ntree_base,
      importance = TRUE
    )
    
    pred_base[test_id] <- predict(fit_base, Xb_test)
    
    # 訓練fold内では base RF の OOB 予測から残差を作る
    resid_train <- y_train - fit_base$predicted
    
    # --------------------------------------------------------
    # 3.2 X_proc から residual PLS5 を作る
    #     注意：PLSは訓練foldだけで作る
    # --------------------------------------------------------
    
    Xp_train0 <- as.data.frame(Xp_train0)
    Xp_test0  <- as.data.frame(Xp_test0)
    
    # 訓練foldで分散ゼロの列は除外
    sds <- apply(Xp_train0, 2, sd, na.rm = TRUE)
    keep <- is.finite(sds) & sds > 0
    
    Xp_train <- as.matrix(Xp_train0[, keep, drop = FALSE])
    Xp_test  <- as.matrix(Xp_test0[, keep, drop = FALSE])
    
    nc <- min(ncomp, ncol(Xp_train), nrow(Xp_train) - 2)
    
    # plsr 用に安全な列名へ変更
    colnames(Xp_train) <- paste0("X", seq_len(ncol(Xp_train)))
    colnames(Xp_test)  <- colnames(Xp_train)
    
    df_pls <- data.frame(
      resid_train = resid_train,
      Xp_train,
      check.names = FALSE
    )
    
    set.seed(seed + 1000 + k)
    
    fit_pls <- plsr(
      resid_train ~ .,
      data = df_pls,
      ncomp = nc,
      scale = TRUE,
      validation = "none",
      method = "simpls"
    )
    
    # --------------------------------------------------------
    # 3.3 訓練・テストを同じPLS軸へ射影
    # --------------------------------------------------------
    
    center <- fit_pls$Xmeans
    scalev <- fit_pls$scale
    
    Xtr_sc <- sweep(Xp_train, 2, center, "-")
    Xtr_sc <- sweep(Xtr_sc, 2, scalev, "/")
    
    Xte_sc <- sweep(Xp_test, 2, center, "-")
    Xte_sc <- sweep(Xte_sc, 2, scalev, "/")
    
    proj <- fit_pls$projection[, seq_len(nc), drop = FALSE]
    
    score_train <- Xtr_sc %*% proj
    score_test  <- Xte_sc %*% proj
    
    score_train <- as.data.frame(score_train)
    score_test  <- as.data.frame(score_test)
    
    names(score_train) <- paste0("PLS", seq_len(nc))
    names(score_test)  <- paste0("PLS", seq_len(nc))
    
    # --------------------------------------------------------
    # 3.4 PLSスコアで残差RF補正
    # --------------------------------------------------------
    
    set.seed(seed + 2000 + k)
    
    fit_resid <- randomForest(
      x = score_train,
      y = resid_train,
      ntree = ntree_resid,
      importance = TRUE
    )
    
    corr_test <- predict(fit_resid, score_test)
    
    pred_final[test_id] <- pred_base[test_id] + corr_test
  }
  
  list(
    pred_base = pred_base,
    pred_final = pred_final,
    resid_base = y - pred_base,
    resid_final = y - pred_final,
    metrics_base = calc_metrics(y, pred_base),
    metrics_final = calc_metrics(y, pred_final)
  )
}


# ============================================================
# 4. 実行
# ============================================================

res_final <- fit_final_oof(
  y = y,
  X_base = X_base,
  X_proc = X_proc,
  fold_id = fold_id,
  ncomp = 5,
  ntree_base = 1000,
  ntree_resid = 1000,
  seed = 123
)

cat("\n=== base model: summary + first_invd + angle_raw ===\n")
print(res_final$metrics_base)

cat("\n=== final model: + residual XprocPLS5 + RF correction ===\n")
print(res_final$metrics_final)


# ============================================================
# 5. OOF予測を保存
# ============================================================

oof <- data.frame(
  file_key = dat$file_key,
  y = y,
  pred_base_angle = res_final$pred_base,
  pred_final = res_final$pred_final,
  resid_base_angle = res_final$resid_base,
  resid_final = res_final$resid_final,
  abs_resid_base_angle = abs(res_final$resid_base),
  abs_resid_final = abs(res_final$resid_final)
)

oof$improve_abs_resid <- oof$abs_resid_base_angle - oof$abs_resid_final

oof$is_top20_final <- rank(
  -oof$abs_resid_final,
  ties.method = "first"
) <= 20

oof <- oof[order(-oof$abs_resid_final), ]

write.csv(
  oof,
  "GPR_handoff/final_model_RF_residualPLS5_OOF_predictions.csv",
  row.names = FALSE
)

cat("\n=== final Top20 residuals ===\n")
print(head(oof, 20))


# ============================================================
# 6. 散布図
# ============================================================

plot(
  y,
  res_final$pred_final,
  pch = 16,
  xlab = "Observed y",
  ylab = "OOF predicted y",
  main = "Final model: angle_raw + residual XprocPLS5 + RF"
)

abline(0, 1, lty = 2)
grid()

text(
  x = min(y, na.rm = TRUE),
  y = max(res_final$pred_final, na.rm = TRUE),
  labels = paste0(
    "R2 = ", round(res_final$metrics_final$R2, 4),
    "\nRMSE = ", round(res_final$metrics_final$RMSE, 4),
    "\nMAE = ", round(res_final$metrics_final$MAE, 4)
  ),
  pos = 4
)