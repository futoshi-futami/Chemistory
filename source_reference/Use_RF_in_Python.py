# ============================================================
# Python wrapper that calls R for the final RF model
#
# Final model:
#   summary + first_invd + angle_raw
#   + residual XprocPLS5 + RF correction
#
# PythonからRscriptを呼び出し，
# R randomForest + R pls::plsr で実行する。
# ============================================================

from pathlib import Path
import subprocess
import shutil
import sys
import pandas as pd


# ============================================================
# 1. paths
# ============================================================

WORKDIR = Path(r"D:/Users/TSdell1/Dropbox/myfile/Dr.Tada/run_all")
HANDOFF = WORKDIR / "GPR_handoff"

BASE_CSV = HANDOFF / "01_base_summary_first_angle.csv"
XPROC_CSV = HANDOFF / "02_Xproc_matched.csv"
FOLD_CSV = HANDOFF / "03_cv_folds_seed123.csv"

R_SCRIPT = HANDOFF / "_run_R_final_RF_model.R"

OUT_PRED = HANDOFF / "final_model_R_randomForest_from_python_OOF_predictions.csv"
OUT_METRICS = HANDOFF / "final_model_R_randomForest_from_python_metrics.csv"


# ============================================================
# 2. basic checks
# ============================================================

for p in [BASE_CSV, XPROC_CSV, FOLD_CSV]:
    if not p.exists():
        raise FileNotFoundError(p)

print("WORKDIR:", WORKDIR)
print("HANDOFF:", HANDOFF)


# ============================================================
# 3. find Rscript
# ============================================================

rscript = shutil.which("Rscript")

if rscript is None:
    # よくあるWindowsの場所を候補にする
    candidates = [
        r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe",
        r"D:\Program Files\R\R-4.5.2\bin\Rscript.exe",
    ]

    for c in candidates:
        if Path(c).exists():
            rscript = c
            break

if rscript is None:
    raise RuntimeError(
        "Rscript が見つかりません。Rをインストールするか，Rscript.exe のパスを指定してください。"
    )

print("Rscript:", rscript)


# ============================================================
# 4. R script content
# ============================================================

r_code = r'''
# ============================================================
# R final model called from Python
#
# Model:
#   summary + first_invd + angle_raw
#   + residual XprocPLS5 + RF correction
# ============================================================

args <- commandArgs(trailingOnly = TRUE)

base_csv <- args[1]
xproc_csv <- args[2]
fold_csv <- args[3]
out_pred <- args[4]
out_metrics <- args[5]

cat("base_csv:", base_csv, "\n")
cat("xproc_csv:", xproc_csv, "\n")
cat("fold_csv:", fold_csv, "\n")
cat("out_pred:", out_pred, "\n")
cat("out_metrics:", out_metrics, "\n")

# ------------------------------------------------------------
# packages
# ------------------------------------------------------------

if (!requireNamespace("randomForest", quietly = TRUE)) {
  stop("R package 'randomForest' is not installed. Run install.packages('randomForest')")
}

if (!requireNamespace("pls", quietly = TRUE)) {
  stop("R package 'pls' is not installed. Run install.packages('pls')")
}

library(randomForest)
library(pls)

# ------------------------------------------------------------
# read data
# ------------------------------------------------------------

dat <- read.csv(base_csv, check.names = FALSE)
Xproc <- read.csv(xproc_csv, check.names = FALSE)
folds <- read.csv(fold_csv, check.names = FALSE)

stopifnot(all(dat$file_key == Xproc$file_key))
stopifnot(all(dat$file_key == folds$file_key))

y <- as.numeric(dat$y)

base_cols <- setdiff(names(dat), c("file_key", "y"))
xproc_cols <- setdiff(names(Xproc), "file_key")

X_base <- dat[, base_cols, drop = FALSE]
X_proc <- Xproc[, xproc_cols, drop = FALSE]

for (v in names(X_base)) {
  X_base[[v]] <- as.numeric(X_base[[v]])
}

for (v in names(X_proc)) {
  X_proc[[v]] <- as.numeric(X_proc[[v]])
}

fold_id <- folds$fold_seed123

cat("\n=== data check ===\n")
cat("n =", length(y), "\n")
cat("base feature p =", ncol(X_base), "\n")
cat("X_proc feature p =", ncol(X_proc), "\n")
print(table(fold_id))

# ------------------------------------------------------------
# metrics
# ------------------------------------------------------------

calc_metrics <- function(y, pred) {
  resid <- y - pred

  data.frame(
    R2 = 1 - sum(resid^2) / sum((y - mean(y))^2),
    RMSE = sqrt(mean(resid^2)),
    MAE = mean(abs(resid)),
    n = length(y)
  )
}

# ------------------------------------------------------------
# final model
# ------------------------------------------------------------

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
    # base RF:
    # summary + first_invd + angle_raw
    # --------------------------------------------------------

    set.seed(seed + k)

    fit_base <- randomForest(
      x = Xb_train,
      y = y_train,
      ntree = ntree_base,
      importance = TRUE
    )

    pred_base[test_id] <- predict(fit_base, Xb_test)

    # 訓練fold内では base RF の OOB予測から残差を作る
    resid_train <- y_train - fit_base$predicted

    # --------------------------------------------------------
    # residual PLS5 from X_proc
    # PLSは訓練foldだけで作る
    # --------------------------------------------------------

    sds <- apply(Xp_train0, 2, sd, na.rm = TRUE)
    keep <- is.finite(sds) & sds > 0

    Xp_train <- as.matrix(Xp_train0[, keep, drop = FALSE])
    Xp_test  <- as.matrix(Xp_test0[, keep, drop = FALSE])

    nc <- min(ncomp, ncol(Xp_train), nrow(Xp_train) - 2)

    if (nc < 1) {
      stop("ncomp became < 1")
    }

    colnames(Xp_train) <- paste0("X", seq_len(ncol(Xp_train)))
    colnames(Xp_test) <- colnames(Xp_train)

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
    # score projection
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
    # residual RF correction
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

# ------------------------------------------------------------
# run
# ------------------------------------------------------------

res <- fit_final_oof(
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
print(res$metrics_base)

cat("\n=== final model: + residual XprocPLS5 + RF correction ===\n")
print(res$metrics_final)

# ------------------------------------------------------------
# save metrics
# ------------------------------------------------------------

metrics_out <- rbind(
  data.frame(
    model = "summary_first_angle_raw",
    res$metrics_base
  ),
  data.frame(
    model = "summary_first_angle_raw_residualPLS5_RF",
    res$metrics_final
  )
)

write.csv(metrics_out, out_metrics, row.names = FALSE)

# ------------------------------------------------------------
# save OOF predictions
# ------------------------------------------------------------

oof <- data.frame(
  file_key = dat$file_key,
  y = y,
  pred_base_angle = res$pred_base,
  pred_final = res$pred_final,
  resid_base_angle = res$resid_base,
  resid_final = res$resid_final,
  abs_resid_base_angle = abs(res$resid_base),
  abs_resid_final = abs(res$resid_final)
)

oof$improve_abs_resid <- oof$abs_resid_base_angle - oof$abs_resid_final
oof$rank_abs_resid_final <- rank(-oof$abs_resid_final, ties.method = "first")
oof$is_top20_final <- oof$rank_abs_resid_final <= 20

oof <- oof[order(-oof$abs_resid_final), ]

write.csv(oof, out_pred, row.names = FALSE)

cat("\n=== final Top20 residuals ===\n")
print(head(oof, 20))

cat("\nDone.\n")
'''

R_SCRIPT.write_text(r_code, encoding="utf-8")


# ============================================================
# 5. run Rscript
# ============================================================

cmd = [
    rscript,
    str(R_SCRIPT),
    str(BASE_CSV),
    str(XPROC_CSV),
    str(FOLD_CSV),
    str(OUT_PRED),
    str(OUT_METRICS),
]

print("\nRunning Rscript...")
print(" ".join(cmd))

result = subprocess.run(
    cmd,
    cwd=str(WORKDIR),
    text=True,
    capture_output=True,
)

print("\n=== R stdout ===")
print(result.stdout)

print("\n=== R stderr ===")
print(result.stderr)

if result.returncode != 0:
    raise RuntimeError(f"Rscript failed with return code {result.returncode}")


# ============================================================
# 6. read results in Python
# ============================================================

metrics = pd.read_csv(OUT_METRICS)
oof = pd.read_csv(OUT_PRED)

print("\n=== metrics read by Python ===")
print(metrics)

print("\n=== final Top20 read by Python ===")
print(oof.head(20))

print("\nSaved files:")
print(OUT_METRICS)
print(OUT_PRED)