# Reproduce the supplied R randomForest + pls model without any machine-specific paths.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("Usage: rf_reference.R base.csv xproc.csv folds.csv out_pred.csv out_metrics.csv rng_sample_kind")
}

base_csv <- args[1]
xproc_csv <- args[2]
fold_csv <- args[3]
out_pred <- args[4]
out_metrics <- args[5]
rng_sample_kind <- args[6]

for (package in c("randomForest", "pls")) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(sprintf("R package '%s' is not installed", package))
  }
}
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(pls))

# randomForest calls sample() for bootstrap rows. R 3.6 changed sample.kind,
# so both current (Rejection) and legacy (Rounding) modes are exposed.
tryCatch(
  suppressWarnings(RNGkind(sample.kind = rng_sample_kind)),
  error = function(error) stop(sprintf("Unsupported rng_sample_kind=%s: %s", rng_sample_kind, error$message))
)

dat <- read.csv(base_csv, check.names = FALSE)
Xproc <- read.csv(xproc_csv, check.names = FALSE)
folds <- read.csv(fold_csv, check.names = FALSE)
stopifnot(all(dat$file_key == Xproc$file_key))
stopifnot(all(dat$file_key == folds$file_key))

y <- as.numeric(dat$y)
X_base <- dat[, setdiff(names(dat), c("file_key", "y")), drop = FALSE]
X_proc <- Xproc[, setdiff(names(Xproc), "file_key"), drop = FALSE]
X_base[] <- lapply(X_base, as.numeric)
X_proc[] <- lapply(X_proc, as.numeric)
fold_id <- as.integer(folds$fold_seed123)

calc_metrics <- function(y, prediction) {
  residual <- y - prediction
  data.frame(
    R2 = 1 - sum(residual^2) / sum((y - mean(y))^2),
    RMSE = sqrt(mean(residual^2)),
    MAE = mean(abs(residual)),
    n = length(y)
  )
}

fit_final_oof <- function(y, X_base, X_proc, fold_id,
                          ncomp = 5, ntree_base = 1000,
                          ntree_resid = 1000, seed = 123) {
  pred_base <- rep(NA_real_, length(y))
  pred_final <- rep(NA_real_, length(y))

  for (fold in sort(unique(fold_id))) {
    train_id <- which(fold_id != fold)
    test_id <- which(fold_id == fold)
    Xb_train <- X_base[train_id, , drop = FALSE]
    Xb_test <- X_base[test_id, , drop = FALSE]
    Xp_train0 <- X_proc[train_id, , drop = FALSE]
    Xp_test0 <- X_proc[test_id, , drop = FALSE]
    y_train <- y[train_id]

    set.seed(seed + fold)
    fit_base <- randomForest(
      x = Xb_train,
      y = y_train,
      ntree = ntree_base,
      importance = TRUE
    )
    pred_base[test_id] <- predict(fit_base, Xb_test)
    residual_train <- y_train - fit_base$predicted

    standard_deviations <- apply(Xp_train0, 2, sd, na.rm = TRUE)
    keep <- is.finite(standard_deviations) & standard_deviations > 0
    Xp_train <- as.matrix(Xp_train0[, keep, drop = FALSE])
    Xp_test <- as.matrix(Xp_test0[, keep, drop = FALSE])
    components <- min(ncomp, ncol(Xp_train), nrow(Xp_train) - 2)
    if (components < 1) stop("ncomp became < 1")
    colnames(Xp_train) <- paste0("X", seq_len(ncol(Xp_train)))
    colnames(Xp_test) <- colnames(Xp_train)

    pls_data <- data.frame(residual_train = residual_train, Xp_train, check.names = FALSE)
    set.seed(seed + 1000 + fold)
    fit_pls <- plsr(
      residual_train ~ .,
      data = pls_data,
      ncomp = components,
      scale = TRUE,
      validation = "none",
      method = "simpls"
    )

    Xtr_scaled <- sweep(Xp_train, 2, fit_pls$Xmeans, "-")
    Xtr_scaled <- sweep(Xtr_scaled, 2, fit_pls$scale, "/")
    Xte_scaled <- sweep(Xp_test, 2, fit_pls$Xmeans, "-")
    Xte_scaled <- sweep(Xte_scaled, 2, fit_pls$scale, "/")
    projection <- fit_pls$projection[, seq_len(components), drop = FALSE]
    score_train <- as.data.frame(Xtr_scaled %*% projection)
    score_test <- as.data.frame(Xte_scaled %*% projection)
    names(score_train) <- paste0("PLS", seq_len(components))
    names(score_test) <- names(score_train)

    set.seed(seed + 2000 + fold)
    fit_residual <- randomForest(
      x = score_train,
      y = residual_train,
      ntree = ntree_resid,
      importance = TRUE
    )
    pred_final[test_id] <- pred_base[test_id] + predict(fit_residual, score_test)
  }
  list(pred_base = pred_base, pred_final = pred_final)
}

result <- fit_final_oof(y, X_base, X_proc, fold_id)
metrics <- rbind(
  data.frame(model = "summary_first_angle_raw", calc_metrics(y, result$pred_base)),
  data.frame(model = "summary_first_angle_raw_residualPLS5_RF", calc_metrics(y, result$pred_final))
)
metrics$rng_sample_kind <- rng_sample_kind
metrics$r_version <- R.version.string
metrics$randomForest_version <- as.character(packageVersion("randomForest"))
metrics$pls_version <- as.character(packageVersion("pls"))

oof <- data.frame(
  file_key = dat$file_key,
  fold = fold_id,
  y = y,
  pred_base_angle = result$pred_base,
  pred_final = result$pred_final,
  resid_base_angle = y - result$pred_base,
  resid_final = y - result$pred_final
)
oof$abs_resid_base_angle <- abs(oof$resid_base_angle)
oof$abs_resid_final <- abs(oof$resid_final)
oof$improve_abs_resid <- oof$abs_resid_base_angle - oof$abs_resid_final
oof$rank_abs_resid_final <- rank(-oof$abs_resid_final, ties.method = "first")
oof$is_top20_final <- oof$rank_abs_resid_final <= 20

write.csv(metrics, out_metrics, row.names = FALSE)
write.csv(oof, out_pred, row.names = FALSE)
print(metrics)
