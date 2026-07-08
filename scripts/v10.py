"""
proctor_gpr.py
==============
Gaussian Process Regression for the LeiGS 2026 Proctor Challenge.

Why a GP instead of the neural net? The dataset is small (~200 training rows).
Neural networks are data-hungry and over-fit at this size, whereas a Gaussian
Process is a *data-efficient*, non-parametric Bayesian model that tends to
generalize better on small tabular data — and it returns a calibrated
**uncertainty** (standard deviation) for every prediction, not just a point
estimate. (Gradient-boosted trees are the other strong small-data choice and are
already covered by proctor_pipeline.py.)

By default it uses an ARD kernel (a separate length-scale per feature), so the GP
learns each feature's relevance instead of treating them all equally — soft,
built-in feature selection that suits many-feature/few-row tabular data. The
learned relevances are reported per target. Use --isotropic for a single shared
length-scale, and --kernel to switch the smoothness kernel.

The physics is kept as a post-hoc guardrail: predicted MDD is clipped to the
Zero-Air-Voids (saturation) line, so outputs stay physically admissible.

It predicts the numerical targets:
    - proctor_owc_pct   (optimum water content, %)
    - proctor_mdd_g_cm3 (maximum dry density, g/cm^3)

reusing the organizers' helpers.py for feature engineering + MICE imputation
(fold-isolated, no leakage), standardizing features, evaluating with the
competition metric NMAE (+ MAE / RMSE / R^2), and writing a submission plus the
per-sample uncertainty.

Run:
    python proctor_gpr.py --data_dir <csvs> --helpers_dir <helpers.py folder>

Needs train.csv, test.csv, helpers.py. Dependencies: scikit-learn, pandas, numpy,
matplotlib, and torch (used only to save/load the model as a .pt file).
"""

import argparse
import importlib.util
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch  # used only to save/load the model as a .pt file
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, DotProduct, Matern,
                                              RBF, RationalQuadratic, WhiteKernel)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

TARGETS = ["proctor_mdd_g_cm3", "proctor_owc_pct"]   # col 0 = MDD, col 1 = OWC


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def nmae(y_true, y_pred):
    """Official competition metric: mean column-wise IQR-normalized MAE."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    q75, q25 = np.percentile(y_true, [75, 25], axis=0)
    iqr = np.where((q75 - q25) == 0, 1e-8, q75 - q25)
    return float(np.mean(mae / iqr))


def regression_metrics(y_true, y_pred):
    err = y_true - y_pred
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot == 0, 1e-12, ss_tot)
    return mae, rmse, r2


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def load_helpers(helpers_dir):
    path = os.path.join("/Users/aliceqi/Documents/GitHub/proctor-prediction-challenge/helper_functions.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"helpers.py not found at {path}")
    spec = importlib.util.spec_from_file_location("helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup_logger(path):
    logger = logging.getLogger("gpr")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def base_feature_engineering(df, H):
    df = H.add_gradation_parameters(df)
    df = H.prepare_features(df)
    return df


def numeric_impute_columns(X, exclude):
    return [c for c in X.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(X[c])]


def preprocess_fold(X_tr, X_va, cols_for_imputation, H, seed):
    """Fold-isolated: missing indicators -> MICE -> feature engineering -> scale."""
    for f in (X_tr, X_va):
        f["atterberg_is_missing"] = f["atterberg_liquid_limit_pct"].isnull().astype(int)
        f["kf_is_missing"] = f["hyd_cond_kf_m_s"].isnull().astype(int)
        f["loi_is_missing"] = f["loss_on_ignition_pct"].isnull().astype(int)
    imputer = H.get_default_mice_imputer(seed=seed)
    X_tr[cols_for_imputation] = imputer.fit_transform(X_tr[cols_for_imputation])
    X_va[cols_for_imputation] = imputer.transform(X_va[cols_for_imputation])
    for f in (X_tr, X_va):
        H.apply_fold_feature_engineering(f)
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr.values), scaler.transform(X_va.values)


def build_kernel(n_features, args):
    """Compose the GP kernel.

    Signal scale (ConstantKernel) x smoothness kernel + measurement noise
    (WhiteKernel). With ARD ("Automatic Relevance Determination") the smoothness
    kernel gets a SEPARATE length-scale per feature, so the GP learns each
    feature's relevance — a short length-scale means the target changes quickly
    with that feature (important); a very long one means the feature is
    effectively ignored. This is soft, built-in feature selection and usually
    helps on tabular data with many features and few rows.
    """
    ls = np.ones(n_features) if args.ard else 1.0
    if args.kernel == "rbf":
        base = RBF(length_scale=ls, length_scale_bounds=(1e-2, 1e3))
    elif args.kernel == "rq":
        # RationalQuadratic is isotropic in sklearn (scalar length-scale only)
        base = RationalQuadratic(length_scale=1.0, alpha=1.0,
                                 length_scale_bounds=(1e-2, 1e3))
    else:  # matern (default) — twice-differentiable, standard for physical data
        base = Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e3), nu=2.5)
    kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * base
              + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)))
    if args.linear:
        # global linear trend (helps targets like OWC that rise steadily with
        # plasticity/fines); the Matern part then models local deviations.
        kernel = kernel + DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
    return kernel


def make_gp(args, n_features):
    """A GP regressor with the composed (optionally ARD) kernel."""
    return GaussianProcessRegressor(
        kernel=build_kernel(n_features, args), normalize_y=True, alpha=1e-8,
        n_restarts_optimizer=args.restarts, random_state=args.seed)


def learned_length_scales(gp, n_features):
    """Extract the per-feature length-scales from a fitted kernel (None if isotropic)."""
    for k, v in gp.kernel_.get_params().items():
        if k.endswith("length_scale"):
            arr = np.atleast_1d(v)
            if arr.size == n_features:
                return arr
    return None


def select_and_fit(Xtr, y_col, args):
    """Fit an ARD GP, then (optionally) drop features the GP judged irrelevant
    and refit on the relevant subset. Returns (fitted_gp, selected_indices).

    Rationale: ARD sends unhelpful features to a huge length-scale (~1e3). Refitting
    on only the informative columns removes that noise, which sharpens the fit and
    makes the kernel optimization better-conditioned. Fold-isolated: selection uses
    only the training data passed in.
    """
    gp = make_gp(args, Xtr.shape[1])
    gp.fit(Xtr, y_col)
    sel = np.arange(Xtr.shape[1])
    if args.select and args.ard:
        ls = learned_length_scales(gp, Xtr.shape[1])
        if ls is not None:
            keep = np.where(ls < args.select_thresh)[0]
            if keep.size < args.select_min:                 # guarantee a minimum
                keep = np.argsort(ls)[:args.select_min]
            if 0 < keep.size < Xtr.shape[1]:
                sel = np.sort(keep)
                gp = make_gp(args, sel.size)
                gp.fit(Xtr[:, sel], y_col)
    return gp, sel


def make_gbt(args):
    """Gradient-boosted trees — a low-variance model that captures feature
    interactions the smooth GP kernel misses. Blending the two lowers error and
    variance because they make different mistakes."""
    return HistGradientBoostingRegressor(
        random_state=args.seed, max_iter=args.gbt_iter,
        learning_rate=args.gbt_lr, max_leaf_nodes=args.gbt_leaves,
        l2_regularization=1.0, early_stopping=False)


def fit_bases(Xtr, ytr, args):
    """Fit GP (+selection) and GBT for every target. Returns (gps, sels, gbts)."""
    gps, sels, gbts = [], [], []
    for j in range(len(TARGETS)):
        gp, sel = select_and_fit(Xtr, ytr[:, j], args)
        gps.append(gp)
        sels.append(sel)
        if args.ensemble:
            gbt = make_gbt(args)
            gbt.fit(Xtr, ytr[:, j])
            gbts.append(gbt)
        else:
            gbts.append(None)
    return gps, sels, gbts


def predict_bases(Xte, gps, sels, gbts):
    """Predict each base model per target. Returns (gp_mean, gp_std, gbt_mean)."""
    gpm = np.zeros((len(Xte), len(TARGETS)))
    gps_ = np.zeros((len(Xte), len(TARGETS)))
    gbm = np.full((len(Xte), len(TARGETS)), np.nan)
    for j in range(len(TARGETS)):
        m, s = gps[j].predict(Xte[:, sels[j]], return_std=True)
        gpm[:, j], gps_[:, j] = m, s
        if gbts[j] is not None:
            gbm[:, j] = gbts[j].predict(Xte)
    return gpm, gps_, gbm


def choose_blend_weights(y_true, gp_pred, gbt_pred, grid=21):
    """Per target, pick the GP weight w in [0,1] minimizing MAE of
    w*GP + (1-w)*GBT on the (out-of-fold) predictions. Returns array of weights."""
    weights = np.ones(y_true.shape[1])
    if np.isnan(gbt_pred).any():
        return weights                       # ensemble off -> pure GP
    for j in range(y_true.shape[1]):
        best_w, best = 1.0, np.inf
        for w in np.linspace(0, 1, grid):
            mae = np.mean(np.abs(y_true[:, j] - (w * gp_pred[:, j]
                                                 + (1 - w) * gbt_pred[:, j])))
            if mae < best:
                best, best_w = mae, w
        weights[j] = best_w
    return weights


def blend_and_clip(gp_pred, gbt_pred, weights, rho_s, H):
    """Blend per target with `weights`, then clip MDD to the saturation line."""
    if np.isnan(gbt_pred).any():
        blended = gp_pred.copy()
    else:
        blended = weights * gp_pred + (1 - weights) * gbt_pred
    owc = np.clip(blended[:, 1], 0, None)
    mdd = np.minimum(blended[:, 0], H.calc_satline(owc, rho_s) * 0.999)
    return np.column_stack([mdd, owc])




# --------------------------------------------------------------------------- #
# Saveable model: fit on all data, persist, reload, predict
# --------------------------------------------------------------------------- #
def _add_indicators(df):
    df["atterberg_is_missing"] = df["atterberg_liquid_limit_pct"].isnull().astype(int)
    df["kf_is_missing"] = df["hyd_cond_kf_m_s"].isnull().astype(int)
    df["loi_is_missing"] = df["loss_on_ignition_pct"].isnull().astype(int)


def fit_full_model(X_base, y, cols_for_imputation, H, args, weights=None):
    """Fit imputer, scaler, and per-target GP(+selection) and GBT on ALL data.

    `weights` are the per-target GP blend weights (from CV); default all-GP.
    Returns a self-contained artifacts dict that reproduces predictions.
    """
    X = X_base.copy()
    _add_indicators(X)
    imputer = H.get_default_mice_imputer(seed=args.seed)
    X[cols_for_imputation] = imputer.fit_transform(X[cols_for_imputation])
    H.apply_fold_feature_engineering(X)
    feature_cols = list(X.columns)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    gps, sels, gbts = fit_bases(Xs, y, args)
    if weights is None:
        weights = np.ones(len(TARGETS))
    return {
        "imputer": imputer,
        "scaler": scaler,
        "gps": gps,
        "gbts": gbts,                                 # None per target if ensemble off
        "weights": np.asarray(weights),               # per-target GP blend weight
        "selected": sels,                             # per-target column indices
        "feature_cols": feature_cols,
        "cols_for_imputation": cols_for_imputation,
        "targets": TARGETS,
        "kernels": [str(gp.kernel_) for gp in gps],   # learned kernel hyperparams
    }


def _transform_with(artifacts, df_fe, H):
    """Apply the saved preprocessing to a (base-feature-engineered) frame."""
    X = df_fe.copy()
    _add_indicators(X)
    X[artifacts["cols_for_imputation"]] = artifacts["imputer"].transform(
        X[artifacts["cols_for_imputation"]])
    H.apply_fold_feature_engineering(X)
    X = X.reindex(columns=artifacts["feature_cols"])   # exact train-time order
    return artifacts["scaler"].transform(X.values)


def gp_predict(artifacts, df_fe, H, clip=True):
    """Predict [MDD, OWC] (GP+GBT blend) + GP stds for a base-FE'd frame."""
    Xs = _transform_with(artifacts, df_fe, H)
    n = len(df_fe)
    sels = artifacts.get("selected", [np.arange(Xs.shape[1])] * len(TARGETS))
    gbts = artifacts.get("gbts", [None] * len(TARGETS))
    weights = artifacts.get("weights", np.ones(len(TARGETS)))
    gp_mean = np.zeros((n, len(TARGETS)))
    gbt_mean = np.full((n, len(TARGETS)), np.nan)
    stds = np.zeros((n, len(TARGETS)))
    for j, (gp, sel) in enumerate(zip(artifacts["gps"], sels)):
        m, s = gp.predict(Xs[:, sel], return_std=True)
        gp_mean[:, j], stds[:, j] = m, s
        if gbts[j] is not None:
            gbt_mean[:, j] = gbts[j].predict(Xs)
    rho_s = df_fe["grain_density_g_cm3"].fillna(2.65).values
    if clip:
        means = blend_and_clip(gp_mean, gbt_mean, np.asarray(weights), rho_s, H)
    else:
        means = (gp_mean if np.isnan(gbt_mean).any()
                 else np.asarray(weights) * gp_mean + (1 - np.asarray(weights)) * gbt_mean)
    return means, stds


def load_model(path):
    """Load a saved GP artifacts dict from a .pt file."""
    return torch.load(path, weights_only=False)


def predict_from_raw(df_raw, artifacts, H, clip=True):
    """Convenience: run base feature engineering, then predict from a raw frame."""
    return gp_predict(artifacts, base_feature_engineering(df_raw.copy(), H), H, clip)


def plot_pred_vs_actual(y_true, y_pred, y_std, target, mae, r2, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=120)
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ms=4, alpha=0.55,
                ecolor="#b0b0b0", elinewidth=0.8, color="#1f7a5c",
                markeredgecolor="white", markeredgewidth=0.4)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="ideal")
    ax.set_xlabel(f"actual {target}", fontweight="bold")
    ax.set_ylabel(f"predicted {target}", fontweight="bold")
    ax.set_title(f"{target}  (GP ±1σ)\nMAE={mae:.3f}  R²={r2:.3f}", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    os.makedirs(args.report_dir, exist_ok=True)
    log_path = args.log if os.path.isabs(args.log) else os.path.join(args.report_dir, args.log)
    logger = setup_logger(log_path)
    logger.info("Gaussian Process regression baseline (small-data model)")
    logger.info("seed=%d  folds=%d  kernel restarts=%d", args.seed, args.folds, args.restarts)

    H = load_helpers(args.helpers_dir)
    H.CFG.seed_everything(args.seed)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    logger.info("train shape=%s  test shape=%s", train.shape, test.shape)

    train = base_feature_engineering(train, H)
    test = base_feature_engineering(test, H)

    y = train[TARGETS].values.astype(float)
    X_base = train.drop(columns=TARGETS)
    exclude = ["id", "atterberg_is_missing", "kf_is_missing", "loi_is_missing"]
    cols_for_imputation = numeric_impute_columns(X_base, exclude)
    logger.info("features=%d  MICE-imputed columns=%d",
                X_base.shape[1], len(cols_for_imputation))

    # ---- stratified-shuffle cross-validation ----
    # StratifiedShuffleSplit needs class labels, so we stratify on quantile bins
    # of the density target (MDD): every split's validation set then spans the
    # full density range. Splits are random subsamples (they may overlap and need
    # not cover every row), so we average each row's predictions over the splits
    # it lands in before computing the aggregate metrics.
    n_strata = int(min(args.val_strata, max(2, len(y) // 10)))
    edges = np.unique(np.quantile(y[:, 0], np.linspace(0, 1, n_strata + 1)))
    strata = np.digitize(y[:, 0], edges[1:-1])
    sss = StratifiedShuffleSplit(n_splits=args.folds, test_size=args.val_frac,
                                 random_state=args.seed)
    logger.info("StratifiedShuffleSplit: %d splits, test_size=%.2f, %d density strata",
                args.folds, args.val_frac, n_strata)

    gp_sum = np.zeros_like(y)         # summed GP predictions per row
    gbt_sum = np.zeros_like(y)        # summed GBT predictions per row
    ssum = np.zeros_like(y)           # summed GP std per row
    cnt = np.zeros(len(y))
    fold_records = []                 # (va, gp_pred, gbt_pred) per split
    for k, (tr, va) in enumerate(sss.split(X_base, strata)):
        Xtr, Xva = preprocess_fold(X_base.iloc[tr].copy(), X_base.iloc[va].copy(),
                                   cols_for_imputation, H, args.seed)
        gps, sels, gbts = fit_bases(Xtr, y[tr], args)
        gpm, gps_std, gbm = predict_bases(Xva, gps, sels, gbts)
        gp_sum[va] += gpm
        gbt_sum[va] += np.nan_to_num(gbm)
        ssum[va] += gps_std
        cnt[va] += 1
        fold_records.append((va, gpm, gbm))
        logger.info("split %d GP-NMAE = %.4f", k + 1, nmae(y[va], gpm))

    seen = cnt > 0
    logger.info("validation coverage: %d/%d rows scored across splits",
                int(seen.sum()), len(y))
    y_seen = y[seen]
    gp_oof = gp_sum[seen] / cnt[seen, None]
    gbt_oof = (gbt_sum[seen] / cnt[seen, None]) if args.ensemble else np.full_like(gp_oof, np.nan)
    oof_std = ssum[seen] / cnt[seen, None]
    rho_s_seen = X_base["grain_density_g_cm3"].fillna(2.65).values[seen]

    # choose per-target blend weights on the out-of-fold predictions
    weights = choose_blend_weights(y_seen, gp_oof, gbt_oof)
    logger.info("\nBlend weights (GP share per target): %s",
                {TARGETS[j]: round(float(weights[j]), 2) for j in range(len(TARGETS))})

    # aggregate NMAE for GP-only, GBT-only, and the blend
    gp_clip = blend_and_clip(gp_oof, np.full_like(gp_oof, np.nan),
                             np.ones(len(TARGETS)), rho_s_seen, H)
    blend_clip = blend_and_clip(gp_oof, gbt_oof, weights, rho_s_seen, H)
    logger.info("\n===== VALIDATION EVALUATION (averaged over splits) =====")
    logger.info("Aggregate NMAE  GP-only: %.4f", nmae(y_seen, gp_clip))
    if args.ensemble:
        gbt_clip = blend_and_clip(gp_oof, gbt_oof, np.zeros(len(TARGETS)), rho_s_seen, H)
        logger.info("Aggregate NMAE  GBT-only: %.4f", nmae(y_seen, gbt_clip))
        logger.info("Aggregate NMAE  BLEND:    %.4f", nmae(y_seen, blend_clip))

    # per-split consistency of the chosen blend (uses stored per-split base preds)
    split_blend = []
    for va, gpm, gbm in fold_records:
        pred = blend_and_clip(gpm, gbm if args.ensemble else np.full_like(gpm, np.nan),
                              weights, X_base["grain_density_g_cm3"].fillna(2.65).values[va], H)
        split_blend.append(nmae(y[va], pred))
    logger.info("Per-split BLEND NMAE: mean %.4f  std %.4f  (min %.4f, max %.4f)",
                float(np.mean(split_blend)), float(np.std(split_blend)),
                float(np.min(split_blend)), float(np.max(split_blend)))

    oof_clip = blend_clip
    mae, rmse, r2 = regression_metrics(y_seen, oof_clip)
    logger.info("%-22s %10s %10s %8s %12s", "target", "MAE", "RMSE", "R2", "mean σ")
    for i, name in enumerate(TARGETS):
        logger.info("%-22s %10.4f %10.4f %8.3f %12.3f", name, mae[i], rmse[i], r2[i],
                    float(np.nanmean(oof_std[:, i])))
        img = os.path.join(args.report_dir, f"gpr_pred_vs_actual_{name}.png")
        plot_pred_vs_actual(y_seen[:, i], oof_clip[:, i], oof_std[:, i], name,
                            mae[i], r2[i], img)
        logger.info("saved -> %s", img)

    # ---- fit final model on all training data, save it, predict test ----
    artifacts = fit_full_model(X_base, y, cols_for_imputation, H, args, weights=weights)
    if args.model_out:
        torch.save(artifacts, args.model_out)
        logger.info("\nSaved GP model -> %s", args.model_out)
        for name, kern in zip(TARGETS, artifacts["kernels"]):
            logger.info("  learned kernel [%s]: %s", name, kern)

    # ARD relevance report: which features the GP found most informative
    if args.ard:
        feat = artifacts["feature_cols"]
        logger.info("\nARD feature relevance (shortest length-scale = most informative):")
        for name, gp, sel in zip(TARGETS, artifacts["gps"], artifacts["selected"]):
            sel_names = [feat[i] for i in sel]
            logger.info("  [%s] kept %d/%d features after selection",
                        name, len(sel), len(feat))
            ls = learned_length_scales(gp, len(sel))
            if ls is None:
                logger.info("      (isotropic kernel — no per-feature relevance)")
                continue
            order = np.argsort(ls)
            for i in order[:args.relevance_top]:
                logger.info("      %-34s length-scale=%.3g", sel_names[i], float(ls[i]))

    # predict the test set from the saved artifacts (test is already base-FE'd)
    preds, stds = gp_predict(artifacts, test, H, clip=True)
    mdd, owc = preds[:, 0], preds[:, 1]

    out = pd.DataFrame({
        "id": test["id"].values,
        "proctor_owc_pct": np.round(owc, 3),
        "proctor_mdd_g_cm3": np.round(mdd, 4),
    })
    out.to_csv(args.out, index=False)

    if args.uncertainty_out:
        unc = pd.DataFrame({
            "id": test["id"].values,
            "owc_std": np.round(stds[:, 1], 3),
            "mdd_std": np.round(stds[:, 0], 4),
        })
        unc.to_csv(args.uncertainty_out, index=False)
        logger.info("Wrote per-sample uncertainty -> %s", args.uncertainty_out)

    logger.info("\nWrote %d predictions -> %s", len(out), args.out)
    logger.info("Run log -> %s", log_path)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=".")
    p.add_argument("--helpers_dir", default=".")
    p.add_argument("--out", default="submission_gpr.csv")
    p.add_argument("--model_out", default="proctor_gpr.pt",
                   help="path to save the fitted GP model as a .pt file (empty to skip)")
    p.add_argument("--uncertainty_out", default="gpr_uncertainty.csv",
                   help="CSV of per-sample predictive std (empty string to skip)")
    p.add_argument("--report_dir", default=".")
    p.add_argument("--log", default="proctor_gpr.log")
    p.add_argument("--folds", type=int, default=5,
                   help="number of StratifiedShuffleSplit splits")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="validation fraction per split (test_size)")
    p.add_argument("--val_strata", type=int, default=5,
                   help="density strata used to stratify the split")
    p.add_argument("--restarts", type=int, default=6,
                   help="kernel hyperparameter optimizer restarts")
    p.add_argument("--kernel", choices=["matern", "rbf", "rq"], default="matern",
                   help="smoothness kernel (matern recommended)")
    p.add_argument("--ard", dest="ard", action="store_true", default=True,
                   help="per-feature length-scales (Automatic Relevance Determination)")
    p.add_argument("--isotropic", dest="ard", action="store_false",
                   help="use a single shared length-scale instead of ARD")
    p.add_argument("--relevance_top", type=int, default=10,
                   help="how many top ARD features to report per target")
    p.add_argument("--select", dest="select", action="store_true", default=True,
                   help="two-stage ARD feature selection (fit, drop noise, refit)")
    p.add_argument("--no_select", dest="select", action="store_false",
                   help="disable ARD feature selection")
    p.add_argument("--select_thresh", type=float, default=100.0,
                   help="keep features whose learned length-scale is below this")
    p.add_argument("--select_min", type=int, default=5,
                   help="always keep at least this many (most relevant) features")
    p.add_argument("--linear", action="store_true",
                   help="add a linear (DotProduct) kernel term for global trends")
    p.add_argument("--ensemble", dest="ensemble", action="store_true", default=True,
                   help="blend the GP with gradient-boosted trees (recommended)")
    p.add_argument("--no_ensemble", dest="ensemble", action="store_false",
                   help="use the GP alone (no blend)")
    p.add_argument("--gbt_iter", type=int, default=400, help="GBT boosting iterations")
    p.add_argument("--gbt_lr", type=float, default=0.05, help="GBT learning rate")
    p.add_argument("--gbt_leaves", type=int, default=31, help="GBT max leaf nodes")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())