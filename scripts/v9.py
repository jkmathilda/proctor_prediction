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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern, WhiteKernel)
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
    path = os.path.join('/Users/aliceqi/Documents/GitHub/proctor-prediction-challenge/helper_functions.py')
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


def make_gp(args):
    """A Matern-kernel GP: smooth-but-flexible, with a learned noise level.

    ConstantKernel scales the signal, Matern(nu=2.5) is the workhorse kernel for
    physical data (twice differentiable), and WhiteKernel absorbs measurement
    noise. normalize_y centers/scales the target internally.
    """
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
              + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)))
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=1e-8,
        n_restarts_optimizer=args.restarts, random_state=args.seed)


def fit_predict_targets(Xtr, ytr, Xte, args):
    """Fit one GP per target; return (means, stds) each shape (n, 2)."""
    means = np.zeros((len(Xte), len(TARGETS)))
    stds = np.zeros((len(Xte), len(TARGETS)))
    for j in range(len(TARGETS)):
        gp = make_gp(args)
        gp.fit(Xtr, ytr[:, j])
        m, s = gp.predict(Xte, return_std=True)
        means[:, j], stds[:, j] = m, s
    return means, stds


# --------------------------------------------------------------------------- #
# Saveable model: fit on all data, persist, reload, predict
# --------------------------------------------------------------------------- #
def _add_indicators(df):
    df["atterberg_is_missing"] = df["atterberg_liquid_limit_pct"].isnull().astype(int)
    df["kf_is_missing"] = df["hyd_cond_kf_m_s"].isnull().astype(int)
    df["loi_is_missing"] = df["loss_on_ignition_pct"].isnull().astype(int)


def fit_full_model(X_base, y, cols_for_imputation, H, args):
    """Fit the imputer, scaler, and one GP per target on ALL training data.

    Returns a self-contained artifacts dict (fitted objects + the exact feature
    column order and kernels) that reproduces predictions without the raw data.
    """
    X = X_base.copy()
    _add_indicators(X)
    imputer = H.get_default_mice_imputer(seed=args.seed)
    X[cols_for_imputation] = imputer.fit_transform(X[cols_for_imputation])
    H.apply_fold_feature_engineering(X)
    feature_cols = list(X.columns)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    gps = []
    for j in range(len(TARGETS)):
        gp = make_gp(args)
        gp.fit(Xs, y[:, j])
        gps.append(gp)
    return {
        "imputer": imputer,
        "scaler": scaler,
        "gps": gps,
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
    """Predict [MDD, OWC] means + stds for a base-FE'd frame using saved artifacts."""
    Xs = _transform_with(artifacts, df_fe, H)
    means = np.zeros((len(df_fe), len(TARGETS)))
    stds = np.zeros((len(df_fe), len(TARGETS)))
    for j, gp in enumerate(artifacts["gps"]):
        m, s = gp.predict(Xs, return_std=True)
        means[:, j], stds[:, j] = m, s
    if clip:
        rho_s = df_fe["grain_density_g_cm3"].fillna(2.65).values
        owc = np.clip(means[:, 1], 0, None)
        mdd = np.minimum(means[:, 0], H.calc_satline(owc, rho_s) * 0.999)
        means = np.column_stack([mdd, owc])
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

    psum = np.zeros_like(y)          # sum of predictions per row
    ssum = np.zeros_like(y)          # sum of predictive std per row
    cnt = np.zeros(len(y))           # how many splits scored each row
    fold_nmae = []
    for k, (tr, va) in enumerate(sss.split(X_base, strata)):
        Xtr, Xva = preprocess_fold(X_base.iloc[tr].copy(), X_base.iloc[va].copy(),
                                   cols_for_imputation, H, args.seed)
        means, stds = fit_predict_targets(Xtr, y[tr], Xva, args)
        psum[va] += means
        ssum[va] += stds
        cnt[va] += 1
        fold_nmae.append(nmae(y[va], means))
        logger.info("split %d NMAE = %.4f", k + 1, fold_nmae[-1])

    seen = cnt > 0                    # rows that appeared in at least one val split
    logger.info("validation coverage: %d/%d rows scored across splits",
                int(seen.sum()), len(y))
    oof = psum[seen] / cnt[seen, None]
    oof_std = ssum[seen] / cnt[seen, None]
    y_seen = y[seen]

    # physical saturation clip on the averaged predictions
    rho_s_tr = X_base["grain_density_g_cm3"].fillna(2.65).values[seen]
    owc_oof = np.clip(oof[:, 1], 0, None)
    mdd_oof = np.minimum(oof[:, 0], H.calc_satline(owc_oof, rho_s_tr) * 0.999)
    oof_clip = np.column_stack([mdd_oof, owc_oof])

    mae, rmse, r2 = regression_metrics(y_seen, oof_clip)
    logger.info("\n===== VALIDATION EVALUATION (averaged over splits) =====")
    logger.info("Per-split NMAE: mean %.4f  std %.4f  (min %.4f, max %.4f)",
                float(np.mean(fold_nmae)), float(np.std(fold_nmae)),
                float(np.min(fold_nmae)), float(np.max(fold_nmae)))
    logger.info("Aggregate NMAE on scored rows: %.4f", nmae(y_seen, oof_clip))
    logger.info("%-22s %10s %10s %8s %12s", "target", "MAE", "RMSE", "R2", "mean σ")
    for i, name in enumerate(TARGETS):
        logger.info("%-22s %10.4f %10.4f %8.3f %12.3f", name, mae[i], rmse[i], r2[i],
                    float(np.nanmean(oof_std[:, i])))
        img = os.path.join(args.report_dir, f"gpr_pred_vs_actual_{name}.png")
        plot_pred_vs_actual(y_seen[:, i], oof_clip[:, i], oof_std[:, i], name,
                            mae[i], r2[i], img)
        logger.info("saved -> %s", img)

    # ---- fit final model on all training data, save it, predict test ----
    artifacts = fit_full_model(X_base, y, cols_for_imputation, H, args)
    if args.model_out:
        torch.save(artifacts, args.model_out)
        logger.info("\nSaved GP model -> %s", args.model_out)
        for name, kern in zip(TARGETS, artifacts["kernels"]):
            logger.info("  learned kernel [%s]: %s", name, kern)

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
    p.add_argument("--restarts", type=int, default=4,
                   help="kernel hyperparameter optimizer restarts")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())