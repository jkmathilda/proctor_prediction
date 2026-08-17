"""
v15test.py
==========
Same model as scripts/v15.py (GP + optional GBT ensemble, MICE-imputed
features minus hyd_cond_hyd_gradient, saturation-line clip on MDD) with two
changes:

1. NO SPLIT: v15.py's StratifiedShuffleSplit cross-validation loop (used
   only to report an out-of-fold NMAE/MAE/RMSE/R^2 and to pick a GP/GBT
   blend weight) is removed entirely. The final model was ALWAYS fit on
   100% of the training data regardless of that split (v15.py's own
   fit_full_model call uses X_base/y directly, never the split subset) --
   the split only ever affected the validation REPORT, never the actual
   submitted predictions. This script skips straight to that full-data fit:
   no held-out NMAE, no blend-weight search, no per-fold plots. Because
   there's no out-of-fold data to search a blend weight on, the GP/GBT
   blend weight defaults to pure GP (weights=None -> all-ones, same
   fallback v15.py itself uses) -- pass --gp_weight to blend in GBT anyway
   at a fixed ratio if you want the ensemble back without a split.

2. --exclude_ids (default [155]): drops train.csv id values before
   fitting, same rationale as model1i.py/model2i.py's default -- id 155 is
   an extreme leverage point (highest Cook's distance in the dataset by a
   wide margin, driven mostly by OWC) that measurably hurt those models'
   fit on the rest of the data. Pass --exclude_ids (with no values) to
   include every row again.

Run:
    python scripts/v15test.py
    python scripts/v15test.py --exclude_ids            # include every row
    python scripts/v15test.py --exclude_ids 155 69      # drop more than one id
"""

import argparse
import importlib.util
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # used only to save/load the model as a .pt file
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, DotProduct, Matern,
                                              RBF, RationalQuadratic, WhiteKernel)
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


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def load_helpers(helpers_dir):
    path = os.path.join(helpers_dir, "helper_functions.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"helper_functions.py not found at {path}")
    spec = importlib.util.spec_from_file_location("helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup_logger(path):
    logger = logging.getLogger("v15test")
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


def build_kernel(n_features, args):
    """Compose the GP kernel -- identical to v15.py's build_kernel."""
    ls = np.ones(n_features) if args.ard else 1.0
    if args.kernel == "rbf":
        base = RBF(length_scale=ls, length_scale_bounds=(1e-2, 1e3))
    elif args.kernel == "rq":
        base = RationalQuadratic(length_scale=1.0, alpha=1.0,
                                 length_scale_bounds=(1e-2, 1e3))
    else:  # matern (default)
        base = Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e3), nu=2.5)
    kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * base
              + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)))
    if args.linear:
        kernel = kernel + DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
    return kernel


def make_gp(args, n_features):
    return GaussianProcessRegressor(
        kernel=build_kernel(n_features, args), normalize_y=True, alpha=1e-8,
        n_restarts_optimizer=args.restarts, random_state=args.seed)


def learned_length_scales(gp, n_features):
    for k, v in gp.kernel_.get_params().items():
        if k.endswith("length_scale"):
            arr = np.atleast_1d(v)
            if arr.size == n_features:
                return arr
    return None


def select_and_fit(Xtr, y_col, args):
    """Fit an ARD GP, then (optionally) drop features the GP judged irrelevant
    and refit on the relevant subset -- identical to v15.py's select_and_fit."""
    gp = make_gp(args, Xtr.shape[1])
    gp.fit(Xtr, y_col)
    sel = np.arange(Xtr.shape[1])
    if args.select and args.ard:
        ls = learned_length_scales(gp, Xtr.shape[1])
        if ls is not None:
            keep = np.where(ls < args.select_thresh)[0]
            if keep.size < args.select_min:
                keep = np.argsort(ls)[:args.select_min]
            if 0 < keep.size < Xtr.shape[1]:
                sel = np.sort(keep)
                gp = make_gp(args, sel.size)
                gp.fit(Xtr[:, sel], y_col)
    return gp, sel


def make_gbt(args):
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

    `weights` are the per-target GP blend weights. No split means there's no
    out-of-fold data to search these on -- default (None) is pure GP
    (all-ones), same fallback v15.py itself uses whenever it has no OOF
    predictions to compare against.
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
        "gbts": gbts,
        "weights": np.asarray(weights),
        "selected": sels,
        "feature_cols": feature_cols,
        "cols_for_imputation": cols_for_imputation,
        "targets": TARGETS,
        "kernels": [str(gp.kernel_) for gp in gps],
    }


def _transform_with(artifacts, df_fe, H):
    X = df_fe.copy()
    _add_indicators(X)
    X[artifacts["cols_for_imputation"]] = artifacts["imputer"].transform(
        X[artifacts["cols_for_imputation"]])
    H.apply_fold_feature_engineering(X)
    X = X.reindex(columns=artifacts["feature_cols"])
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
    return torch.load(path, weights_only=False)


def predict_from_raw(df_raw, artifacts, H, clip=True):
    return gp_predict(artifacts, base_feature_engineering(df_raw.copy(), H), H, clip)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    log_path = args.log
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = setup_logger(log_path)
    logger.info("v15test -- v15.py's GP(+GBT) model, no validation split, full-data fit")
    logger.info("seed=%d  kernel restarts=%d", args.seed, args.restarts)

    H = load_helpers(args.helpers_dir)
    H.CFG.seed_everything(args.seed)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    logger.info("train shape=%s  test shape=%s", train.shape, test.shape)

    if args.exclude_ids:
        before = len(train)
        train = train[~train["id"].isin(args.exclude_ids)].reset_index(drop=True)
        logger.info("excluded ids=%s from training: %d -> %d rows",
                    args.exclude_ids, before, len(train))

    train = base_feature_engineering(train, H)
    test = base_feature_engineering(test, H)

    y = train[TARGETS].values.astype(float)
    X_base = train.drop(columns=TARGETS)
    # v15 vs v10: drop hyd_cond_hyd_gradient outright.
    X_base = X_base.drop(columns=["hyd_cond_hyd_gradient"], errors="ignore")
    exclude = ["id", "atterberg_is_missing", "kf_is_missing", "loi_is_missing"]
    cols_for_imputation = numeric_impute_columns(X_base, exclude)
    logger.info("features=%d  MICE-imputed columns=%d  (hyd_cond_hyd_gradient dropped)",
                X_base.shape[1], len(cols_for_imputation))

    weights = None if args.gp_weight is None else np.array([args.gp_weight, args.gp_weight])
    if weights is not None:
        logger.info("using fixed GP/GBT blend weight=%.2f (no OOF search -- no split)", args.gp_weight)
    else:
        logger.info("no split -> no OOF data to fit a GP/GBT blend weight on; using pure GP "
                    "(pass --gp_weight to force a fixed blend)")

    artifacts = fit_full_model(X_base, y, cols_for_imputation, H, args, weights=weights)
    if args.model_out:
        os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
        torch.save(artifacts, args.model_out)
        logger.info("Saved GP model -> %s", args.model_out)
        for name, kern in zip(TARGETS, artifacts["kernels"]):
            logger.info("  learned kernel [%s]: %s", name, kern)

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

    # in-sample fit quality on the training rows actually used (informational
    # only -- this is NOT an out-of-fold estimate, since there's no split).
    train_preds, _ = gp_predict(artifacts, train.drop(columns=[]), H, clip=True)
    train_nmae = nmae(y, train_preds)
    logger.info("\nIn-sample NMAE on training rows actually fit (NOT out-of-fold, no split): %.4f",
                train_nmae)

    preds, stds = gp_predict(artifacts, test, H, clip=True)
    mdd, owc = preds[:, 0], preds[:, 1]

    out = pd.DataFrame({
        "id": test["id"].values,
        "proctor_owc_pct": np.round(owc, 3),
        "proctor_mdd_g_cm3": np.round(mdd, 4),
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    if args.uncertainty_out:
        os.makedirs(os.path.dirname(args.uncertainty_out) or ".", exist_ok=True)
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
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(repo_root / "data"))
    p.add_argument("--helpers_dir", default=str(repo_root / "src"))
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_v15test.csv"))
    p.add_argument("--model_out", default=str(repo_root / "models" / "v15test.pt"),
                   help="path to save the fitted GP model as a .pt file (empty to skip)")
    p.add_argument("--uncertainty_out", default=str(repo_root / "logs" / "v15test_uncertainty.csv"),
                   help="CSV of per-sample predictive std (empty string to skip)")
    p.add_argument("--log", default=str(repo_root / "logs" / "v15test_run.log"))
    p.add_argument("--exclude_ids", type=int, nargs="*", default=[155],
                   help="train.csv id values to drop before fitting. Defaults to [155] -- "
                        "see model1i.py's --exclude_ids help for why. Pass --exclude_ids "
                        "(with no values) to include every row again.")
    p.add_argument("--gp_weight", type=float, default=None,
                   help="fixed GP share [0,1] for the GP/GBT blend (default: pure GP, since "
                        "there's no split to search a weight on)")
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
                   help="also fit gradient-boosted trees (blended in only if --gp_weight < 1)")
    p.add_argument("--no_ensemble", dest="ensemble", action="store_false",
                   help="use the GP alone, don't fit GBT at all")
    p.add_argument("--gbt_iter", type=int, default=400, help="GBT boosting iterations")
    p.add_argument("--gbt_lr", type=float, default=0.05, help="GBT learning rate")
    p.add_argument("--gbt_leaves", type=int, default=31, help="GBT max leaf nodes")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
