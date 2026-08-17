"""
model1i_v15features.py
=======================
Ablation: model1i's exact fitting approach (Ridge + GPR + XGBoost
`WeightedBlendRegressor`, Optuna-tuned XGBoost, id-155 excluded from
training, StratifiedKFold OOF, OWC isotonic calibration, saturation-line
clip) applied to v15's feature set instead of model1i's own curated
`IMPUTED_FEATURES` (28 cols).

Why: this project has two feature families that were never cross-tested --
model1i's `IMPUTED_FEATURES` (28 cols, MICE-imputed Atterberg/kf/LOI,
fine/coarse-split imputation) and v15's `base_feature_engineering` output
(~38 cols, single MICE imputer over every numeric column, DIN fractions +
log-ratio PSD shape features, hyd_cond_hyd_gradient dropped). Every prior
comparison in this repo confounds feature set AND model architecture at
the same time (model1i's blend vs v15's GP+GBT+ARD, on their own respective
features) -- this script isolates the feature-set variable by holding
model1i's architecture fixed and swapping only the features, the other half
of the 2x2 this project hasn't run (the reverse cell -- v15's ARD-GP+GBT
architecture on IMPUTED_FEATURES -- is not implemented here).

Feature-set differences from real v15.py, both deliberate:
  - Uses v15's SINGLE MICE imputer over all numeric columns (not model1i's
    fine/coarse-grained split) -- this is v15's real imputation, not
    model1i's, since the point is to test v15's feature set as-is.
  - Drops the raw 'id' column before fitting. Real v15.py leaves 'id' in
    its feature matrix as an (accidental, undocumented) model input; that's
    not a real feature of v15's design, so it's excluded here rather than
    faithfully reproduced.
  - No ARD-based feature pruning (v15's GP self-prunes ~38 cols down to
    ~13-17 per target; model1i's GPR component is itself ARD-capable --
    per-feature RBF length-scale -- but doesn't hard-prune). All ~37
    features are handed to the blend as-is.

Run:
    python scripts/model1i_v15features.py
    python scripts/model1i_v15features.py --optuna_trials 0   # skip tuning, fast
"""

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.general_model_impute import (
    WeightedBlendRegressor,
    tune_xgb_with_optuna,
    make_stratified_kfold_splits,
)

TARGETS = ["proctor_mdd_g_cm3", "proctor_owc_pct"]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nmae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    q75, q25 = np.percentile(y_true, [75, 25], axis=0)
    iqr = np.where((q75 - q25) == 0, 1e-8, q75 - q25)
    return float(np.mean(mae / iqr))


def setup_logger(path):
    logger = logging.getLogger("model1i_v15features")
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


def build_v15_features(train_raw, test_raw, v15, H, seed):
    """v15's real feature set: base_feature_engineering + a SINGLE MICE
    imputer fit on train (transformed onto test, never refit on test),
    hyd_cond_hyd_gradient dropped, 'id' dropped (see module docstring)."""
    train_fe = v15.base_feature_engineering(train_raw.copy(), H)
    test_fe = v15.base_feature_engineering(test_raw.copy(), H)

    for f in (train_fe, test_fe):
        f["atterberg_is_missing"] = f["atterberg_liquid_limit_pct"].isnull().astype(int)
        f["kf_is_missing"] = f["hyd_cond_kf_m_s"].isnull().astype(int)
        f["loi_is_missing"] = f["loss_on_ignition_pct"].isnull().astype(int)

    y_train = train_fe[TARGETS].copy()
    X_train = train_fe.drop(columns=TARGETS)
    X_test = test_fe.copy()
    for X in (X_train, X_test):
        X.drop(columns=["hyd_cond_hyd_gradient", "id"], errors="ignore", inplace=True)

    exclude = ["atterberg_is_missing", "kf_is_missing", "loi_is_missing"]
    cols_for_imputation = v15.numeric_impute_columns(X_train, exclude)

    imputer = H.get_default_mice_imputer(seed=seed)
    X_train[cols_for_imputation] = imputer.fit_transform(X_train[cols_for_imputation])
    X_test[cols_for_imputation] = imputer.transform(X_test[cols_for_imputation])

    for X in (X_train, X_test):
        H.apply_fold_feature_engineering(X)

    return X_train, y_train, X_test


def fit_owc_calibrator(oof_pred, y, *, n_folds=10, seed=42, logger):
    oof_pred = np.asarray(oof_pred, dtype=float)
    y = np.asarray(y, dtype=float)
    strata = pd.qcut(y, 5, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    nested_pred = np.full(len(y), np.nan)
    for train_idx, val_idx in skf.split(np.zeros(len(y)), strata):
        fold_cal = IsotonicRegression(out_of_bounds="clip")
        fold_cal.fit(oof_pred[train_idx], y[train_idx])
        nested_pred[val_idx] = fold_cal.predict(oof_pred[val_idx])
    nmae_raw = nmae(y, oof_pred)
    nmae_calibrated = nmae(y, nested_pred)
    logger.info(
        "[proctor_owc_pct] OWC isotonic calibration -- nested %d-fold honest NMAE: "
        "raw=%.6f calibrated=%.6f (delta=%+.6f)",
        n_folds, nmae_raw, nmae_calibrated, nmae_calibrated - nmae_raw,
    )
    final_calibrator = IsotonicRegression(out_of_bounds="clip")
    final_calibrator.fit(oof_pred, y)
    return final_calibrator


def main(args):
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logger = setup_logger(args.log)
    logger.info("model1i_v15features -- model1i's WeightedBlendRegressor (Ridge+GPR+XGBoost) on v15's feature set")
    logger.info("seed=%d  optuna_trials=%d  exclude_ids=%s", args.seed, args.optuna_trials, args.exclude_ids)

    repo_root = Path(__file__).resolve().parent.parent
    v15 = load_module(str(repo_root / "scripts" / "v15.py"), "v15_mod")
    model1i_mod = load_module(str(repo_root / "scripts" / "model1i.py"), "model1i_mod")
    H = v15.load_helpers(args.helpers_dir)
    calc_satline = model1i_mod.load_calc_satline(args.helpers_dir)

    train_raw = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test_raw = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    logger.info("train shape=%s  test shape=%s", train_raw.shape, test_raw.shape)

    if args.exclude_ids:
        before = len(train_raw)
        train_raw = train_raw[~train_raw["id"].isin(args.exclude_ids)].reset_index(drop=True)
        logger.info("excluded ids=%s from training: %d -> %d rows", args.exclude_ids, before, len(train_raw))

    X_train, y_train, X_test = build_v15_features(train_raw, test_raw, v15, H, args.seed)
    logger.info("v15-style feature set: %d columns (MICE-imputed, single imputer, hyd_grad+id dropped)",
                X_train.shape[1])
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("NaNs remain in the v15-style feature matrix after imputation.")

    y_mdd_all = y_train["proctor_mdd_g_cm3"].to_numpy(dtype=float)
    splits = make_stratified_kfold_splits(y_mdd_all, n_splits=args.folds, val_strata=args.val_strata, random_state=args.seed)
    logger.info("validation scheme: StratifiedKFold folds=%d val_strata=%d (stratified on MDD, full coverage)",
                args.folds, args.val_strata)

    preds = {}
    owc_calibrator = None
    for target in TARGETS:
        y = y_train[target].to_numpy(dtype=float)
        if args.optuna_trials > 0:
            logger.info("[%s] tuning XGBoost with Optuna (%d trials)...", target, args.optuna_trials)
            xgb_model, study = tune_xgb_with_optuna(X_train, y, n_trials=args.optuna_trials, random_state=args.seed, splits=splits)
            logger.info("[%s] best CV MAE=%.4f  params=%s", target, study.best_value, study.best_params)
        else:
            xgb_model = None

        model = WeightedBlendRegressor(random_state=args.seed, xgb_model=xgb_model, splits=splits)
        model.fit(X_train, y)
        summary = model.get_training_summary()
        logger.info("[%s] blend weights: ridge=%.3f gpr=%.3f xgboost=%.3f",
                     target, summary["weights"]["ridge"], summary["weights"]["gpr"], summary["weights"]["xgboost"])
        logger.info("[%s] internal OOF (n_validated=%d/%d): R2=%.4f MAE=%.4f RMSE=%.4f NMAE=%.6f",
                     target, summary["n_validated"], len(y_train),
                     summary["blend_oof_metrics"]["r2"], summary["blend_oof_metrics"]["mae"],
                     summary["blend_oof_metrics"]["rmse"], nmae(y, model.blend_oof_prediction_))

        preds[target] = model.predict(X_test)

        if target == "proctor_owc_pct":
            owc_calibrator = fit_owc_calibrator(model.blend_oof_prediction_, y, seed=args.seed, logger=logger)
            preds[target] = owc_calibrator.predict(preds[target])

    rho_s = test_raw["grain_density_g_cm3"].fillna(2.65).to_numpy()
    owc = np.clip(preds["proctor_owc_pct"], 0, None)
    mdd = np.minimum(preds["proctor_mdd_g_cm3"], calc_satline(owc, rho_s) * 0.999)
    n_clipped = int((mdd < preds["proctor_mdd_g_cm3"] - 1e-9).sum())
    logger.info("saturation-line clip applied to %d/%d MDD predictions", n_clipped, len(mdd))

    out = pd.DataFrame({
        "id": test_raw["id"].values,
        "proctor_owc_pct": np.round(owc, 3),
        "proctor_mdd_g_cm3": np.round(mdd, 4),
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    logger.info("Wrote %d predictions -> %s", len(out), args.out)
    logger.info("Run log -> %s", args.log)
    return out


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(repo_root / "data"))
    p.add_argument("--helpers_dir", default=str(repo_root / "src"))
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_model1i_v15features.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "model1i_v15features_run.log"))
    p.add_argument("--exclude_ids", type=int, nargs="*", default=[155])
    p.add_argument("--optuna_trials", type=int, default=50)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val_strata", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
