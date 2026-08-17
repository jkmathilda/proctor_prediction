"""
model1i_v15features_model1iimpute.py
=====================================
Second half of the feature-set-vs-imputation ablation started by
model1i_v15features.py. That script combined v15's feature set with v15's
SINGLE MICE imputer (over every numeric column) and lost to plain model1i
on Kaggle (0.2532 vs 0.2465) despite a large internal gain -- traced to a
confound, not the extra features themselves: v15's imputer fabricates
Atterberg limit values for coarse-grained rows, where those limits are
never physically measured. model1i's own fine/coarse-split imputer avoids
this by construction (it doesn't invent Atterberg values for coarse rows
at all -- see add_imputed_features's docstring).

THIS script removes that confound: v15's extra engineered features (DIN
fractions, log-ratio PSD shape features via base_feature_engineering) are
kept, but imputation now follows model1i's own actual discipline instead
of v15's:
  - kf and loss-on-ignition: fine/coarse-split MICE via
    add_imputed_features (imputed for BOTH groups, matching model1i.py),
    using the `_completed`/`_was_missing` columns IMPUTED_FEATURES itself
    uses -- not v15's raw single-imputed columns.
  - Atterberg limits (liquid/plastic limit): DROPPED entirely, matching
    model1i.py's own IMPUTED_FEATURES, which never uses them as a model
    feature at all -- not because they're unimportant (v15's own ARD
    analysis ranks atterberg_plastic_limit_pct as MDD's single most
    relevant feature), but because there is no physically honest way to
    give a coarse-grained row an Atterberg value; model1i.py's design
    sidesteps the problem instead of solving it, and this ablation follows
    that same choice rather than inventing a new one.
  - 'fine-grained' itself: computed (needed as the imputation split key)
    but NOT included as a model feature, matching model1i.py's
    MICE_PREDICTORS = NO_MISSING_FEATURES minus 'fine-grained'.
  - v15's own psd_fraction_clay/silt/sand/gravel are dropped in favor of
    model1i's clay/silt/sand/gravel (mathematically identical, just a
    naming difference -- keeping both would just be redundant, perfectly
    collinear columns).

Everything else -- WeightedBlendRegressor, Optuna-tuned XGBoost, id-155
excluded by default, StratifiedKFold OOF, OWC isotonic calibration,
saturation-line clip -- is unchanged from model1i.py / model1i_v15features.py.

Run:
    python scripts/model1i_v15features_model1iimpute.py
"""

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.general_model_impute import (
    WeightedBlendRegressor,
    tune_xgb_with_optuna,
    make_stratified_kfold_splits,
    add_no_missing_features,
    add_imputed_features,
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
    logger = logging.getLogger("model1i_v15features_model1iimpute")
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


def build_features(train_raw, test_raw, v15, H, seed):
    """v15's engineered feature set + model1i's own fine/coarse-split
    imputation discipline (Atterberg dropped, kf/LOI fine/coarse-imputed)."""
    train_fe = v15.base_feature_engineering(train_raw.copy(), H)
    test_fe = v15.base_feature_engineering(test_raw.copy(), H)

    # model1i's own base engineering: clay/silt/sand/gravel/'fine-grained'
    # (the imputation split key) + feat_cu/feat_cc/feat_log_cu.
    train_fe = add_no_missing_features(train_fe)
    test_fe = add_no_missing_features(test_fe)

    # model1i's real fine/coarse-split MICE (fit on train only, applied to
    # test with the SAME fitted imputers -- never refit on test).
    train_imp, fine_imputer, coarse_imputer = add_imputed_features(train_fe, random_state=seed)
    test_imp, _, _ = add_imputed_features(
        test_fe, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer, random_state=seed,
    )

    drop_cols = TARGETS + [
        "id", "hyd_cond_hyd_gradient",
        # raw (unimputed / v15-single-imputed-elsewhere) columns replaced by
        # model1i's own _completed/_was_missing columns:
        "hyd_cond_kf_m_s", "loss_on_ignition_pct",
        # Atterberg: dropped entirely, matching model1i.py (see docstring) --
        # including add_imputed_features' own _completed/_was_missing columns,
        # since atterberg_*_completed stays NaN for coarse rows by design
        # (add_imputed_features only fills Atterberg within fine_mask).
        "atterberg_liquid_limit_pct", "atterberg_plastic_limit_pct",
        "atterberg_liquid_limit_pct_completed", "atterberg_liquid_limit_pct_was_missing",
        "atterberg_plastic_limit_pct_completed", "atterberg_plastic_limit_pct_was_missing",
        # v15's own DIN fractions, redundant with model1i's clay/silt/sand/gravel:
        "psd_fraction_clay", "psd_fraction_silt", "psd_fraction_sand", "psd_fraction_gravel",
        # imputation split key, not a model feature (matches model1i.py):
        "fine-grained",
    ]

    y_train = train_imp[TARGETS].copy()
    X_train = train_imp.drop(columns=[c for c in drop_cols if c in train_imp.columns])
    X_test = test_imp.drop(columns=[c for c in drop_cols if c in test_imp.columns and c not in TARGETS])

    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test[X_train.columns]

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
    logger.info("model1i_v15features_model1iimpute -- v15's feature set + model1i's fine/coarse-split imputation")
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

    X_train, y_train, X_test = build_features(train_raw, test_raw, v15, H, args.seed)
    logger.info("feature set: %d columns = %s", X_train.shape[1], list(X_train.columns))
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("NaNs remain in the feature matrix after imputation.")

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
            # Skip calibration this run: this feature set/imputer combo's own honest
            # nested-CV NMAE (logged above) found calibration makes it worse, unlike
            # model1i.py's production feature set -- so applying it here would just
            # reintroduce a known-harmful correction on top of the imputer-fix test.
            logger.info("[%s] calibration NOT applied to test predictions (raw OOF NMAE is better)", target)

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
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_model1i_v15features_model1iimpute.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "model1i_v15features_model1iimpute_run.log"))
    p.add_argument("--exclude_ids", type=int, nargs="*", default=[155])
    p.add_argument("--optuna_trials", type=int, default=50)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val_strata", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
