"""
model1.py
=========

CLI pipeline for training and generating submissions with
general_model.py's WeightedBlendRegressor (Ridge + GPR + XGBoost).

Uses the complete (no-missing-values) feature set, so no imputation is
required. By default, XGBoost hyperparameters are optimized with Optuna
before fitting; setting --optuna_trials 0 skips tuning and uses the
default configuration.

Applies physical constraints by clipping OWC to >= 0 and MDD to the
Zero-Air-Voids (saturation) limit. Trained models are cached with joblib
and automatically reused unless --force_retrain is specified.

Run:
    python scripts/model1.py
    python scripts/model1.py --optuna_trials 0
    python scripts/model1.py --force_retrain
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.general_model import (
    WeightedBlendRegressor,
    tune_xgb_with_optuna,
    make_stratified_kfold_splits,
    make_stratified_shuffle_splits,
    NO_MISSING_FEATURES,
)

TARGETS = ["proctor_mdd_g_cm3", "proctor_owc_pct"]


def nmae(y_true, y_pred):
    """Official competition metric: mean column-wise IQR-normalized MAE."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    q75, q25 = np.percentile(y_true, [75, 25], axis=0)
    iqr = np.where((q75 - q25) == 0, 1e-8, q75 - q25)
    return float(np.mean(mae / iqr))


def load_calc_satline(helpers_dir):
    """Dynamically load helper_functions.py's calc_satline, matching the
    load_helpers pattern used in v10/v14/v15/v16.py."""
    path = os.path.join(helpers_dir, "helper_functions.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"helper_functions.py not found at {path}")
    spec = importlib.util.spec_from_file_location("helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.calc_satline


def add_no_missing_features(df):
    """Derive the NO_MISSING_FEATURES engineered columns (clay/silt/sand/
    gravel/fine-grained/feat_cu/feat_cc/feat_log_cu), matching
    data_analysis.ipynb's feature engineering exactly."""
    df = df.copy()
    df["clay"] = df["psd_passing_at_0_002mm_pct"]
    df["silt"] = df["psd_passing_at_0_063mm_pct"] - df["psd_passing_at_0_002mm_pct"]
    df["sand"] = df["psd_passing_at_2mm_pct"] - df["psd_passing_at_0_063mm_pct"]
    df["gravel"] = 100 - df["psd_passing_at_2mm_pct"]
    df["fine-grained"] = df["clay"] + df["silt"] > 15
    df["feat_cu"] = (
        df["psd_size_at_d60_mm"].replace(0, np.nan)
        / df["psd_size_at_d10_mm"].replace(0, np.nan)
    )
    df["feat_cc"] = (df["psd_size_at_d30_mm"] ** 2) / (
        df["psd_size_at_d60_mm"].replace(0, np.nan)
        * df["psd_size_at_d10_mm"].replace(0, np.nan)
    )
    df["feat_log_cu"] = np.log1p(df["feat_cu"])
    return df


def setup_logger(path):
    logger = logging.getLogger("model1")
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


def main(args):
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logger = setup_logger(args.log)
    logger.info("model1 -- general_model.py's WeightedBlendRegressor (Ridge + GPR + XGBoost)")
    logger.info("seed=%d  optuna_trials=%d", args.seed, args.optuna_trials)

    calc_satline = load_calc_satline(args.helpers_dir)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    logger.info("train shape=%s  test shape=%s", train.shape, test.shape)

    train = add_no_missing_features(train)
    test = add_no_missing_features(test)

    X_train = train[NO_MISSING_FEATURES]
    X_test = test[NO_MISSING_FEATURES]

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError(
            "NO_MISSING_FEATURES should have zero missing values by "
            "construction -- got NaNs. Check for psd_size_at_d10_mm == 0 "
            "rows (feat_cu/feat_cc division)."
        )

    cache_hit = (
        os.path.exists(args.model_out)
        and not args.force_retrain
    )

    if cache_hit:
        logger.info("loading cached models <- %s", args.model_out)
        cached = joblib.load(args.model_out)

        if cached["feature_columns"] != NO_MISSING_FEATURES:
            raise ValueError(
                f"Cached model at {args.model_out} was trained on a "
                "different feature set than the current NO_MISSING_FEATURES "
                "-- rerun with --force_retrain."
            )

        models = cached["models"]
        logger.info(
            "cached model metadata: optuna_trials=%s seed=%s",
            cached["optuna_trials"], cached["seed"],
        )
    else:
        models = {}

    # Quantile-stratified split (on MDD bins), drawn once and reused for
    # both targets' Optuna tuning and blend-weight fitting.
    # --split_kind kfold (default): StratifiedKFold, full coverage.
    # --split_kind shuffle: scripts/v15.py's literal single-holdout scheme.
    splits = None
    if not cache_hit:
        y_mdd_all = train["proctor_mdd_g_cm3"].to_numpy(dtype=float)
        if args.split_kind == "kfold":
            splits = make_stratified_kfold_splits(
                y_mdd_all, n_splits=args.folds, val_strata=args.val_strata, random_state=args.seed,
            )
            logger.info(
                "validation scheme: StratifiedKFold folds=%d val_strata=%d (stratified on MDD, full coverage)",
                args.folds, args.val_strata,
            )
        elif args.split_kind == "shuffle":
            splits = make_stratified_shuffle_splits(
                y_mdd_all, n_splits=args.folds, test_size=args.val_frac,
                val_strata=args.val_strata, random_state=args.seed,
            )
            logger.info(
                "validation scheme: StratifiedShuffleSplit folds=%d val_frac=%.2f val_strata=%d "
                "(v15.py-style single holdout, stratified on MDD)",
                args.folds, args.val_frac, args.val_strata,
            )
        else:
            raise ValueError(f"Unknown --split_kind={args.split_kind!r}")

    preds = {}
    for target in TARGETS:
        y = train[target].to_numpy(dtype=float)

        if cache_hit:
            model = models[target]
        else:
            if args.optuna_trials > 0:
                logger.info("[%s] tuning XGBoost with Optuna (%d trials)...", target, args.optuna_trials)
                xgb_model, study = tune_xgb_with_optuna(
                    X_train, y, n_trials=args.optuna_trials, random_state=args.seed,
                    splits=splits,
                )
                logger.info("[%s] best CV MAE=%.4f  params=%s", target, study.best_value, study.best_params)
            else:
                xgb_model = None  # WeightedBlendRegressor falls back to its hardcoded default

            model = WeightedBlendRegressor(random_state=args.seed, xgb_model=xgb_model, splits=splits)
            model.fit(X_train, y)
            models[target] = model

        summary = model.get_training_summary()
        logger.info(
            "[%s] blend weights: ridge=%.3f gpr=%.3f xgboost=%.3f",
            target, summary["weights"]["ridge"], summary["weights"]["gpr"], summary["weights"]["xgboost"],
        )
        logger.info(
            "[%s] internal OOF (n_validated=%d/%d): R2=%.4f MAE=%.4f RMSE=%.4f",
            target,
            summary["n_validated"], len(train),
            summary["blend_oof_metrics"]["r2"],
            summary["blend_oof_metrics"]["mae"],
            summary["blend_oof_metrics"]["rmse"],
        )

        preds[target] = model.predict(X_test)

    if not cache_hit:
        os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
        joblib.dump(
            {
                "models": models,
                "feature_columns": NO_MISSING_FEATURES,
                "optuna_trials": args.optuna_trials,
                "seed": args.seed,
            },
            args.model_out,
        )
        logger.info("saved fitted models -> %s", args.model_out)

    rho_s = test["grain_density_g_cm3"].fillna(2.65).to_numpy()
    owc = np.clip(preds["proctor_owc_pct"], 0, None)
    mdd = np.minimum(preds["proctor_mdd_g_cm3"], calc_satline(owc, rho_s) * 0.999)
    n_clipped = int((mdd < preds["proctor_mdd_g_cm3"] - 1e-9).sum())
    logger.info("saturation-line clip applied to %d/%d MDD predictions", n_clipped, len(mdd))

    out = pd.DataFrame({
        "id": test["id"].values,
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
    p.add_argument("--helpers_dir", default=str(repo_root))
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_model1.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "model1_run.log"))
    p.add_argument("--model_out", default=str(repo_root / "models" / "model1.joblib"),
                   help="path to cache/load the fitted per-target models")
    p.add_argument("--force_retrain", action="store_true",
                   help="ignore --model_out if it exists and fit fresh anyway")
    p.add_argument("--optuna_trials", type=int, default=50,
                   help="Optuna trials for XGBoost tuning per target (0 to skip tuning)")
    p.add_argument("--split_kind", choices=["kfold", "shuffle"], default="kfold",
                   help="'kfold': StratifiedKFold, full coverage (default). "
                        "'shuffle': v15.py's single StratifiedShuffleSplit holdout")
    p.add_argument("--folds", type=int, default=5,
                   help="StratifiedKFold fold count (split_kind=kfold) or number of "
                        "StratifiedShuffleSplit draws (split_kind=shuffle, default 1 there "
                        "matches v15.py -- pass --folds 1 explicitly for the literal v15.py scheme)")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="holdout fraction, only used when split_kind=shuffle")
    p.add_argument("--val_strata", type=int, default=5,
                   help="MDD quantile strata used to stratify the split")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

