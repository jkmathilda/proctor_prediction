"""
model1h.py
==========
Identical to model1i.py in every respect (same WeightedBlendRegressor architecture, same
IMPUTED_FEATURES, same MICE imputation, same stratified CV, same saturation clip, same
CLI/caching conventions) with exactly one change: the XGBoost component of the Ridge +
GPR + XGBoost blend is replaced with a HistGradientBoostingRegressor, Optuna-tuned the
same way (tune_histgbt_with_optuna, same 50-trial default, same objective).

Why: a direct, equal-tuning-effort comparison (50 Optuna trials each, same
IMPUTED_FEATURES, same StratifiedKFold-on-MDD-quantile splits) found HistGBT beats
XGBoost on this dataset -- MDD is a near-tie (R^2 0.8522 vs 0.8533), OWC is a real win for
HistGBT (R^2 0.7983 vs 0.7888), and combined NMAE favors HistGBT (0.2495 vs 0.2549). This
also lines up with docs/260804.md's existing finding that v15 (HistGBT + GP blend) is the
best Kaggle submission logged so far (0.2383), beating model1i's XGBoost-based blend
(0.2610) -- this script isolates just the boosting-library variable to test that
connection directly, on the general-model family specifically.

Note on naming: WeightedBlendRegressor's third blend component is still called
`xgb_model`/reported as `"xgboost"` in get_training_summary() -- that parameter/key name
is shared code used by model1i.py/model2i.py too, not renamed here to avoid touching a
class other scripts depend on. In this script's own logging, that slot holds a
HistGradientBoostingRegressor, not an XGBRegressor -- read "xgboost" in the summary output
as "the third blend component" for this run specifically.

By default HistGBT's hyperparameters are Optuna-tuned on the full training set
(--optuna_trials, default 50); pass --optuna_trials 0 to skip tuning.

The physics guardrail from v10-v16/model1.py/model1i.py is kept: predicted OWC is
clipped to >= 0, and predicted MDD is clipped to the Zero-Air-Voids (saturation) line.

The fitted per-target models AND the fitted fine/coarse MICE imputers are cached to
--model_out via joblib -- reused on a later run instead of retuning/refitting/re-imputing,
unless --force_retrain is passed.

Run:
    python scripts/model1h.py
    python scripts/model1h.py --optuna_trials 0   # skip tuning, fast
    python scripts/model1h.py --force_retrain     # ignore cached model
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

from sklearn.ensemble import HistGradientBoostingRegressor

from src.general_model_impute import (
    WeightedBlendRegressor,
    tune_histgbt_with_optuna,
    add_imputed_features,
    add_no_missing_features,
    make_stratified_kfold_splits,
    make_stratified_shuffle_splits,
    IMPUTED_FEATURES,
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
    load_helpers pattern used in v10/v14/v15/v16.py/model1i.py."""
    path = os.path.join(helpers_dir, "helper_functions.py")
    if not os.path.exists(path):
        # helper_functions.py lives in src/ after the repo reorg -- fall back there if
        # the caller passed the repo root (model1i.py's default) rather than src/.
        alt_path = os.path.join(helpers_dir, "src", "helper_functions.py")
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"helper_functions.py not found at {path} or {alt_path}")
    spec = importlib.util.spec_from_file_location("helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.calc_satline


def setup_logger(path):
    logger = logging.getLogger("model1h")
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
    logger.info("model1h -- general_model_impute.py's WeightedBlendRegressor (Ridge + GPR + HistGBT)")
    logger.info("(the blend's 3rd component / 'xgboost' summary key is HistGradientBoostingRegressor here, not XGBoost -- see module docstring)")
    logger.info("seed=%d  optuna_trials=%d", args.seed, args.optuna_trials)

    calc_satline = load_calc_satline(args.helpers_dir)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    logger.info("train shape=%s  test shape=%s", train.shape, test.shape)

    train = add_no_missing_features(train)
    test = add_no_missing_features(test)

    cache_hit = (
        os.path.exists(args.model_out)
        and not args.force_retrain
    )

    if cache_hit:
        logger.info("loading cached models <- %s", args.model_out)
        cached = joblib.load(args.model_out)

        if cached["feature_columns"] != IMPUTED_FEATURES:
            raise ValueError(
                f"Cached model at {args.model_out} was trained on a "
                "different feature set than the current IMPUTED_FEATURES "
                "-- rerun with --force_retrain."
            )

        models = cached["models"]
        fine_imputer = cached["fine_imputer"]
        coarse_imputer = cached["coarse_imputer"]
        logger.info(
            "cached model metadata: optuna_trials=%s seed=%s",
            cached["optuna_trials"], cached["seed"],
        )

        # Models are already fit -- only test needs to be imputed, reusing
        # the imputers fit on training data (never refit on test).
        test_imputed, _, _ = add_imputed_features(
            test, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer,
            random_state=args.seed,
        )
        train_imputed = None
    else:
        models = {}
        logger.info("fitting MICE imputers on train.csv (fine/coarse-grained split)...")
        train_imputed, fine_imputer, coarse_imputer = add_imputed_features(
            train, random_state=args.seed,
        )
        test_imputed, _, _ = add_imputed_features(
            test, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer,
            random_state=args.seed,
        )

    X_test = test_imputed[IMPUTED_FEATURES]
    if X_test.isna().any().any():
        raise ValueError(
            "IMPUTED_FEATURES contains NaNs in the test set after "
            "imputation -- check for a missingness pattern the imputer "
            "wasn't fit to handle."
        )

    if not cache_hit:
        X_train = train_imputed[IMPUTED_FEATURES]
        if X_train.isna().any().any():
            raise ValueError(
                "IMPUTED_FEATURES contains NaNs in the training set after "
                "imputation -- this should not happen."
            )

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
        if cache_hit:
            model = models[target]
        else:
            y = train[target].to_numpy(dtype=float)

            if args.optuna_trials > 0:
                logger.info("[%s] tuning HistGBT with Optuna (%d trials)...", target, args.optuna_trials)
                histgbt_model, study = tune_histgbt_with_optuna(
                    X_train, y, n_trials=args.optuna_trials, random_state=args.seed,
                    splits=splits,
                )
                logger.info("[%s] best CV MAE=%.4f  params=%s", target, study.best_value, study.best_params)
            else:
                # NOTE: WeightedBlendRegressor's xgb_model=None fallback is XGBoost
                # (make_default_target_xgb_model), not HistGBT -- pass an explicit
                # default-hyperparameter HistGBT instance here instead, so skipping
                # tuning still uses HistGBT rather than silently reverting to XGBoost.
                histgbt_model = HistGradientBoostingRegressor(random_state=args.seed)

            model = WeightedBlendRegressor(random_state=args.seed, xgb_model=histgbt_model, splits=splits)
            model.fit(X_train, y)
            models[target] = model

        summary = model.get_training_summary()
        logger.info(
            "[%s] blend weights: ridge=%.3f gpr=%.3f histgbt=%.3f",
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
                "fine_imputer": fine_imputer,
                "coarse_imputer": coarse_imputer,
                "feature_columns": IMPUTED_FEATURES,
                "optuna_trials": args.optuna_trials,
                "seed": args.seed,
            },
            args.model_out,
        )
        logger.info("saved fitted models + imputers -> %s", args.model_out)

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
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_model1h.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "model1h_run.log"))
    p.add_argument("--model_out", default=str(repo_root / "models" / "model1h.joblib"),
                   help="path to cache/load the fitted per-target models + MICE imputers")
    p.add_argument("--force_retrain", action="store_true",
                   help="ignore --model_out if it exists and fit fresh anyway")
    p.add_argument("--optuna_trials", type=int, default=50,
                   help="Optuna trials for HistGBT tuning per target (0 to skip tuning)")
    p.add_argument("--split_kind", choices=["kfold", "shuffle"], default="kfold",
                   help="'kfold': StratifiedKFold, full coverage (default). "
                        "'shuffle': v15.py's single StratifiedShuffleSplit holdout")
    p.add_argument("--folds", type=int, default=5,
                   help="StratifiedKFold fold count (split_kind=kfold) or number of "
                        "StratifiedShuffleSplit draws (split_kind=shuffle -- pass --folds 1 "
                        "explicitly for the literal v15.py scheme)")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="holdout fraction, only used when split_kind=shuffle")
    p.add_argument("--val_strata", type=int, default=5,
                   help="MDD quantile strata used to stratify the split")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
