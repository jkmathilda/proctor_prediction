"""
model2i.py
==========
Same base architecture as model1i.py -- general_model_impute.py's
WeightedBlendRegressor (Ridge + GPR + XGBoost) on IMPUTED_FEATURES -- plus a
second stage on top of it, applied to BOTH targets: mdd_fine_correction.py's
MDDFineGrainedResidualCorrector for MDD and owc_fine_correction.py's
OWCFineGrainedResidualCorrector for OWC. Both correct the general model's
residual on fine-grained rows only, using data_analysis.ipynb's "C: Full
specialist" feature group (cells 129-137 for MDD; the best of the three
specialist feature sets it compared there):

    MDD (fine-grained rows only): general R^2 0.807  + Group C correction R^2 0.826  (beta ~0.72)
    OWC (fine-grained rows only): general R^2 0.767  + Group C correction R^2 0.779  (beta ~0.93)

data_analysis.ipynb never fit an analogous specialist correction for OWC --
OWCFineGrainedResidualCorrector is this repo's own extension, validated
against its own OOF numbers rather than a notebook citation (see
owc_fine_correction.py's docstring, including a caveat: OWC's gain is real
but much more fragile than MDD's across different CV seeds). Both correctors
are the same target-agnostic class (MDDFineGrainedResidualCorrector's
fit()/predict() never actually reference MDD internally), which always uses
a weighted ensemble of XGBoost and GPR as its residual model -- an earlier
version picked per-run among XGBoost-alone/GPR-alone/ensemble by OOF MSE, but
that just added flip-flopping without changing the outcome (GPR-alone never
won; the ensemble's own weight-fitting already collapses toward whichever
model is better). Coarse-grained rows are left untouched for both targets --
Atterberg limits/PI/LOI/kf (the specialist features) are only physically
measured for fine-grained soils in the first place.

The Group C specialist features (MICE-completed Atterberg limits/PI/LOI/kf
plus missingness flags) are derived from the SAME fine/coarse MICE imputers
model1i.py already fits for IMPUTED_FEATURES -- model2i.py does not run a
second, separate MICE fit (mdd_fine_correction.add_group_c_features exists
for callers that haven't already imputed these columns; model2i.py has, so
it only adds the one extra derived column, feat_pi_completed).

By default XGBoost's hyperparameters are Optuna-tuned on the full training
set (--optuna_trials, default 50) for the two general models (MDD, OWC);
pass --optuna_trials 0 to skip tuning. Each Group C corrector (MDD's and
OWC's) has its own Optuna budget (--specialist_optuna_trials, default 50,
shared between the two) for tuning the XGBoost half of its ensemble; pass
--specialist_optuna_trials 0 to fall back to fixed XGBoost hyperparameters
there instead.

VALIDATION SCHEME: quantile-stratified 5-fold (--folds, --val_strata;
defaults 5/5), stratified on MDD quantile bins (matching scripts/v15.py's
own choice to always stratify on MDD, col 0, even when also fitting OWC),
drawn ONCE and reused across every stage: both general models' Optuna
tuning, both general models' blend-weight fitting, AND both Group C
correctors -- not four independently-drawn splits. Every final model
(Ridge/GPR/XGBoost per target, each corrector's XGBoost+GPR ensemble) is
still refit on ALL 201 training rows afterward, same as before. See
general_model_impute.make_stratified_kfold_splits and
MDDFineGrainedResidualCorrector's `stratified_split` parameter.

This replaced an earlier version that matched v15.py's literal scheme -- a
single 80/20 StratifiedShuffleSplit holdout, reused everywhere. That beat
plain (unstratified) KFold on the actual Kaggle leaderboard, confirming
quantile stratification is a real win on this small, skewed dataset -- but
the single small holdout starved the Group C correction stage: only ~18 of
the ~89 fine-grained rows fell in the general model's ~40-row validated
slice, and the correctors' own 20% split of THAT left just ~4 rows to fit
beta on, which landed near its upper bound (1.39 of 1.5) and needed the
saturation-line clip on 3/87 test predictions -- the old plain-KFold scheme
never needed that. StratifiedKFold keeps the stratification (every fold's
validation set still spans the full MDD range) while restoring full
coverage (every row, including every fine-grained row, gets validated
across the 5 folds) -- same statistical footing the original KFold scheme
had, without giving up what made stratification help on Kaggle.

The physics guardrail from v10-v16/model1.py/model1i.py is kept: predicted
OWC (after its own Group C correction) is clipped to >= 0, and predicted MDD
(after its Group C correction) is clipped to the Zero-Air-Voids (saturation)
line.

The fitted per-target general models, both Group C correctors, AND the
fitted fine/coarse MICE imputers are cached to --model_out via joblib --
reused on a later run instead of retuning/refitting/re-imputing, unless
--force_retrain is passed.

Run:
    python scripts/model2i.py
    python scripts/model2i.py --optuna_trials 0 --specialist_optuna_trials 0   # skip tuning, fast
    python scripts/model2i.py --force_retrain     # ignore cached model
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

from src.general_model_impute import (
    WeightedBlendRegressor,
    tune_xgb_with_optuna,
    add_imputed_features,
    add_no_missing_features,
    make_stratified_kfold_splits,
    make_stratified_shuffle_splits,
    IMPUTED_FEATURES,
)
from src.mdd_fine_correction import (
    MDDFineGrainedResidualCorrector,
    add_specialist_derived_features,
    SPECIALIST_RAW_FEATURES,
)
from src.owc_fine_correction import OWCFineGrainedResidualCorrector

TARGETS = ["proctor_mdd_g_cm3", "proctor_owc_pct"]
MDD_TARGET = "proctor_mdd_g_cm3"
OWC_TARGET = "proctor_owc_pct"

CORRECTOR_CLASSES = {
    MDD_TARGET: MDDFineGrainedResidualCorrector,
    OWC_TARGET: OWCFineGrainedResidualCorrector,
}


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


def setup_logger(path):
    logger = logging.getLogger("model2i")
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
    logger.info("model2i -- general_model_impute.py's WeightedBlendRegressor (Ridge + GPR + XGBoost) "
                "+ Group C fine-grained specialist correction on both MDD and OWC")
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
        correctors = cached["correctors"]
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
        correctors = {}
        logger.info("fitting MICE imputers on train.csv (fine/coarse-grained split)...")
        train_imputed, fine_imputer, coarse_imputer = add_imputed_features(
            train, random_state=args.seed,
        )
        test_imputed, _, _ = add_imputed_features(
            test, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer,
            random_state=args.seed,
        )

        # Quantile-stratified split (on MDD bins, matching v15.py's own
        # choice even for OWC), drawn once and reused for every stage below.
        # --split_kind kfold (default): StratifiedKFold, full coverage.
        # --split_kind shuffle: scripts/v15.py's literal single-holdout scheme.
        y_mdd_all = train[MDD_TARGET].to_numpy(dtype=float)
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

    test_imputed = add_specialist_derived_features(test_imputed)
    X_test = test_imputed[IMPUTED_FEATURES]
    X_test_specialist = test_imputed[SPECIALIST_RAW_FEATURES]
    fine_mask_test = test_imputed["fine-grained"].to_numpy(dtype=bool)

    if X_test.isna().any().any():
        raise ValueError(
            "IMPUTED_FEATURES contains NaNs in the test set after "
            "imputation -- check for a missingness pattern the imputer "
            "wasn't fit to handle."
        )

    if not cache_hit:
        train_imputed = add_specialist_derived_features(train_imputed)
        X_train = train_imputed[IMPUTED_FEATURES]
        X_train_specialist = train_imputed[SPECIALIST_RAW_FEATURES]
        fine_mask_train = train_imputed["fine-grained"].to_numpy(dtype=bool)

        if X_train.isna().any().any():
            raise ValueError(
                "IMPUTED_FEATURES contains NaNs in the training set after "
                "imputation -- this should not happen."
            )

    preds = {}
    for target in TARGETS:
        if cache_hit:
            model = models[target]
        else:
            y = train[target].to_numpy(dtype=float)

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

    # ---------------------------------------------------------------
    # Group C fine-grained correction, applied to BOTH targets
    # ---------------------------------------------------------------
    for target in TARGETS:
        if target not in correctors:
            y_target = train[target].to_numpy(dtype=float)
            corrector_cls = CORRECTOR_CLASSES[target]
            corrector = corrector_cls(
                general_model=models[target],
                optuna_trials=args.specialist_optuna_trials,
                random_state=args.seed,
                stratified_split=True,
                split_kind=args.split_kind,
                val_strata=args.val_strata,
                val_frac=args.val_frac,
                folds=args.folds,
            )
            corrector.fit(X_train, X_train_specialist, y_target, fine_mask_train)
            correctors[target] = corrector

        corrector = correctors[target]
        corrector_summary = corrector.get_training_summary()
        logger.info(
            "[%s] Group C correction (n_fine_grained=%d, n_usable=%d, n_validated=%d): "
            "residual_model=%s candidates=%s",
            target, corrector_summary["n_fine_grained"], corrector_summary["n_usable"],
            corrector_summary["n_validated"], corrector_summary["residual_model_type"],
            corrector_summary["candidate_oof_metrics"],
        )
        logger.info(
            "[%s] Group C correction: beta=%.3f "
            "residual_oof_r2=%.4f general_oof_r2=%.4f corrected_oof_r2=%.4f "
            "general_oof_rmse=%.4f corrected_oof_rmse=%.4f",
            target, corrector_summary["beta"],
            corrector_summary["residual_oof_r2"], corrector_summary["general_oof_r2"],
            corrector_summary["corrected_oof_r2"], corrector_summary["general_oof_rmse"],
            corrector_summary["corrected_oof_rmse"],
        )

        preds[target] = corrector.predict(X_test, X_test_specialist, fine_mask_test)

    if not cache_hit:
        os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
        joblib.dump(
            {
                "models": models,
                "correctors": correctors,
                "fine_imputer": fine_imputer,
                "coarse_imputer": coarse_imputer,
                "feature_columns": IMPUTED_FEATURES,
                "optuna_trials": args.optuna_trials,
                "seed": args.seed,
            },
            args.model_out,
        )
        logger.info("saved fitted models + correctors + imputers -> %s", args.model_out)

    rho_s = test["grain_density_g_cm3"].fillna(2.65).to_numpy()
    owc = np.clip(preds[OWC_TARGET], 0, None)
    mdd = np.minimum(preds[MDD_TARGET], calc_satline(owc, rho_s) * 0.999)
    n_clipped = int((mdd < preds[MDD_TARGET] - 1e-9).sum())
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
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_model2i.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "model2i_run.log"))
    p.add_argument("--model_out", default=str(repo_root / "models" / "model2i.joblib"),
                   help="path to cache/load the fitted per-target models + both Group C correctors + MICE imputers")
    p.add_argument("--force_retrain", action="store_true",
                   help="ignore --model_out if it exists and fit fresh anyway")
    p.add_argument("--optuna_trials", type=int, default=50,
                   help="Optuna trials for XGBoost tuning per general-model target (0 to skip tuning)")
    p.add_argument("--specialist_optuna_trials", type=int, default=50,
                   help="Optuna trials for tuning the XGBoost half of each Group C corrector's "
                        "ensemble (0 to fall back to fixed hyperparameters there instead)")
    p.add_argument("--split_kind", choices=["kfold", "shuffle"], default="kfold",
                   help="'kfold': StratifiedKFold, full coverage (default). "
                        "'shuffle': v15.py's single StratifiedShuffleSplit holdout")
    p.add_argument("--folds", type=int, default=5,
                   help="StratifiedKFold fold count (split_kind=kfold) or number of "
                        "StratifiedShuffleSplit draws (split_kind=shuffle -- pass --folds 1 "
                        "explicitly for the literal v15.py scheme), reused across every "
                        "pipeline stage")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="holdout fraction, only used when split_kind=shuffle")
    p.add_argument("--val_strata", type=int, default=5,
                   help="MDD quantile strata used to stratify the split")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
