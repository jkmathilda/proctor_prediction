"""
evaluate_validation_schemes.py
===============================
Answers a question model2i.py's own internal "OOF" metrics can't: which
inner validation scheme (plain KFold, quantile-stratified KFold, or a
v15.py-style single stratified holdout) actually generalizes best -- judged
against a genuinely held-out slice of train.csv that NO scheme's tuning
(Optuna, blend weights, ensemble weights, beta) ever sees.

Every metric model2i.py itself reports is computed from data used, directly
or indirectly, to fit something -- even "out-of-fold" predictions come from
folds whose split assignment was also used to pick hyperparameters (see
tune_xgb_with_optuna's `splits` argument). That's fine for comparing
candidates against EACH OTHER within one scheme, but it can't tell you
whether a scheme that looks good internally (e.g. the v15-style single
holdout's suspiciously high fine-grained R^2 in an earlier run, fit on just
4 rows) actually generalizes, or is just overfit to its own tiny validation
slice. The only way to know is a split that's held out from ALL of that.

Procedure
---------
1. Draw ONE frozen outer stratified 80/20 split of train.csv (--outer_frac,
   stratified on MDD quantile bins via make_stratified_shuffle_splits,
   --outer_seed -- deliberately a different seed than any inner scheme's,
   so there's no accidental correlation). This outer split is NEVER passed
   into any scheme's fitting/tuning below -- MICE imputers, Optuna, blend
   weights, ensemble weights, and beta are all fit using ONLY the outer
   training rows.

2. For each candidate inner scheme (--schemes, default all three):
     "plain_kfold"      -- model2i.py's original scheme (KFold(5), no
                            stratification).
     "stratified_kfold" -- model2i.py's CURRENT scheme (StratifiedKFold(5)
                            on MDD quantile bins, full coverage).
     "single_holdout"   -- scripts/v15.py's literal scheme (one
                            StratifiedShuffleSplit 80/20 holdout, reused
                            everywhere) -- the version that beat plain KFold
                            on the actual Kaggle leaderboard, but showed
                            visible overfitting symptoms internally (see
                            mdd_fine_correction.py's module docstring).
   fit model2i.py's full pipeline (general MDD/OWC models + both Group C
   correctors) using ONLY the outer training rows, with that scheme as the
   inner validation split. Then predict on the frozen outer test rows (which
   have real observed MDD/OWC -- they're carved from train.csv, not
   test.csv) and score against them.

3. Report R^2/MAE/RMSE per scheme per target on this outer test set --
   the closest thing available to an honest, Kaggle-independent read on
   which scheme actually generalizes, without spending a submission.

CAVEAT: --outer_frac shrinks the already-small training data further (an
80/20 outer split leaves ~161 training rows, ~71 of them fine-grained), and
the outer test set itself is only ~40 rows -- still a small sample, just an
HONEST one (never touched during fitting/tuning, unlike every number
model2i.py itself reports). Run this once per scheme comparison, not
repeatedly while tweaking -- checking against the outer set over and over
reintroduces the same overfitting-to-a-small-set problem one level up.

Run:
    python scripts/evaluate_validation_schemes.py
    python scripts/evaluate_validation_schemes.py --schemes stratified_kfold single_holdout
    python scripts/evaluate_validation_schemes.py --optuna_trials 20 --specialist_optuna_trials 20   # faster
"""

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

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
    calculate_regression_metrics,
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

SCHEME_NAMES = ["plain_kfold", "stratified_kfold", "single_holdout"]


def load_calc_satline(helpers_dir):
    path = os.path.join(helpers_dir, "helper_functions.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"helper_functions.py not found at {path}")
    spec = importlib.util.spec_from_file_location("helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.calc_satline


def setup_logger(path):
    logger = logging.getLogger("evaluate_validation_schemes")
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


def general_model_splits(scheme, y_mdd_outer_train, args):
    """(splits, corrector_kwargs) for a scheme, both built from the OUTER
    TRAINING rows' MDD values only -- never the outer test rows."""
    if scheme == "plain_kfold":
        return None, dict(stratified_split=False)

    if scheme == "stratified_kfold":
        splits = make_stratified_kfold_splits(
            y_mdd_outer_train, n_splits=args.folds,
            val_strata=args.val_strata, random_state=args.seed,
        )
        return splits, dict(
            stratified_split=True, split_kind="kfold",
            folds=args.folds, val_strata=args.val_strata,
        )

    if scheme == "single_holdout":
        splits = make_stratified_shuffle_splits(
            y_mdd_outer_train, n_splits=1, test_size=args.val_frac,
            val_strata=args.val_strata, random_state=args.seed,
        )
        return splits, dict(
            stratified_split=True, split_kind="shuffle",
            folds=1, val_frac=args.val_frac, val_strata=args.val_strata,
        )

    raise ValueError(f"Unknown scheme={scheme!r}")


def run_scheme(scheme, train_outer, test_outer, calc_satline, args, logger):
    """Fit model2i.py's full pipeline on train_outer only (using `scheme` as
    the inner validation split), then score on test_outer -- rows the
    scheme's fitting/tuning never saw."""
    logger.info("=" * 70)
    logger.info("scheme=%s", scheme)

    # MICE imputers fit on train_outer ONLY -- test_outer must never
    # influence them, or scores here would be optimistic vs a real holdout.
    train_imputed, fine_imputer, coarse_imputer = add_imputed_features(
        train_outer, random_state=args.seed,
    )
    test_imputed, _, _ = add_imputed_features(
        test_outer, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer,
        random_state=args.seed,
    )
    train_imputed = add_specialist_derived_features(train_imputed)
    test_imputed = add_specialist_derived_features(test_imputed)

    X_train = train_imputed[IMPUTED_FEATURES]
    X_train_specialist = train_imputed[SPECIALIST_RAW_FEATURES]
    fine_mask_train = train_imputed["fine-grained"].to_numpy(dtype=bool)

    X_test = test_imputed[IMPUTED_FEATURES]
    X_test_specialist = test_imputed[SPECIALIST_RAW_FEATURES]
    fine_mask_test = test_imputed["fine-grained"].to_numpy(dtype=bool)

    splits, corrector_kwargs = general_model_splits(
        scheme, train_outer[MDD_TARGET].to_numpy(dtype=float), args,
    )

    models = {}
    preds = {}
    for target in TARGETS:
        y = train_outer[target].to_numpy(dtype=float)

        if args.optuna_trials > 0:
            xgb_model, _ = tune_xgb_with_optuna(
                X_train, y, n_trials=args.optuna_trials, random_state=args.seed,
                splits=splits,
            )
        else:
            xgb_model = None

        model = WeightedBlendRegressor(random_state=args.seed, xgb_model=xgb_model, splits=splits)
        model.fit(X_train, y)
        models[target] = model
        preds[target] = model.predict(X_test)

    correctors = {}
    for target in TARGETS:
        y_target = train_outer[target].to_numpy(dtype=float)
        corrector_cls = CORRECTOR_CLASSES[target]
        corrector = corrector_cls(
            general_model=models[target],
            optuna_trials=args.specialist_optuna_trials,
            random_state=args.seed,
            **corrector_kwargs,
        )
        corrector.fit(X_train, X_train_specialist, y_target, fine_mask_train)
        correctors[target] = corrector
        preds[target] = corrector.predict(X_test, X_test_specialist, fine_mask_test)

        summary = corrector.get_training_summary()
        logger.info(
            "[%s] beta=%.3f n_usable=%d n_validated=%d corrected_oof_r2=%.4f (inner, not the outer score below)",
            target, summary["beta"], summary["n_usable"], summary["n_validated"], summary["corrected_oof_r2"],
        )

    rho_s = test_outer["grain_density_g_cm3"].fillna(2.65).to_numpy()
    owc = np.clip(preds[OWC_TARGET], 0, None)
    mdd = np.minimum(preds[MDD_TARGET], calc_satline(owc, rho_s) * 0.999)

    y_mdd_true = test_outer[MDD_TARGET].to_numpy(dtype=float)
    y_owc_true = test_outer[OWC_TARGET].to_numpy(dtype=float)

    mdd_metrics = calculate_regression_metrics(y_mdd_true, mdd)
    owc_metrics = calculate_regression_metrics(y_owc_true, owc)

    logger.info(
        "[OUTER TEST, n=%d, never seen during fitting/tuning] MDD: R2=%.4f MAE=%.4f RMSE=%.4f",
        len(test_outer), mdd_metrics.r2, mdd_metrics.mae, mdd_metrics.rmse,
    )
    logger.info(
        "[OUTER TEST, n=%d, never seen during fitting/tuning] OWC: R2=%.4f MAE=%.4f RMSE=%.4f",
        len(test_outer), owc_metrics.r2, owc_metrics.mae, owc_metrics.rmse,
    )

    return {
        "scheme": scheme,
        "mdd": mdd_metrics.as_dict(),
        "owc": owc_metrics.as_dict(),
    }


def main(args):
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logger = setup_logger(args.log)
    logger.info("evaluate_validation_schemes -- honest outer-holdout comparison of inner validation schemes")
    logger.info("outer_frac=%.2f outer_seed=%d  inner seed=%d  schemes=%s", args.outer_frac, args.outer_seed, args.seed, args.schemes)

    calc_satline = load_calc_satline(args.helpers_dir)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    train = add_no_missing_features(train)
    logger.info("train shape=%s", train.shape)

    outer_split = make_stratified_shuffle_splits(
        train[MDD_TARGET].to_numpy(dtype=float),
        n_splits=1, test_size=args.outer_frac, val_strata=args.val_strata,
        random_state=args.outer_seed,
    )
    outer_train_idx, outer_test_idx = outer_split[0]
    train_outer = train.iloc[outer_train_idx].reset_index(drop=True)
    test_outer = train.iloc[outer_test_idx].reset_index(drop=True)
    logger.info(
        "outer split: train_outer=%d rows (fine-grained=%d), test_outer=%d rows (fine-grained=%d) -- FROZEN, never used for fitting/tuning below",
        len(train_outer), int(train_outer["fine-grained"].sum()),
        len(test_outer), int(test_outer["fine-grained"].sum()),
    )

    results = []
    for scheme in args.schemes:
        results.append(run_scheme(scheme, train_outer, test_outer, calc_satline, args, logger))

    logger.info("=" * 70)
    logger.info("SUMMARY (outer test set, n=%d, honest generalization estimate)", len(test_outer))
    logger.info("%-20s %10s %10s %10s   %10s %10s %10s", "scheme", "MDD_R2", "MDD_MAE", "MDD_RMSE", "OWC_R2", "OWC_MAE", "OWC_RMSE")
    for r in results:
        logger.info(
            "%-20s %10.4f %10.4f %10.4f   %10.4f %10.4f %10.4f",
            r["scheme"], r["mdd"]["r2"], r["mdd"]["mae"], r["mdd"]["rmse"],
            r["owc"]["r2"], r["owc"]["mae"], r["owc"]["rmse"],
        )
    logger.info("Run log -> %s", args.log)

    return results


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(repo_root / "data"))
    p.add_argument("--helpers_dir", default=str(repo_root))
    p.add_argument("--log", default=str(repo_root / "logs" / "evaluate_validation_schemes.log"))
    p.add_argument("--schemes", nargs="+", default=SCHEME_NAMES, choices=SCHEME_NAMES,
                   help="which inner validation schemes to compare")
    p.add_argument("--outer_frac", type=float, default=0.2,
                   help="fraction of train.csv held out as the frozen outer test set")
    p.add_argument("--outer_seed", type=int, default=7,
                   help="seed for the outer split -- deliberately distinct from --seed")
    p.add_argument("--optuna_trials", type=int, default=50,
                   help="Optuna trials for the general models' XGBoost (0 to skip tuning)")
    p.add_argument("--specialist_optuna_trials", type=int, default=50,
                   help="Optuna trials for each Group C corrector's XGBoost")
    p.add_argument("--folds", type=int, default=5,
                   help="StratifiedKFold fold count for the stratified_kfold scheme")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="holdout fraction for the single_holdout scheme's inner split")
    p.add_argument("--val_strata", type=int, default=5,
                   help="MDD quantile strata used for stratification (outer split and inner schemes)")
    p.add_argument("--seed", type=int, default=42,
                   help="seed used for all inner fitting/tuning (MICE, Optuna, splits)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
