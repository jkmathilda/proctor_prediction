"""
model2ic.py
===========
Same as model2i.py in every respect except the Group C correction stage:
model2i.py's MDDFineGrainedResidualCorrector/OWCFineGrainedResidualCorrector
are replaced by
mdd_fine_correction_confidence.ConfidenceWeightedFineGrainedResidualCorrector
(one target-agnostic class used for both MDD and OWC, same as model2i.py's
two correctors already were the same underlying class).

THE DIFFERENCE: model2i.py's beta is a single global number applied
uniformly to every fine-grained row's correction, regardless of how
confident the residual model actually is at that specific row. This class
adds a per-row confidence factor -- derived from the GPR half's own
predictive standard deviation -- that shrinks the correction wherever the
residual model is unsure, so the correction stage can't dominate a
prediction it has little real basis for:

    corrected_prediction = general_prediction + beta * confidence * residual_prediction

confidence = 1/gpr_std, normalized to (0, 1] against the most-confident
training row, so it's a shrinkage-only factor -- it can only pull toward the
general model's raw prediction, never push the correction stronger than
plain beta alone would. See src/mdd_fine_correction_confidence.py's module
docstring for the full mechanism.

Everything else is unchanged from model2i.py: same general model
(general_model_impute.WeightedBlendRegressor on IMPUTED_FEATURES), same
fine-grained-only hard split (coarse rows untouched), same SPECIALIST_FEATURES_C
feature set, same quantile-stratified validation scheme, same --exclude_ids
asymmetry (drops rows from the GENERAL model's fit only -- the correction
stage still trains on them, via
src.held_out_general_model.GeneralModelWithHeldOutRows), same saturation-line
physics guardrail, same joblib caching.

Run:
    python scripts/model2ic.py
    python scripts/model2ic.py --optuna_trials 0 --specialist_optuna_trials 0   # skip tuning, fast
    python scripts/model2ic.py --force_retrain     # ignore cached model
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
    add_specialist_derived_features,
    SPECIALIST_RAW_FEATURES,
)
from src.mdd_fine_correction_confidence import ConfidenceWeightedFineGrainedResidualCorrector
from src.held_out_general_model import GeneralModelWithHeldOutRows

TARGETS = ["proctor_mdd_g_cm3", "proctor_owc_pct"]
MDD_TARGET = "proctor_mdd_g_cm3"
OWC_TARGET = "proctor_owc_pct"


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
    logger = logging.getLogger("model2ic")
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
    logger.info("model2ic -- general_model_impute.py's WeightedBlendRegressor (Ridge + GPR + XGBoost) "
                "+ confidence-weighted Group C fine-grained specialist correction on both MDD and OWC")
    logger.info("seed=%d  optuna_trials=%d  beta_max=%.2f", args.seed, args.optuna_trials, args.beta_max)

    calc_satline = load_calc_satline(args.helpers_dir)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    logger.info("train shape=%s  test shape=%s", train.shape, test.shape)

    train = add_no_missing_features(train)
    test = add_no_missing_features(test)

    general_mask = np.ones(len(train), dtype=bool)
    if args.exclude_ids:
        general_mask = ~train["id"].isin(args.exclude_ids).to_numpy()
        logger.info(
            "excluding ids=%s from the GENERAL model's training only (%d -> %d rows); "
            "the Group C correction stage still trains on all %d rows",
            args.exclude_ids, len(train), int(general_mask.sum()), len(train),
        )
    n_general = int(general_mask.sum())

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

        y_mdd_general = train.loc[general_mask, MDD_TARGET].to_numpy(dtype=float)
        if args.split_kind == "kfold":
            splits = make_stratified_kfold_splits(
                y_mdd_general, n_splits=args.folds, val_strata=args.val_strata, random_state=args.seed,
            )
            logger.info(
                "validation scheme: StratifiedKFold folds=%d val_strata=%d (stratified on MDD, full coverage)",
                args.folds, args.val_strata,
            )
        elif args.split_kind == "shuffle":
            splits = make_stratified_shuffle_splits(
                y_mdd_general, n_splits=args.folds, test_size=args.val_frac,
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

        X_train_full = train_imputed[IMPUTED_FEATURES]
        X_train_specialist_full = train_imputed[SPECIALIST_RAW_FEATURES]
        fine_mask_train_full = train_imputed["fine-grained"].to_numpy(dtype=bool)

        X_train_general = X_train_full.loc[general_mask]

        if X_train_full.isna().any().any():
            raise ValueError(
                "IMPUTED_FEATURES contains NaNs in the training set after "
                "imputation -- this should not happen."
            )

    preds = {}
    for target in TARGETS:
        if cache_hit:
            model = models[target]
        else:
            y_general = train.loc[general_mask, target].to_numpy(dtype=float)

            if args.optuna_trials > 0:
                logger.info("[%s] tuning XGBoost with Optuna (%d trials)...", target, args.optuna_trials)
                xgb_model, study = tune_xgb_with_optuna(
                    X_train_general, y_general, n_trials=args.optuna_trials, random_state=args.seed,
                    splits=splits,
                )
                logger.info("[%s] best CV MAE=%.4f  params=%s", target, study.best_value, study.best_params)
            else:
                xgb_model = None

            model = WeightedBlendRegressor(random_state=args.seed, xgb_model=xgb_model, splits=splits)
            model.fit(X_train_general, y_general)
            models[target] = model

        summary = model.get_training_summary()
        logger.info(
            "[%s] blend weights: ridge=%.3f gpr=%.3f xgboost=%.3f",
            target, summary["weights"]["ridge"], summary["weights"]["gpr"], summary["weights"]["xgboost"],
        )
        logger.info(
            "[%s] internal OOF (n_validated=%d/%d): R2=%.4f MAE=%.4f RMSE=%.4f",
            target,
            summary["n_validated"], n_general,
            summary["blend_oof_metrics"]["r2"],
            summary["blend_oof_metrics"]["mae"],
            summary["blend_oof_metrics"]["rmse"],
        )

        preds[target] = model.predict(X_test)

    # ---------------------------------------------------------------
    # Confidence-weighted Group C correction, applied to BOTH targets.
    # ---------------------------------------------------------------
    for target in TARGETS:
        if target not in correctors:
            y_target_full = train[target].to_numpy(dtype=float)

            if args.exclude_ids:
                excluded_index = X_train_full.index[~general_mask]
                general_model_for_correction = GeneralModelWithHeldOutRows(
                    general_model=models[target],
                    extra_X=X_train_full.loc[excluded_index],
                    extra_index=excluded_index,
                )
            else:
                general_model_for_correction = models[target]

            corrector = ConfidenceWeightedFineGrainedResidualCorrector(
                general_model=general_model_for_correction,
                optuna_trials=args.specialist_optuna_trials,
                random_state=args.seed,
                beta_bounds=(0.0, args.beta_max),
                split_kind=args.split_kind,
                val_strata=args.val_strata,
                val_frac=args.val_frac,
                folds=args.folds,
                confidence_eps=args.confidence_eps,
            )
            corrector.fit(X_train_full, X_train_specialist_full, y_target_full, fine_mask_train_full)
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
            "[%s] Group C correction: beta=%.3f mean_confidence=%.3f min_confidence=%.3f "
            "residual_oof_r2=%.4f general_oof_r2=%.4f corrected_oof_r2=%.4f "
            "general_oof_rmse=%.4f corrected_oof_rmse=%.4f",
            target, corrector_summary["beta"],
            corrector_summary["mean_confidence"], corrector_summary["min_confidence"],
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
    p.add_argument("--helpers_dir", default=str(repo_root / "src"))
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_model2ic.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "model2ic_run.log"))
    p.add_argument("--model_out", default=str(repo_root / "models" / "model2ic.joblib"),
                   help="path to cache/load the fitted per-target models + both confidence-weighted "
                        "Group C correctors + MICE imputers")
    p.add_argument("--force_retrain", action="store_true",
                   help="ignore --model_out if it exists and fit fresh anyway")
    p.add_argument("--exclude_ids", type=int, nargs="*", default=[155],
                   help="train.csv id values to drop from the GENERAL model's training only -- "
                        "the Group C correction stage still trains on these rows. Defaults to "
                        "[155] -- see model1i.py's --exclude_ids help for why. Pass --exclude_ids "
                        "(with no values) to include every row again.")
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
                        "StratifiedShuffleSplit draws (split_kind=shuffle), reused across every "
                        "pipeline stage")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="holdout fraction, only used when split_kind=shuffle")
    p.add_argument("--val_strata", type=int, default=5,
                   help="MDD quantile strata used to stratify the split")
    p.add_argument("--confidence_eps", type=float, default=1e-3,
                   help="added to each fine-grained row's GPR predictive std before inverting "
                        "to a confidence weight, to avoid dividing by ~0 wherever the GPR is "
                        "extremely certain")
    p.add_argument("--beta_max", type=float, default=1.5,
                   help="upper bound for beta -- same 1.5 cap model2i.py uses. Confidence is "
                        "normalized to top out at exactly 1.0 for the single most-confident "
                        "training row, so beta=1.5 there gives the IDENTICAL max per-row "
                        "correction strength model2i.py's uncapped-by-confidence scheme could "
                        "already reach; every less-confident row gets strictly less. Raising "
                        "this would let beta compensate for confidence shrinkage and could let "
                        "the correction exceed model2i.py's strength on weak-signal rows --"
                        "defeating the point of adding confidence weighting in the first place.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
