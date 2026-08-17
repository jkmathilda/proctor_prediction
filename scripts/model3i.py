"""
model3i.py
==========
Same architecture as model2i.py -- general_model_impute.py's
WeightedBlendRegressor (Ridge + GPR + XGBoost) on IMPUTED_FEATURES, plus
mdd_fine_correction.py's Group C fine-grained residual correction on BOTH
targets (MDDFineGrainedResidualCorrector / OWCFineGrainedResidualCorrector)
-- with exactly one change: the correction's blend strength.

model2i.py blends the specialist correction back into the general model's
prediction with a SINGLE GLOBAL weight, fit once by bounded scalar
optimization over every fine-grained row:

    corrected_i = general_i + beta * residual_i,   beta constant for all i

model3i.py instead uses a PER-SAMPLE weight, beta_i, driven by the
specialist's own predictive confidence at that specific row (its GPR
residual model's out-of-fold predictive std -- how sure the specialist is
at this exact point in feature space, not just on average):

    confidence_i = clip(1 - (gpr_std_i - std_lo) / (std_hi - std_lo), 0, 1)
    beta_i       = beta_lo_conf + (beta_hi_conf - beta_lo_conf) * confidence_i
    corrected_i  = general_i + beta_i * residual_i

std_lo/std_hi are the 10th/90th percentile of the GPR's own out-of-fold
predictive std (fold-honest -- a fresh GPR clone per fold, std only on that
fold's held-out rows), so a row's confidence is judged against how
uncertain this specialist typically is on genuinely unseen fine-grained
rows. beta_lo_conf/beta_hi_conf are fit jointly (bounded 2-parameter
optimization, same MSE objective and same (0, 1.5) bounds as model2i's
scalar beta) initialized at model2i's own scalar-optimal beta -- so this
per-sample search can never do worse, on the OOF objective it's fit to,
than plain model2i; if per-sample confidence doesn't actually help on a
given target, the optimizer is free to collapse beta_lo_conf ~= beta_hi_conf,
recovering model2i's behavior exactly. Low-confidence rows (specialist
unsure here) lean toward beta_lo_conf -- more general-model, less specialist;
high-confidence rows lean toward beta_hi_conf.

Motivation: model2i.py's Kaggle result (0.2667) was WORSE than plain
model1i (0.2465) despite the correction improving its own fine-grained-
subset OOF metric -- a real negative result, not just noise (see
docs/260804.md Section 1/model2i writeup). One plausible failure mode: a
single global beta forces the SAME correction strength onto every
fine-grained row, including ones the specialist genuinely can't say much
about -- diluting the general model's already-decent prediction with a
low-confidence guess. Scaling beta down exactly where the specialist is
uncertain, and up where it's confident, is a natural fix to test -- but
this is a genuinely new, unvalidated mechanism (no notebook precedent, no
prior Kaggle submission), not a known improvement. Compare its own OOF
numbers against model2i's and model1i's before trusting it; it may not
beat model1i either -- see this project's running heuristic
(docs/260804.md) that small internal deltas rarely survive a real Kaggle
submission at this dataset's size.

This script scraps the earlier model3ic.py (4-soil-type proportional
specialist blend, GP-confidence-weighted only BETWEEN its four specialists,
still a single global beta against the general model) -- an unrelated,
more complex mechanism that never got a Kaggle submission. model3i.py
replaces it with this much narrower, single-lever change on top of
model2i's already-submitted architecture, to isolate exactly one thing:
global beta vs. confidence-weighted per-sample beta.

Everything else -- the general model, the fine/coarse hard-gate 89-row
correction, Optuna budgets, --exclude_ids asymmetry, validation scheme
(quantile-stratified 5-fold reused across every stage), the saturation-line
MDD clip -- is identical to model2i.py. See its module docstring for the
full detail on those; this docstring only covers the delta.

The fitted per-target general models, both Group C correctors (now with
per-sample beta_lo_conf_/beta_hi_conf_/std_lo_/std_hi_ instead of a single
beta_), AND the fitted fine/coarse MICE imputers are cached to --model_out
via joblib, same as model2i.py.

Run:
    python scripts/model3i.py
    python scripts/model3i.py --optuna_trials 0 --specialist_optuna_trials 0   # skip tuning, fast
    python scripts/model3i.py --force_retrain     # ignore cached model
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
from src.held_out_general_model import GeneralModelWithHeldOutRows

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
    logger = logging.getLogger("model3i")
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
    logger.info("model3i -- model2i (WeightedBlendRegressor + Group C fine-grained "
                "correction on both MDD and OWC), with a per-sample confidence-weighted "
                "beta_i (GPR predictive std) instead of model2i's single global beta")
    logger.info("seed=%d  optuna_trials=%d", args.seed, args.optuna_trials)

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
    # Row index of the general model's own training rows -- needed below for
    # the full-set OOF reconstruction regardless of cache_hit (unlike
    # X_train_general, which is only built in the non-cached branch).
    general_index = train.index[general_mask]

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
        # choice even for OWC), drawn once over the GENERAL model's own
        # training rows (excluding --exclude_ids) and reused for every
        # stage below, including the Group C correctors.
        # --split_kind kfold (default): StratifiedKFold, full coverage.
        # --split_kind shuffle: scripts/v15.py's literal single-holdout scheme.
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

        # Full-data views (every row, including anything --exclude_ids
        # dropped from the general model) -- what the Group C correction
        # stage trains on.
        X_train_full = train_imputed[IMPUTED_FEATURES]
        X_train_specialist_full = train_imputed[SPECIALIST_RAW_FEATURES]
        fine_mask_train_full = train_imputed["fine-grained"].to_numpy(dtype=bool)

        # General-model-only view -- excludes --exclude_ids.
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
                xgb_model = None  # WeightedBlendRegressor falls back to its hardcoded default

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
    # Group C fine-grained correction, applied to BOTH targets, with
    # per_sample_beta=True (the one change from model2i.py). Trains on ALL
    # rows (X_train_full etc.) even though the general model above was only
    # fit on general_mask's rows -- GeneralModelWithHeldOutRows lets the
    # excluded rows still contribute a (non-leaked) OOF-style prediction for
    # this stage. With no --exclude_ids, general_mask is all-True and this
    # is exactly model2i's own single-model behavior, just with per-sample beta.
    # ---------------------------------------------------------------
    full_set_nmae = {}
    for target in TARGETS:
        if target not in correctors:
            y_target_full = train[target].to_numpy(dtype=float)
            corrector_cls = CORRECTOR_CLASSES[target]

            if args.exclude_ids:
                excluded_index = X_train_full.index[~general_mask]
                general_model_for_correction = GeneralModelWithHeldOutRows(
                    general_model=models[target],
                    extra_X=X_train_full.loc[excluded_index],
                    extra_index=excluded_index,
                )
            else:
                general_model_for_correction = models[target]

            corrector = corrector_cls(
                general_model=general_model_for_correction,
                optuna_trials=args.specialist_optuna_trials,
                random_state=args.seed,
                stratified_split=True,
                split_kind=args.split_kind,
                val_strata=args.val_strata,
                val_frac=args.val_frac,
                folds=args.folds,
                per_sample_beta=True,
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
            "[%s] Group C correction: mean_beta_i=%.3f (scalar model2i-equivalent baseline=%.3f, "
            "beta_lo_conf=%.3f beta_hi_conf=%.3f, gpr_std [10th,90th]=[%.4f,%.4f]) "
            "residual_oof_r2=%.4f general_oof_r2=%.4f corrected_oof_r2=%.4f "
            "general_oof_rmse=%.4f corrected_oof_rmse=%.4f",
            target, corrector_summary["beta"], corrector_summary["beta_scalar_baseline"],
            corrector_summary["beta_lo_conf"], corrector_summary["beta_hi_conf"],
            corrector_summary["gpr_std_lo"], corrector_summary["gpr_std_hi"],
            corrector_summary["residual_oof_r2"], corrector_summary["general_oof_r2"],
            corrector_summary["corrected_oof_r2"], corrector_summary["general_oof_rmse"],
            corrector_summary["corrected_oof_rmse"],
        )

        preds[target] = corrector.predict(X_test, X_test_specialist, fine_mask_test)

        # -----------------------------------------------------------
        # Full-set OOF NMAE: the corrector summary above only scores
        # fine-grained rows. Reconstruct the actual combined OOF
        # prediction for every one of the general model's own n_general
        # rows (general model's OOF for coarse-grained rows, Group C's
        # corrected OOF for fine-grained rows) so this is comparable,
        # apples-to-apples, to model1i.py's/model2i.py's/model4i.py's
        # full-set NMAE logged over the same n_general rows (general_mask,
        # id 155 excluded by default).
        # -----------------------------------------------------------
        y_general_target = train.loc[general_mask, target].to_numpy(dtype=float)
        general_oof_df = models[target].get_oof_results(y=y_general_target, index=general_index)

        combined_oof = general_oof_df["blend_oof_prediction"].copy()
        fine_index = corrector.oof_index_.intersection(general_index)
        corrected_series = pd.Series(
            corrector.corrected_oof_prediction_fine_, index=corrector.oof_index_,
        )
        combined_oof.loc[fine_index] = corrected_series.loc[fine_index]
        observed_series = general_oof_df["observed"]

        # Under split_kind=shuffle, coverage is partial -- StratifiedShuffleSplit
        # draws may not cover every row, so get_oof_results() leaves NaN for
        # rows no split validated (see its docstring). Score only rows with a
        # genuine OOF value; split_kind=kfold gives full coverage, so this is
        # a no-op there.
        scoreable = combined_oof.notna()
        combined_oof = combined_oof[scoreable]
        observed_series = observed_series[scoreable]

        full_r2 = r2_score(observed_series, combined_oof)
        full_mae = mean_absolute_error(observed_series, combined_oof)
        full_rmse = mean_squared_error(observed_series, combined_oof) ** 0.5
        full_nmae = nmae(observed_series.to_numpy(), combined_oof.to_numpy())
        full_set_nmae[target] = full_nmae

        logger.info(
            "[%s] FULL-SET OOF incl. Group C correction (n=%d/%d, %d fine-grained rows "
            "corrected): R2=%.4f MAE=%.4f RMSE=%.4f NMAE=%.6f",
            target, len(combined_oof), n_general, len(fine_index),
            full_r2, full_mae, full_rmse, full_nmae,
        )

    logger.info(
        "combined full-set OOF NMAE (mean of MDD+OWC, n=%d, matching model1i.py's/"
        "model2i.py's evaluation set): %.6f",
        n_general, sum(full_set_nmae.values()) / len(full_set_nmae),
    )

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
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_model3i.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "model3i_run.log"))
    p.add_argument("--model_out", default=str(repo_root / "models" / "model3i.joblib"),
                   help="path to cache/load the fitted per-target models + both Group C correctors + MICE imputers")
    p.add_argument("--force_retrain", action="store_true",
                   help="ignore --model_out if it exists and fit fresh anyway")
    p.add_argument("--exclude_ids", type=int, nargs="*", default=[155],
                   help="train.csv id values to drop from the GENERAL model's training only -- "
                        "the Group C correction stage still trains on these rows (see model2i.py's "
                        "module docstring's --exclude_ids ASYMMETRY section, unchanged here). "
                        "Defaults to [155] -- see model1i.py's --exclude_ids help for why. Pass "
                        "--exclude_ids (with no values) to include every row again.")
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
