"""
model6i.py
==========
MDD is exactly model1i.py's stage: WeightedBlendRegressor (Ridge + GPR +
XGBoost, Optuna-tuned) on the standard 28-column IMPUTED_FEATURES.

OWC uses a separately-tuned WeightedBlendRegressor on an AUGMENTED feature
set: IMPUTED_FEATURES plus features v15.py's helper_functions.py computes
but model1i.py never uses:

    - log_diff_psd_size_at_d{20..98}_mm (10 cols): log10(d_i) - log10(d10),
      PSD shape relative to the finest measured size
    - psd_C_U_is_low: binary indicator, coefficient of uniformity < 5
    - atterberg_plasticity_index (PI), skempton_index, hazen_interaction,
      casagrande_interaction: all derived from the Atterberg liquid/plastic
      limits, which model1i.py's add_imputed_features() ALREADY MICE-imputes
      for fine-grained rows -- IMPUTED_FEATURES just never includes the
      result. That's the single biggest gap this closes.

Why MDD doesn't get these too: an earlier symmetric test (adding these
features to BOTH targets) found OWC improved (0.2776 -> 0.2717 NMAE) but
MDD got slightly worse (0.1936 -> 0.1954) -- these features carry real
OWC signal but are net noise for MDD, which is already well-explained by
the base feature set. Since MDD and OWC are fit as fully independent
models here (unlike model5i.py's cross-target-feature design), there's no
cost to giving each target its own feature set.

Coarse-grained rows have no measured Atterberg limits (non-plastic soils
-- not physically applicable). Their Atterberg-derived columns are set to
0.0 (an explicit "not applicable" placeholder), matching
helper_functions.py's apply_fold_feature_engineering rule for genuinely
coarse rows, rather than imputed as if a value existed. The Atterberg
imputer itself is model1i.py's own (BayesianRidge, no custom
constraints) -- a version with an added min_value=0.0 floor and an
explicit fine-grained input flag was also tested and performed slightly
WORSE (see project history), so this uses the simpler, better-performing
configuration.

Honest OOF validation (StratifiedKFold folds=5 val_strata=5, id155
excluded, same methodology as model1i.py):

    MDD (model1i's own):        NMAE 0.193642  (unchanged from model1i.py)
    OWC (augmented features):   NMAE 0.271667  (model1i.py alone: 0.277554)
    Combined:                   NMAE 0.232655  (model1i.py alone: 0.235598)

This is NOT model5i.py's approach (which adds the OTHER target's own OOF
PREDICTION as a cross-feature, on the ORIGINAL 28-column feature set) --
model6i.py never uses either target's prediction as an input to the
other; the improvement here comes entirely from richer OWC features.
The two ideas are independent and have not been combined/tested together.

Run:
    python scripts/model6i.py
    python scripts/model6i.py --optuna_trials 0   # skip tuning, fast
    python scripts/model6i.py --force_retrain     # ignore cached model
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
    IMPUTED_FEATURES,
)

MDD_TARGET = "proctor_mdd_g_cm3"
OWC_TARGET = "proctor_owc_pct"
TARGETS = [MDD_TARGET, OWC_TARGET]

MDD_FEATURES = IMPUTED_FEATURES

OWC_EXTRA_FEATURES = [
    f"log_diff_psd_size_at_d{pct}_mm" for pct in [20, 30, 40, 50, 60, 70, 80, 90, 95, 98]
] + [
    "psd_C_U_is_low",
    "atterberg_liquid_limit_pct_completed",
    "atterberg_plastic_limit_pct_completed",
    "atterberg_plasticity_index",
    "skempton_index",
    "hazen_interaction",
    "casagrande_interaction",
]
OWC_FEATURES = IMPUTED_FEATURES + OWC_EXTRA_FEATURES

FEATURE_SETS = {MDD_TARGET: MDD_FEATURES, OWC_TARGET: OWC_FEATURES}


def nmae(y_true, y_pred):
    """Official competition metric: mean column-wise IQR-normalized MAE."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    q75, q25 = np.percentile(y_true, [75, 25], axis=0)
    iqr = np.where((q75 - q25) == 0, 1e-8, q75 - q25)
    return float(np.mean(mae / iqr))


def add_owc_extra_features(df):
    """Adds OWC_EXTRA_FEATURES on top of an add_imputed_features()-processed
    dataframe (already has atterberg_*_completed, log10_hyd_cond_kf_m_s_completed).
    Coarse-grained rows' Atterberg-derived columns are set to 0.0 -- "not
    applicable", not imputed as if a value existed."""
    df = df.copy()

    d10 = df["psd_size_at_d10_mm"].clip(lower=1e-10)
    for pct in [20, 30, 40, 50, 60, 70, 80, 90, 95, 98]:
        col = f"psd_size_at_d{pct}_mm"
        df[f"log_diff_psd_size_at_d{pct}_mm"] = np.log10(df[col].clip(lower=1e-10)) - np.log10(d10)

    df["psd_C_U_is_low"] = (df["feat_cu"] < 5).astype(int)

    ll = df["atterberg_liquid_limit_pct_completed"].copy()
    pl = df["atterberg_plastic_limit_pct_completed"].copy()
    still_missing = ll.isna() | pl.isna()  # true for coarse-grained rows
    ll[still_missing] = 0.0
    pl[still_missing] = 0.0
    df["atterberg_liquid_limit_pct_completed"] = ll
    df["atterberg_plastic_limit_pct_completed"] = pl

    df["atterberg_plasticity_index"] = ll - pl
    df["skempton_index"] = (df["atterberg_plasticity_index"] * df["clay"]) / 100.0
    df["hazen_interaction"] = df["log10_hyd_cond_kf_m_s_completed"] - 2 * df["log_diff_psd_size_at_d20_mm"]
    df["casagrande_interaction"] = df["atterberg_plasticity_index"] - 0.73 * (ll - 20)

    return df


def load_calc_satline(helpers_dir):
    path = os.path.join(helpers_dir, "helper_functions.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"helper_functions.py not found at {path}")
    spec = importlib.util.spec_from_file_location("helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.calc_satline


def setup_logger(path):
    logger = logging.getLogger("model6i")
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


def fit_target(X_train, y, X_test, *, splits, optuna_trials, seed, logger, tag):
    if optuna_trials > 0:
        logger.info("%s tuning XGBoost with Optuna (%d trials)...", tag, optuna_trials)
        xgb_model, study = tune_xgb_with_optuna(
            X_train, y, n_trials=optuna_trials, random_state=seed, splits=splits,
        )
        logger.info("%s best CV MAE=%.4f  params=%s", tag, study.best_value, study.best_params)
    else:
        xgb_model = None

    model = WeightedBlendRegressor(random_state=seed, xgb_model=xgb_model, splits=splits)
    model.fit(X_train, y)

    summary = model.get_training_summary()
    w = summary["weights"]
    m = summary["blend_oof_metrics"]
    oof_nmae = nmae(y, model.blend_oof_prediction_)
    logger.info("%s blend weights: ridge=%.3f gpr=%.3f xgboost=%.3f", tag, w["ridge"], w["gpr"], w["xgboost"])
    logger.info(
        "%s OOF (n_validated=%d/%d): R2=%.4f MAE=%.4f RMSE=%.4f NMAE=%.6f",
        tag, summary["n_validated"], len(y), m["r2"], m["mae"], m["rmse"], oof_nmae,
    )

    return model, model.predict(X_test)


def main(args):
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logger = setup_logger(args.log)
    logger.info(
        "model6i -- MDD is model1i's own 28-feature model; OWC gets an augmented "
        "feature set (Atterberg plasticity features model1i imputes but never uses, "
        "plus log-diff PSD ratios and a C_U-low flag) -- see module docstring"
    )
    logger.info("seed=%d  optuna_trials=%d", args.seed, args.optuna_trials)

    calc_satline = load_calc_satline(args.helpers_dir)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    logger.info("train shape=%s  test shape=%s", train.shape, test.shape)

    if args.exclude_ids:
        before = len(train)
        train = train[~train["id"].isin(args.exclude_ids)].reset_index(drop=True)
        logger.info("excluded ids=%s from training: %d -> %d rows", args.exclude_ids, before, len(train))

    train = add_no_missing_features(train)
    test = add_no_missing_features(test)

    cache_hit = os.path.exists(args.model_out) and not args.force_retrain

    if cache_hit:
        logger.info("loading cached models <- %s", args.model_out)
        cached = joblib.load(args.model_out)
        if cached["feature_sets"] != FEATURE_SETS:
            raise ValueError(
                f"Cached model at {args.model_out} was trained on different feature "
                "sets than the current MDD_FEATURES/OWC_FEATURES -- rerun with --force_retrain."
            )
        models = cached["models"]
        fine_imputer = cached["fine_imputer"]
        coarse_imputer = cached["coarse_imputer"]
        test_imputed, _, _ = add_imputed_features(
            test, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer, random_state=args.seed,
        )
    else:
        models = {}
        logger.info("fitting MICE imputers on train.csv (fine/coarse-grained split)...")
        train_imputed, fine_imputer, coarse_imputer = add_imputed_features(train, random_state=args.seed)
        test_imputed, _, _ = add_imputed_features(
            test, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer, random_state=args.seed,
        )
        train_imputed = add_owc_extra_features(train_imputed)

    test_imputed = add_owc_extra_features(test_imputed)

    for target in TARGETS:
        cols = FEATURE_SETS[target]
        if test_imputed[cols].isna().any().any():
            raise ValueError(f"[{target}] feature set contains NaNs in the test set after imputation.")

    if not cache_hit:
        for target in TARGETS:
            cols = FEATURE_SETS[target]
            if train_imputed[cols].isna().any().any():
                raise ValueError(f"[{target}] feature set contains NaNs in the training set after imputation.")

        y_mdd_all = train[MDD_TARGET].to_numpy(dtype=float)
        splits = make_stratified_kfold_splits(
            y_mdd_all, n_splits=args.folds, val_strata=args.val_strata, random_state=args.seed,
        )
        logger.info(
            "validation scheme: StratifiedKFold folds=%d val_strata=%d (stratified on MDD, full coverage, "
            "reused for both targets despite their different feature sets)",
            args.folds, args.val_strata,
        )

    preds = {}
    for target in TARGETS:
        cols = FEATURE_SETS[target]

        if cache_hit:
            model = models[target]
        else:
            X_train = train_imputed[cols]
            y = train[target].to_numpy(dtype=float)
            model, _ = fit_target(
                X_train, y, test_imputed[cols], splits=splits, optuna_trials=args.optuna_trials,
                seed=args.seed, logger=logger, tag=f"[{target}]",
            )
            models[target] = model

        preds[target] = model.predict(test_imputed[cols])

    if not cache_hit:
        os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
        joblib.dump(
            {
                "models": models,
                "fine_imputer": fine_imputer,
                "coarse_imputer": coarse_imputer,
                "feature_sets": FEATURE_SETS,
                "optuna_trials": args.optuna_trials,
                "seed": args.seed,
            },
            args.model_out,
        )
        logger.info("saved fitted models + imputers -> %s", args.model_out)

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
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_model6i.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "model6i_run.log"))
    p.add_argument("--model_out", default=str(repo_root / "models" / "model6i.joblib"))
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--exclude_ids", type=int, nargs="*", default=[155])
    p.add_argument("--optuna_trials", type=int, default=50)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val_strata", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
