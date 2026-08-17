"""
model5i.py
==========
Asymmetric two-stage extension of model1i.py, built on top of an earlier
symmetric version tested in this project's development history.

MDD and OWC are the same physical Proctor compaction curve for a given
soil (Pearson r = -0.76 in this dataset), but every model tried so far
(model1.py through model4i.py, v15) predicts them as fully independent
regression problems on the geotechnical features alone. Checking model1i's
own residuals confirms real signal is being left on the table: even after
IMPUTED_FEATURES already explains most of the raw -0.76 target correlation,
the residual correlation is still -0.32 -- when model1i overpredicts MDD
for a row, it tends to underpredict OWC for that same row, beyond what the
shared features capture.

The FIRST version of this script added a cross-target OOF feature
SYMMETRICALLY (MDD's model got OWC's stage-1 OOF prediction as an extra
feature, and vice versa). Honest OOF validation showed a real but
asymmetric effect: OWC improved (0.2776 -> 0.2753 NMAE) but MDD got WORSE
(0.1936 -> 0.1986), for a net-negative combined result. The likely reason:
OWC's own model is noisier (NMAE ~0.28) than MDD's (~0.19), so feeding
OWC's OOF into MDD's model mostly injects variance, not signal, while the
reverse direction -- feeding MDD's comparatively cleaner OOF into OWC's
model -- is a net win. Cross-target stacking like this only pays off when
the donor target is cleaner than the recipient; here that only holds in
one direction.

THIS VERSION is therefore asymmetric:
    - MDD: pure model1i.py stage 1, no cross-feature. (Also independently
      confirmed safe by a real Kaggle submission: mixing in even a small
      12% model4i share for MDD changed the leaderboard score by only
      0.0002 -- MDD is robust to small perturbations, but there's no
      reason to introduce a change here that the honest OOF already says
      is net-negative.)
    - OWC: stage 2 adds ONE extra feature -- MDD's stage-1 out-of-fold
      prediction (never the real MDD value, which is never available at
      test time either). A fresh WeightedBlendRegressor is fit on
      IMPUTED_FEATURES + that one cross-target column, using the SAME
      StratifiedKFold splits stage 1 used.

Why reusing the same splits (rather than a second, independently-nested
CV) is NOT leakage: OWC's stage-2 OOF prediction for row i only ever comes
from a model trained on rows outside row i's own fold, and the MDD OOF
feature value it's given for row i was itself computed by a stage-1 MDD
model that never saw row i's true OWC (or MDD) label either. This is the
identical pattern model2i.py's Group C correction already uses in this
codebase (a corrector fit on top of the general model's OWN out-of-fold
`blend_oof_prediction`, validated via a further round of CV on top of that
already-OOF feature) -- standard OOF-stacking practice, not a new risk.

At test time: stage 1 predicts MDD on the test set first, and that
prediction becomes the cross-target feature OWC's stage 2 uses for its own
test prediction.

NOTE: this is mechanistically unrelated to the model1i+model4i blend
(scripts/blend_model1i_model4i.py), which was confirmed on the real
leaderboard to make OWC WORSE (0.2580 vs model1i alone's 0.2465) despite
a better honest OOF number -- that failure was specific to leaning on
model4i, a model already known to generalize worse. This script never
touches model4i; the cross-feature comes from model1i's own MDD model.

The physics guardrail (OWC clipped >= 0, MDD clipped to the Zero-Air-Voids
saturation line) is applied to the final predictions, same as model1i.py.

Run:
    python scripts/model5i.py
    python scripts/model5i.py --optuna_trials 0   # skip tuning, fast
    python scripts/model5i.py --force_retrain     # ignore cached model
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

# Only OWC gets a stage-2 cross-feature model -- see module docstring for
# why the symmetric version (MDD also getting one) was net-negative.
STAGE2_TARGETS = [OWC_TARGET]
CROSS_FEATURE_COL = "mdd_stage1_oof"


def nmae(y_true, y_pred):
    """Official competition metric: mean column-wise IQR-normalized MAE."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    q75, q25 = np.percentile(y_true, [75, 25], axis=0)
    iqr = np.where((q75 - q25) == 0, 1e-8, q75 - q25)
    return float(np.mean(mae / iqr))


def load_calc_satline(helpers_dir):
    path = os.path.join(helpers_dir, "helper_functions.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"helper_functions.py not found at {path}")
    spec = importlib.util.spec_from_file_location("helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.calc_satline


def setup_logger(path):
    logger = logging.getLogger("model5i")
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


def fit_stage(X_train, y, X_test, *, splits, optuna_trials, seed, logger, tag):
    """Fit an Optuna-tuned WeightedBlendRegressor, log its OOF metrics, and
    return (model, oof_prediction, test_prediction, oof_nmae)."""
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
    logger.info(
        "%s blend weights: ridge=%.3f gpr=%.3f xgboost=%.3f", tag, w["ridge"], w["gpr"], w["xgboost"],
    )
    logger.info(
        "%s OOF (n_validated=%d/%d): R2=%.4f MAE=%.4f RMSE=%.4f NMAE=%.6f",
        tag, summary["n_validated"], len(y), m["r2"], m["mae"], m["rmse"], oof_nmae,
    )

    return model, model.blend_oof_prediction_, model.predict(X_test), oof_nmae


def main(args):
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logger = setup_logger(args.log)
    logger.info(
        "model5i -- asymmetric two-stage model1i: MDD is pure stage 1, OWC's stage 2 "
        "adds MDD's stage-1 OOF prediction as an extra feature (residual correlation "
        "-0.32 after IMPUTED_FEATURES -- see module docstring for why only OWC gets this)"
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
        if cached["feature_columns"] != IMPUTED_FEATURES:
            raise ValueError(
                f"Cached model at {args.model_out} was trained on a different "
                "feature set than the current IMPUTED_FEATURES -- rerun with --force_retrain."
            )
        stage1_models = cached["stage1_models"]
        owc_stage2_model = cached["owc_stage2_model"]
        fine_imputer = cached["fine_imputer"]
        coarse_imputer = cached["coarse_imputer"]
        test_imputed, _, _ = add_imputed_features(
            test, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer, random_state=args.seed,
        )
        train_imputed = None
    else:
        stage1_models, owc_stage2_model = {}, None
        logger.info("fitting MICE imputers on train.csv (fine/coarse-grained split)...")
        train_imputed, fine_imputer, coarse_imputer = add_imputed_features(train, random_state=args.seed)
        test_imputed, _, _ = add_imputed_features(
            test, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer, random_state=args.seed,
        )

    X_test = test_imputed[IMPUTED_FEATURES]
    if X_test.isna().any().any():
        raise ValueError("IMPUTED_FEATURES contains NaNs in the test set after imputation.")

    if not cache_hit:
        X_train = train_imputed[IMPUTED_FEATURES]
        if X_train.isna().any().any():
            raise ValueError("IMPUTED_FEATURES contains NaNs in the training set after imputation.")

        y_mdd_all = train[MDD_TARGET].to_numpy(dtype=float)
        splits = make_stratified_kfold_splits(
            y_mdd_all, n_splits=args.folds, val_strata=args.val_strata, random_state=args.seed,
        )
        logger.info(
            "validation scheme: StratifiedKFold folds=%d val_strata=%d (stratified on MDD, full coverage, "
            "reused for stage 1 AND OWC's stage 2)",
            args.folds, args.val_strata,
        )

    # ================================================================
    # Stage 1: exactly model1i.py, independently per target. MDD's
    # result here IS the final MDD prediction (no stage 2 for MDD).
    # ================================================================
    stage1_oof, stage1_test_pred = {}, {}

    for target in TARGETS:
        if cache_hit:
            model = stage1_models[target]
            stage1_oof[target] = model.blend_oof_prediction_
        else:
            y = train[target].to_numpy(dtype=float)
            model, oof, _, oof_nmae = fit_stage(
                X_train, y, X_test, splits=splits, optuna_trials=args.optuna_trials,
                seed=args.seed, logger=logger, tag=f"[stage1:{target}]",
            )
            stage1_models[target] = model
            stage1_oof[target] = oof

        stage1_test_pred[target] = model.predict(X_test)

    # ================================================================
    # Stage 2: OWC only. IMPUTED_FEATURES + MDD's stage-1 OOF.
    # ================================================================
    if cache_hit:
        X_test_owc_stage2 = X_test.copy()
        X_test_owc_stage2[CROSS_FEATURE_COL] = stage1_test_pred[MDD_TARGET]
    else:
        y_owc = train[OWC_TARGET].to_numpy(dtype=float)

        X_train_owc_stage2 = X_train.copy()
        X_train_owc_stage2[CROSS_FEATURE_COL] = stage1_oof[MDD_TARGET]

        X_test_owc_stage2 = X_test.copy()
        X_test_owc_stage2[CROSS_FEATURE_COL] = stage1_test_pred[MDD_TARGET]

        owc_stage2_model, _, _, owc_stage2_nmae = fit_stage(
            X_train_owc_stage2, y_owc, X_test_owc_stage2, splits=splits, optuna_trials=args.optuna_trials,
            seed=args.seed, logger=logger, tag="[stage2:proctor_owc_pct]",
        )

        owc_stage1_nmae = nmae(y_owc, stage1_oof[OWC_TARGET])
        logger.info(
            "[proctor_owc_pct] stage1 -> stage2 NMAE: %.6f -> %.6f (delta=%+.6f)",
            owc_stage1_nmae, owc_stage2_nmae, owc_stage2_nmae - owc_stage1_nmae,
        )

    final_test_pred = {
        MDD_TARGET: stage1_test_pred[MDD_TARGET],
        OWC_TARGET: owc_stage2_model.predict(X_test_owc_stage2),
    }

    if not cache_hit:
        os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
        joblib.dump(
            {
                "stage1_models": stage1_models,
                "owc_stage2_model": owc_stage2_model,
                "fine_imputer": fine_imputer,
                "coarse_imputer": coarse_imputer,
                "feature_columns": IMPUTED_FEATURES,
                "cross_feature_col": CROSS_FEATURE_COL,
                "optuna_trials": args.optuna_trials,
                "seed": args.seed,
            },
            args.model_out,
        )
        logger.info("saved stage1 models + OWC stage2 model + imputers -> %s", args.model_out)

    rho_s = test["grain_density_g_cm3"].fillna(2.65).to_numpy()
    owc = np.clip(final_test_pred[OWC_TARGET], 0, None)
    mdd = np.minimum(final_test_pred[MDD_TARGET], calc_satline(owc, rho_s) * 0.999)
    n_clipped = int((mdd < final_test_pred[MDD_TARGET] - 1e-9).sum())
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
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_model5i.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "model5i_run.log"))
    p.add_argument("--model_out", default=str(repo_root / "models" / "model5i.joblib"))
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--exclude_ids", type=int, nargs="*", default=[155])
    p.add_argument("--optuna_trials", type=int, default=50)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val_strata", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
