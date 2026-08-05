"""
tabpfn_baseline.py
===================
Baseline using TabPFN (Prior Labs' pretrained tabular foundation model) --
no hyperparameter tuning and no gradient-descent training from scratch:
TabPFNRegressor does in-context learning over the training rows at inference
time, using transformer weights pretrained on millions of synthetic tabular
datasets. Included as a baseline specifically because it was designed for
exactly this dataset's regime (small n, no manual tuning) -- n=201 is well
inside TabPFN's intended operating range.

Same convention as every other model in this project (docs/260804.md
Section 8): MDD and OWC are fit as two independent models, not jointly.
Same feature set as model1i (IMPUTED_FEATURES, 28 MICE-imputed universal
columns) and the same StratifiedKFold-on-MDD-quantile validation scheme
(general_model_impute.make_stratified_kfold_splits) model1i.py uses, so this
script's OOF NMAE is directly comparable to the model1/model1i/model2i
family's already-logged internal numbers in docs/260804.md.

Uses TabPFN-2 specifically (`ModelVersion.V2`), not the package's own default model
version -- the pip package `tabpfn` (>=8.x) defaults to v2.5/v2.6/v3, which require a
one-time browser-based license acceptance + TABPFN_TOKEN (see
https://github.com/PriorLabs/tabpfn). TabPFN-2's code and weights are Apache 2.0
(Prior Labs License, open + attribution) and run fully locally after downloading weights
from Hugging Face on first use -- no token, no license click-through. v2 is the older
model, but avoids the auth wall entirely, which matters more for a quick baseline than
squeezing out the latest version's accuracy gain.

Run:
    python scripts/tabpfn_baseline.py
    python scripts/tabpfn_baseline.py --folds 3   # faster, fewer folds while testing
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from src.general_model_impute import (
    IMPUTED_FEATURES,
    add_imputed_features,
    add_no_missing_features,
    make_stratified_kfold_splits,
)
from src.helper_functions import calc_satline, calculate_nmae

TARGETS = ["proctor_mdd_g_cm3", "proctor_owc_pct"]


def setup_logger(path):
    logger = logging.getLogger("tabpfn")
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
    logger.info("tabpfn -- TabPFNRegressor baseline (pretrained, no tuning), per-target independent fit")
    logger.info("seed=%d folds=%d val_strata=%d", args.seed, args.folds, args.val_strata)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    logger.info("train shape=%s  test shape=%s", train.shape, test.shape)

    train_feat = add_no_missing_features(train)
    test_feat = add_no_missing_features(test)

    logger.info("fitting MICE imputers on train.csv (fine/coarse-grained split)...")
    train_imputed, fine_imputer, coarse_imputer = add_imputed_features(
        train_feat, random_state=args.seed,
    )
    test_imputed, _, _ = add_imputed_features(
        test_feat, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer,
        random_state=args.seed,
    )

    X_train = train_imputed[IMPUTED_FEATURES].astype(float).to_numpy()
    X_test = test_imputed[IMPUTED_FEATURES].astype(float).to_numpy()
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        raise ValueError(
            "IMPUTED_FEATURES contains NaNs after imputation -- this should not happen."
        )

    y_mdd_all = train["proctor_mdd_g_cm3"].to_numpy(dtype=float)
    splits = make_stratified_kfold_splits(
        y_mdd_all, n_splits=args.folds, val_strata=args.val_strata, random_state=args.seed,
    )
    logger.info(
        "validation scheme: StratifiedKFold folds=%d val_strata=%d (stratified on MDD, "
        "same scheme model1i.py uses, for direct NMAE comparability)",
        args.folds, args.val_strata,
    )

    oof = {t: np.full(len(train), np.nan) for t in TARGETS}
    test_preds_per_fold = {t: [] for t in TARGETS}

    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        logger.info("fold %d/%d: train=%d val=%d", fold_i + 1, len(splits), len(tr_idx), len(va_idx))
        for target in TARGETS:
            y = train[target].to_numpy(dtype=float)
            model = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2,
                random_state=args.seed,
                ignore_pretraining_limits=args.ignore_pretraining_limits,
            )
            model.fit(X_train[tr_idx], y[tr_idx])
            oof[target][va_idx] = model.predict(X_train[va_idx])
            # Also predict test.csv from each fold's model, averaged across folds at the
            # end -- TabPFN has no tuning to redo on a final refit the way a tuned GBM
            # does, so per-fold test-prediction averaging is a cheap variance reduction
            # (consistent with the ensembling spirit of TabPFN's own internal
            # n_estimators), rather than a separate "refit on everything" step.
            test_preds_per_fold[target].append(model.predict(X_test))

    for target in TARGETS:
        y_true = train[target].to_numpy(dtype=float)
        r2 = r2_score(y_true, oof[target])
        mae = np.mean(np.abs(y_true - oof[target]))
        logger.info("[%s] OOF R2=%.4f MAE=%.4f", target, r2, mae)

    y_true_all = train[TARGETS].to_numpy(dtype=float)
    oof_all = np.column_stack([oof[t] for t in TARGETS])
    nmae = calculate_nmae(y_true_all, oof_all)
    logger.info("Combined OOF NMAE = %.4f", nmae)

    preds = {t: np.mean(np.column_stack(test_preds_per_fold[t]), axis=1) for t in TARGETS}

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
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_tabpfn.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "tabpfn_run.log"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val_strata", type=int, default=5,
                   help="MDD quantile strata used to stratify the split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ignore_pretraining_limits", action="store_true",
        help="TabPFN warns/restricts above certain n_samples/n_features -- this dataset "
             "(201 rows, 28 features) is well within default limits, so this shouldn't be "
             "needed; exposed for flexibility only",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
