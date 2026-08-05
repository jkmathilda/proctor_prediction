"""
tabicl_baseline.py
===================
Baseline using TabICLv2 (Jingang Qu et al. / soda-inria's tabular foundation model),
via FinetunedTabICLRegressor -- unlike tabpfn_baseline.py's frozen in-context inference,
this one actually gradient-fine-tunes the pretrained checkpoint on this dataset's own
training rows (AdamW, cosine-with-warmup, gradient clipping, early stopping against a
held-out split) before predicting -- a genuinely different bet from tabpfn_baseline.py,
not just a different pretrained model.

Same convention as every other model in this project (docs/260804.md Section 8): MDD and
OWC are fit as two independent models, not jointly. Same feature set as model1i
(IMPUTED_FEATURES, 28 MICE-imputed universal columns) and the same
StratifiedKFold-on-MDD-quantile validation scheme (general_model_impute.
make_stratified_kfold_splits) model1i.py/tabpfn_baseline.py use, so this script's OOF
NMAE is directly comparable to both.

Fine-tuning needs its own held-out split for early stopping (FinetunedTabICLRegressor's
required X_val/y_val) -- carved from each outer CV fold's TRAINING rows only, never the
fold's held-out test rows, same leak-free convention model3.py uses for its inner
early-stopping split.

No token/license wall (unlike tabpfn_baseline.py's default TabPFN version) -- TabICL is
open source including pretraining, per its README. Checkpoints auto-download from
Hugging Face on first use.

Install: pip install tabicl[finetune]

Run:
    python scripts/tabicl_baseline.py
    python scripts/tabicl_baseline.py --folds 3 --epochs 10   # faster while testing
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabicl import FinetunedTabICLRegressor

from src.general_model_impute import (
    IMPUTED_FEATURES,
    add_imputed_features,
    add_no_missing_features,
    make_stratified_kfold_splits,
)
from src.helper_functions import calc_satline, calculate_nmae

TARGETS = ["proctor_mdd_g_cm3", "proctor_owc_pct"]


def setup_logger(path):
    logger = logging.getLogger("tabicl_baseline")
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


def fit_one_fold(X_tr, y_tr, args, seed):
    """Carve an inner early-stopping validation slice out of this fold's TRAINING rows
    only (never the outer fold's held-out test rows), then fine-tune."""
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=args.inner_val_frac, random_state=seed,
    )
    model = FinetunedTabICLRegressor(
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        random_state=seed,
        verbose=False,
    )
    model.fit(X_tr2, y_tr2, X_val=X_val, y_val=y_val)
    return model


def main(args):
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logger = setup_logger(args.log)
    logger.info("tabicl_baseline -- FinetunedTabICLRegressor, per-target independent fit")
    logger.info(
        "seed=%d folds=%d val_strata=%d epochs=%d lr=%.1e patience=%d",
        args.seed, args.folds, args.val_strata, args.epochs, args.lr, args.patience,
    )

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
        "same scheme model1i.py/tabpfn_baseline.py use, for direct NMAE comparability)",
        args.folds, args.val_strata,
    )

    oof = {t: np.full(len(train), np.nan) for t in TARGETS}
    test_preds_per_fold = {t: [] for t in TARGETS}

    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        logger.info("fold %d/%d: train=%d val=%d", fold_i + 1, len(splits), len(tr_idx), len(va_idx))
        for target in TARGETS:
            y = train[target].to_numpy(dtype=float)
            seed = args.seed + fold_i
            model = fit_one_fold(X_train[tr_idx], y[tr_idx], args, seed)
            oof[target][va_idx] = model.predict(X_train[va_idx])
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
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_tabicl.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "tabicl_run.log"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val_strata", type=int, default=5,
                   help="MDD quantile strata used to stratify the split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30, help="FinetunedTabICLRegressor default")
    p.add_argument("--lr", type=float, default=1e-5, help="FinetunedTabICLRegressor default")
    p.add_argument("--patience", type=int, default=8, help="early-stopping patience, epochs")
    p.add_argument(
        "--inner_val_frac", type=float, default=0.15,
        help="fraction of each outer-CV fold's TRAINING rows carved out for the "
             "fine-tuning early-stopping split -- never the outer fold's held-out rows",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
