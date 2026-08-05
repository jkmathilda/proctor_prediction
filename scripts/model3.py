"""
model3.py
=========
CLI pipeline for src/model3.py's SharedEncoderTargetPopulationHead -- see
docs/01-plan/features/model3.plan.md and docs/02-design/features/model3.design.md for the
full architecture rationale.

Not gated against model1/model1i/model2i's scores -- this is a from-scratch architecture
exploration (plan's scope note). Results are logged/reported to docs/260804.md regardless
of outcome, same as every other model in this project.

Evaluation is RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats) stratified on
`fine-grained`, reporting per-leaf (MDD-fine, MDD-coarse, OWC-fine, OWC-coarse) OOF R^2 and
combined NMAE as mean +/- std over repeats -- never a single-split point estimate
(docs/260804.md Section 3's project-wide finding that single splits at this n read
optimistic/noisy).

Run:
    python scripts/model3.py
    python scripts/model3.py --repeats 1 --folds 3 --epochs 30   # quick smoke test
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.general_model_impute import add_no_missing_features
from src.helper_functions import calc_satline, calculate_nmae
from src.model3 import (
    MODEL3_FEATURES,
    TARGET_COLS,
    SharedEncoderTargetPopulationHead,
    get_device,
    prepare_model3_features,
    predict_model3,
    train_model3,
)


def setup_logger(path):
    logger = logging.getLogger("model3")
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


def make_loaders(X, fine_flag, Y, batch_size, val_frac, seed):
    """Carve an inner early-stopping validation slice out of the TRAINING rows passed in
    -- stratified on fine_flag so both populations stay represented. This is always a
    subset of the current outer-CV fold's training rows, never the fold's held-out test
    rows (design.md Section 4's leak-free convention, matching eda.ipynb)."""
    n = len(X)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, val_idx = next(sss.split(np.zeros(n), fine_flag))

    def to_loader(idx, shuffle):
        ds = TensorDataset(
            torch.tensor(X[idx], dtype=torch.float32),
            torch.tensor(fine_flag[idx], dtype=torch.float32),
            torch.tensor(Y[idx], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return to_loader(tr_idx, True), to_loader(val_idx, False)


def fit_one_fold(X_tr, fine_tr, Y_tr, args, seed, device):
    """Fold-isolated scaling + inner early-stopping split + train. Returns the fitted
    model and the scalers needed to transform/inverse-transform this fold's held-out rows
    (or test.csv, for the final refit)."""
    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = StandardScaler().fit(Y_tr)
    X_tr_s = x_scaler.transform(X_tr)
    Y_tr_s = y_scaler.transform(Y_tr)

    train_loader, val_loader = make_loaders(
        X_tr_s, fine_tr, Y_tr_s, args.batch_size, args.inner_val_frac, seed,
    )

    torch.manual_seed(seed)
    model = SharedEncoderTargetPopulationHead(
        n_features=X_tr.shape[1],
        trunk_units=tuple(args.trunk_units),
        target_units=tuple(args.target_units),
        head_units=tuple(args.head_units),
        trunk_dropout=args.trunk_dropout,
        head_dropout=args.head_dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    history = train_model3(
        model, train_loader, val_loader, optimizer, criterion,
        epochs=args.epochs, patience=args.patience, device=device,
    )
    return model, x_scaler, y_scaler, history


def main(args):
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logger = setup_logger(args.log)
    logger.info("model3 -- hierarchical shared encoder (trunk -> MDD/OWC encoders -> fine/coarse heads)")
    logger.info(
        "seed=%d repeats=%d folds=%d trunk_units=%s target_units=%s head_units=%s",
        args.seed, args.repeats, args.folds, args.trunk_units, args.target_units, args.head_units,
    )

    device = get_device()
    logger.info("device=%s", device)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    logger.info("train shape=%s  test shape=%s", train.shape, test.shape)

    train_feat = add_no_missing_features(train)
    test_feat = add_no_missing_features(test)

    logger.info("fitting MICE imputers on train.csv (fine/coarse-grained split)...")
    train_imputed, fine_imputer, coarse_imputer = prepare_model3_features(
        train_feat, random_state=args.seed,
    )
    test_imputed, _, _ = prepare_model3_features(
        test_feat, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer,
        random_state=args.seed,
    )

    X_all = train_imputed[MODEL3_FEATURES].astype(float).to_numpy()
    fine_all = train_imputed["fine-grained"].astype(float).to_numpy()
    Y_all = train_imputed[TARGET_COLS].astype(float).to_numpy()

    if np.isnan(X_all).any():
        raise ValueError(
            "MODEL3_FEATURES contains NaNs in the training set after imputation -- this "
            "should not happen."
        )

    n = len(X_all)
    n_params = sum(p.numel() for p in SharedEncoderTargetPopulationHead(
        n_features=X_all.shape[1], trunk_units=tuple(args.trunk_units),
        target_units=tuple(args.target_units), head_units=tuple(args.head_units),
    ).parameters())
    logger.info("model size: %d params (plan.md Section 6 budget: ~1,880)", n_params)

    # ---- Repeated stratified CV (stratified on fine-grained) ----
    rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
    repeat_oof = [np.full((n, 2), np.nan) for _ in range(args.repeats)]

    for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(X_all, fine_all)):
        repeat_i = fold_i // args.folds
        fold_in_repeat = fold_i % args.folds
        seed = args.seed + fold_i

        model, x_scaler, y_scaler, history = fit_one_fold(
            X_all[tr_idx], fine_all[tr_idx], Y_all[tr_idx], args, seed, device,
        )

        X_te_s = x_scaler.transform(X_all[te_idx])
        pred_s = predict_model3(
            model,
            torch.tensor(X_te_s, dtype=torch.float32),
            torch.tensor(fine_all[te_idx], dtype=torch.float32),
            device=device,
        )
        repeat_oof[repeat_i][te_idx] = y_scaler.inverse_transform(pred_s)

        val_mdd = history["val_mdd_loss"][-1] if history["val_mdd_loss"] else float("nan")
        val_owc = history["val_owc_loss"][-1] if history["val_owc_loss"] else float("nan")
        logger.info(
            "repeat %d fold %d: epochs_run=%d train_loss=%.4f val_loss=%.4f (mdd=%.4f owc=%.4f)",
            repeat_i, fold_in_repeat, len(history["train_loss"]),
            history["train_loss"][-1],
            history["val_loss"][-1] if history["val_loss"] else float("nan"),
            val_mdd, val_owc,
        )
        # Decision 1.3's checkpoint: flag (don't fail) if one target's val loss looks like
        # it's dominating -- more than 5x the other's, on standardized-scale losses that
        # should otherwise be roughly comparable.
        if val_mdd > 0 and val_owc > 0 and (val_mdd / val_owc > 5 or val_owc / val_mdd > 5):
            logger.info(
                "  NOTE: mdd/owc val loss ratio=%.2f -- one target may be dominating "
                "training (docs/260804.md Section 8 failure mode); watch this if it "
                "persists across folds",
                val_mdd / val_owc if val_owc > 0 else float("inf"),
            )

    # ---- Per-leaf metrics, mean +/- std over repeats ----
    fine_mask = fine_all.astype(bool)
    leaf_defs = {
        "mdd_fine": (0, fine_mask), "mdd_coarse": (0, ~fine_mask),
        "owc_fine": (1, fine_mask), "owc_coarse": (1, ~fine_mask),
    }
    leaf_r2s = {name: [] for name in leaf_defs}
    nmaes = []

    for r in range(args.repeats):
        pred_r = repeat_oof[r]
        if np.isnan(pred_r).any():
            raise RuntimeError(
                f"repeat {r} has missing OOF predictions -- RepeatedStratifiedKFold "
                "should give full coverage within each repeat."
            )
        for name, (col, mask) in leaf_defs.items():
            leaf_r2s[name].append(r2_score(Y_all[mask, col], pred_r[mask, col]))
        nmaes.append(calculate_nmae(Y_all, pred_r))

    logger.info("=" * 60)
    logger.info("OOF results (mean +/- std over %d repeats):", args.repeats)
    for name, vals in leaf_r2s.items():
        logger.info("  [%s] R^2 = %.3f +/- %.3f", name, np.mean(vals), np.std(vals))
    logger.info("  Combined NMAE = %.4f +/- %.4f", np.mean(nmaes), np.std(nmaes))
    logger.info("=" * 60)

    # ---- Final refit on all rows ----
    logger.info("refitting final model on all %d rows...", n)
    final_model, final_x_scaler, final_y_scaler, final_history = fit_one_fold(
        X_all, fine_all, Y_all, args, args.seed, device,
    )
    logger.info(
        "final refit: epochs_run=%d train_loss=%.4f val_loss=%.4f",
        len(final_history["train_loss"]), final_history["train_loss"][-1],
        final_history["val_loss"][-1] if final_history["val_loss"] else float("nan"),
    )

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    torch.save({
        "model_state": final_model.state_dict(),
        "model_kwargs": {
            "n_features": X_all.shape[1],
            "trunk_units": tuple(args.trunk_units),
            "target_units": tuple(args.target_units),
            "head_units": tuple(args.head_units),
            "trunk_dropout": args.trunk_dropout,
            "head_dropout": args.head_dropout,
        },
        "x_scaler": final_x_scaler,
        "y_scaler": final_y_scaler,
        "feature_columns": MODEL3_FEATURES,
        "fine_imputer": fine_imputer,
        "coarse_imputer": coarse_imputer,
        "args": vars(args),
        "oof_leaf_r2": {name: (float(np.mean(v)), float(np.std(v))) for name, v in leaf_r2s.items()},
        "oof_nmae": (float(np.mean(nmaes)), float(np.std(nmaes))),
    }, args.model_out)
    logger.info("saved final model -> %s", args.model_out)

    # ---- Predict on test.csv ----
    X_test = test_imputed[MODEL3_FEATURES].astype(float).to_numpy()
    fine_test = test_imputed["fine-grained"].astype(float).to_numpy()
    if np.isnan(X_test).any():
        raise ValueError("MODEL3_FEATURES contains NaNs in the test set after imputation.")

    X_test_s = final_x_scaler.transform(X_test)
    pred_test_s = predict_model3(
        final_model,
        torch.tensor(X_test_s, dtype=torch.float32),
        torch.tensor(fine_test, dtype=torch.float32),
        device=device,
    )
    pred_test = final_y_scaler.inverse_transform(pred_test_s)

    # ---- Saturation clip -- both targets come from this model's own forward pass, no
    # cross-model OWC dependency needed (design.md Section 6) ----
    rho_s = test["grain_density_g_cm3"].fillna(2.65).to_numpy()
    owc = np.clip(pred_test[:, 1], 0, None)
    mdd = np.minimum(pred_test[:, 0], calc_satline(owc, rho_s) * 0.999)
    n_clipped = int((mdd < pred_test[:, 0] - 1e-9).sum())
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
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_model3.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "model3_run.log"))
    p.add_argument("--model_out", default=str(repo_root / "models" / "model3.pt"))
    p.add_argument("--repeats", type=int, default=5, help="RepeatedStratifiedKFold repeat count")
    p.add_argument("--folds", type=int, default=5, help="RepeatedStratifiedKFold fold count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trunk_units", type=int, nargs="+", default=[24, 16])
    p.add_argument("--target_units", type=int, nargs="+", default=[12, 8])
    p.add_argument("--head_units", type=int, nargs="+", default=[4])
    p.add_argument("--trunk_dropout", type=float, default=0.2)
    p.add_argument("--head_dropout", type=float, default=0.15)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument(
        "--inner_val_frac", type=float, default=0.2,
        help="fraction of each outer-CV fold's TRAINING rows carved out for early-stopping "
             "only -- never the outer fold's held-out test rows",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
