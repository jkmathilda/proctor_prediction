"""
blend_v15_model1i.py
=====================
Prediction-level blend of v15 (ARD-GP + HistGBT on MICE-imputed engineered
features, hyd_cond_hyd_gradient dropped) and model1i (Ridge + GPR + XGBoost
on MICE-imputed IMPUTED_FEATURES, fine/coarse-split imputation) -- the two
best-scoring, most architecturally distinct models in this repo (v15: 0.2383
Kaggle NMAE; model1i: 0.2610). blend_model1i_model4i.py already showed
blending across genuinely different pipelines gives a real (if modest) gain
that survives contact with the leaderboard (0.2610 -> 0.2582); v15 and
model1i share NO components (no common base learner, no common feature set,
no common imputation split), which is more architectural distance than
model1i/model4i had (both shared XGBoost) -- so this pairing is this
project's best remaining hypothesis for beating v15's 0.2383 outright.

METHODOLOGY -- leave-one-seed-out nested CV, matching how
blend_model1i_model4i.py's weights were validated (not model2i's plain OOF,
which looked better internally and then lost on Kaggle):

For each of --n_seeds seeds, BOTH models get a fresh honest out-of-fold
prediction via the SAME kind of split (make_stratified_kfold_splits,
full coverage, stratified on MDD quantile bins -- the split used by
model1i/model2i/model5i):
  - model1i: WeightedBlendRegressor(splits=splits), Optuna-tuned XGBoost,
    exactly as model1i.py computes its own internal OOF. Fit on the 200
    rows that survive model1i's own default --exclude_ids [155].
  - v15: the SAME GP(+GBT ARD-selected, blend-weight-chosen) recipe
    scripts/v15.py's main() uses, but with make_stratified_kfold_splits
    (full coverage) in place of v15.py's own single/repeated
    StratifiedShuffleSplit -- everything else (kernel, feature selection,
    GBT hyperparameters, blend-weight choice) is v15.py's real logic,
    reused via direct import so this can't silently drift from the
    production script. Fit on the full 201 rows (v15.py never excludes
    id 155).

Both OOF sets are restricted to the 200 ids they share (id 155 has no
model1i OOF by construction) before any weight fitting or evaluation.

Per seed, per target, a grid search picks the v15-share weight w in [0,1]
minimizing NMAE on that seed's pooled OOF. The HONEST reported number is
leave-one-seed-out: for each held-out seed, w is fit on the OTHER seeds'
averaged OOF and evaluated on the held-out seed's OOF, then averaged over
all rotations -- so the weight is never evaluated on the same predictions
it was chosen from. The FINAL production weight (used for the actual
blended submission) is fit on all seeds' pooled OOF together.

CAVEATS:
  - model1i's OWC isotonic calibration (a small, separately-validated
    correction, ~-0.0015 NMAE) is NOT reproduced inside this OOF sweep for
    simplicity -- the comparison uses model1i's raw (uncalibrated) OOF
    blend prediction. The final submission blend below uses the actual
    submission_model1i.csv (which IS calibrated), so the weight found here
    is a reasonable but not perfectly re-validated carry-over for the
    calibrated output, same caveat blend_model1i_model4i.py already flags
    for its own OWC weight.
  - v15's feature matrix includes the raw 'id' column as an actual model
    input (a pre-existing quirk of v15.py, not introduced here) -- this
    script reuses v15.py's real fit_bases/predict_bases functions
    unmodified, so that quirk is faithfully replicated rather than fixed,
    to keep the OOF honestly representative of the real production v15.
  - UNVALIDATED ON THE ACTUAL LEADERBOARD until submitted -- exactly the
    same caveat blend_model1i_model4i.py carries.

Run:
    python scripts/blend_v15_model1i.py                  # full run (~10-15 min)
    python scripts/blend_v15_model1i.py --n_seeds 1 --optuna_trials 5   # quick smoke test
"""

import argparse
import importlib.util
import logging
import os
import sys
from argparse import Namespace
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
    IMPUTED_FEATURES,
)

TARGETS = ["proctor_mdd_g_cm3", "proctor_owc_pct"]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup_logger(path):
    logger = logging.getLogger("blend_v15_model1i")
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


def nmae_target(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = np.mean(np.abs(y_true - y_pred))
    q75, q25 = np.percentile(y_true, [75, 25])
    iqr = q75 - q25 if q75 != q25 else 1e-8
    return float(mae / iqr)


def model1i_oof_for_seed(train_excl155, calc_satline, seed, optuna_trials, folds, val_strata, logger):
    """Reproduces model1i.py's own internal OOF (uncalibrated), for the given seed."""
    train_imputed, _, _ = add_imputed_features(train_excl155, random_state=seed)
    X = train_imputed[IMPUTED_FEATURES]
    y_mdd_all = train_excl155["proctor_mdd_g_cm3"].to_numpy(dtype=float)
    splits = make_stratified_kfold_splits(y_mdd_all, n_splits=folds, val_strata=val_strata, random_state=seed)

    oof = {}
    for target in TARGETS:
        y = train_excl155[target].to_numpy(dtype=float)
        if optuna_trials > 0:
            xgb_model, study = tune_xgb_with_optuna(X, y, n_trials=optuna_trials, random_state=seed, splits=splits)
            logger.info("[seed %d][model1i][%s] best CV MAE=%.4f", seed, target, study.best_value)
        else:
            xgb_model = None
        model = WeightedBlendRegressor(random_state=seed, xgb_model=xgb_model, splits=splits)
        model.fit(X, y)
        oof[target] = model.blend_oof_prediction_.copy()

    rho_s = train_excl155["grain_density_g_cm3"].fillna(2.65).to_numpy()
    owc = np.clip(oof["proctor_owc_pct"], 0, None)
    mdd = np.minimum(oof["proctor_mdd_g_cm3"], calc_satline(owc, rho_s) * 0.999)
    return {"proctor_mdd_g_cm3": mdd, "proctor_owc_pct": owc}, train_excl155["id"].to_numpy()


def v15_oof_for_seed(train_full, v15, H, seed, folds, val_strata, restarts, logger):
    """Reproduces v15.py's own GP(+GBT, ARD-selected, blend-weight-chosen) OOF
    recipe exactly, swapping its single/repeated StratifiedShuffleSplit for
    make_stratified_kfold_splits (full coverage) so every row gets a real OOF
    prediction from one pass, instead of averaging over possibly-overlapping
    random holdouts."""
    train_fe = v15.base_feature_engineering(train_full.copy(), H)
    ids = train_fe["id"].to_numpy()
    y = train_fe[TARGETS].values.astype(float)
    X_base = train_fe.drop(columns=TARGETS)
    X_base = X_base.drop(columns=["hyd_cond_hyd_gradient"], errors="ignore")
    exclude = ["id", "atterberg_is_missing", "kf_is_missing", "loi_is_missing"]
    cols_for_imputation = v15.numeric_impute_columns(X_base, exclude)

    splits = make_stratified_kfold_splits(y[:, 0], n_splits=folds, val_strata=val_strata, random_state=seed)
    v15_args = Namespace(
        kernel="matern", ard=True, restarts=restarts, select=True,
        select_thresh=100.0, select_min=5, linear=False, ensemble=True,
        gbt_iter=400, gbt_lr=0.05, gbt_leaves=31, seed=seed,
    )

    gp_sum = np.zeros_like(y)
    gbt_sum = np.zeros_like(y)
    for k, (tr, va) in enumerate(splits):
        Xtr, Xva = v15.preprocess_fold(X_base.iloc[tr].copy(), X_base.iloc[va].copy(),
                                        cols_for_imputation, H, seed)
        gps, sels, gbts = v15.fit_bases(Xtr, y[tr], v15_args)
        gpm, gps_std, gbm = v15.predict_bases(Xva, gps, sels, gbts)
        gp_sum[va] += gpm
        gbt_sum[va] += np.nan_to_num(gbm)
        logger.info("[seed %d][v15] fold %d/%d done", seed, k + 1, len(splits))

    gp_oof = gp_sum  # KFold -> every row hit exactly once, no need to divide by cnt
    gbt_oof = gbt_sum
    weights = v15.choose_blend_weights(y, gp_oof, gbt_oof)
    rho_s = X_base["grain_density_g_cm3"].fillna(2.65).values
    blended = v15.blend_and_clip(gp_oof, gbt_oof, weights, rho_s, H)
    logger.info("[seed %d][v15] GP-share weights: %s", seed,
                {TARGETS[j]: round(float(weights[j]), 2) for j in range(len(TARGETS))})
    return {TARGETS[j]: blended[:, j] for j in range(len(TARGETS))}, ids


def align(oof_a, ids_a, oof_b, ids_b):
    """Restrict both OOF dicts to the ids they share, same row order."""
    common = np.intersect1d(ids_a, ids_b)
    ia = {id_: i for i, id_ in enumerate(ids_a)}
    ib = {id_: i for i, id_ in enumerate(ids_b)}
    idx_a = np.array([ia[i] for i in common])
    idx_b = np.array([ib[i] for i in common])
    a = {t: oof_a[t][idx_a] for t in TARGETS}
    b = {t: oof_b[t][idx_b] for t in TARGETS}
    return a, b, common


def best_weight(y_true, pred_v15, pred_model1i, grid=41):
    best_w, best_nmae = 1.0, np.inf
    for w in np.linspace(0, 1, grid):
        blended = w * pred_v15 + (1 - w) * pred_model1i
        n = nmae_target(y_true, blended)
        if n < best_nmae:
            best_nmae, best_w = n, w
    return best_w, best_nmae


def main(args):
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logger = setup_logger(args.log)
    logger.info("blend_v15_model1i -- leave-one-seed-out nested CV blend weight search")
    logger.info("n_seeds=%d folds=%d optuna_trials=%d", args.n_seeds, args.folds, args.optuna_trials)

    repo_root = Path(__file__).resolve().parent.parent
    v15 = load_module(str(repo_root / "scripts" / "v15.py"), "v15_mod")
    model1i_mod = load_module(str(repo_root / "scripts" / "model1i.py"), "model1i_mod")
    H = v15.load_helpers(args.helpers_dir)
    calc_satline = model1i_mod.load_calc_satline(args.helpers_dir)

    train_raw = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    # model1i needs add_no_missing_features' engineered cols (incl. the
    # 'fine-grained' imputation-split key); v15 must NOT see them -- its own
    # base_feature_engineering builds its feature set from the raw columns,
    # and real v15.py never calls add_no_missing_features. Keep the two
    # frames separate so each model's OOF is computed on its real feature set.
    train_for_model1i = add_no_missing_features(train_raw.copy())
    train_excl155 = train_for_model1i[~train_for_model1i["id"].isin(args.exclude_ids)].reset_index(drop=True)
    logger.info("train shape=%s  model1i training rows (excl %s)=%d",
                train_raw.shape, args.exclude_ids, len(train_excl155))

    per_seed = []  # list of (y_true_dict, v15_pred_dict, model1i_pred_dict) restricted to common ids
    for seed in args.seeds:
        m1i_oof, m1i_ids = model1i_oof_for_seed(
            train_excl155, calc_satline, seed, args.optuna_trials, args.folds, args.val_strata, logger)
        v15_oof, v15_ids = v15_oof_for_seed(
            train_raw, v15, H, seed, args.folds, args.val_strata, args.restarts, logger)
        v15_a, m1i_a, common_ids = align(v15_oof, v15_ids, m1i_oof, m1i_ids)
        y_true = {t: train_raw.set_index("id").loc[common_ids, t].to_numpy(dtype=float) for t in TARGETS}

        for t in TARGETS:
            w, n = best_weight(y_true[t], v15_a[t], m1i_a[t])
            nmae_v15_alone = nmae_target(y_true[t], v15_a[t])
            nmae_m1i_alone = nmae_target(y_true[t], m1i_a[t])
            logger.info("[seed %d][%s] NMAE v15-alone=%.4f model1i-alone=%.4f  best-w(v15)=%.2f -> NMAE=%.4f",
                        seed, t, nmae_v15_alone, nmae_m1i_alone, w, n)

        per_seed.append((y_true, v15_a, m1i_a))

    # ---- leave-one-seed-out honesty check ----
    logger.info("\n===== LEAVE-ONE-SEED-OUT (honest) =====")
    held_out_nmaes = {t: [] for t in TARGETS}
    if len(per_seed) < 2:
        logger.info("n_seeds < 2 -- skipping leave-one-seed-out (needs >=2 seeds); "
                     "using per-seed weight above as a placeholder, NOT an honest estimate.")
    for held_out in range(len(per_seed) if len(per_seed) >= 2 else 0):
        train_seeds = [i for i in range(len(per_seed)) if i != held_out]
        for t in TARGETS:
            # average the training seeds' OOF predictions row-wise (all share common_ids
            # per seed, but different seeds may share different common-id sets in principle;
            # here they're identical since exclude_ids/ids are seed-independent)
            y_pool = per_seed[train_seeds[0]][0][t]
            v15_pool = np.mean([per_seed[i][1][t] for i in train_seeds], axis=0)
            m1i_pool = np.mean([per_seed[i][2][t] for i in train_seeds], axis=0)
            w, _ = best_weight(y_pool, v15_pool, m1i_pool)

            y_ho = per_seed[held_out][0][t]
            v15_ho = per_seed[held_out][1][t]
            m1i_ho = per_seed[held_out][2][t]
            blended_ho = w * v15_ho + (1 - w) * m1i_ho
            n_ho = nmae_target(y_ho, blended_ho)
            held_out_nmaes[t].append(n_ho)
            logger.info("held-out seed idx=%d [%s] weight-fit-on-others w(v15)=%.2f -> held-out NMAE=%.4f",
                        held_out, t, w, n_ho)

    if held_out_nmaes[TARGETS[0]]:
        overall = []
        for t in TARGETS:
            mean_n = float(np.mean(held_out_nmaes[t]))
            logger.info("[%s] leave-one-seed-out honest NMAE: mean=%.4f std=%.4f", t, mean_n, float(np.std(held_out_nmaes[t])))
            overall.append(mean_n)
        logger.info("Combined leave-one-seed-out honest NMAE (mean of MDD+OWC): %.4f", float(np.mean(overall)))

    # ---- final production weight: fit on ALL seeds pooled ----
    logger.info("\n===== FINAL WEIGHT (fit on all seeds pooled) =====")
    final_weights = {}
    for t in TARGETS:
        y_pool = per_seed[0][0][t]
        v15_pool = np.mean([per_seed[i][1][t] for i in range(len(per_seed))], axis=0)
        m1i_pool = np.mean([per_seed[i][2][t] for i in range(len(per_seed))], axis=0)
        w, n = best_weight(y_pool, v15_pool, m1i_pool)
        final_weights[t] = w
        logger.info("[%s] final weight w(v15)=%.2f  pooled-fit NMAE=%.4f", t, w, n)

    # ---- write the actual blended submission from the existing test-set CSVs ----
    v15_sub = pd.read_csv(args.v15_submission).sort_values("id").reset_index(drop=True)
    m1i_sub = pd.read_csv(args.model1i_submission).sort_values("id").reset_index(drop=True)
    if not (v15_sub["id"].to_numpy() == m1i_sub["id"].to_numpy()).all():
        raise ValueError("v15 and model1i submissions do not cover the same ids.")

    w_mdd = final_weights["proctor_mdd_g_cm3"]
    w_owc = final_weights["proctor_owc_pct"]
    blended_sub = pd.DataFrame({
        "id": v15_sub["id"],
        "proctor_mdd_g_cm3": w_mdd * v15_sub["proctor_mdd_g_cm3"] + (1 - w_mdd) * m1i_sub["proctor_mdd_g_cm3"],
        "proctor_owc_pct": w_owc * v15_sub["proctor_owc_pct"] + (1 - w_owc) * m1i_sub["proctor_owc_pct"],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    blended_sub.to_csv(args.out, index=False)
    logger.info("Wrote %d blended predictions -> %s", len(blended_sub), args.out)
    logger.info("Run log -> %s", args.log)
    return blended_sub


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(repo_root / "data"))
    p.add_argument("--helpers_dir", default=str(repo_root / "src"))
    p.add_argument("--v15_submission", default=str(repo_root / "submissions" / "submission_v15_ensemble+mice_nohydgrad.csv"))
    p.add_argument("--model1i_submission", default=str(repo_root / "submissions" / "submission_model1i.csv"))
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_blend_v15_model1i.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "blend_v15_model1i_run.log"))
    p.add_argument("--exclude_ids", type=int, nargs="*", default=[155])
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--seed_start", type=int, default=42)
    p.add_argument("--optuna_trials", type=int, default=50)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val_strata", type=int, default=5)
    p.add_argument("--restarts", type=int, default=6, help="GP kernel optimizer restarts (v15 default 6)")
    args = p.parse_args()
    args.seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    return args


if __name__ == "__main__":
    main(parse_args())
