"""
v7.py
=====
KNN diversity model for the LeiGS 2026 Proctor Prediction Challenge.

Extends v6.py's leak-free CV pipeline (fold-isolated MICE imputation ->
feature engineering -> K-fold CV -> registry-based ensemble) with two
additions, per docs/01-plan/features/proctor-prediction.plan.md §12 and
docs/02-design/features/proctor-prediction.design.md §11:

1. ``fine_grained`` flag (psd_fraction_clay + psd_fraction_silt > 15) -- the
   one genuinely new column from the requested clay/silt/sand/gravel/
   fine-grained/ip set. The other five already exist verbatim in
   helper_functions.py as psd_fraction_clay/silt/sand/gravel and
   atterberg_plasticity_index (see design.md §11.1).
2. A ``KNN`` candidate model family (scaled KNeighborsRegressor), evaluated
   inside the *same* CV run as v6's ExtraTrees/HistGB/Ridge/PINN candidates
   so all out-of-fold predictions are fold-aligned and can be validly
   blended -- report standalone KNN CV NMAE first, then test whether
   blending it in beats the best non-KNN candidate alone.
3. CV folds stratified on dominant soil-composition bucket (Sand/Gravel/Fine,
   see ``soil_composition_strata``) instead of a plain shuffled KFold, so
   each fold sees a similar clay/silt/sand/gravel mix -- v1-v6 all used plain
   KFold, which at n=201 can by chance concentrate rare soil compositions
   into one or two folds.

Run:
    python v7.py --data_dir ../data --helpers_dir .. --sub_dir ..
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v6 as V6  # reuse the registry-based CV pipeline, NMAE metric, PINN wrapper, etc.


# --------------------------------------------------------------------------- #
# Soil-composition-stratified CV folds
# --------------------------------------------------------------------------- #
def soil_composition_strata(df):
    """Dominant DIN soil-type bucket (Sand/Gravel/Fine) per sample, used only to
    stratify CV folds so each fold sees a similar clay/silt/sand/gravel mix --
    v6/v1-v5 use a plain shuffled KFold, which at n=201 can by chance
    concentrate rare soil compositions into one or two folds. Reuses the same
    DIN boundaries as scripts/data_split_by_soil.py.

    Clay is merged into the Silt bucket: train.csv has only 1 clay-dominant
    sample (scripts/data_split_by_soil.py's dominant_soil_type counts:
    Sand=129, Gravel=45, Silt=26, Clay=1), and StratifiedKFold requires every
    class to have at least n_splits members.
    """
    clay = df["psd_passing_at_0_002mm_pct"].clip(lower=0)
    silt = (df["psd_passing_at_0_063mm_pct"] - df["psd_passing_at_0_002mm_pct"]).clip(lower=0)
    sand = (df["psd_passing_at_2mm_pct"] - df["psd_passing_at_0_063mm_pct"]).clip(lower=0)
    gravel = (100 - df["psd_passing_at_2mm_pct"]).clip(lower=0)
    fractions = pd.DataFrame({"Sand": sand, "Gravel": gravel, "Fine": clay + silt})
    return fractions.idxmax(axis=1)


def run_cv_pipeline_stratified(X_base, y_base, model_configs, cols_for_imputation,
                               scaler_groups, H, cv_splits, seed=42):
    """Identical to v6.run_cv_pipeline's body, except the fold splits are
    supplied by the caller (a stratified split) instead of being generated
    internally by a plain KFold. v6.py itself is left untouched so its
    already-documented results stay reproducible."""
    feat_skew, feat_out, feat_std = scaler_groups
    registry = {label: [] for label, _, _ in model_configs}
    oof = {label: np.full((len(X_base), y_base.shape[1]), np.nan)
           for label, _, _ in model_configs}
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(cv_splits):
        X_tr, X_va = X_base.iloc[tr_idx].copy(), X_base.iloc[va_idx].copy()
        y_tr, y_va = y_base.iloc[tr_idx], y_base.iloc[va_idx]

        for f in (X_tr, X_va):
            f["atterberg_is_missing"] = f["atterberg_liquid_limit_pct"].isnull().astype(int)
            f["kf_is_missing"] = f["hyd_cond_kf_m_s"].isnull().astype(int)
            f["loi_is_missing"] = f["loss_on_ignition_pct"].isnull().astype(int)

        imputer = H.get_default_mice_imputer(seed=seed)
        X_tr[cols_for_imputation] = imputer.fit_transform(X_tr[cols_for_imputation])
        X_va[cols_for_imputation] = imputer.transform(X_va[cols_for_imputation])

        for f in (X_tr, X_va):
            H.apply_fold_feature_engineering(f)

        for label, template, scaled in model_configs:
            scaler = None
            if scaled:
                valid = set(X_tr.columns)
                ct = H.get_column_preprocessor(
                    [c for c in feat_skew if c in valid],
                    [c for c in feat_out if c in valid],
                    [c for c in feat_std if c in valid],
                )
                Xtr_p = ct.fit_transform(X_tr)
                Xva_p = ct.transform(X_va)
                scaler = ct
            elif label == "PINN":
                Xtr_p, Xva_p = X_tr, X_va
            else:
                Xtr_p, Xva_p = X_tr.values, X_va.values

            model = clone(template)
            model.fit(Xtr_p, y_tr)
            preds = np.asarray(model.predict(Xva_p))

            score = V6.nmae(y_va.values, preds)
            rows.append({"Modell": label, "Skaliert": scaled,
                         "Fold": fold, "NMAE": score})
            oof[label][va_idx] = preds
            registry[label].append({
                "fitted_imputer": imputer,
                "fitted_scaler": scaler,
                "fitted_model": model,
            })

    return registry, pd.DataFrame(rows), oof


# --------------------------------------------------------------------------- #
# New feature: fine_grained flag
# --------------------------------------------------------------------------- #
def add_fine_grained_flag(df):
    """The only genuinely new column requested (design.md §11.1) -- everything
    else (clay/silt/sand/gravel/ip) already exists as psd_fraction_clay/silt/
    sand/gravel and atterberg_plasticity_index via helper_functions.py."""
    df["fine_grained"] = (
        (df["psd_fraction_clay"] + df["psd_fraction_silt"]) > 15
    ).astype(int)
    return df


def base_feature_engineering_v7(df, H):
    df = V6.base_feature_engineering(df, H)
    df = add_fine_grained_flag(df)
    return df


# --------------------------------------------------------------------------- #
# KNN candidates -- a small explicit sweep, appended to v6's default model set
# --------------------------------------------------------------------------- #
KNN_GRID = [
    dict(n_neighbors=k, weights=w, metric=m)
    for k in (3, 5, 7, 9, 11, 15)
    for w in ("uniform", "distance")
    for m in ("euclidean", "manhattan")
]


def build_knn_configs():
    """(label, estimator, scaled=True) tuples -- KNN is scale-sensitive, unlike
    the tree-based candidates in v6.base_model_configs (design.md §11.4)."""
    configs = []
    for params in KNN_GRID:
        label = f"KNN[{V6._short(params)}]"
        configs.append((label, KNeighborsRegressor(**params), True))
    return configs


def base_model_configs_v7(args):
    return V6.base_model_configs(args) + build_knn_configs()


# --------------------------------------------------------------------------- #
# Blend search: does adding KNN's OOF predictions to the best non-KNN
# candidate improve on the non-KNN candidate alone?
# --------------------------------------------------------------------------- #
def best_blend_weight(y_true, pred_a, pred_b, steps=21):
    """Grid-search a fixed blend weight alpha in [0, 1] minimizing NMAE for
    alpha*pred_a + (1-alpha)*pred_b. Mirrors v5's fixed-weight blend approach
    (plan.md §11.4) rather than fitting a meta-learner, since only two OOF
    arrays are being combined here."""
    best_alpha, best_score = 0.0, np.inf
    for alpha in np.linspace(0.0, 1.0, steps):
        blend = alpha * pred_a + (1 - alpha) * pred_b
        score = V6.nmae(y_true, blend)
        if score < best_score:
            best_alpha, best_score = alpha, score
    return best_alpha, best_score


def main(args):
    import os
    os.makedirs(args.report_dir, exist_ok=True)
    log_path = args.log if os.path.isabs(args.log) else os.path.join(args.report_dir, args.log)
    logger = V6.setup_logger(log_path)
    logger.info("Proctor v7 (KNN diversity) run")
    logger.info("seed=%d  folds=%d  data_dir=%s", args.seed, args.folds, args.data_dir)

    H = V6.load_helpers(args.helpers_dir)
    H.CFG.seed_everything(args.seed)

    import pandas as pd
    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    logger.info("train shape=%s  test shape=%s", train.shape, test.shape)

    train = base_feature_engineering_v7(train, H)
    test = base_feature_engineering_v7(test, H)

    y_base = train[V6.TARGETS].copy()
    X_base = train.drop(columns=V6.TARGETS)

    exclude = ["id", "atterberg_is_missing", "kf_is_missing", "loi_is_missing"]
    cols_for_imputation = V6.numeric_impute_columns(X_base, exclude)
    logger.info("base feature columns: %d (incl. fine_grained)", X_base.shape[1])

    imp_full = H.impute_missing_values(X_base, cols_for_imputation, target_cols=[],
                                        seed=args.seed)
    H.apply_fold_feature_engineering(imp_full)
    diag_exclude = exclude + [c for c in imp_full.columns
                               if imp_full[c].dropna().isin([0, 1]).all()]
    df_metrics, num_cols = H.run_distribution_diagnostics(imp_full, diag_exclude, [])
    scaler_groups = H.get_scaler_assignments(imp_full, df_metrics, num_cols)
    logger.info("scaler groups -> PowerTransformer:%d  RobustScaler:%d  StandardScaler:%d",
                len(scaler_groups[0]), len(scaler_groups[1]), len(scaler_groups[2]))

    model_configs = base_model_configs_v7(args)
    knn_labels = [label for label, _, _ in model_configs if label.startswith("KNN[")]
    logger.info("candidate models: %d non-KNN + %d KNN sweep configs",
                len(model_configs) - len(knn_labels), len(knn_labels))

    strata = soil_composition_strata(X_base)
    logger.info("\nSoil-composition strata for CV stratification (dominant fraction):\n%s",
                strata.value_counts().to_string())
    cv_splits = list(StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    ).split(X_base, strata))

    registry, results, oof = run_cv_pipeline_stratified(
        X_base, y_base, model_configs, cols_for_imputation, scaler_groups, H,
        cv_splits, seed=args.seed)

    fold_local_summary = (results.groupby("Modell")["NMAE"].agg(["mean", "std"]).sort_values("mean"))
    logger.info("\n=== Per-fold NMAE (v6.run_cv_pipeline's own IQR-per-fold table -- "
                "reference only, NOT used for model selection) ===\n%s",
                fold_local_summary.round(4).to_string())

    # v6.run_cv_pipeline scores each fold with an IQR computed from that fold's
    # own ~40-row y_va subset, not the fixed global IQR that plan.md §6 /
    # design.md §5.2 specify ("computed once from all available training
    # targets... do not recompute per fold"). Score every candidate's full OOF
    # array against the fixed global IQR instead -- this is what decides
    # best_knn / best_non_knn and the blend below.
    oof_nmae = pd.Series({label: V6.nmae(y_base.values, oof[label]) for label in oof},
                         name="NMAE").sort_values()
    logger.info("\n=== OOF NMAE, fixed global IQR (used for model selection) ===\n%s",
                oof_nmae.round(4).to_string())

    best_knn = oof_nmae.loc[oof_nmae.index.isin(knn_labels)].index[0]
    best_non_knn = oof_nmae.loc[~oof_nmae.index.isin(knn_labels)].index[0]
    logger.info("\nBest standalone KNN config: %s  (OOF NMAE %.4f)",
                best_knn, oof_nmae[best_knn])
    logger.info("Best non-KNN config (v6 candidates): %s  (OOF NMAE %.4f)",
                best_non_knn, oof_nmae[best_non_knn])

    alpha, blend_score = best_blend_weight(y_base.values, oof[best_non_knn], oof[best_knn])
    logger.info(
        "\nBlend search: %.2f*%s + %.2f*%s -> NMAE %.4f (vs. best-non-KNN-alone %.4f)",
        alpha, best_non_knn, 1 - alpha, best_knn, blend_score, oof_nmae[best_non_knn])

    ship_blend = blend_score < oof_nmae[best_non_knn]
    logger.info("KNN blend %s the best non-KNN candidate.",
                "BEATS" if ship_blend else "does NOT beat")

    rho_s_tr = X_base["grain_density_g_cm3"].fillna(2.65).values
    chosen_oof = (alpha * oof[best_non_knn] + (1 - alpha) * oof[best_knn]
                  if ship_blend else oof[best_non_knn])
    V6.evaluate_and_log(logger, y_base.values, chosen_oof, V6.TARGETS, H, rho_s_tr,
                        args.report_dir, args.n_bins)

    pred_non_knn = V6.predict_registry(test, best_non_knn, registry, cols_for_imputation, H)
    if ship_blend:
        pred_knn = V6.predict_registry(test, best_knn, registry, cols_for_imputation, H)
        pred = alpha * pred_non_knn + (1 - alpha) * pred_knn
    else:
        pred = pred_non_knn
    mdd, owc = pred[:, 0], np.clip(pred[:, 1], 0, None)

    rho_s = test["grain_density_g_cm3"].fillna(2.65).values
    mdd = np.minimum(mdd, H.calc_satline(owc, rho_s) * 0.999)

    out = pd.DataFrame({
        "id": test["id"].values,
        "proctor_owc_pct": np.round(owc, 3),
        "proctor_mdd_g_cm3": np.round(mdd, 4),
    })
    out.to_csv(args.out, index=False)
    logger.info("\nWrote %d predictions -> %s", len(out), args.out)
    logger.info("Run log saved -> %s", log_path)
    return out, oof_nmae


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="../data")
    p.add_argument("--helpers_dir", default="..")
    p.add_argument("--out", default="submission_v7.csv")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--pinn_epochs", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", default="proctor_v7_run.log")
    p.add_argument("--report_dir", default="../docs/v7_logs")
    p.add_argument("--n_bins", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
