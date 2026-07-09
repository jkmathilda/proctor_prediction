"""
Physics-consistent ML pipeline for the LeiGS 2026 Proctor Prediction Challenge.

Paradigm (see proctor_ml_paradigm.md):
  1. Predict wopt directly (L1 models).
  2. Predict logit(Sr_opt) (degree of saturation at optimum) with predicted
     wopt as a chained feature -> reconstruct rho_Pr analytically. This
     guarantees rho_Pr <= zero-air-voids line for the sample's own rho_s.
  3. Blend physics-reconstructed rho_Pr with a direct rho_Pr model
     (blend weight alpha chosen by out-of-fold NMAE), then soft-clip to ZAV.
  4. Repeated K-fold CV scored with the exact competition NMAE (MAE / IQR).

Only numpy / pandas / scikit-learn required. Trees use native-NaN
HistGradientBoosting; linear models get median imputation + missing flags.

Usage:  python proctor_pipeline.py [train.csv] [test.csv] [submission.csv]
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

RNG = 42
RHO_W = 1.0          # g/cm3
RHO_S_DEFAULT = 2.65  # quartzitic median (challenge statement)
T_W, T_R = "proctor_owc_pct", "proctor_mdd_g_cm3"

# ----------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------
def zav_density(w_pct, rho_s):
    """Zero-air-voids (saturation line) dry density."""
    return rho_s * RHO_W / (RHO_W + (w_pct / 100.0) * rho_s)

def saturation_at_optimum(w_pct, rho_d, rho_s):
    """Degree of saturation Sr at the Proctor optimum."""
    e = rho_s / rho_d - 1.0
    return (w_pct / 100.0) * rho_s / (RHO_W * np.maximum(e, 1e-6))

def density_from_sr(w_pct, sr, rho_s):
    """Invert: rho_d from (wopt, Sr). Sr in (0,1] => rho_d <= ZAV."""
    return rho_s / (1.0 + (w_pct / 100.0) * rho_s / (sr * RHO_W))

def logit(p):
    p = np.clip(p, 0.02, 0.999)
    return np.log(p / (1 - p))

def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))

# ----------------------------------------------------------------------
# Feature engineering (geotechnically motivated; see methodology doc)
# ----------------------------------------------------------------------
PSD_PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98]

def build_features(df):
    f = pd.DataFrame(index=df.index)
    d = {}
    for p in PSD_PCTS:
        col = f"psd_size_at_d{p}_mm"
        if col in df:
            d[p] = df[col].astype(float)
            f[f"logD{p}"] = np.log10(d[p].clip(lower=1e-5))

    # Gradation / packing
    if 10 in d and 60 in d and 30 in d:
        cu = d[60] / d[10].clip(lower=1e-5)
        cc = d[30] ** 2 / (d[10] * d[60]).clip(lower=1e-9)
        f["logCu"] = np.log10(cu.clip(lower=1e-3))
        f["Cc"] = cc.clip(0, 20)
    logs = np.column_stack([np.log10(d[p].clip(lower=1e-5)) for p in PSD_PCTS if p in d])
    f["gsd_mean_log"] = logs.mean(axis=1)          # graphic mean
    f["gsd_sort_log"] = logs.std(axis=1)           # sorting / spread
    if 90 in d and 30 in d:
        f["slope_coarse"] = np.log10(d[90].clip(lower=1e-5)) - np.log10(d[30].clip(lower=1e-5))

    # Fraction features
    fines = df.get("psd_passing_0_063mm_pct")
    clay = df.get("psd_passing_0_002mm_pct")
    p2 = df.get("psd_passing_2mm_pct")
    if fines is not None:
        f["fines"] = fines
        if clay is not None:
            f["clay"] = clay
            f["silt"] = (fines - clay).clip(lower=0)
        if p2 is not None:
            f["sand"] = (p2 - fines).clip(lower=0)
            f["gravel"] = (100 - p2).clip(lower=0)

    # Plasticity
    ll = df.get("atterberg_liquid_limit_pct")
    pl = df.get("atterberg_plastic_limit_pct")
    if ll is not None and pl is not None:
        pi = ll - pl
        f["LL"], f["PL"], f["PI"] = ll, pl, pi
        f["is_plastic"] = pi.notna().astype(float)
        # Non-plastic soils: missing PI physically means PI = 0
        f["PI_phys"] = pi.fillna(0.0)
        if clay is not None:
            f["activity"] = pi / clay.clip(lower=1.0)
        # Empirical anchors as *features* (Gurtug-Sridharan-type)
        f["wopt_emp_PL"] = 0.92 * pl
        f["wopt_emp_LL"] = 0.30 * ll

    # Organics, hydraulics, test conditions
    loi = df.get("loss_on_ignition_pct")
    if loi is not None:
        f["LOI"] = loi
        if fines is not None:
            f["LOI_x_fines"] = loi * fines / 100.0
    kf = df.get("hyd_cond_kf_m_s")
    if kf is not None:
        f["log_kf"] = np.log10(kf.clip(lower=1e-13))
        if 10 in d:
            f["hazen_resid"] = f["log_kf"] - 2 * np.log10(d[10].clip(lower=1e-5))
    if "hyd_cond_hyd_gradient" in df:
        f["hyd_grad"] = df["hyd_cond_hyd_gradient"]
    if "proctor_diam_mm" in df:
        f["mold_B"] = (df["proctor_diam_mm"] > 120).astype(float)
    if "psd_has_sedimentation" in df:
        f["has_sedim"] = df["psd_has_sedimentation"].astype(bool).astype(float)
    f["rho_s"] = df.get("grain_density_g_cm3", pd.Series(RHO_S_DEFAULT, index=df.index)) \
                   .fillna(RHO_S_DEFAULT)
    return f

# ----------------------------------------------------------------------
# Models: L1 gradient boosting (native NaN) + ridge (imputed) -> average
# ----------------------------------------------------------------------
def make_gbm(seed):
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_depth=3, max_iter=150,
        learning_rate=0.08, min_samples_leaf=8, l2_regularization=1.0,
        max_leaf_nodes=15, random_state=seed)

def make_ridge():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(), Ridge(alpha=10.0))

def fit_predict(X_tr, y_tr, X_te, seed=RNG):
    """Blend of 2-seed GBM + ridge."""
    preds = []
    for s in (seed, seed + 1):
        m = make_gbm(s).fit(X_tr, y_tr)
        preds.append(m.predict(X_te))
    r = make_ridge().fit(X_tr, y_tr)
    preds.append(r.predict(X_te))
    return 0.7 * np.mean(preds[:2], axis=0) + 0.3 * preds[2]

def oof_predict(X, y, n_splits=4, seed=RNG):
    """Out-of-fold predictions (used for the chained wopt feature)."""
    oof = np.zeros(len(y))
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        oof[va] = fit_predict(X.iloc[tr], y[tr], X.iloc[va], seed)
    return oof

# ----------------------------------------------------------------------
# Competition metric
# ----------------------------------------------------------------------
def nmae(y_true_w, y_pred_w, y_true_r, y_pred_r, iqr_w, iqr_r):
    return 0.5 * (np.mean(np.abs(y_true_w - y_pred_w)) / iqr_w +
                  np.mean(np.abs(y_true_r - y_pred_r)) / iqr_r)

# ----------------------------------------------------------------------
# Full paradigm: chained physics model, evaluated fold-honestly
# ----------------------------------------------------------------------
def predict_targets(X_tr, w_tr, r_tr, rho_s_tr, X_te, rho_s_te,
                    alpha_grid=(0.0, 0.25, 0.5, 0.65, 0.8, 1.0), seed=RNG):
    """Train on (X_tr, targets), predict (wopt, rhoPr) for X_te.
    alpha = weight of the physics-reconstructed density in the blend,
    selected on inner OOF predictions of the *training* data only."""
    # -- target reparameterization
    sr_tr = saturation_at_optimum(w_tr, r_tr, rho_s_tr)
    z_tr = logit(sr_tr)

    # -- Model A: wopt (plus inner OOF for leakage-free chaining/blending)
    w_oof = oof_predict(X_tr, w_tr, seed=seed)
    w_te = fit_predict(X_tr, w_tr, X_te, seed)

    # -- Model B: logit(Sr) with chained wopt feature
    Xb_tr = X_tr.copy(); Xb_tr["wopt_hat"] = w_oof
    Xb_te = X_te.copy(); Xb_te["wopt_hat"] = w_te
    z_oof = oof_predict(Xb_tr, z_tr, seed=seed + 10)
    z_te = fit_predict(Xb_tr, z_tr, Xb_te, seed + 10)
    r_phys_oof = density_from_sr(w_oof, inv_logit(z_oof), rho_s_tr)
    r_phys_te = density_from_sr(w_te, inv_logit(z_te), rho_s_te)

    # -- direct density model
    r_dir_oof = oof_predict(X_tr, r_tr, seed=seed + 20)
    r_dir_te = fit_predict(X_tr, r_tr, X_te, seed + 20)

    # -- pick alpha on inner OOF
    iqr_r = np.subtract(*np.percentile(r_tr, [75, 25]))
    best = min(alpha_grid, key=lambda a: np.mean(
        np.abs(r_tr - (a * r_phys_oof + (1 - a) * r_dir_oof))) / iqr_r)
    r_te = best * r_phys_te + (1 - best) * r_dir_te

    # -- physics sanity gate
    w_te = np.clip(w_te, 3.0, 30.0)
    r_te = np.minimum(r_te, 0.99 * zav_density(w_te, rho_s_te))
    r_te = np.clip(r_te, 1.3, 2.4)
    return w_te, r_te, best

# ----------------------------------------------------------------------
def main(train_path="train.csv", test_path="test.csv", out_path="submission.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X = build_features(train)
    X_te = build_features(test)
    X_te = X_te.reindex(columns=X.columns)          # align columns
    w = train[T_W].values
    r = train[T_R].values
    rho_s = X["rho_s"].values
    rho_s_te = X_te["rho_s"].values
    iqr_w = np.subtract(*np.percentile(w, [75, 25]))
    iqr_r = np.subtract(*np.percentile(r, [75, 25]))

    # ---------------- repeated K-fold CV with the competition metric
    import os
    n_reps = int(os.environ.get("N_REPS", 3))
    scores, alphas = [], []
    for rep in range(n_reps):
        kf = KFold(5, shuffle=True, random_state=100 + rep)
        pw, pr = np.zeros(len(w)), np.zeros(len(r))
        for tr, va in kf.split(X):
            pw[va], pr[va], a = predict_targets(
                X.iloc[tr], w[tr], r[tr], rho_s[tr],
                X.iloc[va], rho_s[va], seed=RNG + rep)
            alphas.append(a)
        scores.append(nmae(w, pw, r, pr, iqr_w, iqr_r))
    print(f"CV NMAE: {np.mean(scores):.4f} +/- {np.std(scores):.4f} "
          f"(reps: {[f'{s:.4f}' for s in scores]})")
    print(f"physics-blend alpha (median across folds): {np.median(alphas):.2f}")

    # ---------------- fit on all training data, predict test
    w_hat, r_hat, a = predict_targets(X, w, r, rho_s, X_te, rho_s_te)
    print(f"final alpha={a:.2f} | wopt in [{w_hat.min():.1f},{w_hat.max():.1f}]%"
          f" | rhoPr in [{r_hat.min():.3f},{r_hat.max():.3f}] g/cm3")
    assert np.all(r_hat <= zav_density(w_hat, rho_s_te) + 1e-9), "ZAV violated!"
    sr_hat = saturation_at_optimum(w_hat, r_hat, rho_s_te)
    print(f"implied Sr at optimum: [{sr_hat.min():.2f},{sr_hat.max():.2f}]")

    pd.DataFrame({"id": test["id"], T_W: np.round(w_hat, 2),
                  T_R: np.round(r_hat, 3)}).to_csv(out_path, index=False)
    print(f"wrote {out_path}")

if __name__ == "__main__":
    main(*sys.argv[1:4])