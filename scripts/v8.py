"""
Physics-Informed Neural Network (PINN) for the LeiGS 2026 Proctor Prediction Challenge.

Predicts the two Proctor compaction targets from soil-classification features:
    - proctor_owc_pct  (optimum water content, wopt, %)
    - proctor_mdd_g_cm3 (maximum dry density, rho_Pr, g/cm^3)

The "physics" is the Zero-Air-Voids (saturation) line. For a soil with grain
(particle) density rho_s, the dry density at water content w (%) cannot exceed the
fully-saturated value:

        rho_d_sat(w) = rho_s / (1 + (w/100) * rho_s / rho_w),   rho_w = 1.0 g/cm^3

A real Proctor optimum sits *below* this line (some air voids remain), with degree of
saturation Sr typically ~0.7-0.95. The loss combines the data fit with several
physics conditions (each with its own weight so it can be tuned or switched off):

  (1) saturation bound  — predicted MDD may not exceed the ZAV line (hard).
  (2) degree-of-saturation band — implied Sr kept in a plausible range (soft).
  (3) rho_d < rho_s     — dry density stays below the solid grain density (hard).
  (4) range boxes       — OWC and MDD kept inside physically sane limits (soft).
  (5) air-voids bound    — n_a at the optimum stays below ~12% (soft).
  (6) monotonicity priors (optional) — MDD should not increase with fines content,
      and OWC should not decrease with the liquid limit (soft gradient penalties).

These constraints inject domain knowledge, improve generalization on unseen soils, and
keep predictions physically admissible.

Usage:
    python proctor_pinn.py --data_dir /path/to/csvs --out submission.csv

Expects train.csv (with the two target columns) and test.csv (with an `id` column).
Dependencies: torch, numpy, pandas, scikit-learn.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

RHO_W = 1.0          # density of water, g/cm^3
TARGETS = ["proctor_owc_pct", "proctor_mdd_g_cm3"]

# ----------------------------------------------------------------------------- #
# Feature engineering
# ----------------------------------------------------------------------------- #
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive geotechnically meaningful features from the raw columns."""
    df = df.copy()

    d10 = df.get("psd_size_at_d10_mm")
    d30 = df.get("psd_size_at_d30_mm")
    d60 = df.get("psd_size_at_d60_mm")

    # Coefficient of uniformity Cu = D60 / D10  (width/steepness of grading curve)
    if d10 is not None and d60 is not None:
        df["feat_Cu"] = d60 / d10.replace(0, np.nan)
    # Coefficient of curvature Cc = D30^2 / (D10 * D60)  (shape/continuity of grading)
    if d10 is not None and d30 is not None and d60 is not None:
        df["feat_Cc"] = (d30 ** 2) / (d10.replace(0, np.nan) * d60.replace(0, np.nan))

    # Plasticity index PI = LL - PL  (how clayey/workable a fine-grained soil is)
    ll = df.get("atterberg_liquid_limit_pct")
    pl = df.get("atterberg_plastic_limit_pct")
    if ll is not None and pl is not None:
        df["feat_PI"] = ll - pl

    # log-permeability (kf spans many orders of magnitude)
    kf = df.get("hyd_cond_kf_m_s")
    if kf is not None:
        df["feat_log_kf"] = np.log10(kf.clip(lower=1e-15))

    # log of characteristic grain sizes (they span orders of magnitude)
    for col in ["psd_size_at_d10_mm", "psd_size_at_d30_mm", "psd_size_at_d60_mm"]:
        if col in df.columns:
            df["feat_log_" + col] = np.log10(df[col].clip(lower=1e-6))

    return df


def build_feature_matrix(df: pd.DataFrame, feature_cols, medians):
    """Impute missing values with training medians and append missingness flags."""
    X = df[feature_cols].copy()
    miss = X.isna().astype(float)
    miss.columns = [c + "_isna" for c in feature_cols]
    X = X.fillna(medians)
    X = pd.concat([X, miss], axis=1)
    return X


# ----------------------------------------------------------------------------- #
# Model
# ----------------------------------------------------------------------------- #
class ProctorPINN(nn.Module):
    def __init__(self, n_in, hidden=(128, 128, 64), p_drop=0.1):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(p_drop)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 2)  # -> [owc_norm, mdd_norm]

    def forward(self, x):
        return self.head(self.body(x))


def saturation_density(w_pct, rho_s):
    """Zero-air-voids dry density (g/cm^3) at water content w (%) for grain density rho_s."""
    return rho_s / (1.0 + (w_pct / 100.0) * rho_s / RHO_W)


def load_model(path, device="cpu"):
    """Load a saved checkpoint and return (model, ckpt) ready for inference."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ProctorPINN(ckpt["n_in"], p_drop=ckpt["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict_df(df, model, ckpt, device="cpu"):
    """Predict OWC & MDD for a raw dataframe using a loaded model. Returns a DataFrame
    with id, proctor_owc_pct, proctor_mdd_g_cm3 (physically clipped to the saturation line)."""
    df = add_engineered_features(df)
    for c in ckpt["feature_cols"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    X = build_feature_matrix(df, ckpt["feature_cols"], ckpt["medians"]).values.astype(np.float32)
    X = (X - ckpt["x_mu"]) / ckpt["x_sd"]
    with torch.no_grad():
        pred = model(torch.tensor(X, device=device)).cpu().numpy() * ckpt["y_sd"] + ckpt["y_mu"]
    rho_s = df["grain_density_g_cm3"].fillna(2.65).values.astype(np.float32)
    owc = np.clip(pred[:, 0], 0, None)
    mdd = np.minimum(pred[:, 1], saturation_density(owc, rho_s) * 0.999)
    out = pd.DataFrame({"proctor_owc_pct": np.round(owc, 3),
                        "proctor_mdd_g_cm3": np.round(mdd, 4)})
    if "id" in df.columns:
        out.insert(0, "id", df["id"].values)
    return out


# ----------------------------------------------------------------------------- #
# Training
# ----------------------------------------------------------------------------- #
def nmae(y_true, y_pred, iqr):
    """Mean column-wise IQR-normalized MAE (the competition metric)."""
    mae = np.abs(y_true - y_pred).mean(axis=0)
    return float((mae / iqr).mean())


def per_target_metrics(y_true, y_pred):
    """Per-target MAE, RMSE, and R^2 (each a length-2 array [owc, mdd])."""
    err = y_true - y_pred
    mae = np.abs(err).mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    ss_res = (err ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot == 0, 1e-12, ss_tot)
    return mae, rmse, r2


def bootstrap_metrics(y_true, y_pred, iqr, n_boot=1000, seed=42):
    """Bootstrap 95% confidence intervals for NMAE, MAE, RMSE, and R^2.

    Resamples the (y_true, y_pred) rows with replacement ``n_boot`` times and
    recomputes each metric, giving an uncertainty band — valuable on a small
    laboratory dataset where a single point estimate can be noisy. NMAE uses the
    fixed IQR from the full data (the metric's normalizer is defined on the true
    target distribution). Returns {metric: (mean, lo95, hi95)}, where per-target
    metrics carry length-2 arrays [owc, mdd].
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc = {"nmae": [], "mae": [], "rmse": [], "r2": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        acc["nmae"].append(nmae(yt, yp, iqr))
        m, r, rr = per_target_metrics(yt, yp)
        acc["mae"].append(m); acc["rmse"].append(r); acc["r2"].append(rr)

    def ci(vals):
        a = np.array(vals)
        return a.mean(axis=0), np.percentile(a, 2.5, axis=0), np.percentile(a, 97.5, axis=0)

    return {k: ci(v) for k, v in acc.items()}


def plot_metric_curves(history, target_names, path):
    """Plot validation MAE, RMSE, and R^2 vs. epoch (one panel each, per target)."""
    ep = history["epoch"]
    units = {"proctor_owc_pct": "%", "proctor_mdd_g_cm3": "g/cm³"}
    colors = ["#2c5d88", "#b84a39"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=120)

    for k, (key, ylabel) in enumerate(
            [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²")]):
        ax = axes[k]
        arr = np.array(history[key])  # shape (n_evals, 2)
        for j, name in enumerate(target_names):
            u = units.get(name, "")
            lbl = f"{name}" + (f" [{u}]" if u and key != "r2" else "")
            ax.plot(ep, arr[:, j], marker="o", ms=3, color=colors[j % 2], label=lbl)
        ax.set_title(f"Validation {ylabel} vs. epoch", fontweight="bold")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=8)
        if key == "r2":
            ax.set_ylim(top=1.02)

    fig.suptitle("PINN training — validation metrics", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    # Feature columns = everything except targets and id, that exists in both frames.
    drop = set(TARGETS + ["id"])
    feature_cols = [c for c in train_df.columns
                    if c not in drop and c in test_df.columns
                    and pd.api.types.is_numeric_dtype(train_df[c])]

    # Cast boolean-like columns to numeric if present.
    for c in feature_cols:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

    medians = train_df[feature_cols].median()
    Xtr_df = build_feature_matrix(train_df, feature_cols, medians)
    feat_names = list(Xtr_df.columns)
    Xtr = Xtr_df.values.astype(np.float32)
    Xte = build_feature_matrix(test_df, feature_cols, medians).values.astype(np.float32)

    # Column indices used by the optional monotonicity conditions. Standardization
    # is a positive scaling, so gradient signs are the same in standardized space.
    mono_fines_idx = [i for i, c in enumerate(feat_names)
                      if "passing_0_063" in c and not c.endswith("_isna")]
    mono_ll_idx = [i for i, c in enumerate(feat_names)
                   if "liquid_limit" in c and not c.endswith("_isna")]

    Y = train_df[TARGETS].values.astype(np.float32)            # [owc, mdd]
    rho_s_tr = train_df["grain_density_g_cm3"].fillna(2.65).values.astype(np.float32)
    rho_s_te = test_df["grain_density_g_cm3"].fillna(2.65).values.astype(np.float32)

    # Standardize features and targets.
    x_mu, x_sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - x_mu) / x_sd
    Xte = (Xte - x_mu) / x_sd
    y_mu, y_sd = Y.mean(0), Y.std(0) + 1e-8
    Yn = (Y - y_mu) / y_sd

    iqr = np.subtract(*np.percentile(Y, [75, 25], axis=0))     # for the NMAE metric

    # ---- Stratified hold-out validation split ----
    # The targets are continuous, so we stratify on quantile bins of the density
    # target (MDD). This makes the validation set span the full target range
    # (no fold that accidentally omits the very dense or very loose soils),
    # giving a more representative NMAE/R^2 estimate than a plain random split.
    strat_target = Y[:, 1]                                      # column 1 = MDD
    n_strata = int(min(args.val_strata, max(2, len(Y) // 10)))
    edges = np.unique(np.quantile(strat_target, np.linspace(0, 1, n_strata + 1)))
    strata = np.digitize(strat_target, edges[1:-1])            # bin index per sample
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac,
                                     random_state=args.seed)
        tr_idx, val_idx = next(sss.split(np.zeros(len(Y)), strata))
        print(f"Stratified hold-out: {len(tr_idx)} train / {len(val_idx)} val "
              f"({n_strata} density strata)")
    except ValueError:
        # too few samples per stratum -> fall back to a plain random split
        idx = np.random.permutation(len(Y))
        n_val = max(1, int(args.val_frac * len(Y)))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        print(f"Stratification not feasible -> random hold-out "
              f"({len(tr_idx)} train / {len(val_idx)} val)")

    def to_t(a):
        return torch.tensor(a, device=device)

    Xtr_t, Yn_t = to_t(Xtr), to_t(Yn)
    rho_t = to_t(rho_s_tr)
    y_mu_t, y_sd_t = to_t(y_mu), to_t(y_sd)

    model = ProctorPINN(Xtr.shape[1], p_drop=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    tr_mask = torch.tensor(tr_idx, device=device)
    va_mask = torch.tensor(val_idx, device=device)

    conditions = ["data(Huber)", f"saturation-bound(w={args.w_bound})",
                  f"Sr-band(w={args.w_sr})", f"rho_d<rho_s(w={args.w_rho})",
                  f"range-box(w={args.w_range})", f"air-voids<={args.na_max}(w={args.w_na})"]
    if args.w_mono > 0 and (mono_fines_idx or mono_ll_idx):
        conditions.append(f"monotonicity(w={args.w_mono})")
    print("Physics conditions active:", ", ".join(conditions))

    best_val, best_state = np.inf, None
    history = {"epoch": [], "mae": [], "rmse": [], "r2": [], "nmae": []}
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        pred_n = model(Xtr_t[tr_mask])

        # ---- data loss (Huber on standardized targets, robust to outliers) ----
        data_loss = nn.functional.smooth_l1_loss(pred_n, Yn_t[tr_mask])

        # ---- physics loss: de-standardize to physical units ----
        pred_phys = pred_n * y_sd_t + y_mu_t
        owc_pred = pred_phys[:, 0].clamp(min=0.0)
        mdd_pred = pred_phys[:, 1]
        rho_s_b = rho_t[tr_mask]
        rho_sat = rho_s_b / (1.0 + (owc_pred / 100.0) * rho_s_b / RHO_W)

        # (1) hard bound: MDD must not exceed the saturation line
        violation = torch.relu(mdd_pred - rho_sat)
        phys_bound = (violation ** 2).mean()

        # (2) soft: implied degree of saturation Sr should sit in a plausible band.
        #     Using void ratio e from rho_d:  e = rho_s/rho_d - 1
        #     Sr = w * rho_s / (e * rho_w * 100)
        e = (rho_s_b / mdd_pred.clamp(min=1e-3)) - 1.0
        Sr = (owc_pred * rho_s_b) / (e.clamp(min=1e-3) * RHO_W * 100.0)
        sr_lo, sr_hi = 0.6, 1.0
        phys_sr = (torch.relu(sr_lo - Sr) + torch.relu(Sr - sr_hi)).pow(2).mean()

        # (3) fundamental bound: dry density must stay below the solid grain
        #     density (you can't pack looser grains denser than solid rock).
        phys_rho = torch.relu(mdd_pred - rho_s_b).pow(2).mean()

        # (4) plausible range boxes — keep predictions inside physically sane
        #     limits so the model can't extrapolate to nonsense on odd soils.
        #     (% and g/cm^3 kept in separate terms so units don't mix.)
        owc_lo, owc_hi = args.owc_range
        mdd_lo, mdd_hi = args.mdd_range
        phys_range = (torch.relu(owc_lo - owc_pred).pow(2).mean()
                      + torch.relu(owc_pred - owc_hi).pow(2).mean()
                      + torch.relu(mdd_lo - mdd_pred).pow(2).mean()
                      + torch.relu(mdd_pred - mdd_hi).pow(2).mean())

        # (5) air-voids upper bound: at a real Proctor optimum a few percent of
        #     air remains, but not too much. n_a = 1 - rho_d/rho_s - (w/100)*rho_d/rho_w.
        na = 1.0 - (mdd_pred / rho_s_b) - (owc_pred / 100.0) * mdd_pred / RHO_W
        phys_na = torch.relu(na - args.na_max).pow(2).mean()

        loss = (data_loss
                + args.w_bound * phys_bound
                + args.w_sr * phys_sr
                + args.w_rho * phys_rho
                + args.w_range * phys_range
                + args.w_na * phys_na)

        # (6) optional monotonicity priors (gradient-sign penalties): established
        #     geotechnical trends — MDD should not rise with fines content, and
        #     OWC should not fall as the liquid limit rises. Enforced softly by
        #     penalizing the wrong-signed input gradient.
        if args.w_mono > 0 and (mono_fines_idx or mono_ll_idx):
            xb = Xtr_t[tr_mask].detach().clone().requires_grad_(True)
            out = model(xb) * y_sd_t + y_mu_t
            g_mdd = torch.autograd.grad(out[:, 1].sum(), xb, create_graph=True)[0]
            g_owc = torch.autograd.grad(out[:, 0].sum(), xb, create_graph=True)[0]
            mono = out.new_zeros(())
            for j in mono_fines_idx:          # MDD ↑ with fines is discouraged
                mono = mono + torch.relu(g_mdd[:, j]).pow(2).mean()
            for j in mono_ll_idx:             # OWC ↓ with liquid limit is discouraged
                mono = mono + torch.relu(-g_owc[:, j]).pow(2).mean()
            loss = loss + args.w_mono * mono

        loss.backward()
        opt.step()
        sched.step()

        # ---- validation on the competition metric ----
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                vp = model(Xtr_t[va_mask]).cpu().numpy() * y_sd + y_mu
            vy = Y[val_idx]
            score = nmae(vy, vp, iqr)
            mae_v, rmse_v, r2_v = per_target_metrics(vy, vp)
            history["epoch"].append(epoch + 1)
            history["mae"].append(mae_v)
            history["rmse"].append(rmse_v)
            history["r2"].append(r2_v)
            history["nmae"].append(score)
            if score < best_val:
                best_val = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"epoch {epoch+1:4d} | data {data_loss.item():.4f} "
                  f"| bound {phys_bound.item():.5f} | Sr {phys_sr.item():.5f} "
                  f"| val NMAE {score:.4f} | R2 {np.round(r2_v, 3)}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest validation NMAE: {best_val:.4f}")

    # ---- graph validation MAE / RMSE / R^2 over training ----
    if history["epoch"]:
        plot_metric_curves(history, TARGETS, args.metrics_plot)
        # also report final per-target metrics of the best model
        model.eval()
        with torch.no_grad():
            vp = model(Xtr_t[va_mask]).cpu().numpy() * y_sd + y_mu
        fmae, frmse, fr2 = per_target_metrics(Y[val_idx], vp)
        for j, name in enumerate(TARGETS):
            print(f"  {name:22s}  MAE={fmae[j]:.4f}  RMSE={frmse[j]:.4f}  R2={fr2[j]:.3f}")
        print(f"Saved metric curves -> {args.metrics_plot}")

        # ---- bootstrap 95% confidence intervals on the validation metrics ----
        if args.bootstrap > 0:
            bs = bootstrap_metrics(Y[val_idx], vp, iqr, args.bootstrap, args.seed)
            mean, lo, hi = bs["nmae"]
            print(f"\nBootstrap 95% CIs (B={args.bootstrap} resamples):")
            print(f"  NMAE                    {mean:.4f}  [{lo:.4f}, {hi:.4f}]")
            for key in ("mae", "rmse", "r2"):
                mean, lo, hi = bs[key]
                for j, name in enumerate(TARGETS):
                    print(f"  {key.upper():4s} {name:18s} {mean[j]:.4f}  "
                          f"[{lo[j]:.4f}, {hi[j]:.4f}]")

    # ---- save the trained model + everything needed to reuse it ----
    # We store the weights AND the preprocessing (feature list, medians, scalers) so the
    # checkpoint is fully self-contained: load_model() below can rebuild the exact pipeline
    # without the training data.
    ckpt = {
        "state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "medians": medians,                       # pandas Series, indexed by feature name
        "x_mu": x_mu, "x_sd": x_sd,                # feature standardization
        "y_mu": y_mu, "y_sd": y_sd,                # target standardization
        "n_in": Xtr.shape[1],
        "dropout": args.dropout,
        "targets": TARGETS,
        "val_nmae": best_val,
    }
    torch.save(ckpt, args.model_out)
    print(f"Saved model -> {args.model_out}  (val NMAE {best_val:.4f})")

    # ---- predict test set ----
    model.eval()
    with torch.no_grad():
        te_pred = model(to_t(Xte)).cpu().numpy() * y_sd + y_mu

    # Enforce the physical bound at inference too (clip MDD to saturation line).
    owc_out = np.clip(te_pred[:, 0], 0, None)
    rho_sat_te = saturation_density(owc_out, rho_s_te)
    mdd_out = np.minimum(te_pred[:, 1], rho_sat_te * 0.999)

    sub = pd.DataFrame({
        "id": test_df["id"].values,
        "proctor_owc_pct": np.round(owc_out, 3),
        "proctor_mdd_g_cm3": np.round(mdd_out, 4),
    })
    sub.to_csv(args.out, index=False)
    print(f"Wrote {len(sub)} predictions -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=".", help="folder containing train.csv & test.csv")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--model_out", default="proctor_pinn.pt", help="path to save the trained model")
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--w_bound", type=float, default=5.0, help="weight of saturation-bound loss")
    p.add_argument("--w_sr", type=float, default=0.5, help="weight of degree-of-saturation loss")
    p.add_argument("--w_rho", type=float, default=5.0,
                   help="weight of the rho_d < grain-density bound")
    p.add_argument("--w_range", type=float, default=1.0,
                   help="weight of the plausible OWC/MDD range boxes")
    p.add_argument("--w_na", type=float, default=0.5,
                   help="weight of the air-voids upper-bound condition")
    p.add_argument("--na_max", type=float, default=0.12,
                   help="max plausible air-void fraction at the optimum (0.12 = 12%%)")
    p.add_argument("--owc_range", type=float, nargs=2, default=[2.0, 40.0],
                   help="plausible optimum water content range [low high] in %%")
    p.add_argument("--mdd_range", type=float, nargs=2, default=[1.2, 2.4],
                   help="plausible max dry density range [low high] in g/cm^3")
    p.add_argument("--w_mono", type=float, default=0.0,
                   help="weight of monotonicity priors (0 = off; try ~0.1 to enable)")
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--metrics_plot", default="pinn_metrics.png",
                   help="output path for the MAE / RMSE / R^2 training graph")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="validation fraction for the stratified hold-out split")
    p.add_argument("--val_strata", type=int, default=5,
                   help="number of density-based strata for StratifiedShuffleSplit")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="bootstrap resamples for metric 95%% CIs (0 to disable)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())