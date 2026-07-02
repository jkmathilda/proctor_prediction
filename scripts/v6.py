"""
proctor_pipeline.py
===================
End-to-end driver for the LeiGS 2026 Proctor Prediction Challenge that wires the
organizers' ``helpers.py`` toolkit into a complete, leak-free modeling pipeline.

``helpers.py`` ships every building block (MICE imputation, domain feature
engineering, scaling, the NMAE metric, and the registry-based prediction
helpers) but no main loop that actually trains models and produces the fold
artifacts that ``predict_with_pipeline_registry`` / ``generate_oof_predictions``
consume. This module implements exactly that missing piece:

    load -> pre-CV feature engineering -> K-fold CV with fold-isolated
    (imputer + optional scaler + model) -> NMAE model selection ->
    registry-based ensemble prediction on test -> saturation-line clip ->
    submission.csv

It also registers the physics-informed neural net from ``proctor_pinn.py`` as a
candidate model via a scikit-learn-compatible wrapper, so the PINN competes
against the tree/linear baselines inside the same CV machinery. The PINN is
optional: if PyTorch is not installed, the pipeline simply skips it and runs the
sklearn models.

Run:
    python proctor_pipeline.py --data_dir /folder/with/csvs --helpers_dir .

Expects ``train.csv``, ``test.csv``, ``sample_submission.csv`` and ``helpers.py``.
"""

import argparse
import importlib.util
import logging
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless: helper plotting calls won't open windows
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

# Target order is fixed throughout: column 0 = MDD, column 1 = OWC.
TARGETS = ["proctor_mdd_g_cm3", "proctor_owc_pct"]


# --------------------------------------------------------------------------- #
# Official competition metric — NMAE (Mean Column-wise Normalized MAE)
# --------------------------------------------------------------------------- #
def nmae(y_true, y_pred):
    r"""Mean column-wise IQR-normalized mean absolute error (the leaderboard score).

    Implements the official LeiGS 2026 metric exactly:

        NMAE = (1/2) * sum_{j=1..2} [ ( (1/n) * sum_{i=1..n} |y_ij - yhat_ij| )
                                       / IQR(y_j) ]

    Each target column's MAE is divided by that column's interquartile range
    (IQR = Q75 - Q25 of the true values), so the two targets — despite very
    different ranges (MDD ~1.6-2.2 g/cm^3, OWC ~5-25 %) — contribute 50/50.
    Lower is better; 0.0 is error-free. Absolute (not squared) error keeps it
    robust to lab outliers. This is numerically identical to
    ``helpers.calculate_nmae``; it is duplicated here so the pipeline computes
    the exact leaderboard metric with no external dependency.

    Args:
        y_true: ground-truth array, shape (n_samples, n_targets).
        y_pred: predicted array, same shape.

    Returns:
        Macro-averaged NMAE across target columns (float).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    mae_per_col = np.mean(np.abs(y_true - y_pred), axis=0)
    q75, q25 = np.percentile(y_true, [75, 25], axis=0)
    iqr_per_col = np.where((q75 - q25) == 0, 1e-8, q75 - q25)
    return float(np.mean(mae_per_col / iqr_per_col))


# --------------------------------------------------------------------------- #
# helpers.py loader
# --------------------------------------------------------------------------- #
def load_helpers(helpers_dir):
    """Import the organizers' helpers.py from an arbitrary directory."""
    path = os.path.join(helpers_dir, "helper_functions.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"helper_functions.py not found at {path}")
    spec = importlib.util.spec_from_file_location("helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Physics-informed NN wrapped as a scikit-learn multi-output regressor
# --------------------------------------------------------------------------- #
class ProctorPINNRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper around the saturation-constrained PINN.

    Predicts [MDD, OWC]. The physics term penalizes any predicted MDD above the
    Zero-Air-Voids line for that sample's grain density. Operates on UNSCALED
    feature frames so it can read ``grain_density_g_cm3`` in physical units; it
    standardizes internally. Requires PyTorch.
    """

    def __init__(self, hidden=(128, 128, 64), dropout=0.1, lr=2e-3,
                 epochs=1500, w_bound=5.0, w_sr=0.5, seed=42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.w_bound = w_bound
        self.w_sr = w_sr
        self.seed = seed

    # -- helpers ----------------------------------------------------------- #
    def _to_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        # numpy path: reuse the column order captured during fit
        return pd.DataFrame(X, columns=self._columns_)

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self._columns_ = list(Xdf.columns)
        self._gd_col_ = ("grain_density_g_cm3" if "grain_density_g_cm3" in self._columns_
                         else None)
        rho_s = (Xdf[self._gd_col_].fillna(2.65).values.astype(np.float32)
                 if self._gd_col_ else np.full(len(Xdf), 2.65, np.float32))

        Xv = Xdf.values.astype(np.float32)
        yv = np.asarray(y, dtype=np.float32)
        if yv.ndim == 1:
            yv = yv[:, None]

        # standardize features + targets
        self._x_mu_, self._x_sd_ = Xv.mean(0), Xv.std(0) + 1e-8
        self._y_mu_, self._y_sd_ = yv.mean(0), yv.std(0) + 1e-8
        Xn = (Xv - self._x_mu_) / self._x_sd_
        Yn = (yv - self._y_mu_) / self._y_sd_

        layers, d = [], Xn.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(self.dropout)]
            d = h
        layers += [nn.Linear(d, 2)]
        net = nn.Sequential(*layers)

        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        Xt = torch.tensor(Xn)
        Yt = torch.tensor(Yn)
        rt = torch.tensor(rho_s)
        ymu = torch.tensor(self._y_mu_)
        ysd = torch.tensor(self._y_sd_)

        net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pn = net(Xt)
            data = nn.functional.smooth_l1_loss(pn, Yt)
            phys = pn * ysd + ymu                      # back to physical units
            owc = phys[:, 1].clamp(min=0.0)            # col 1 = OWC
            mdd = phys[:, 0]                            # col 0 = MDD
            rho_sat = rt / (1.0 + (owc / 100.0) * rt)  # ZAV line
            bound = torch.relu(mdd - rho_sat).pow(2).mean()
            e = (rt / mdd.clamp(min=1e-3)) - 1.0
            Sr = (owc * rt) / (e.clamp(min=1e-3) * 100.0)
            sr = (torch.relu(0.6 - Sr) + torch.relu(Sr - 1.0)).pow(2).mean()
            (data + self.w_bound * bound + self.w_sr * sr).backward()
            opt.step()
            sched.step()

        self._net_ = net
        return self

    def predict(self, X):
        import torch
        Xdf = self._to_frame(X)
        Xn = (Xdf.values.astype(np.float32) - self._x_mu_) / self._x_sd_
        self._net_.eval()
        with torch.no_grad():
            out = self._net_(torch.tensor(Xn)).numpy() * self._y_sd_ + self._y_mu_
        return out


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Pre-CV feature engineering (run once on the raw train/test frames)
# --------------------------------------------------------------------------- #
def base_feature_engineering(df, H):
    """C_U/C_C + DIN fractions + log-ratio PSD features, via helpers."""
    df = H.add_gradation_parameters(df)   # psd_C_U, psd_C_C
    df = H.prepare_features(df)            # fractions + log_diff_* features
    return df


def numeric_impute_columns(X, exclude):
    """All numeric feature columns (so MICE returns a complete matrix)."""
    cols = [c for c in X.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(X[c])]
    return cols


# --------------------------------------------------------------------------- #
# The missing piece: CV loop that builds the fold-artifact registry
# --------------------------------------------------------------------------- #
def run_cv_pipeline(X_base, y_base, model_configs, cols_for_imputation,
                    scaler_groups, H, n_splits=10, seed=42):
    """K-fold CV with per-fold isolated (imputer -> optional scaler -> model).

    Returns
    -------
    registry : dict[label] -> list of fold-artifact dicts
        Each artifact = {'fitted_imputer','fitted_scaler','fitted_model'} — the
        exact shape ``helpers.predict_with_pipeline_registry`` expects.
    results  : pandas.DataFrame  (one row per model x fold, with NMAE)
    oof      : dict[label] -> ndarray (n_samples, n_targets)
        Out-of-fold predictions: every row predicted by a model that never saw it
        in training. Used for honest metrics / confusion matrices.
    """
    feat_skew, feat_out, feat_std = scaler_groups
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    registry = {label: [] for label, _, _ in model_configs}
    oof = {label: np.full((len(X_base), y_base.shape[1]), np.nan)
           for label, _, _ in model_configs}
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_base)):
        X_tr, X_va = X_base.iloc[tr_idx].copy(), X_base.iloc[va_idx].copy()
        y_tr, y_va = y_base.iloc[tr_idx], y_base.iloc[va_idx]

        # missing indicators (must exist before fold feature engineering)
        for f in (X_tr, X_va):
            f["atterberg_is_missing"] = f["atterberg_liquid_limit_pct"].isnull().astype(int)
            f["kf_is_missing"] = f["hyd_cond_kf_m_s"].isnull().astype(int)
            f["loi_is_missing"] = f["loss_on_ignition_pct"].isnull().astype(int)

        # fold-isolated MICE: fit on train only, transform val
        imputer = H.get_default_mice_imputer(seed=seed)
        X_tr[cols_for_imputation] = imputer.fit_transform(X_tr[cols_for_imputation])
        X_va[cols_for_imputation] = imputer.transform(X_va[cols_for_imputation])

        # deterministic post-imputation feature engineering (in place)
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
                # PINN needs named, unscaled columns to read grain density
                Xtr_p, Xva_p = X_tr, X_va
            else:
                Xtr_p, Xva_p = X_tr.values, X_va.values

            model = clone(template)
            model.fit(Xtr_p, y_tr)
            preds = np.asarray(model.predict(Xva_p))

            score = nmae(y_va.values, preds)
            rows.append({"Modell": label, "Skaliert": scaled,
                         "Fold": fold, "NMAE": score})
            oof[label][va_idx] = preds
            registry[label].append({
                "fitted_imputer": imputer,
                "fitted_scaler": scaler,
                "fitted_model": model,
            })

    return registry, pd.DataFrame(rows), oof


def predict_registry(df_test, label, registry, cols_for_imputation, H):
    """Ensemble-average a config's fold models over the raw test frame.

    Mirrors helpers.predict_with_pipeline_registry, but also supports the PINN
    config (which consumes a named, unscaled DataFrame instead of .values).
    """
    preds = []
    for art in registry[label]:
        X = df_test.copy()
        X["atterberg_is_missing"] = X["atterberg_liquid_limit_pct"].isnull().astype(int)
        X["kf_is_missing"] = X["hyd_cond_kf_m_s"].isnull().astype(int)
        X["loi_is_missing"] = X["loss_on_ignition_pct"].isnull().astype(int)
        X[cols_for_imputation] = art["fitted_imputer"].transform(X[cols_for_imputation])
        H.apply_fold_feature_engineering(X)
        if art["fitted_scaler"] is not None:
            Xp = art["fitted_scaler"].transform(X)
        elif label == "PINN":
            Xp = X
        else:
            Xp = X.values
        preds.append(np.asarray(art["fitted_model"].predict(Xp)))
    return np.mean(preds, axis=0)


# --------------------------------------------------------------------------- #
# Hyperparameter tuning (random search over each model family)
# --------------------------------------------------------------------------- #
# Each family maps parameter name -> list of candidate values. Random search
# samples combinations from these; every sampled config becomes its own labelled
# candidate and is scored by the existing CV/NMAE machinery, so tuning needs no
# special selection code — the standard "best mean NMAE wins" logic handles it.
SEARCH_SPACES = {
    "ExtraTrees": dict(n_estimators=[300, 500, 800],
                       max_depth=[None, 12, 20, 30],
                       min_samples_leaf=[1, 2, 4],
                       max_features=["sqrt", 0.5, 1.0]),
    "HistGB": dict(learning_rate=[0.03, 0.05, 0.1],
                   max_iter=[200, 400, 600],
                   max_leaf_nodes=[15, 31, 63],
                   l2_regularization=[0.0, 0.5, 1.0]),
    "Ridge": dict(alpha=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]),
    "PINN": dict(hidden=[(128, 128, 64), (256, 128), (128, 64)],
                 dropout=[0.05, 0.1, 0.15],
                 lr=[1e-3, 2e-3],
                 w_bound=[2.0, 5.0, 10.0],
                 w_sr=[0.25, 0.5, 1.0]),
}


def _short(params):
    """Compact, readable stringification of a param dict for a config label."""
    def fmt(v):
        return "x".join(map(str, v)) if isinstance(v, tuple) else str(v)
    return ",".join(f"{k}={fmt(v)}" for k, v in params.items())


def _build_config(family, params, args):
    """Return a (label, estimator_template, use_scaler) tuple for one config."""
    label = f"{family}[{_short(params)}]"
    if family == "ExtraTrees":
        return (label, ExtraTreesRegressor(random_state=args.seed, n_jobs=-1,
                                           **params), False)
    if family == "HistGB":
        return (label, MultiOutputRegressor(
            HistGradientBoostingRegressor(random_state=args.seed, **params)), False)
    if family == "Ridge":
        return (label, Ridge(**params), True)
    if family == "PINN":
        return (label, ProctorPINNRegressor(seed=args.seed, epochs=args.pinn_epochs,
                                            **params), False)
    raise ValueError(f"unknown model family: {family}")


def base_model_configs(args):
    """The default (untuned) candidate set."""
    cfgs = [
        ("ExtraTrees", ExtraTreesRegressor(n_estimators=400, random_state=args.seed,
                                           n_jobs=-1), False),
        ("HistGB", MultiOutputRegressor(
            HistGradientBoostingRegressor(random_state=args.seed)), False),
        ("Ridge", Ridge(alpha=1.0), True),
    ]
    if torch_available():
        cfgs.append(("PINN", ProctorPINNRegressor(epochs=args.pinn_epochs,
                                                   seed=args.seed), False))
    return cfgs


def make_model_configs(args, logger):
    """Build the candidate list — either the defaults or a tuned random search."""
    if not args.tune:
        cfgs = base_model_configs(args)
        if not torch_available():
            logger.info("[info] PyTorch not available -> skipping PINN candidate.")
        logger.info("candidate models (untuned): %s",
                    [f"{l}{'(scaled)' if s else ''}" for l, _, s in cfgs])
        return cfgs

    families = ([f.strip() for f in args.tune_models.split(",") if f.strip()]
                if args.tune_models else list(SEARCH_SPACES))
    rng = np.random.default_rng(args.seed)
    cfgs = []
    for fam in families:
        if fam not in SEARCH_SPACES:
            logger.info("[warn] unknown family '%s' -> skipped", fam)
            continue
        if fam == "PINN" and not torch_available():
            logger.info("[info] PyTorch not available -> skipping PINN tuning.")
            continue
        space = SEARCH_SPACES[fam]
        seen, tries = set(), 0
        n_added = 0
        # sample distinct combinations (cap attempts to avoid infinite loops)
        while n_added < args.tune_iter and tries < args.tune_iter * 20:
            tries += 1
            params = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
            key = tuple(sorted((k, str(v)) for k, v in params.items()))
            if key in seen:
                continue
            seen.add(key)
            cfgs.append(_build_config(fam, params, args))
            n_added += 1
        logger.info("[tune] %s -> %d sampled configs", fam, n_added)

    logger.info("TUNING: %d total candidate configs across %d families",
                len(cfgs), len(families))
    return cfgs


# --------------------------------------------------------------------------- #
# Optuna (Bayesian / TPE) tuner backend — same search space, smarter sampling
# --------------------------------------------------------------------------- #
def precompute_folds(X_base, y_base, cols_for_imputation, H, n_splits, seed):
    """Run the per-fold preprocessing (missing indicators -> MICE -> feature
    engineering) ONCE and cache the results, so a tuner can evaluate many model
    configs without repeating the expensive imputation each time.

    Returns a list of (X_tr, X_va, y_tr, y_va) tuples — imputed & engineered,
    unscaled. Scaling (if a config needs it) is applied per-config downstream.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, va_idx in kf.split(X_base):
        X_tr, X_va = X_base.iloc[tr_idx].copy(), X_base.iloc[va_idx].copy()
        for f in (X_tr, X_va):
            f["atterberg_is_missing"] = f["atterberg_liquid_limit_pct"].isnull().astype(int)
            f["kf_is_missing"] = f["hyd_cond_kf_m_s"].isnull().astype(int)
            f["loi_is_missing"] = f["loss_on_ignition_pct"].isnull().astype(int)
        imputer = H.get_default_mice_imputer(seed=seed)
        X_tr[cols_for_imputation] = imputer.fit_transform(X_tr[cols_for_imputation])
        X_va[cols_for_imputation] = imputer.transform(X_va[cols_for_imputation])
        for f in (X_tr, X_va):
            H.apply_fold_feature_engineering(f)
        folds.append((X_tr, X_va, y_base.iloc[tr_idx], y_base.iloc[va_idx]))
    return folds


def evaluate_config_on_folds(template, scaled, folds, scaler_groups, H):
    """Mean CV NMAE for one model config over the cached folds."""
    feat_skew, feat_out, feat_std = scaler_groups
    is_pinn = isinstance(template, ProctorPINNRegressor)
    scores = []
    for X_tr, X_va, y_tr, y_va in folds:
        if scaled:
            valid = set(X_tr.columns)
            ct = H.get_column_preprocessor(
                [c for c in feat_skew if c in valid],
                [c for c in feat_out if c in valid],
                [c for c in feat_std if c in valid])
            Xtr_p, Xva_p = ct.fit_transform(X_tr), ct.transform(X_va)
        elif is_pinn:
            Xtr_p, Xva_p = X_tr, X_va
        else:
            Xtr_p, Xva_p = X_tr.values, X_va.values
        model = clone(template)
        model.fit(Xtr_p, y_tr)
        scores.append(nmae(y_va.values, np.asarray(model.predict(Xva_p))))
    return float(np.mean(scores))


def optuna_search(args, logger, X_base, y_base, cols_for_imputation,
                  scaler_groups, H):
    """TPE (Bayesian) search over SEARCH_SPACES; returns the best config per family.

    Falls back to random search (``make_model_configs``) if optuna is unavailable.
    Search space is identical to random search: each parameter is chosen from its
    listed values (sampled by index so tuples/None/mixed types are handled).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        logger.info("[warn] optuna not installed -> falling back to random search.")
        return make_model_configs(args, logger)

    families = ([f.strip() for f in args.tune_models.split(",") if f.strip()]
                if args.tune_models else list(SEARCH_SPACES))
    folds = precompute_folds(X_base, y_base, cols_for_imputation, H,
                             args.folds, args.seed)

    winners = []
    for fam in families:
        if fam not in SEARCH_SPACES:
            logger.info("[warn] unknown family '%s' -> skipped", fam)
            continue
        if fam == "PINN" and not torch_available():
            logger.info("[info] PyTorch not available -> skipping PINN tuning.")
            continue
        space = SEARCH_SPACES[fam]

        def objective(trial, space=space, fam=fam):
            # sample each parameter by index -> robust to tuples/None/mixed types
            params = {k: space[k][trial.suggest_categorical(k, list(range(len(space[k]))))]
                      for k in space}
            label, template, scaled = _build_config(fam, params, args)
            return evaluate_config_on_folds(template, scaled, folds, scaler_groups, H)

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=args.tune_iter, show_progress_bar=False)

        best_params = {k: space[k][study.best_params[k]] for k in space}
        winners.append(_build_config(fam, best_params, args))
        logger.info("[optuna] %s: best CV NMAE %.4f  params=%s  (%d trials)",
                    fam, study.best_value, _short(best_params), args.tune_iter)

    logger.info("OPTUNA tuning complete -> %d winning configs (1 per family)",
                len(winners))
    return winners


# --------------------------------------------------------------------------- #
# Logging & evaluation (regression metrics + binned "confusion / accuracy")
# --------------------------------------------------------------------------- #
def setup_logger(path):
    """Log to both the given file (overwrite) and the console."""
    logger = logging.getLogger("proctor")
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


def regression_metrics(y_true, y_pred):
    """Per-target MAE, RMSE, and R^2."""
    err = y_true - y_pred
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot == 0, 1e-12, ss_tot)
    return mae, rmse, r2


def binned_confusion(y_true_col, y_pred_col, n_bins=3):
    """Turn a continuous target into ordinal bins and build a confusion matrix.

    Bin edges are the quantiles of the TRUE values, so both actual and predicted
    values are placed on the same scale. Returns (confusion_df, bin_accuracy,
    labels). Bin accuracy = fraction of samples whose predicted bin == true bin.
    """
    if n_bins == 3:
        names = ["Low", "Medium", "High"]
    else:
        names = [f"Q{i + 1}" for i in range(n_bins)]

    edges = np.unique(np.quantile(y_true_col, np.linspace(0, 1, n_bins + 1)))
    labels = names[:len(edges) - 1]
    inner = edges.copy().astype(float)
    inner[0], inner[-1] = -np.inf, np.inf  # capture any out-of-range predictions

    tb = pd.Series(pd.cut(np.asarray(y_true_col), bins=inner, labels=labels,
                          include_lowest=True))
    pb = pd.Series(pd.cut(np.asarray(y_pred_col), bins=inner, labels=labels,
                          include_lowest=True))
    cm = (pd.crosstab(tb, pb, dropna=False)
          .reindex(index=labels, columns=labels, fill_value=0))
    cm.index.name, cm.columns.name = "actual", "predicted"
    acc = float((tb.astype(str).values == pb.astype(str).values).mean())
    return cm, acc, labels


def plot_confusion(cm, labels, title, path):
    """Save a confusion-matrix heatmap (rows = actual, cols = predicted)."""
    m = cm.values.astype(int)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=120)
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted bin", fontweight="bold")
    ax.set_ylabel("Actual bin", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    thresh = m.max() / 2 if m.max() > 0 else 0.5
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, m[i, j], ha="center", va="center",
                    color="white" if m[i, j] > thresh else "black",
                    fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate_and_log(logger, y_true, y_pred, target_names, H, rho_s,
                     report_dir, n_bins):
    """Log OOF regression metrics + binned confusion/accuracy; save CM images."""
    # apply the same physical saturation clip used for the final submission
    owc = np.clip(y_pred[:, 1], 0, None)
    mdd = np.minimum(y_pred[:, 0], H.calc_satline(owc, rho_s) * 0.999)
    y_pred = np.column_stack([mdd, owc])

    mae, rmse, r2 = regression_metrics(y_true, y_pred)
    score = nmae(y_true, y_pred)

    logger.info("\n===== OUT-OF-FOLD EVALUATION (held-out predictions) =====")
    logger.info("Overall NMAE (competition metric): %.4f", score)
    logger.info("%-22s %10s %10s %8s", "target", "MAE", "RMSE", "R2")
    for i, name in enumerate(target_names):
        logger.info("%-22s %10.4f %10.4f %8.3f", name, mae[i], rmse[i], r2[i])

    logger.info("\n----- Binned classification view (%d bins per target) -----",
                n_bins)
    for i, name in enumerate(target_names):
        cm, acc, labels = binned_confusion(y_true[:, i], y_pred[:, i], n_bins)
        logger.info("\n[%s]  bin accuracy = %.3f  (bins by tertiles of actual)",
                    name, acc)
        logger.info("confusion matrix (rows=actual, cols=predicted):\n%s",
                    cm.to_string())
        img = os.path.join(report_dir, f"confusion_{name}.png")
        plot_confusion(cm, labels, f"Confusion — {name}  (acc={acc:.2f})", img)
        logger.info("saved -> %s", img)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    os.makedirs(args.report_dir, exist_ok=True)
    log_path = args.log if os.path.isabs(args.log) else os.path.join(args.report_dir, args.log)
    logger = setup_logger(log_path)
    logger.info("Proctor pipeline run @ %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("seed=%d  folds=%d  data_dir=%s", args.seed, args.folds, args.data_dir)

    H = load_helpers(args.helpers_dir)
    H.CFG.seed_everything(args.seed)

    train, test, sub = H.load_and_summarize_data(args.data_dir, lang="en")
    logger.info("train shape=%s  test shape=%s", train.shape, test.shape)

    # pre-CV feature engineering on both frames (identical construction)
    train = base_feature_engineering(train, H)
    test = base_feature_engineering(test, H)

    y_base = train[TARGETS].copy()
    X_base = train.drop(columns=TARGETS)

    exclude = ["id", "atterberg_is_missing", "kf_is_missing", "loi_is_missing"]
    cols_for_imputation = numeric_impute_columns(X_base, exclude)
    logger.info("base feature columns: %d", X_base.shape[1])
    logger.info("MICE-imputed columns (%d): %s",
                len(cols_for_imputation), cols_for_imputation)

    # scaler group assignment (computed once on a quick full-train imputation)
    imp_full = H.impute_missing_values(X_base, cols_for_imputation, target_cols=[],
                                       seed=args.seed)
    H.apply_fold_feature_engineering(imp_full)
    diag_exclude = exclude + [c for c in imp_full.columns
                              if imp_full[c].dropna().isin([0, 1]).all()]
    df_metrics, num_cols = H.run_distribution_diagnostics(imp_full, diag_exclude, [])
    scaler_groups = H.get_scaler_assignments(imp_full, df_metrics, num_cols)
    logger.info("scaler groups -> PowerTransformer:%d  RobustScaler:%d  StandardScaler:%d",
                len(scaler_groups[0]), len(scaler_groups[1]), len(scaler_groups[2]))

    # candidate models — defaults, or a tuned search when --tune is set
    logger.info("tuning: %s%s", "ON" if args.tune else "off",
                f" ({args.tuner})" if args.tune else "")
    if args.tune and args.tuner == "optuna":
        model_configs = optuna_search(args, logger, X_base, y_base,
                                      cols_for_imputation, scaler_groups, H)
    else:
        model_configs = make_model_configs(args, logger)

    registry, results, oof = run_cv_pipeline(
        X_base, y_base, model_configs, cols_for_imputation, scaler_groups, H,
        n_splits=args.folds, seed=args.seed)

    summary = (results.groupby("Modell")["NMAE"]
               .agg(["mean", "std"]).sort_values("mean"))
    if args.tune:
        logger.info("\n=== Tuning leaderboard (top %d of %d configs) ===\n%s",
                    min(args.top_k, len(summary)), len(summary),
                    summary.head(args.top_k).round(4).to_string())
    else:
        logger.info("\n=== Cross-validated NMAE (lower is better) ===\n%s",
                    summary.round(4).to_string())

    best = summary.index[0]
    best_template = {l: t for l, t, _ in model_configs}[best]
    logger.info("\nSelected model: %s  (mean NMAE %.4f, std %.4f)",
                best, summary.loc[best, "mean"], summary.loc[best, "std"])
    logger.info("model class: %s", type(best_template).__name__)
    try:
        params = best_template.get_params(deep=False)
        logger.info("model hyperparameters: %s",
                    {k: v for k, v in params.items() if not hasattr(v, "get_params")})
    except Exception:
        pass

    # honest evaluation on out-of-fold predictions of the selected model
    rho_s_tr = X_base["grain_density_g_cm3"].fillna(2.65).values
    evaluate_and_log(logger, y_base.values, oof[best], TARGETS, H, rho_s_tr,
                     args.report_dir, args.n_bins)

    # predict test with the winning config's fold ensemble
    pred = predict_registry(test, best, registry, cols_for_imputation, H)
    mdd, owc = pred[:, 0], np.clip(pred[:, 1], 0, None)

    # physical safety net: clip MDD to the saturation line per sample
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
    return out, summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=".")
    p.add_argument("--helpers_dir", default=".")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--pinn_epochs", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", default="proctor_run.log", help="run log filename")
    p.add_argument("--report_dir", default=".", help="where log + confusion PNGs go")
    p.add_argument("--n_bins", type=int, default=3,
                   help="bins per target for the confusion-matrix / accuracy view")
    p.add_argument("--tune", action="store_true",
                   help="enable hyperparameter tuning")
    p.add_argument("--tuner", choices=["random", "optuna"], default="random",
                   help="tuning backend: random search or optuna TPE (Bayesian)")
    p.add_argument("--tune_iter", type=int, default=8,
                   help="configs sampled per family (random) or trials per family (optuna)")
    p.add_argument("--tune_models", default="",
                   help="comma-separated families to tune "
                        "(e.g. 'ExtraTrees,HistGB'); default = all")
    p.add_argument("--top_k", type=int, default=10,
                   help="how many top configs to show in the tuning leaderboard")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())