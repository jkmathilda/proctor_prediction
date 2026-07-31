"""
v17.py
======
Joint (multi-output) version of v15.py for the LeiGS 2026 Proctor Challenge.

WHAT CHANGED vs v15
-------------------
v15 fit two *independent* models per target: a separate GP (own kernel, own
length-scales, own selected features) for MDD and for OWC, plus a separate GBT
for each. The only place the targets ever met was the post-hoc Zero-Air-Voids
clip. That throws away the strong physical coupling between them -- on Proctor
data MDD and OWC are markedly negatively correlated (wetter optimum => looser
maximum dry density), and v15 could not exploit it.

v17 predicts both targets **at the same time**, on both halves of the ensemble:

1. GP side -- Intrinsic Coregionalization Model (ICM).
   ONE Gaussian Process over an augmented input space (x, t), where t is the
   task index (0 = MDD, 1 = OWC):

       K((x,t),(x',t')) = B[t,t'] * k_base(x,x')   +   noise[t] * delta

   k_base is the shared (ARD-Matern by default) input kernel, and B is a full,
   *learned* 2x2 task covariance. Its off-diagonal is the MDD-OWC correlation,
   fit by marginal-likelihood alongside every other hyperparameter. Because B
   is learned rather than assumed, the model can discover the coupling instead
   of having it imposed. Prediction for one target now borrows strength from
   the other target's observations, which is exactly what helps at n ~ 200.

   Parameterization (see ICMKernel):
       B = [[v0,           rho*sqrt(v0*v1)],
            [rho*sqrt(v0*v1),           v1]],    rho = (r^2 - 1) / (r^2 + 1)

   v0, v1 > 0 are per-task signal variances; r > 0 is an unconstrained raw
   correlation parameter. The rational map keeps rho in (-1, 1) so B stays
   positive definite for *any* r, and it is symmetric in log(r) -- so
   sklearn's log-space hyperparameter search and its random restarts are
   unbiased with respect to the sign of the correlation (r = 1 <=> rho = 0).
   For 2 tasks (v0, v1, rho) is the fully general PSD 2x2, so nothing is lost
   by not using an explicit low-rank LMC.

   Noise is per-task (TaskWhiteKernel), since MDD (~g/cm^3) and OWC (~%) have
   very different measurement scales. Targets are z-scored per column before
   stacking so the shared kernel sees comparable magnitudes.

2. GBT side -- RegressorChain, order OWC -> MDD.
   The first tree model predicts OWC from the features; the second predicts
   MDD from the features *plus the predicted OWC*. That mirrors the physics
   (the compaction curve ties MDD to the water content at its peak) and makes
   the tree half joint as well. Chaining uses cross-validated predictions
   (--chain_cv) so the MDD stage is trained on realistic OWC estimates rather
   than on ground truth it will not have at test time.

TRADE-OFF TO BE AWARE OF
------------------------
ICM ties both targets to ONE shared set of ARD length-scales, so feature
relevance is now common to MDD and OWC (a single shared selected subset
instead of v15's two). That is the price of sharing statistical strength.
If a feature matters to only one target, an ICM will compromise. Compare the
reported NMAE against v15 before adopting it -- run --independent to get the
v15-style per-target GP back through this same script for an apples-to-apples
check.

Everything else is unchanged from v15: hyd_cond_hyd_gradient is still dropped
outright (not imputed, not a predictor), fold-isolated MICE imputation, the
organizers' feature engineering, StratifiedShuffleSplit validation, the
Zero-Air-Voids clip as a physical guardrail, and the NMAE / MAE / RMSE / R^2
reporting plus per-sample uncertainty.

Run:
    python v17.py --data_dir <csvs> --helpers_dir <helpers folder>

Needs train.csv, test.csv, helper_functions.py.
Dependencies: scikit-learn, pandas, numpy, matplotlib, torch (save/load only).
"""

import argparse
import importlib.util
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch  # used only to save/load the model as a .pt file
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, DotProduct,
                                              Hyperparameter, Kernel, Matern,
                                              RBF, RationalQuadratic,
                                              WhiteKernel)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler

TARGETS = ["proctor_mdd_g_cm3", "proctor_owc_pct"]   # task 0 = MDD, task 1 = OWC
N_TASKS = len(TARGETS)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def nmae(y_true, y_pred):
    """Official competition metric: mean column-wise IQR-normalized MAE."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    q75, q25 = np.percentile(y_true, [75, 25], axis=0)
    iqr = np.where((q75 - q25) == 0, 1e-8, q75 - q25)
    return float(np.mean(mae / iqr))


def regression_metrics(y_true, y_pred):
    err = y_true - y_pred
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot == 0, 1e-12, ss_tot)
    return mae, rmse, r2


# --------------------------------------------------------------------------- #
# Multi-output kernels
# --------------------------------------------------------------------------- #
def _raw_to_rho(r):
    """Map an unconstrained positive raw parameter to a correlation in (-1,1).

    rho(r) = (r^2 - 1) / (r^2 + 1).  r = 1 -> rho = 0, and rho(1/r) = -rho(r),
    so the map is antisymmetric in log(r): sklearn optimizes log-hyperparameters,
    which makes the search (and the random restarts drawn uniformly from the log
    bounds) symmetric about zero correlation instead of biased toward one sign.
    """
    u = float(r) ** 2
    return (u - 1.0) / (u + 1.0)


def _drho_dlogr(r):
    """d rho / d log(r), needed because sklearn differentiates w.r.t. log-params."""
    u = float(r) ** 2
    return 4.0 * u / (u + 1.0) ** 2


class ICMKernel(Kernel):
    """Intrinsic Coregionalization Model kernel over augmented inputs.

    Inputs are rows of the form [x_1 ... x_d, t] where the LAST column is an
    integer task index. The kernel is

        K((x,t),(x',t')) = B[t,t'] * k_base(x, x')

    with B the 2x2 task covariance built from (task_var, rho_raw). Learning B
    jointly with k_base is what makes this a genuine multi-output GP: the
    off-diagonal of B is the learned cross-target correlation, and it is what
    lets an observation of OWC inform the posterior over MDD.

    Note there is deliberately NO ConstantKernel amplitude on k_base -- the
    diagonal of B already carries the per-task signal variance, and including
    both would be unidentifiable.
    """

    def __init__(self, base_kernel, task_var=np.ones(N_TASKS), rho_raw=1.0,
                 task_var_bounds=(1e-3, 1e3), rho_raw_bounds=(0.05, 20.0),
                 n_tasks=N_TASKS):
        # rho_raw_bounds are reciprocal (0.05 = 1/20) so the reachable
        # correlation range is symmetric: rho in [-0.995, +0.995]. Capping just
        # short of +-1 keeps B comfortably positive definite -- at |rho| = 1 the
        # task covariance is rank-1 and the 2n x 2n Gram matrix becomes singular.
        self.base_kernel = base_kernel
        self.task_var = task_var
        self.rho_raw = rho_raw
        self.task_var_bounds = task_var_bounds
        self.rho_raw_bounds = rho_raw_bounds
        self.n_tasks = n_tasks

    # -- sklearn plumbing --------------------------------------------------- #
    def get_params(self, deep=True):
        params = dict(base_kernel=self.base_kernel, task_var=self.task_var,
                      rho_raw=self.rho_raw, task_var_bounds=self.task_var_bounds,
                      rho_raw_bounds=self.rho_raw_bounds, n_tasks=self.n_tasks)
        if deep:
            for k, v in self.base_kernel.get_params().items():
                params["base_kernel__" + k] = v
        return params

    @property
    def hyperparameter_task_var(self):
        return Hyperparameter("task_var", "numeric", self.task_var_bounds,
                              len(np.atleast_1d(self.task_var)))

    @property
    def hyperparameter_rho_raw(self):
        return Hyperparameter("rho_raw", "numeric", self.rho_raw_bounds)

    @property
    def hyperparameters(self):
        """Explicit ordering: base kernel params, then task_var, then rho_raw.

        The gradient stacking in __call__ must follow exactly this order.
        (The Kernel base class would otherwise order these alphabetically via
        dir(), which is fragile here.)
        """
        r = [Hyperparameter("base_kernel__" + hp.name, hp.value_type,
                            hp.bounds, hp.n_elements, hp.fixed)
             for hp in self.base_kernel.hyperparameters]
        r.append(self.hyperparameter_task_var)
        r.append(self.hyperparameter_rho_raw)
        return r

    # -- task covariance ---------------------------------------------------- #
    def task_covariance(self):
        """The learned 2x2 task covariance B."""
        v = np.atleast_1d(np.asarray(self.task_var, dtype=float))
        rho = _raw_to_rho(self.rho_raw)
        off = rho * np.sqrt(v[0] * v[1])
        return np.array([[v[0], off], [off, v[1]]])

    def task_correlation(self):
        """The learned MDD-OWC correlation, in (-1, 1)."""
        return _raw_to_rho(self.rho_raw)

    def _task_matrices(self):
        """B and its derivatives w.r.t. the LOG hyperparameters."""
        v = np.atleast_1d(np.asarray(self.task_var, dtype=float))
        rho = _raw_to_rho(self.rho_raw)
        s = np.sqrt(v[0] * v[1])
        off = rho * s
        B = np.array([[v[0], off], [off, v[1]]])
        # d B / d log(v0):  dB00/dv0 * v0 = v0 ;  dB01/dv0 * v0 = 0.5*rho*s
        dB_dlogv0 = np.array([[v[0], 0.5 * off], [0.5 * off, 0.0]])
        dB_dlogv1 = np.array([[0.0, 0.5 * off], [0.5 * off, v[1]]])
        # d B / d log(r) = (dB/drho) * (drho/dlog r)
        dB_dlogr = np.array([[0.0, s], [s, 0.0]]) * _drho_dlogr(self.rho_raw)
        return B, [dB_dlogv0, dB_dlogv1], dB_dlogr

    # -- kernel evaluation -------------------------------------------------- #
    @staticmethod
    def _split(X):
        X = np.asarray(X, dtype=float)
        return X[:, :-1], X[:, -1].astype(int)

    def __call__(self, X, Y=None, eval_gradient=False):
        Xf, ti = self._split(X)
        if Y is None:
            Yf, tj = Xf, ti
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            Yf, tj = self._split(Y)

        B, dB_dlogv, dB_dlogr = self._task_matrices()
        Bij = B[np.ix_(ti, tj)]

        if not eval_gradient:
            return Bij * self.base_kernel(Xf, Yf)

        Kb, Kb_grad = self.base_kernel(Xf, eval_gradient=True)
        K = Bij * Kb

        grads = [Kb_grad * Bij[:, :, np.newaxis]]                 # base kernel
        for g in dB_dlogv:                                        # task variances
            grads.append((g[np.ix_(ti, tj)] * Kb)[:, :, np.newaxis])
        grads.append((dB_dlogr[np.ix_(ti, tj)] * Kb)[:, :, np.newaxis])  # rho
        return K, np.dstack(grads)

    def diag(self, X):
        Xf, t = self._split(X)
        B, _, _ = self._task_matrices()
        return np.diag(B)[t] * self.base_kernel.diag(Xf)

    def is_stationary(self):
        return False

    def __repr__(self):
        return "ICM(base={0}, task_var={1}, rho={2:.3f})".format(
            self.base_kernel,
            np.array2string(np.atleast_1d(np.asarray(self.task_var, float)),
                            precision=3),
            self.task_correlation())


class TaskWhiteKernel(Kernel):
    """Independent measurement noise with a SEPARATE level per task.

    A plain WhiteKernel on the stacked data would force MDD (~1.9 g/cm^3) and
    OWC (~15 %) to share one noise level. Even after z-scoring the targets
    their observational noise differs, so give each task its own.
    """

    def __init__(self, noise_level=np.full(N_TASKS, 0.1),
                 noise_level_bounds=(1e-6, 1e1), n_tasks=N_TASKS):
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds
        self.n_tasks = n_tasks

    @property
    def hyperparameter_noise_level(self):
        return Hyperparameter("noise_level", "numeric", self.noise_level_bounds,
                              len(np.atleast_1d(self.noise_level)))

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, dtype=float)
        t = X[:, -1].astype(int)
        nl = np.atleast_1d(np.asarray(self.noise_level, dtype=float))
        if Y is not None:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            return np.zeros((X.shape[0], np.asarray(Y).shape[0]))
        K = np.diag(nl[t])
        if eval_gradient:
            grad = np.zeros((t.size, t.size, nl.size))
            for i in range(nl.size):
                grad[:, :, i] = np.diag(np.where(t == i, nl[i], 0.0))
            return K, grad
        return K

    def diag(self, X):
        t = np.asarray(X, dtype=float)[:, -1].astype(int)
        return np.atleast_1d(np.asarray(self.noise_level, dtype=float))[t]

    def is_stationary(self):
        return False

    def __repr__(self):
        return "TaskWhite(noise_level={0})".format(
            np.array2string(np.atleast_1d(np.asarray(self.noise_level, float)),
                            precision=3))


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def load_helpers(helpers_dir):
    path = os.path.join(helpers_dir, "helper_functions.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"helper_functions.py not found at {path}")
    spec = importlib.util.spec_from_file_location("helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup_logger(path):
    logger = logging.getLogger("icm_gpr")
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


def base_feature_engineering(df, H):
    df = H.add_gradation_parameters(df)
    df = H.prepare_features(df)
    return df


def numeric_impute_columns(X, exclude):
    return [c for c in X.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(X[c])]


def preprocess_fold(X_tr, X_va, cols_for_imputation, H, seed):
    """Fold-isolated: missing indicators -> MICE -> feature engineering -> scale."""
    for f in (X_tr, X_va):
        f["atterberg_is_missing"] = f["atterberg_liquid_limit_pct"].isnull().astype(int)
        f["kf_is_missing"] = f["hyd_cond_kf_m_s"].isnull().astype(int)
        f["loi_is_missing"] = f["loss_on_ignition_pct"].isnull().astype(int)
    imputer = H.get_default_mice_imputer(seed=seed)
    X_tr[cols_for_imputation] = imputer.fit_transform(X_tr[cols_for_imputation])
    X_va[cols_for_imputation] = imputer.transform(X_va[cols_for_imputation])
    for f in (X_tr, X_va):
        H.apply_fold_feature_engineering(f)
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr.values), scaler.transform(X_va.values)


# --------------------------------------------------------------------------- #
# Joint GP model
# --------------------------------------------------------------------------- #
def build_base_kernel(n_features, args):
    """The shared input kernel k_base -- SHARED by both targets under ICM.

    With ARD it carries a separate length-scale per feature, so the GP still
    learns feature relevance; the difference from v15 is that there is now one
    relevance profile serving both targets rather than two independent ones.
    """
    ls = np.ones(n_features) if args.ard else 1.0
    if args.kernel == "rbf":
        return RBF(length_scale=ls, length_scale_bounds=(1e-2, 1e3))
    if args.kernel == "rq":
        # RationalQuadratic is isotropic in sklearn (scalar length-scale only)
        return RationalQuadratic(length_scale=1.0, alpha=1.0,
                                 length_scale_bounds=(1e-2, 1e3))
    # matern (default) -- twice-differentiable, standard for physical data
    return Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e3), nu=2.5)


def build_icm_kernel(n_features, args):
    """ICM signal + per-task noise (+ optional global linear trend)."""
    kernel = (ICMKernel(build_base_kernel(n_features, args))
              + TaskWhiteKernel(noise_level=np.full(N_TASKS, 0.1)))
    if args.linear:
        # A linear trend shared across tasks but scaled per task: wrapping
        # DotProduct in its own ICM block keeps the multi-output structure.
        kernel = kernel + ICMKernel(
            DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3)))
    return kernel


def build_independent_kernel(n_features, args):
    """v15-style single-target kernel, kept for --independent A/B comparisons."""
    ls = np.ones(n_features) if args.ard else 1.0
    if args.kernel == "rbf":
        base = RBF(length_scale=ls, length_scale_bounds=(1e-2, 1e3))
    elif args.kernel == "rq":
        base = RationalQuadratic(length_scale=1.0, alpha=1.0,
                                 length_scale_bounds=(1e-2, 1e3))
    else:
        base = Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e3), nu=2.5)
    kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * base
              + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)))
    if args.linear:
        kernel = kernel + DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
    return kernel


class JointGP:
    """A single GP that predicts BOTH targets simultaneously via ICM.

    fit(X, Y) with Y of shape (n, 2) stacks the data into 2n augmented rows
    [x, t] and fits ONE GaussianProcessRegressor over them, so every
    hyperparameter -- length-scales, task variances, the cross-target
    correlation, per-task noise -- is estimated from one joint marginal
    likelihood. predict(X) returns (n, 2) means and (n, 2) standard deviations.

    Targets are z-scored per column before stacking (the shared kernel needs
    comparable magnitudes) and predictions are mapped back afterwards.
    """

    def __init__(self, args, n_features):
        self.args = args
        self.n_features = n_features

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _augment(X):
        """(n, d) -> (2n, d+1): all rows as task 0, then all rows as task 1."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        return np.vstack([np.hstack([X, np.full((n, 1), float(t))])
                          for t in range(N_TASKS)])

    def _unstack(self, flat, n):
        return np.column_stack([flat[t * n:(t + 1) * n] for t in range(N_TASKS)])

    # -- api ---------------------------------------------------------------- #
    def fit(self, X, Y):
        Y = np.asarray(Y, dtype=float)
        self.y_scaler_ = StandardScaler().fit(Y)
        Yz = self.y_scaler_.transform(Y)
        Xa = self._augment(X)
        ya = np.concatenate([Yz[:, t] for t in range(N_TASKS)])
        self.gp_ = GaussianProcessRegressor(
            kernel=build_icm_kernel(self.n_features, self.args),
            normalize_y=False,          # we standardize per target ourselves
            alpha=1e-8,
            n_restarts_optimizer=self.args.restarts,
            random_state=self.args.seed)
        self.gp_.fit(Xa, ya)
        return self

    def predict(self, X, return_std=True):
        n = np.asarray(X).shape[0]
        flat_mean, flat_std = self.gp_.predict(self._augment(X), return_std=True)
        mean = self.y_scaler_.inverse_transform(self._unstack(flat_mean, n))
        std = self._unstack(flat_std, n) * self.y_scaler_.scale_
        return (mean, std) if return_std else mean

    def predict_task_cov(self, X):
        """Full 2x2 predictive covariance per test point -- (n, 2, 2).

        This is the part an independent-model setup simply cannot produce: the
        posterior cross-covariance between predicted MDD and predicted OWC for
        the same sample. Useful for jointly-consistent sampling or for error
        bars on any derived quantity that mixes the two.
        """
        n = np.asarray(X).shape[0]
        _, cov = self.gp_.predict(self._augment(X), return_cov=True)
        scale = self.y_scaler_.scale_
        out = np.zeros((n, N_TASKS, N_TASKS))
        for a in range(N_TASKS):
            for b in range(N_TASKS):
                block = np.diag(cov[a * n:(a + 1) * n, b * n:(b + 1) * n])
                out[:, a, b] = block * scale[a] * scale[b]
        return out

    # -- introspection ------------------------------------------------------ #
    def _icm_block(self):
        for part in (getattr(self.gp_.kernel_, "k1", None),
                     getattr(self.gp_.kernel_, "k2", None),
                     self.gp_.kernel_):
            if isinstance(part, ICMKernel):
                return part
            if part is not None and isinstance(getattr(part, "k1", None), ICMKernel):
                return part.k1
        return None

    def task_correlation(self):
        blk = self._icm_block()
        return None if blk is None else blk.task_correlation()

    def task_covariance(self):
        blk = self._icm_block()
        return None if blk is None else blk.task_covariance()

    def length_scales(self):
        return learned_length_scales(self.gp_.kernel_, self.n_features)

    def kernel_repr(self):
        return str(self.gp_.kernel_)


class IndependentGPs:
    """v15 behaviour (one GP per target), exposed behind the JointGP interface.

    Only used with --independent, so the joint and separate approaches can be
    compared under an otherwise identical pipeline.
    """

    def __init__(self, args, n_features):
        self.args = args
        self.n_features = n_features

    def fit(self, X, Y):
        self.gps_ = []
        for j in range(N_TASKS):
            gp = GaussianProcessRegressor(
                kernel=build_independent_kernel(self.n_features, self.args),
                normalize_y=True, alpha=1e-8,
                n_restarts_optimizer=self.args.restarts,
                random_state=self.args.seed)
            gp.fit(X, np.asarray(Y, dtype=float)[:, j])
            self.gps_.append(gp)
        return self

    def predict(self, X, return_std=True):
        n = np.asarray(X).shape[0]
        mean = np.zeros((n, N_TASKS))
        std = np.zeros((n, N_TASKS))
        for j, gp in enumerate(self.gps_):
            mean[:, j], std[:, j] = gp.predict(X, return_std=True)
        return (mean, std) if return_std else mean

    def predict_task_cov(self, X):
        _, std = self.predict(X)
        out = np.zeros((len(std), N_TASKS, N_TASKS))
        for j in range(N_TASKS):
            out[:, j, j] = std[:, j] ** 2       # zero cross-covariance by design
        return out

    def task_correlation(self):
        return None

    def task_covariance(self):
        return None

    def length_scales(self):
        # relevance of the first target's kernel, for reporting only
        return learned_length_scales(self.gps_[0].kernel_, self.n_features)

    def kernel_repr(self):
        return " | ".join(f"[{n}] {gp.kernel_}" for n, gp in zip(TARGETS, self.gps_))


def learned_length_scales(kernel, n_features):
    """Per-feature length-scales from a fitted kernel (None if isotropic)."""
    for k, v in kernel.get_params().items():
        if k.endswith("length_scale"):
            arr = np.atleast_1d(v)
            if arr.size == n_features:
                return arr
    return None


def make_gp(args, n_features):
    return (IndependentGPs(args, n_features) if args.independent
            else JointGP(args, n_features))


def select_and_fit(Xtr, Y, args):
    """Fit the joint GP, then optionally drop features ARD judged irrelevant
    and refit on the survivors. Returns (fitted_model, selected_indices).

    Under ICM there is ONE shared length-scale vector, so selection now yields
    a single subset used by both targets (v15 selected per target). Selection
    is fold-isolated: it only ever sees the training rows passed in.
    """
    gp = make_gp(args, Xtr.shape[1]).fit(Xtr, Y)
    sel = np.arange(Xtr.shape[1])
    if args.select and args.ard:
        ls = gp.length_scales()
        if ls is not None:
            keep = np.where(ls < args.select_thresh)[0]
            if keep.size < args.select_min:                 # guarantee a minimum
                keep = np.argsort(ls)[:args.select_min]
            if 0 < keep.size < Xtr.shape[1]:
                sel = np.sort(keep)
                gp = make_gp(args, sel.size).fit(Xtr[:, sel], Y)
    return gp, sel


# --------------------------------------------------------------------------- #
# Joint GBT model
# --------------------------------------------------------------------------- #
def make_gbt(args):
    """Gradient-boosted trees -- a low-variance model that captures feature
    interactions the smooth GP kernel misses. Blending the two lowers error and
    variance because they make different mistakes."""
    return HistGradientBoostingRegressor(
        random_state=args.seed, max_iter=args.gbt_iter,
        learning_rate=args.gbt_lr, max_leaf_nodes=args.gbt_leaves,
        l2_regularization=1.0, early_stopping=False)


def make_chain(args):
    """RegressorChain over both targets: predict OWC first, then MDD using the
    predicted OWC as an extra feature.

    TARGETS = [MDD, OWC], so order=[1, 0] means OWC (index 1) is the first link.
    That direction matches the physics -- the compaction curve fixes MDD at the
    optimum water content, so knowing OWC is informative about MDD, much more
    than the reverse.

    cv > 1 makes the chain feed *cross-validated* OWC predictions to the MDD
    stage during fitting. Without it the MDD model trains on ground-truth OWC
    but predicts from estimated OWC, and the resulting exposure bias inflates
    test error.
    """
    return RegressorChain(base_estimator=make_gbt(args), order=[1, 0],
                          cv=(args.chain_cv if args.chain_cv and args.chain_cv > 1
                              else None),
                          random_state=args.seed)


# --------------------------------------------------------------------------- #
# Fit / predict / blend
# --------------------------------------------------------------------------- #
def fit_bases(Xtr, ytr, args):
    """Fit the joint GP (+selection) and the chained GBT. Returns (gp, sel, chain)."""
    gp, sel = select_and_fit(Xtr, ytr, args)
    chain = None
    if args.ensemble:
        chain = make_chain(args)
        chain.fit(Xtr, ytr)
    return gp, sel, chain


def predict_bases(Xte, gp, sel, chain):
    """Predict both targets from each base model. Returns (gp_mean, gp_std, chain_mean)."""
    gp_mean, gp_std = gp.predict(Xte[:, sel])
    if chain is None:
        chain_mean = np.full((len(Xte), N_TASKS), np.nan)
    else:
        chain_mean = np.asarray(chain.predict(Xte), dtype=float)
    return gp_mean, gp_std, chain_mean


def choose_blend_weights(y_true, gp_pred, gbt_pred, grid=21):
    """Per target, pick the GP weight w in [0,1] minimizing MAE of
    w*GP + (1-w)*chain on the (out-of-fold) predictions."""
    weights = np.ones(y_true.shape[1])
    if np.isnan(gbt_pred).any():
        return weights                       # ensemble off -> pure GP
    for j in range(y_true.shape[1]):
        best_w, best = 1.0, np.inf
        for w in np.linspace(0, 1, grid):
            mae = np.mean(np.abs(y_true[:, j] - (w * gp_pred[:, j]
                                                 + (1 - w) * gbt_pred[:, j])))
            if mae < best:
                best, best_w = mae, w
        weights[j] = best_w
    return weights


def blend_and_clip(gp_pred, gbt_pred, weights, rho_s, H):
    """Blend per target with `weights`, then clip MDD to the saturation line.

    The Zero-Air-Voids clip is kept even though the model is now joint: ICM
    learns the MDD-OWC correlation statistically, but nothing in the GP
    guarantees the hard physical bound, so the guardrail still earns its place.
    """
    if np.isnan(gbt_pred).any():
        blended = gp_pred.copy()
    else:
        blended = weights * gp_pred + (1 - weights) * gbt_pred
    owc = np.clip(blended[:, 1], 0, None)
    mdd = np.minimum(blended[:, 0], H.calc_satline(owc, rho_s) * 0.999)
    return np.column_stack([mdd, owc])


# --------------------------------------------------------------------------- #
# Saveable model: fit on all data, persist, reload, predict
# --------------------------------------------------------------------------- #
def _add_indicators(df):
    df["atterberg_is_missing"] = df["atterberg_liquid_limit_pct"].isnull().astype(int)
    df["kf_is_missing"] = df["hyd_cond_kf_m_s"].isnull().astype(int)
    df["loi_is_missing"] = df["loss_on_ignition_pct"].isnull().astype(int)


def fit_full_model(X_base, y, cols_for_imputation, H, args, weights=None):
    """Fit imputer, scaler, the joint GP and the chained GBT on ALL data."""
    X = X_base.copy()
    _add_indicators(X)
    imputer = H.get_default_mice_imputer(seed=args.seed)
    X[cols_for_imputation] = imputer.fit_transform(X[cols_for_imputation])
    H.apply_fold_feature_engineering(X)
    feature_cols = list(X.columns)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    gp, sel, chain = fit_bases(Xs, y, args)

    if weights is None:
        weights = np.ones(N_TASKS)
    return {
        "imputer": imputer,
        "scaler": scaler,
        "gp": gp,                                     # joint model (both targets)
        "chain": chain,                               # None if ensemble off
        "weights": np.asarray(weights),               # per-target GP blend weight
        "selected": sel,                              # shared column indices
        "feature_cols": feature_cols,
        "cols_for_imputation": cols_for_imputation,
        "targets": TARGETS,
        "kernel": gp.kernel_repr(),                   # learned kernel hyperparams
        "task_correlation": gp.task_correlation(),    # learned MDD-OWC corr
        "task_covariance": gp.task_covariance(),
        "joint": not args.independent,
    }


def _transform_with(artifacts, df_fe, H):
    """Apply the saved preprocessing to a (base-feature-engineered) frame."""
    X = df_fe.copy()
    _add_indicators(X)
    X[artifacts["cols_for_imputation"]] = artifacts["imputer"].transform(
        X[artifacts["cols_for_imputation"]])
    H.apply_fold_feature_engineering(X)
    X = X.reindex(columns=artifacts["feature_cols"])   # exact train-time order
    return artifacts["scaler"].transform(X.values)


def gp_predict(artifacts, df_fe, H, clip=True):
    """Predict [MDD, OWC] (joint GP + chained GBT blend) + GP stds."""
    Xs = _transform_with(artifacts, df_fe, H)
    sel = artifacts.get("selected", np.arange(Xs.shape[1]))
    chain = artifacts.get("chain")
    weights = np.asarray(artifacts.get("weights", np.ones(N_TASKS)))

    gp_mean, stds, chain_mean = predict_bases(Xs, artifacts["gp"], sel, chain)

    rho_s = df_fe["grain_density_g_cm3"].fillna(2.65).values
    if clip:
        means = blend_and_clip(gp_mean, chain_mean, weights, rho_s, H)
    else:
        means = (gp_mean if np.isnan(chain_mean).any()
                 else weights * gp_mean + (1 - weights) * chain_mean)
    return means, stds


def predict_task_covariance(artifacts, df_fe, H):
    """Per-sample 2x2 posterior covariance of [MDD, OWC] from the joint GP."""
    Xs = _transform_with(artifacts, df_fe, H)
    sel = artifacts.get("selected", np.arange(Xs.shape[1]))
    return artifacts["gp"].predict_task_cov(Xs[:, sel])


def load_model(path):
    """Load a saved artifacts dict from a .pt file.

    v15 could be unpickled anywhere because it only ever pickled sklearn
    objects. v17 saves custom classes (ICMKernel, TaskWhiteKernel, JointGP),
    and when the model was written by running this file as a script those
    classes were pickled under the module name "__main__". Reloading from a
    different entry point would then fail with
    "Can't get attribute 'JointGP' on <module '__main__'>", so re-publish them
    into __main__ first. Always load through this function rather than calling
    torch.load directly.
    """
    import __main__ as _main
    for cls in (ICMKernel, TaskWhiteKernel, JointGP, IndependentGPs):
        if not hasattr(_main, cls.__name__):
            setattr(_main, cls.__name__, cls)
    return torch.load(path, weights_only=False)


def predict_from_raw(df_raw, artifacts, H, clip=True):
    """Convenience: run base feature engineering, then predict from a raw frame."""
    return gp_predict(artifacts, base_feature_engineering(df_raw.copy(), H), H, clip)


def plot_pred_vs_actual(y_true, y_pred, y_std, target, mae, r2, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=120)
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ms=4, alpha=0.55,
                ecolor="#b0b0b0", elinewidth=0.8, color="#1f7a5c",
                markeredgecolor="white", markeredgewidth=0.4)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="ideal")
    ax.set_xlabel(f"actual {target}", fontweight="bold")
    ax.set_ylabel(f"predicted {target}", fontweight="bold")
    ax.set_title(f"{target}  (joint GP ±1σ)\nMAE={mae:.3f}  R²={r2:.3f}",
                 fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_joint_residuals(y_true, y_pred, path):
    """MDD vs OWC residual scatter -- shows the coupling the joint model targets."""
    res = y_true - y_pred
    r = np.corrcoef(res[:, 0], res[:, 1])[0, 1]
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=120)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax.scatter(res[:, 1], res[:, 0], s=22, alpha=0.6, color="#1f7a5c",
               edgecolor="white", linewidth=0.4)
    ax.set_xlabel("OWC residual (%)", fontweight="bold")
    ax.set_ylabel("MDD residual (g/cm³)", fontweight="bold")
    ax.set_title(f"Residual coupling\nPearson r = {r:.3f}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return float(r)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    os.makedirs(args.report_dir, exist_ok=True)
    log_path = args.log if os.path.isabs(args.log) else os.path.join(args.report_dir, args.log)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = setup_logger(log_path)

    mode = "INDEPENDENT (v15-style)" if args.independent else "JOINT ICM multi-output"
    logger.info("Gaussian Process regression -- %s", mode)
    logger.info("seed=%d  splits=%d  kernel restarts=%d  ensemble=%s (chain cv=%s)",
                args.seed, args.folds, args.restarts, args.ensemble, args.chain_cv)

    H = load_helpers(args.helpers_dir)
    H.CFG.seed_everything(args.seed)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    logger.info("train shape=%s  test shape=%s", train.shape, test.shape)

    train = base_feature_engineering(train, H)
    test = base_feature_engineering(test, H)

    y = train[TARGETS].values.astype(float)
    X_base = train.drop(columns=TARGETS)

    # inherited from v15: drop hyd_cond_hyd_gradient outright -- not imputed,
    # not used as a predictor.
    X_base = X_base.drop(columns=["hyd_cond_hyd_gradient"], errors="ignore")

    exclude = ["id", "atterberg_is_missing", "kf_is_missing", "loi_is_missing"]
    cols_for_imputation = numeric_impute_columns(X_base, exclude)
    logger.info("features=%d  MICE-imputed columns=%d  (hyd_cond_hyd_gradient dropped)",
                X_base.shape[1], len(cols_for_imputation))
    logger.info("empirical corr(MDD, OWC) in training targets: %.3f",
                float(np.corrcoef(y[:, 0], y[:, 1])[0, 1]))

    # ---- stratified-shuffle cross-validation ----
    n_strata = int(min(args.val_strata, max(2, len(y) // 10)))
    edges = np.unique(np.quantile(y[:, 0], np.linspace(0, 1, n_strata + 1)))
    strata = np.digitize(y[:, 0], edges[1:-1])
    sss = StratifiedShuffleSplit(n_splits=args.folds, test_size=args.val_frac,
                                 random_state=args.seed)
    logger.info("StratifiedShuffleSplit: %d splits, test_size=%.2f, %d density strata",
                args.folds, args.val_frac, n_strata)

    gp_sum = np.zeros_like(y)
    gbt_sum = np.zeros_like(y)
    ssum = np.zeros_like(y)
    cnt = np.zeros(len(y))
    fold_records = []
    fold_corrs = []

    for k, (tr, va) in enumerate(sss.split(X_base, strata)):
        Xtr, Xva = preprocess_fold(X_base.iloc[tr].copy(), X_base.iloc[va].copy(),
                                   cols_for_imputation, H, args.seed)
        gp, sel, chain = fit_bases(Xtr, y[tr], args)
        gpm, gp_std, gbm = predict_bases(Xva, gp, sel, chain)

        gp_sum[va] += gpm
        gbt_sum[va] += np.nan_to_num(gbm)
        ssum[va] += gp_std
        cnt[va] += 1
        fold_records.append((va, gpm, gbm))

        corr = gp.task_correlation()
        if corr is not None:
            fold_corrs.append(corr)
            logger.info("split %d GP-NMAE = %.4f   learned corr(MDD,OWC) = %+.3f",
                        k + 1, nmae(y[va], gpm), corr)
        else:
            logger.info("split %d GP-NMAE = %.4f", k + 1, nmae(y[va], gpm))

    seen = cnt > 0
    logger.info("validation coverage: %d/%d rows scored across splits",
                int(seen.sum()), len(y))

    y_seen = y[seen]
    gp_oof = gp_sum[seen] / cnt[seen, None]
    gbt_oof = (gbt_sum[seen] / cnt[seen, None]) if args.ensemble else np.full_like(gp_oof, np.nan)
    oof_std = ssum[seen] / cnt[seen, None]
    rho_s_seen = X_base["grain_density_g_cm3"].fillna(2.65).values[seen]

    weights = choose_blend_weights(y_seen, gp_oof, gbt_oof)
    logger.info("\nBlend weights (GP share per target): %s",
                {TARGETS[j]: round(float(weights[j]), 2) for j in range(N_TASKS)})

    gp_clip = blend_and_clip(gp_oof, np.full_like(gp_oof, np.nan),
                             np.ones(N_TASKS), rho_s_seen, H)
    blend_clip = blend_and_clip(gp_oof, gbt_oof, weights, rho_s_seen, H)

    logger.info("\n===== VALIDATION EVALUATION (averaged over splits) =====")
    logger.info("Aggregate NMAE  GP-only:  %.4f", nmae(y_seen, gp_clip))
    if args.ensemble:
        gbt_clip = blend_and_clip(gp_oof, gbt_oof, np.zeros(N_TASKS), rho_s_seen, H)
        logger.info("Aggregate NMAE  CHAIN-only: %.4f", nmae(y_seen, gbt_clip))
        logger.info("Aggregate NMAE  BLEND:      %.4f", nmae(y_seen, blend_clip))

    split_blend = []
    for va, gpm, gbm in fold_records:
        pred = blend_and_clip(gpm, gbm if args.ensemble else np.full_like(gpm, np.nan),
                              weights, X_base["grain_density_g_cm3"].fillna(2.65).values[va], H)
        split_blend.append(nmae(y[va], pred))
    logger.info("Per-split BLEND NMAE: mean %.4f  std %.4f  (min %.4f, max %.4f)",
                float(np.mean(split_blend)), float(np.std(split_blend)),
                float(np.min(split_blend)), float(np.max(split_blend)))
    if fold_corrs:
        logger.info("Learned task correlation across splits: mean %+.3f  std %.3f",
                    float(np.mean(fold_corrs)), float(np.std(fold_corrs)))

    oof_clip = blend_clip
    mae, rmse, r2 = regression_metrics(y_seen, oof_clip)
    logger.info("%-22s %10s %10s %8s %12s", "target", "MAE", "RMSE", "R2", "mean σ")
    for i, name in enumerate(TARGETS):
        logger.info("%-22s %10.4f %10.4f %8.3f %12.3f", name, mae[i], rmse[i], r2[i],
                    float(np.nanmean(oof_std[:, i])))
        img = os.path.join(args.report_dir, f"v17_pred_vs_actual_{name}.png")
        plot_pred_vs_actual(y_seen[:, i], oof_clip[:, i], oof_std[:, i], name,
                            mae[i], r2[i], img)
        logger.info("saved -> %s", img)

    res_img = os.path.join(args.report_dir, "v17_residual_coupling.png")
    res_corr = plot_joint_residuals(y_seen, oof_clip, res_img)
    logger.info("residual corr(MDD, OWC) = %+.3f  (leftover coupling the model "
                "did not capture)  -> %s", res_corr, res_img)

    # ---- fit final model on all training data, save it, predict test ----
    artifacts = fit_full_model(X_base, y, cols_for_imputation, H, args, weights=weights)
    if args.model_out:
        os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
        torch.save(artifacts, args.model_out)
        logger.info("\nSaved model -> %s", args.model_out)
        logger.info("  learned kernel: %s", artifacts["kernel"])

    if artifacts["task_covariance"] is not None:
        B = artifacts["task_covariance"]
        logger.info("\nLearned task covariance B (on standardized targets):")
        logger.info("        %-22s %-22s", *TARGETS)
        for i, name in enumerate(TARGETS):
            logger.info("  %-20s %10.4f %20.4f", name, B[i, 0], B[i, 1])
        logger.info("  => learned corr(MDD, OWC) = %+.3f",
                    artifacts["task_correlation"])

    # ARD relevance report -- now a SHARED profile serving both targets
    if args.ard:
        feat = artifacts["feature_cols"]
        sel = artifacts["selected"]
        sel_names = [feat[i] for i in sel]
        logger.info("\nARD feature relevance (shared across both targets; "
                    "shortest length-scale = most informative):")
        logger.info("  kept %d/%d features after selection", len(sel), len(feat))
        ls = artifacts["gp"].length_scales()
        if ls is None:
            logger.info("      (isotropic kernel -- no per-feature relevance)")
        else:
            for i in np.argsort(ls)[:args.relevance_top]:
                logger.info("      %-34s length-scale=%.3g", sel_names[i], float(ls[i]))

    # predict the test set from the saved artifacts (test is already base-FE'd)
    preds, stds = gp_predict(artifacts, test, H, clip=True)
    mdd, owc = preds[:, 0], preds[:, 1]

    out = pd.DataFrame({
        "id": test["id"].values,
        "proctor_owc_pct": np.round(owc, 3),
        "proctor_mdd_g_cm3": np.round(mdd, 4),
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    if args.uncertainty_out:
        os.makedirs(os.path.dirname(args.uncertainty_out) or ".", exist_ok=True)
        unc = pd.DataFrame({
            "id": test["id"].values,
            "owc_std": np.round(stds[:, 1], 3),
            "mdd_std": np.round(stds[:, 0], 4),
        })
        # the joint model can also report the posterior cross-correlation
        if not args.independent:
            cov = predict_task_covariance(artifacts, test, H)
            denom = np.sqrt(cov[:, 0, 0] * cov[:, 1, 1])
            unc["pred_corr_mdd_owc"] = np.round(
                np.divide(cov[:, 0, 1], np.where(denom == 0, 1e-12, denom)), 3)
        unc.to_csv(args.uncertainty_out, index=False)
        logger.info("Wrote per-sample uncertainty -> %s", args.uncertainty_out)

    logger.info("\nWrote %d predictions -> %s", len(out), args.out)
    logger.info("Run log -> %s", log_path)
    return out


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(repo_root / "data"))
    p.add_argument("--helpers_dir", default=str(repo_root))
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_v17_icm_joint.csv"))
    p.add_argument("--model_out", default=str(repo_root / "models" / "v17_icm_joint.pt"),
                   help="path to save the fitted model as a .pt file (empty to skip)")
    p.add_argument("--uncertainty_out", default=str(repo_root / "docs" / "v17_logs" / "v17_uncertainty.csv"),
                   help="CSV of per-sample predictive std (empty string to skip)")
    p.add_argument("--report_dir", default=str(repo_root / "figures"))
    p.add_argument("--log", default=str(repo_root / "docs" / "v17_logs" / "v17_run.log"))

    p.add_argument("--folds", type=int, default=1,
                   help="number of StratifiedShuffleSplit splits")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="validation fraction per split (test_size)")
    p.add_argument("--val_strata", type=int, default=5,
                   help="density strata used to stratify the split")

    p.add_argument("--restarts", type=int, default=4,
                   help="kernel hyperparameter optimizer restarts (the joint GP "
                        "factorizes a 2n x 2n matrix, so this costs ~8x a v15 fit)")
    p.add_argument("--kernel", choices=["matern", "rbf", "rq"], default="matern",
                   help="shared smoothness kernel (matern recommended)")
    p.add_argument("--ard", dest="ard", action="store_true", default=True,
                   help="per-feature length-scales (Automatic Relevance Determination)")
    p.add_argument("--isotropic", dest="ard", action="store_false",
                   help="use a single shared length-scale instead of ARD")
    p.add_argument("--relevance_top", type=int, default=10,
                   help="how many top ARD features to report")

    p.add_argument("--independent", action="store_true",
                   help="fall back to v15-style separate per-target GPs "
                        "(for A/B comparison against the joint ICM model)")

    p.add_argument("--select", dest="select", action="store_true", default=True,
                   help="two-stage ARD feature selection (fit, drop noise, refit)")
    p.add_argument("--no_select", dest="select", action="store_false",
                   help="disable ARD feature selection")
    p.add_argument("--select_thresh", type=float, default=100.0,
                   help="keep features whose learned length-scale is below this")
    p.add_argument("--select_min", type=int, default=5,
                   help="always keep at least this many (most relevant) features")
    p.add_argument("--linear", action="store_true",
                   help="add a linear (DotProduct) ICM term for global trends")

    p.add_argument("--ensemble", dest="ensemble", action="store_true", default=True,
                   help="blend the GP with a chained gradient-boosted-tree model")
    p.add_argument("--no_ensemble", dest="ensemble", action="store_false",
                   help="use the joint GP alone (no blend)")
    p.add_argument("--chain_cv", type=int, default=5,
                   help="folds for RegressorChain's cross-validated chaining "
                        "(<=1 disables it and chains on ground-truth OWC)")
    p.add_argument("--gbt_iter", type=int, default=400, help="GBT boosting iterations")
    p.add_argument("--gbt_lr", type=float, default=0.05, help="GBT learning rate")
    p.add_argument("--gbt_leaves", type=int, default=31, help="GBT max leaf nodes")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())