from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import optuna

from scipy.optimize import minimize, minimize_scalar

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import NotFittedError
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 -- required before importing IterativeImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, RidgeCV

from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Equivalent to data_analysis.ipynb's `params_av` ("available"): the feature
# set with zero missing values in train.csv (drops the Atterberg limits,
# hydraulic conductivity, loss-on-ignition, feat_pi, and id columns). Every
# model-architecture experiment in that notebook -- the Ridge/GPR/XGBoost
# blend search and the GlobalResidualRegressor fit -- was run on this set,
# not the full raw column list.
#
# Known limitation for OWC: with this feature set, neither XGBoost nor GPR
# (alone or blended) can predict Ridge's OWC residual better than the mean
# (negative OOF R² on the residual), so GlobalResidualRegressor's fitted
# alpha collapses to 0 for OWC and the hybrid prediction reduces to plain
# Ridge (OOF R² ~0.22). This is a real feature-availability tradeoff, not a
# bug: the Atterberg-limit-derived features that explain water content best
# are exactly the ones this list excludes for having missing values. MDD is
# unaffected (hybrid R² ~0.81, matching the notebook). Kept as params_av for
# both targets by deliberate choice, not an oversight.
NO_MISSING_FEATURES: list[str] = [
    "psd_size_at_d10_mm",
    "psd_size_at_d20_mm",
    "psd_size_at_d30_mm",
    "psd_size_at_d40_mm",
    "psd_size_at_d50_mm",
    "psd_size_at_d60_mm",
    "psd_size_at_d70_mm",
    "psd_size_at_d80_mm",
    "psd_size_at_d90_mm",
    "psd_size_at_d95_mm",
    "psd_size_at_d98_mm",
    "psd_has_sedimentation",
    "psd_passing_at_0_002mm_pct",
    "psd_passing_at_0_063mm_pct",
    "psd_passing_at_2mm_pct",
    "proctor_diam_mm",
    "grain_density_g_cm3",
    "clay",
    "silt",
    "sand",
    "gravel",
    "fine-grained",
    "feat_cu",
    "feat_cc",
    "feat_log_cu",
]


def add_no_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive NO_MISSING_FEATURES' engineered columns (clay/silt/sand/gravel/
    fine-grained/feat_cu/feat_cc/feat_log_cu) from the raw PSD columns,
    matching data_analysis.ipynb's feature engineering exactly -- identical
    to general_model.py's version of the same function (both files define
    the same NO_MISSING_FEATURES columns). `fine-grained` is kept here (used
    below by add_imputed_features as the fine/coarse split key) even though
    MICE_PREDICTORS/IMPUTED_FEATURES exclude it as a model feature.

    Kept alongside NO_MISSING_FEATURES/MICE_PREDICTORS/IMPUTED_FEATURES so a
    caller only needs this module to go from raw train/test data to any of
    this file's feature matrices -- adding a new feature and its derivation
    here, in the same place, keeps the constants and the code that fills
    them from drifting apart.
    """
    df = df.copy()
    df["clay"] = df["psd_passing_at_0_002mm_pct"]
    df["silt"] = df["psd_passing_at_0_063mm_pct"] - df["psd_passing_at_0_002mm_pct"]
    df["sand"] = df["psd_passing_at_2mm_pct"] - df["psd_passing_at_0_063mm_pct"]
    df["gravel"] = 100 - df["psd_passing_at_2mm_pct"]
    df["fine-grained"] = df["clay"] + df["silt"] > 12
    df["feat_cu"] = (
        df["psd_size_at_d60_mm"].replace(0, np.nan)
        / df["psd_size_at_d10_mm"].replace(0, np.nan)
    )
    df["feat_cc"] = (df["psd_size_at_d30_mm"] ** 2) / (
        df["psd_size_at_d60_mm"].replace(0, np.nan)
        * df["psd_size_at_d10_mm"].replace(0, np.nan)
    )
    df["feat_log_cu"] = np.log1p(df["feat_cu"])
    return df


# data_analysis.ipynb's `mice_predictors`: the predictor set used for MICE
# imputation and as the base of every imputed feature set tested. Identical
# to NO_MISSING_FEATURES minus `fine-grained` -- `fine-grained` is used
# below as the fine/coarse split key, not as a MICE predictor or model
# feature.
MICE_PREDICTORS: list[str] = [
    c for c in NO_MISSING_FEATURES if c != "fine-grained"
]


# data_analysis.ipynb's "Three blend with imputed data" section (cells
# 106-111) systematically compared four feature sets built on top of
# MICE_PREDICTORS -- "Universal only" (no imputed columns, i.e.
# MICE_PREDICTORS alone), "+ kf", "+ LOI", and "+ kf + LOI" -- across both
# targets:
#
#     experiment            target   r2      rmse    mae
#     Universal + kf + LOI  MDD      0.8619  0.0520  0.0402
#     Universal + LOI       MDD      0.8594  0.0525  0.0400
#     Universal + kf        MDD      0.8546  0.0534  0.0411
#     Universal only        MDD      0.8506  0.0541  0.0410
#     Universal + LOI       OWC      0.8079  1.4713  1.1235
#     Universal + kf + LOI  OWC      0.8048  1.4829  1.1434
#     Universal + kf        OWC      0.8027  1.4909  1.1336
#     Universal only        OWC      0.8006  1.4989  1.1501
#
# CAVEAT -- the comparison above has preprocessing leakage: the notebook (and
# add_imputed_features, as originally written) fits MICE on ALL rows once,
# before any CV split, so a held-out row's own feature values could
# influence the MICE equations used to impute OTHER rows in its own
# validation fold. Not target leakage (MDD/OWC are never inputs to MICE),
# but enough to make the deltas above optimistic. Re-run with MICE refit
# fresh INSIDE each outer fold (fit on that fold's training rows only,
# applied to its validation rows -- see scripts/leakage_check pattern), same
# WeightedBlendRegressor, same 5-fold split:
#
#     experiment                            target   r2
#     Universal + imputed kf + LOI (fold-isolated)  MDD  0.8467
#     Universal only                                MDD  0.8374
#     Universal only                                OWC  0.7694
#     Universal + imputed kf + LOI (fold-isolated)  OWC  0.7651
#
# Conclusion: for MDD the imputed-feature gain survives fold isolation
# (+0.009 R^2, same direction/order of magnitude as the leaky estimate) --
# genuine signal, not a leakage artifact. For OWC the gain does NOT survive:
# it flips to a small LOSS (-0.004 R^2) once the imputer no longer sees
# validation-fold rows. So the leaky benchmark's OWC ranking was misleading;
# model1i.py (which uses IMPUTED_FEATURES for both targets) should not be
# assumed to beat model1.py's NO_MISSING_FEATURES-only approach for OWC
# specifically without re-validating -- MDD is the target where these
# imputed columns are actually earning their keep.
#
# "Universal + kf + LOI" is the outright best for MDD and within 0.003 R^2
# of OWC's individual best ("Universal + LOI") while beating every other
# option on both targets simultaneously -- the single-best/near-best,
# consistent-across-targets choice, unlike "+ kf" or "+ LOI" alone which
# each trail badly on the other target. Atterberg limits are also
# MICE-imputed by add_imputed_features below (fine-grained rows only -- see
# its docstring) but were never added to any of the four feature sets the
# notebook actually benchmarked, so they're excluded here too.
IMPUTED_FEATURES: list[str] = MICE_PREDICTORS + [
    "log10_hyd_cond_kf_m_s_completed",
    "hyd_cond_kf_m_s_was_missing",
    "loss_on_ignition_pct_completed",
    "loss_on_ignition_pct_was_missing",
]


def impute_features(
    input_df: pd.DataFrame,
    impute_cols: list[str],
    predictors: list[str] | None = None,
    log_transform_cols: list[str] | None = None,
    applicable_mask: Any | None = None,
    estimator: Any | None = None,
    imputer: IterativeImputer | None = None,
    max_iter: int = 20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, IterativeImputer]:
    """
    Impute multiple missing features jointly using MICE --
    data_analysis.ipynb's ``impute_features``.

    Adds ``{col}_completed`` and ``{col}_was_missing`` for every column in
    ``impute_cols``, plus ``log10_{col}_completed`` for any column in
    ``log_transform_cols`` (imputation happens in log10 space for those,
    then is converted back). Only rows where ``applicable_mask`` is True are
    used to fit/apply the imputer -- e.g. Atterberg limits are only
    imputed for fine-grained rows (see add_imputed_features).

    Pass a previously-returned ``imputer`` to transform new data (e.g. a
    test set) with the rules already learned on training data, instead of
    fitting a new one on it. The notebook itself always fits fresh since it
    never needed a train/test split, but reusing a fitted imputer (rather
    than refitting on unseen data) is required to avoid leakage.
    """

    output_df = input_df.copy()

    if predictors is None:
        predictors = []

    if log_transform_cols is None:
        log_transform_cols = []

    if estimator is None:
        estimator = BayesianRidge()

    # Remove duplicates while preserving order
    mice_columns = list(dict.fromkeys(impute_cols + predictors))

    missing_columns = [col for col in mice_columns if col not in output_df.columns]

    if missing_columns:
        raise KeyError(f"These columns are not in the dataframe: {missing_columns}")

    if applicable_mask is None:
        applicable_mask = pd.Series(True, index=output_df.index)
    else:
        applicable_mask = pd.Series(applicable_mask, index=output_df.index).astype(bool)

    if not applicable_mask.any():
        # IterativeImputer.fit()/.transform() both reject 0-row input --
        # this happens whenever a batch (a fold, a single new sample, a
        # test split) contains only one of the fine/coarse groups. Fitting
        # is a genuine error (nothing to learn the MICE equations from);
        # transforming with an already-fitted imputer is a no-op (there are
        # no applicable rows to fill), so it should degrade gracefully
        # instead of crashing.
        if imputer is None:
            raise ValueError(
                "Cannot fit an imputer: applicable_mask selects 0 rows."
            )

        for col in impute_cols:
            completed_col = f"{col}_completed"
            missing_indicator = f"{col}_was_missing"

            if completed_col not in output_df.columns:
                output_df[completed_col] = output_df[col].copy()

            if missing_indicator not in output_df.columns:
                output_df[missing_indicator] = output_df[col].isna().astype(int)

            if col in log_transform_cols:
                log_completed_col = f"log10_{completed_col}"

                if log_completed_col not in output_df.columns:
                    output_df[log_completed_col] = np.nan

        return output_df, imputer

    # Work only on rows where these variables are applicable
    mice_data = output_df.loc[applicable_mask, mice_columns].copy()

    # Ensure numeric input
    mice_data = mice_data.apply(pd.to_numeric, errors="coerce")

    # Apply log10 transformation where requested
    transformed_column_names = {}

    for col in log_transform_cols:
        if col not in mice_columns:
            raise ValueError(
                f"{col!r} is in log_transform_cols but not in "
                "impute_cols or predictors."
            )

        transformed_name = f"__log10_{col}"
        transformed_column_names[col] = transformed_name

        values = mice_data[col].astype(float)

        # Non-positive values are invalid for log10
        valid_positive = values > 0

        mice_data[transformed_name] = np.nan
        mice_data.loc[valid_positive, transformed_name] = np.log10(values.loc[valid_positive])

        mice_data = mice_data.drop(columns=col)

    # Names actually passed into IterativeImputer
    model_columns = mice_data.columns.tolist()

    if imputer is None:
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            tol=1e-3,
            initial_strategy="median",
            imputation_order="ascending",
            skip_complete=True,
            sample_posterior=False,
            random_state=random_state,
        )
        imputed_array = imputer.fit_transform(mice_data)
    else:
        imputed_array = imputer.transform(mice_data)

    imputed_data = pd.DataFrame(imputed_array, index=mice_data.index, columns=model_columns)

    # Convert log-transformed variables back to their original scale
    for original_col, transformed_col in transformed_column_names.items():
        imputed_data[original_col] = 10 ** imputed_data[transformed_col]

    # Create or update completed columns and missing indicators
    for col in impute_cols:
        completed_col = f"{col}_completed"
        missing_indicator = f"{col}_was_missing"

        # Create these only on the first call.
        # If they already exist, preserve previous imputations.
        if completed_col not in output_df.columns:
            output_df[completed_col] = output_df[col].copy()

        if missing_indicator not in output_df.columns:
            output_df[missing_indicator] = output_df[col].isna().astype(int)

        # Fill only originally missing values inside the current subset
        fill_mask = applicable_mask & output_df[col].isna()

        output_df.loc[fill_mask, completed_col] = imputed_data.loc[fill_mask, col]

        # Recalculate the log-completed column using all currently completed values
        if col in log_transform_cols:
            log_completed_col = f"log10_{completed_col}"

            output_df[log_completed_col] = np.nan

            positive_mask = (
                output_df[completed_col].notna()
                & np.isfinite(output_df[completed_col])
                & (output_df[completed_col] > 0)
            )

            output_df.loc[positive_mask, log_completed_col] = np.log10(
                output_df.loc[positive_mask, completed_col]
            )

    return output_df, imputer


def add_imputed_features(
    df: pd.DataFrame,
    fine_imputer: IterativeImputer | None = None,
    coarse_imputer: IterativeImputer | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, IterativeImputer, IterativeImputer]:
    """
    MICE-impute Atterberg limits, loss-on-ignition, and hydraulic
    conductivity, matching data_analysis.ipynb's fine/coarse-grained split
    (the ``impute_features`` calls building ``df_impute``): Atterberg limits
    are only imputed for fine-grained rows, because they are not physically
    measured for non-plastic coarse soils -- imputing them for coarse rows
    would be inventing a lab measurement that was never applicable in the
    first place. Loss-on-ignition and hydraulic conductivity (kf, in log10
    space) are imputed for both fine- and coarse-grained rows.

    Requires a ``fine-grained`` column but does not treat it as a MICE
    predictor or model feature -- it is only the split key.

    Pass previously-fitted ``fine_imputer``/``coarse_imputer`` (as returned
    from an earlier call on training data) to transform new data instead of
    fitting fresh ones, to avoid leaking test/holdout rows into the MICE fit.
    """

    if "fine-grained" not in df.columns:
        raise KeyError("add_imputed_features requires a 'fine-grained' column.")

    fine_mask = df["fine-grained"].astype(bool)
    coarse_mask = ~fine_mask

    fine_targets = [
        "atterberg_liquid_limit_pct",
        "atterberg_plastic_limit_pct",
        "loss_on_ignition_pct",
        "hyd_cond_kf_m_s",
    ]

    df_imputed, fine_imputer = impute_features(
        input_df=df,
        impute_cols=fine_targets,
        predictors=MICE_PREDICTORS,
        log_transform_cols=["hyd_cond_kf_m_s"],
        applicable_mask=fine_mask,
        imputer=fine_imputer,
        random_state=random_state,
    )

    coarse_targets = [
        "loss_on_ignition_pct",
        "hyd_cond_kf_m_s",
    ]

    df_imputed, coarse_imputer = impute_features(
        input_df=df_imputed,
        impute_cols=coarse_targets,
        predictors=MICE_PREDICTORS,
        log_transform_cols=["hyd_cond_kf_m_s"],
        applicable_mask=coarse_mask,
        imputer=coarse_imputer,
        random_state=random_state,
    )

    return df_imputed, fine_imputer, coarse_imputer


@dataclass
class RegressionMetrics:
    """Container for standard regression metrics."""

    r2: float
    mae: float
    rmse: float

    def as_dict(self) -> dict[str, float]:
        return {
            "r2": self.r2,
            "mae": self.mae,
            "rmse": self.rmse,
        }


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Calculate R², MAE and RMSE."""

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    return RegressionMetrics(
        r2=float(r2_score(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),
    )


def make_default_ridge_model() -> Pipeline:
    """
    Construct the default global linear model.

    Standardization is required because Ridge penalizes coefficients,
    and the predictors have different numerical scales.
    """

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-5, 5, 101)))
        ]
    )


def make_default_residual_model(random_state: int = 42) -> XGBRegressor:
    """
    Construct a conservative XGBoost model for residual correction.

    The residual model should be smaller and more regularized than a
    full target model because it only needs to explain what Ridge missed.
    """

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        random_state=random_state,
        # n_jobs=-1 intermittently segfaults the interpreter when many
        # XGBRegressor.fit() calls happen in one process (observed ~1-in-4
        # runs on Apple Silicon, reproducible independent of this code).
        # n_jobs=1 is slower but reliable; verified across 10+ repeated fits.
        n_jobs=1,
    )


def make_default_target_xgb_model(random_state: int = 42) -> XGBRegressor:
    """
    Construct the XGBoost config data_analysis.ipynb used to predict the
    *target* directly (cells 53 and 84-86's blend), not a residual. Less
    regularized than ``make_default_residual_model`` (min_child_weight 3 vs
    5, reg_alpha 0.0 vs 0.1, reg_lambda 1.0 vs 5.0) because it has to explain
    the full nonlinear signal on its own rather than Ridge's leftover error.
    """

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=random_state,
        # see make_default_residual_model's comment: n_jobs=-1 segfaults
        # intermittently on this codebase's dev machine.
        n_jobs=1,
    )


def make_stratified_shuffle_splits(
    y: Any,
    n_splits: int = 1,
    test_size: float = 0.2,
    val_strata: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    scripts/v15.py's validation scheme: StratifiedShuffleSplit binned on
    quantiles of y, "so every split's validation set then spans the full
    density range" (v15.py's own comment), rather than KFold's plain random
    partition. Defaults (n_splits=1, test_size=0.2, val_strata=5) match
    v15.py's CLI defaults (--folds 1 --val_frac 0.2 --val_strata 5) -- a
    single 80/20 holdout, NOT a K-fold partition: unlike KFold, most rows
    never land in any split's validation set, and (with n_splits > 1)
    coverage still isn't guaranteed since splits are independent random
    draws that can overlap.

    Returns a list of (train_idx, val_idx) positional-index pairs so a
    caller can draw the split ONCE and reuse the identical partition across
    every stage of a pipeline (general-model tuning, blend-weight fitting,
    a correction stage on top) instead of each stage drawing its own --
    v15.py's actual design uses exactly one split for everything.
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    n_strata = int(min(val_strata, max(2, len(y_arr) // 10)))
    edges = np.unique(np.quantile(y_arr, np.linspace(0, 1, n_strata + 1)))
    strata = np.digitize(y_arr, edges[1:-1])
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state,
    )
    return list(sss.split(np.zeros(len(y_arr)), strata))


def make_stratified_kfold_splits(
    y: Any,
    n_splits: int = 5,
    val_strata: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Full-coverage counterpart to make_stratified_shuffle_splits: StratifiedKFold
    binned on quantiles of y (same digitize-based stratification, so each
    fold's validation set still spans the target's full range -- the whole
    point of v15.py's stratification), instead of v15.py's own single
    StratifiedShuffleSplit holdout.

    Why this exists: on this dataset (201 rows, ~89 fine-grained), a single
    80/20 holdout starves any downstream stage that only sees a subset of
    those rows -- e.g. MDDFineGrainedResidualCorrector's stratified_split
    mode ended up fitting beta on just 4 validated fine-grained rows, which
    showed up as a near-boundary beta (1.39 of a 1.5 cap) and 3/87 test
    predictions needing the saturation-line clip that the old KFold scheme
    never triggered -- classic small-sample overfitting. Stratifying by
    target quantile IS still worth keeping (it's what improved the actual
    Kaggle leaderboard score over plain KFold), but a single small holdout
    isn't the only way to get that -- StratifiedKFold gives every row a
    validation-fold prediction across the n_splits folds (like plain KFold),
    while still guaranteeing each fold's validation set is quantile-balanced
    (unlike plain KFold's uncontrolled random shuffle).

    Returns a list of (train_idx, val_idx) positional-index pairs, same
    contract as make_stratified_shuffle_splits/_oof_predict_from_splits, so
    it's a drop-in replacement wherever that function is used.
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    n_strata = int(min(val_strata, max(2, len(y_arr) // 10)))
    edges = np.unique(np.quantile(y_arr, np.linspace(0, 1, n_strata + 1)))
    strata = np.digitize(y_arr, edges[1:-1])
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state,
    )
    return list(skf.split(np.zeros(len(y_arr)), strata))


def _oof_predict_from_splits(
    make_estimator: Any,
    X: Any,
    y: Any,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    cross_val_predict's counterpart for splits that don't guarantee full
    coverage (StratifiedShuffleSplit splits, unlike KFold's exhaustive
    partition): fits a fresh estimator per split on its train rows,
    predicts on its validation rows, and averages predictions for any row
    that lands in more than one split's validation set -- the same
    accumulate-then-average pattern v15.py's main() uses (gp_sum/cnt).

    Returns (oof_prediction, validated): validated marks which rows got at
    least one validation-fold prediction; oof_prediction is 0.0
    (meaningless, must be filtered out via validated) wherever it's False.
    """
    is_frame = isinstance(X, pd.DataFrame)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    n = len(y_arr)
    pred_sum = np.zeros(n)
    cnt = np.zeros(n)

    X_arr = X if is_frame else np.asarray(X, dtype=float)

    for tr, va in splits:
        model = make_estimator()
        if is_frame:
            model.fit(X_arr.iloc[tr], y_arr[tr])
            pred = model.predict(X_arr.iloc[va])
        else:
            model.fit(X_arr[tr], y_arr[tr])
            pred = model.predict(X_arr[va])
        pred_sum[va] += np.asarray(pred, dtype=float)
        cnt[va] += 1

    validated = cnt > 0
    oof = np.zeros(n)
    oof[validated] = pred_sum[validated] / cnt[validated]
    return oof, validated


def tune_xgb_with_optuna(
    X: Any,
    y: Any,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[XGBRegressor, optuna.Study]:
    """
    Optuna-tuned replacement for make_default_target_xgb_model /
    make_default_residual_model's hardcoded hyperparameters.

    Searches n_estimators, learning_rate, max_depth, min_child_weight,
    subsample, colsample_bytree, reg_alpha, reg_lambda with TPE sampling.
    Objective is mean CV MAE (matches the competition NMAE metric's
    numerator) over a fixed KFold split, so every trial is evaluated on the
    same folds -- comparable to how alpha/blend weights are estimated
    elsewhere in this module.

    splits:
        Externally-provided (train_idx, val_idx) pairs (e.g. from
        make_stratified_shuffle_splits) to evaluate every trial on instead
        of a fresh KFold -- lets a caller (model2i.py) reuse the exact same
        held-out split across every stage of its pipeline, v15.py-style. If
        None (default), falls back to the original KFold(n_splits,
        shuffle=True, random_state)-based cross_val_score, unchanged for
        every existing caller (model1.py, model1i.py) that doesn't pass it.

    Returns an unfitted XGBRegressor built from the best trial's params
    (caller fits it, same as any other component passed into
    WeightedBlendRegressor/GlobalResidualRegressor/BlendedResidualModel) and
    the Optuna study, for inspecting trial history if needed.
    """

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)

    if splits is None:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective="reg:squarederror",
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            random_state=random_state,
            # segfault-avoidance, same reason as the hardcoded defaults.
            n_jobs=1,
        )

        if splits is None:
            scores = cross_val_score(
                XGBRegressor(**params),
                X_arr,
                y_arr,
                cv=cv,
                scoring="neg_mean_absolute_error",
            )
            return float(-scores.mean())

        oof, validated = _oof_predict_from_splits(
            lambda: XGBRegressor(**params), X_arr, y_arr, splits,
        )
        return float(mean_absolute_error(y_arr[validated], oof[validated]))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=1,
        **study.best_params,
    )

    return best_model, study


def make_default_gpr_model(
    n_features: int,
    random_state: int = 42,
) -> TransformedTargetRegressor:
    """
    Construct the GPR used in data_analysis.ipynb's residual-correction and
    Ridge/GPR/XGBoost blend experiments: ARD-capable RBF (a per-feature
    length-scale, since ``length_scale`` is given as a vector) + white noise,
    on top of standardized features.

    Wrapped in ``TransformedTargetRegressor`` so the target is standardized
    before fitting and predictions are inverse-transformed automatically --
    this mirrors the manual residual-scaling data_analysis.ipynb did by hand
    (``ridge_gpr_residual_oof``'s ``residual_scaler``) before fitting GPR on
    Ridge residuals, since GPR kernel optimization is not scale-invariant and
    residual magnitudes can be much smaller than the raw target.
    """

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=False,
        n_restarts_optimizer=3,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("gpr", gpr),
        ]
    )

    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler(),
    )


class BlendedResidualModel(BaseEstimator, RegressorMixin):
    """
    Weighted blend of XGBoost and a Gaussian Process, for residual correction.

    data_analysis.ipynb's "Model Architecture Testing" section found that
    blending model predictions consistently beat using any single model --
    both for the direct Ridge/GPR/XGBoost ensemble (weights fit by
    constrained optimization: ~0.09/0.37/0.55 for MDD, 0.0/0.27/0.73 for
    OWC) and for GlobalResidualRegressor itself, whose default single-model
    (XGBoost-only) residual correction this class replaces.

    The blend weight is estimated the same way that notebook estimated its
    weights and alpha: by minimizing MSE of the blended out-of-fold
    prediction via bounded scalar optimization. It is not hardcoded, because
    the notebook's fitted weights were for predicting the *target* directly,
    not a residual, and the right ratio is dataset- and target-dependent.

    Parameters
    ----------
    xgb_model, gpr_model:
        Sub-models to blend. Defaults to ``make_default_residual_model`` and
        ``make_default_gpr_model`` respectively.

    weight:
        XGBoost's share of the blend, in [0, 1]. If None (default),
        estimated from out-of-fold predictions.

    n_splits:
        CV folds used to build out-of-fold predictions for weight fitting.
    """

    def __init__(
        self,
        xgb_model: Any | None = None,
        gpr_model: Any | None = None,
        weight: float | None = None,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.xgb_model = xgb_model
        self.gpr_model = gpr_model
        self.weight = weight
        self.n_splits = n_splits
        self.random_state = random_state

    def _get_xgb(self) -> Any:
        if self.xgb_model is None:
            return make_default_residual_model(random_state=self.random_state)

        return clone(self.xgb_model)

    def _get_gpr(self, n_features: int) -> Any:
        if self.gpr_model is None:
            return make_default_gpr_model(
                n_features=n_features,
                random_state=self.random_state,
            )

        return clone(self.gpr_model)

    def fit(self, X: Any, y: Any) -> "BlendedResidualModel":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)

        cv = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        xgb_oof = cross_val_predict(
            self._get_xgb(), X_arr, y_arr, cv=cv,
        )

        gpr_oof = cross_val_predict(
            self._get_gpr(X_arr.shape[1]), X_arr, y_arr, cv=cv,
        )

        if self.weight is None:
            def objective(w: float) -> float:
                blended = w * xgb_oof + (1.0 - w) * gpr_oof
                return float(mean_squared_error(y_arr, blended))

            result = minimize_scalar(
                objective, bounds=(0.0, 1.0), method="bounded",
            )
            self.weight_ = float(result.x)
        else:
            self.weight_ = float(np.clip(self.weight, 0.0, 1.0))

        self.xgb_model_ = self._get_xgb()
        self.xgb_model_.fit(X_arr, y_arr)

        self.gpr_model_ = self._get_gpr(X_arr.shape[1])
        self.gpr_model_.fit(X_arr, y_arr)

        self.is_fitted_ = True

        return self

    def predict(self, X: Any) -> np.ndarray:
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError(
                "BlendedResidualModel must be fitted before calling predict."
            )

        X_arr = np.asarray(X, dtype=float)

        xgb_pred = self.xgb_model_.predict(X_arr)
        gpr_pred = self.gpr_model_.predict(X_arr)

        return self.weight_ * xgb_pred + (1.0 - self.weight_) * gpr_pred


def make_default_blended_residual_model(
    random_state: int = 42,
) -> BlendedResidualModel:
    """Construct the default (XGBoost + GPR blend) residual-correction model."""

    return BlendedResidualModel(random_state=random_state)


class _ValidatedRegressorMixin:
    """
    Shared X/y validation for the sklearn-style regressors in this module.

    Both GlobalResidualRegressor and WeightedBlendRegressor need identical
    input contracts (no missing values, consistent columns between fit and
    predict, etc.), so this is factored out rather than duplicated.
    """

    @staticmethod
    def _validate_y(y: Any) -> np.ndarray:
        y_array = np.asarray(
            y,
            dtype=float,
        ).reshape(-1)

        if not np.isfinite(y_array).all():
            raise ValueError(
                "y contains NaN or infinite values."
            )

        return y_array

    def _validate_X_fit(
        self,
        X: Any,
    ) -> pd.DataFrame | np.ndarray:
        if isinstance(X, pd.DataFrame):
            if X.columns.duplicated().any():
                duplicated = X.columns[
                    X.columns.duplicated()
                ].tolist()

                raise ValueError(
                    f"X contains duplicate columns: {duplicated}"
                )

            if X.isna().any().any():
                missing_columns = X.columns[
                    X.isna().any()
                ].tolist()

                raise ValueError(
                    "X contains missing values in columns: "
                    f"{missing_columns}"
                )

            X_checked = X.copy()
            self.feature_names_in_ = np.asarray(
                X_checked.columns,
                dtype=object,
            )

        else:
            X_checked = np.asarray(
                X,
                dtype=float,
            )

            if X_checked.ndim != 2:
                raise ValueError(
                    "X must be a two-dimensional array."
                )

            if not np.isfinite(X_checked).all():
                raise ValueError(
                    "X contains NaN or infinite values."
                )

        self.n_features_in_ = X_checked.shape[1]

        return X_checked

    def _validate_X_predict(
        self,
        X: Any,
    ) -> pd.DataFrame | np.ndarray:
        if isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names_in_"):
                expected = list(self.feature_names_in_)
                received = list(X.columns)

                missing = [
                    col
                    for col in expected
                    if col not in received
                ]

                extra = [
                    col
                    for col in received
                    if col not in expected
                ]

                if missing:
                    raise ValueError(
                        "Prediction data is missing columns: "
                        f"{missing}"
                    )

                if extra:
                    raise ValueError(
                        "Prediction data contains unexpected "
                        f"columns: {extra}"
                    )

                X = X.loc[:, expected].copy()

            if X.isna().any().any():
                raise ValueError(
                    "Prediction data contains missing values."
                )

            X_checked = X

        else:
            X_checked = np.asarray(
                X,
                dtype=float,
            )

            if X_checked.ndim != 2:
                raise ValueError(
                    "X must be a two-dimensional array."
                )

            if not np.isfinite(X_checked).all():
                raise ValueError(
                    "X contains NaN or infinite values."
                )

        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "Incorrect number of prediction features. "
                f"Expected {self.n_features_in_}, "
                f"received {X_checked.shape[1]}."
            )

        return X_checked

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError(
                f"{type(self).__name__} must be fitted before calling predict."
            )


class GlobalResidualRegressor(_ValidatedRegressorMixin, BaseEstimator, RegressorMixin):
    """
    Two-stage global regression model.

    Stage 1
    -------
    Ridge estimates the dominant global linear relationship:

        base_prediction = Ridge(X)

    Stage 2
    -------
    A nonlinear model predicts the residual left by Ridge:

        ridge_residual = y - base_oof_prediction

        residual_prediction = residual_model(X)

    By default this is a weighted blend of XGBoost and a Gaussian Process
    (BlendedResidualModel) rather than XGBoost alone -- data_analysis.ipynb
    found blending consistently beat any single residual model.

    Final prediction
    ----------------

        final_prediction = base_prediction + alpha * residual_prediction

    Parameters
    ----------
    base_model:
        Global model used to capture the dominant relationship.
        Usually a standardized RidgeCV pipeline.

    residual_model:
        Nonlinear model trained on out-of-fold base-model residuals.
        Defaults to BlendedResidualModel (XGBoost + GPR, OOF-weighted).

    alpha:
        Residual-correction weight. If None, alpha is estimated from
        out-of-fold residual-model predictions.

    alpha_bounds:
        Lower and upper limits for the learned alpha.

    n_splits:
        Number of CV folds used to construct residual targets.

    random_state:
        Random seed for shuffled cross-validation.
    """

    def __init__(
        self,
        base_model: Any | None = None,
        residual_model: Any | None = None,
        alpha: float | None = None,
        alpha_bounds: tuple[float, float] = (0.0, 1.5),
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.base_model = base_model
        self.residual_model = residual_model
        self.alpha = alpha
        self.alpha_bounds = alpha_bounds
        self.n_splits = n_splits
        self.random_state = random_state

    def _get_base_model(self) -> Any:
        if self.base_model is None:
            return make_default_ridge_model()

        return clone(self.base_model)

    def _get_residual_model(self) -> Any:
        if self.residual_model is None:
            return make_default_blended_residual_model(
                random_state=self.random_state
            )

        return clone(self.residual_model)

    def _make_cv(self) -> KFold:
        if self.n_splits < 2:
            raise ValueError(
                "n_splits must be at least 2."
            )

        return KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

    @staticmethod
    def _estimate_alpha(
        residual_target: np.ndarray,
        residual_prediction: np.ndarray,
        bounds: tuple[float, float],
    ) -> float:
        """
        Find alpha minimizing

            ||residual_target
              - alpha * residual_prediction||².

        The unconstrained least-squares solution is

            alpha =
                (r_hat^T r) / (r_hat^T r_hat).

        It is then clipped to alpha_bounds.
        """

        residual_target = np.asarray(
            residual_target,
            dtype=float,
        ).reshape(-1)

        residual_prediction = np.asarray(
            residual_prediction,
            dtype=float,
        ).reshape(-1)

        denominator = float(
            np.dot(
                residual_prediction,
                residual_prediction,
            )
        )

        if denominator <= 1e-12:
            return 0.0

        numerator = float(
            np.dot(
                residual_prediction,
                residual_target,
            )
        )

        alpha_unconstrained = numerator / denominator

        lower, upper = bounds

        if lower > upper:
            raise ValueError(
                "alpha_bounds must satisfy lower <= upper."
            )

        return float(
            np.clip(
                alpha_unconstrained,
                lower,
                upper,
            )
        )

    def fit(
        self,
        X: Any,
        y: Any,
    ) -> "GlobalResidualRegressor":
        """
        Fit Ridge and the nonlinear residual-correction model.
        """

        X_checked = self._validate_X_fit(X)
        y_checked = self._validate_y(y)

        if len(X_checked) != len(y_checked):
            raise ValueError(
                "X and y contain different numbers of samples."
            )

        if len(y_checked) < self.n_splits:
            raise ValueError(
                "The number of samples must be at least "
                "as large as n_splits."
            )

        cv = self._make_cv()

        # ---------------------------------------------------------
        # Stage 1: unbiased out-of-fold Ridge predictions
        # ---------------------------------------------------------
        base_template = self._get_base_model()

        base_oof = cross_val_predict(
            estimator=base_template,
            X=X_checked,
            y=y_checked,
            cv=cv,
            method="predict",
            n_jobs=None,
        )

        ridge_residual_target = (
            y_checked - base_oof
        )

        # ---------------------------------------------------------
        # Stage 2: OOF residual-model predictions
        #
        # These predictions are used to estimate alpha without
        # evaluating the residual model on its own training rows.
        # ---------------------------------------------------------
        residual_template = (
            self._get_residual_model()
        )

        residual_oof = cross_val_predict(
            estimator=residual_template,
            X=X_checked,
            y=ridge_residual_target,
            cv=cv,
            method="predict",
            n_jobs=None,
        )

        if self.alpha is None:
            alpha_fitted = self._estimate_alpha(
                residual_target=ridge_residual_target,
                residual_prediction=residual_oof,
                bounds=self.alpha_bounds,
            )
        else:
            lower, upper = self.alpha_bounds

            alpha_fitted = float(
                np.clip(
                    self.alpha,
                    lower,
                    upper,
                )
            )

        global_oof = (
            base_oof
            + alpha_fitted * residual_oof
        )

        # ---------------------------------------------------------
        # Refit both model components on all available data
        # ---------------------------------------------------------
        self.base_model_ = self._get_base_model()
        self.residual_model_ = (
            self._get_residual_model()
        )

        self.base_model_.fit(
            X_checked,
            y_checked,
        )

        self.residual_model_.fit(
            X_checked,
            ridge_residual_target,
        )

        # Store fitted quantities for diagnostics
        self.alpha_ = alpha_fitted

        self.base_oof_prediction_ = base_oof
        self.residual_target_ = (
            ridge_residual_target
        )
        self.residual_oof_prediction_ = (
            residual_oof
        )
        self.global_oof_prediction_ = global_oof

        self.base_oof_metrics_ = (
            calculate_regression_metrics(
                y_checked,
                base_oof,
            )
        )

        self.global_oof_metrics_ = (
            calculate_regression_metrics(
                y_checked,
                global_oof,
            )
        )

        self.is_fitted_ = True

        return self

    def predict_components(
        self,
        X: Any,
    ) -> pd.DataFrame:
        """
        Return Ridge, residual-correction and final predictions.
        """

        self._check_is_fitted()

        X_checked = self._validate_X_predict(X)

        base_prediction = (
            self.base_model_.predict(X_checked)
        )

        raw_residual_prediction = (
            self.residual_model_.predict(X_checked)
        )

        weighted_residual_correction = (
            self.alpha_
            * raw_residual_prediction
        )

        final_prediction = (
            base_prediction
            + weighted_residual_correction
        )

        if isinstance(X, pd.DataFrame):
            index = X.index
        else:
            index = pd.RangeIndex(
                len(final_prediction)
            )

        return pd.DataFrame(
            {
                "ridge_prediction":
                    base_prediction,

                "raw_residual_prediction":
                    raw_residual_prediction,

                "weighted_residual_correction":
                    weighted_residual_correction,

                "final_prediction":
                    final_prediction,
            },
            index=index,
        )

    def predict(
        self,
        X: Any,
    ) -> np.ndarray:
        """Return the final hybrid prediction."""

        components = self.predict_components(X)

        return components[
            "final_prediction"
        ].to_numpy()

    def get_oof_results(
        self,
        y: Any | None = None,
        index: Any | None = None,
    ) -> pd.DataFrame:
        """
        Return the training out-of-fold predictions and residuals.

        These are the values that should be used for residual
        diagnostics and for training a later soil-specific correction.
        """

        self._check_is_fitted()

        n_samples = len(
            self.global_oof_prediction_
        )

        if index is None:
            index = pd.RangeIndex(n_samples)

        results = pd.DataFrame(
            {
                "ridge_oof_prediction":
                    self.base_oof_prediction_,

                "ridge_residual":
                    self.residual_target_,

                "residual_model_oof_prediction":
                    self.residual_oof_prediction_,

                "global_oof_prediction":
                    self.global_oof_prediction_,
            },
            index=index,
        )

        if y is not None:
            y_array = self._validate_y(y)

            if len(y_array) != n_samples:
                raise ValueError(
                    "Provided y does not match the fitted "
                    "sample count."
                )

            results.insert(
                0,
                "observed",
                y_array,
            )

            results[
                "remaining_residual"
            ] = (
                y_array
                - self.global_oof_prediction_
            )

        return results

    def get_training_summary(
        self,
    ) -> dict[str, Any]:
        """Return fitted alpha and OOF performance."""

        self._check_is_fitted()

        return {
            "alpha": self.alpha_,
            "ridge_oof_metrics":
                self.base_oof_metrics_.as_dict(),
            "global_hybrid_oof_metrics":
                self.global_oof_metrics_.as_dict(),
        }


class WeightedBlendRegressor(_ValidatedRegressorMixin, BaseEstimator, RegressorMixin):
    """
    Direct weighted blend of Ridge, GPR, and XGBoost predictions.

    data_analysis.ipynb's "Model Architecture Testing" section compared this
    against Ridge + residual correction (what GlobalResidualRegressor does)
    and found blending wins on both targets:

        MDD: blend OOF R² 0.850  vs  residual-correction hybrid R² 0.781
        OWC: blend OOF R² 0.805  vs  residual-correction hybrid R² 0.302

    Unlike GlobalResidualRegressor, none of the three base models predicts
    another model's residual -- each is fit directly on y:

        ridge_prediction = Ridge(X)
        gpr_prediction    = GPR(X)
        xgb_prediction    = XGBoost(X)

        final_prediction = w_ridge * ridge_prediction
                          + w_gpr   * gpr_prediction
                          + w_xgb   * xgb_prediction

    The weights are fit on out-of-fold predictions by constrained
    optimization (SLSQP: each weight in [0, 1], weights sum to 1, minimizing
    MSE) -- the same ``find_weights`` procedure data_analysis.ipynb used.

    Parameters
    ----------
    ridge_model, gpr_model, xgb_model:
        Base models to blend. Default to ``make_default_ridge_model``,
        ``make_default_gpr_model``, and ``make_default_target_xgb_model``
        respectively -- the latter is less regularized than
        GlobalResidualRegressor's residual-model default, since it has to
        explain the full target rather than Ridge's leftover error.

    n_splits:
        CV folds used to build out-of-fold predictions for weight fitting.
        Ignored if `splits` is passed.

    random_state:
        Random seed for shuffled cross-validation and base-model fitting.

    splits:
        Externally-provided (train_idx, val_idx) pairs (e.g. from
        make_stratified_shuffle_splits) to use instead of a fresh
        KFold(n_splits) -- lets a caller reuse the exact same held-out
        split across a whole pipeline, v15.py-style. Unlike KFold, these
        splits aren't required to cover every row; rows never landing in
        any split's validation set get `validated=False` in
        get_oof_results() and NaN OOF predictions/metrics computed only
        over the validated subset. If None (default), behavior is
        unchanged from before this parameter existed: a fresh
        KFold(n_splits, shuffle=True, random_state) via cross_val_predict,
        covering every row.
    """

    def __init__(
        self,
        ridge_model: Any | None = None,
        gpr_model: Any | None = None,
        xgb_model: Any | None = None,
        n_splits: int = 5,
        random_state: int = 42,
        splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.ridge_model = ridge_model
        self.gpr_model = gpr_model
        self.xgb_model = xgb_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.splits = splits

    def _get_ridge(self) -> Any:
        if self.ridge_model is None:
            return make_default_ridge_model()

        return clone(self.ridge_model)

    def _get_gpr(self, n_features: int) -> Any:
        if self.gpr_model is None:
            return make_default_gpr_model(
                n_features=n_features,
                random_state=self.random_state,
            )

        return clone(self.gpr_model)

    def _get_xgb(self) -> Any:
        if self.xgb_model is None:
            return make_default_target_xgb_model(random_state=self.random_state)

        return clone(self.xgb_model)

    def _make_cv(self) -> KFold:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

        return KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

    @staticmethod
    def _fit_weights(
        predictions: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Find non-negative weights summing to 1 that minimize the blended
        prediction's SSE against y -- data_analysis.ipynb's ``find_weights``.
        """

        n_models = predictions.shape[1]

        # SSE rather than MSE, so the objective isn't unnecessarily tiny
        # relative to ftol (matches data_analysis.ipynb's find_weights).
        def objective(weights: np.ndarray) -> float:
            residuals = y - predictions @ weights
            return float(np.sum(residuals ** 2))

        result = minimize(
            objective,
            x0=np.full(n_models, 1.0 / n_models),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints={
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - 1.0,
            },
            options={
                "ftol": 1e-12,
                "maxiter": 5000,
                "disp": False,
            },
        )

        if not result.success:
            # Graceful degradation instead of crashing (matches
            # data_analysis.ipynb's find_weights): fall back to a one-hot
            # weight vector on whichever single model has the lowest MSE.
            mse = np.mean((predictions - y[:, None]) ** 2, axis=0)
            best = np.argmin(mse)
            weights = np.zeros(n_models)
            weights[best] = 1.0
        else:
            weights = result.x

        # Remove insignificant numerical noise and renormalize.
        weights[np.abs(weights) < 1e-10] = 0.0
        weights /= weights.sum()

        return weights

    def fit(self, X: Any, y: Any) -> "WeightedBlendRegressor":
        """Fit Ridge, GPR, and XGBoost, and the weights that blend them."""

        X_checked = self._validate_X_fit(X)
        y_checked = self._validate_y(y)

        if len(X_checked) != len(y_checked):
            raise ValueError(
                "X and y contain different numbers of samples."
            )

        n_features = (
            X_checked.shape[1]
            if isinstance(X_checked, np.ndarray)
            else len(X_checked.columns)
        )

        if self.splits is not None:
            # ---------------------------------------------------------
            # v15.py-style: externally-provided splits, not guaranteed to
            # cover every row -- see _oof_predict_from_splits.
            # ---------------------------------------------------------
            ridge_oof, validated = _oof_predict_from_splits(
                self._get_ridge, X_checked, y_checked, self.splits,
            )
            gpr_oof, _ = _oof_predict_from_splits(
                lambda: self._get_gpr(n_features), X_checked, y_checked, self.splits,
            )
            xgb_oof, _ = _oof_predict_from_splits(
                self._get_xgb, X_checked, y_checked, self.splits,
            )

            if not validated.any():
                raise ValueError(
                    "No rows were validated by any split -- check "
                    "test_size/n_splits, or pass splits=None for KFold."
                )

            predictions = np.column_stack([ridge_oof, gpr_oof, xgb_oof])
            self.weights_ = self._fit_weights(
                predictions[validated], y_checked[validated],
            )

            blend_oof = np.full(len(y_checked), np.nan)
            blend_oof[validated] = predictions[validated] @ self.weights_
            ridge_oof = np.where(validated, ridge_oof, np.nan)
            gpr_oof = np.where(validated, gpr_oof, np.nan)
            xgb_oof = np.where(validated, xgb_oof, np.nan)

            self.ridge_oof_metrics_ = calculate_regression_metrics(y_checked[validated], ridge_oof[validated])
            self.gpr_oof_metrics_ = calculate_regression_metrics(y_checked[validated], gpr_oof[validated])
            self.xgb_oof_metrics_ = calculate_regression_metrics(y_checked[validated], xgb_oof[validated])
            self.blend_oof_metrics_ = calculate_regression_metrics(y_checked[validated], blend_oof[validated])
            self.validated_ = validated
        else:
            if len(y_checked) < self.n_splits:
                raise ValueError(
                    "The number of samples must be at least "
                    "as large as n_splits."
                )

            cv = self._make_cv()

            ridge_oof = cross_val_predict(
                self._get_ridge(), X_checked, y_checked, cv=cv,
            )

            gpr_oof = cross_val_predict(
                self._get_gpr(n_features), X_checked, y_checked, cv=cv,
            )

            xgb_oof = cross_val_predict(
                self._get_xgb(), X_checked, y_checked, cv=cv,
            )

            predictions = np.column_stack([ridge_oof, gpr_oof, xgb_oof])
            self.weights_ = self._fit_weights(predictions, y_checked)
            blend_oof = predictions @ self.weights_

            self.ridge_oof_metrics_ = calculate_regression_metrics(y_checked, ridge_oof)
            self.gpr_oof_metrics_ = calculate_regression_metrics(y_checked, gpr_oof)
            self.xgb_oof_metrics_ = calculate_regression_metrics(y_checked, xgb_oof)
            self.blend_oof_metrics_ = calculate_regression_metrics(y_checked, blend_oof)
            self.validated_ = np.ones(len(y_checked), dtype=bool)

        # ---------------------------------------------------------
        # Refit all three base models on all available data
        # ---------------------------------------------------------
        self.ridge_model_ = self._get_ridge()
        self.ridge_model_.fit(X_checked, y_checked)

        self.gpr_model_ = self._get_gpr(n_features)
        self.gpr_model_.fit(X_checked, y_checked)

        self.xgb_model_ = self._get_xgb()
        self.xgb_model_.fit(X_checked, y_checked)

        # Store fitted quantities for diagnostics
        self.ridge_oof_prediction_ = ridge_oof
        self.gpr_oof_prediction_ = gpr_oof
        self.xgb_oof_prediction_ = xgb_oof
        self.blend_oof_prediction_ = blend_oof

        self.is_fitted_ = True

        return self

    def predict_components(self, X: Any) -> pd.DataFrame:
        """Return each base model's prediction plus the final blend."""

        self._check_is_fitted()

        X_checked = self._validate_X_predict(X)

        ridge_prediction = self.ridge_model_.predict(X_checked)
        gpr_prediction = self.gpr_model_.predict(X_checked)
        xgb_prediction = self.xgb_model_.predict(X_checked)

        predictions = np.column_stack(
            [ridge_prediction, gpr_prediction, xgb_prediction]
        )
        final_prediction = predictions @ self.weights_

        if isinstance(X, pd.DataFrame):
            index = X.index
        else:
            index = pd.RangeIndex(len(final_prediction))

        return pd.DataFrame(
            {
                "ridge_prediction": ridge_prediction,
                "gpr_prediction": gpr_prediction,
                "xgb_prediction": xgb_prediction,
                "final_prediction": final_prediction,
            },
            index=index,
        )

    def predict(self, X: Any) -> np.ndarray:
        """Return the final blended prediction."""

        components = self.predict_components(X)

        return components["final_prediction"].to_numpy()

    def get_oof_results(
        self,
        y: Any | None = None,
        index: Any | None = None,
    ) -> pd.DataFrame:
        """
        Return the training out-of-fold predictions and residuals.

        These are the values that should be used for residual diagnostics
        and for training a later soil-specific correction.

        `validated` marks which rows actually received an OOF prediction --
        always all-True when fit with KFold (splits=None); when fit with
        externally-provided splits (v15.py-style), only rows that landed in
        some split's validation set are True, and every other column is NaN
        for the rest -- a caller building a correction on top (e.g.
        MDDFineGrainedResidualCorrector) must filter to `validated` before
        using blend_oof_prediction/remaining_residual as inputs, to avoid
        training on NaN or on leaked (in-sample) predictions.
        """

        self._check_is_fitted()

        n_samples = len(self.blend_oof_prediction_)

        if index is None:
            index = pd.RangeIndex(n_samples)

        results = pd.DataFrame(
            {
                "ridge_oof_prediction": self.ridge_oof_prediction_,
                "gpr_oof_prediction": self.gpr_oof_prediction_,
                "xgb_oof_prediction": self.xgb_oof_prediction_,
                "blend_oof_prediction": self.blend_oof_prediction_,
                "validated": self.validated_,
            },
            index=index,
        )

        if y is not None:
            y_array = self._validate_y(y)

            if len(y_array) != n_samples:
                raise ValueError(
                    "Provided y does not match the fitted sample count."
                )

            results.insert(0, "observed", y_array)
            results["remaining_residual"] = y_array - self.blend_oof_prediction_

        return results

    def get_training_summary(self) -> dict[str, Any]:
        """Return fitted blend weights and each model's OOF performance."""

        self._check_is_fitted()

        return {
            "weights": {
                "ridge": float(self.weights_[0]),
                "gpr": float(self.weights_[1]),
                "xgboost": float(self.weights_[2]),
            },
            "n_validated": int(self.validated_.sum()),
            "ridge_oof_metrics": self.ridge_oof_metrics_.as_dict(),
            "gpr_oof_metrics": self.gpr_oof_metrics_.as_dict(),
            "xgb_oof_metrics": self.xgb_oof_metrics_.as_dict(),
            "blend_oof_metrics": self.blend_oof_metrics_.as_dict(),
        }