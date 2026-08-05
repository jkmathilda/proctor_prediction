"""
mdd_fine_correction.py
======================

Residual correction model for MDD predictions on fine-grained soils.

Wraps an already-fitted general model, learns out-of-fold residuals, and
trains a weighted ensemble of Optuna-tuned XGBoost and Gaussian Process
Regression to predict residuals from specialist soil features
(Atterberg limits, PI, LOI, kf). Corrected predictions are obtained by
adding the predicted residual to the general model prediction.

Only fine-grained soils are corrected; coarse-grained soils use the
general model prediction unchanged. Supports both standard K-Fold and
quantile-stratified K-Fold cross-validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from xgboost import XGBRegressor

from src.general_model_impute import (
    add_imputed_features,
    make_default_gpr_model,
    make_stratified_kfold_splits,
    make_stratified_shuffle_splits,
    tune_xgb_with_optuna,
    _oof_predict_from_splits,
)


# data_analysis.ipynb's specialist_features_c (cell 134) -- the best-scoring
# of the three specialist feature groups it compared. `blend_oof_prediction`
# is the general model's own prediction, included as a specialist feature so
# the residual model can see how confident/wrong the general model already
# is at a given point, not just the raw soil properties.
SPECIALIST_FEATURES_C: list[str] = [
    "blend_oof_prediction",
    "atterberg_liquid_limit_pct_completed",
    "atterberg_plastic_limit_pct_completed",
    "feat_pi_completed",
    "loss_on_ignition_pct_completed",
    "log10_hyd_cond_kf_m_s_completed",
    "atterberg_liquid_limit_pct_was_missing",
    "atterberg_plastic_limit_pct_was_missing",
    "loss_on_ignition_pct_was_missing",
    "hyd_cond_kf_m_s_was_missing",
]

# SPECIALIST_FEATURES_C minus `blend_oof_prediction` -- the columns that come
# from raw/imputed data (add_specialist_derived_features), as opposed to the
# one column that comes from whichever general model is being corrected.
# Kept alongside SPECIALIST_FEATURES_C so callers don't each recompute this
# split themselves.
SPECIALIST_RAW_FEATURES: list[str] = [
    c for c in SPECIALIST_FEATURES_C if c != "blend_oof_prediction"
]


def add_specialist_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive SPECIALIST_RAW_FEATURES' one non-imputed column,
    feat_pi_completed = LL_completed - PL_completed, matching cell 130's
    dff_impute["feat_pi_completed"].

    Assumes `df` has already been through
    general_model_impute.add_imputed_features (or add_group_c_features
    below) -- i.e. the atterberg_*_completed/loss_on_ignition_pct_completed/
    log10_hyd_cond_kf_m_s_completed/*_was_missing columns already exist.
    Kept separate from add_group_c_features so a caller that has already run
    add_imputed_features for its own feature set (e.g. IMPUTED_FEATURES)
    doesn't need to MICE-impute the same columns a second time just to get
    this one derived column.
    """
    df = df.copy()
    df["feat_pi_completed"] = (
        df["atterberg_liquid_limit_pct_completed"] - df["atterberg_plastic_limit_pct_completed"]
    )
    return df


def add_group_c_features(
    df: pd.DataFrame,
    fine_imputer: Any | None = None,
    coarse_imputer: Any | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Any, Any]:
    """
    Derive every SPECIALIST_FEATURES_C column except `blend_oof_prediction`
    (that one comes from whichever general model is being corrected, not
    from raw data -- see MDDFineGrainedResidualCorrector).

    Reuses general_model_impute.add_imputed_features for the MICE-imputed
    Atterberg/LOI/kf columns (fine-grained-only for Atterberg limits,
    matching data_analysis.ipynb's df_impute construction), then
    add_specialist_derived_features for feat_pi_completed.

    For a caller that hasn't already MICE-imputed these columns for some
    other feature set. If it has (e.g. general_model_impute.IMPUTED_FEATURES),
    call add_specialist_derived_features directly on that already-imputed
    data instead, to avoid fitting a second, redundant imputer.

    Pass previously-fitted fine_imputer/coarse_imputer (as returned from an
    earlier call on training data) to transform test data without
    refitting/leaking, same convention as add_imputed_features.
    """
    df_out, fine_imputer, coarse_imputer = add_imputed_features(
        df,
        fine_imputer=fine_imputer,
        coarse_imputer=coarse_imputer,
        random_state=random_state,
    )

    df_out = add_specialist_derived_features(df_out)

    return df_out, fine_imputer, coarse_imputer


def make_default_specialist_xgb(random_state: int = 42) -> XGBRegressor:
    """data_analysis.ipynb's evaluate_specialist() model (cell 135), unchanged."""
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
    )


class MDDFineGrainedResidualCorrector(BaseEstimator, RegressorMixin):
    """
    Fits a residual model on SPECIALIST_FEATURES_C to predict the residual
    a general model leaves on fine-grained MDD rows, then blends it back in
    at a learned strength:

        remaining_residual   = observed - general_model.predict(X)
        residual_prediction  = residual_model(specialist_features_c)
        corrected_prediction = general_prediction + beta * residual_prediction

    The residual model is always a weighted ensemble of XGBoost (Optuna-tuned
    by default, optuna_trials > 0; make_default_specialist_xgb()'s fixed
    hyperparameters otherwise) and general_model_impute.make_default_gpr_model()
    (which tunes its own kernel by marginal-likelihood optimization as part of
    fit() -- Optuna is not applied to it). The blend weight is fit by bounded
    scalar optimization on out-of-fold predictions, the same procedure
    general_model_impute's BlendedResidualModel uses -- see ensemble_weight_.

    This class used to pick per-run among XGBoost-alone / GPR-alone / this
    ensemble by comparing their OOF MSE (data_analysis.ipynb's
    evaluate_specialist() only ever tried XGBoost, so there was no notebook
    citation for that choice either way). In practice XGBoost-alone never
    beat the ensemble by a meaningful margin across repeated runs and GPR-alone
    was consistently worse than no correction at all (negative OOF R^2) -- and
    the ensemble's own weight-fitting already collapses toward whichever of
    the two is better on a given fit, so the extra comparison just added
    run-to-run flip-flopping without changing the outcome. Simplified to
    always use the ensemble; see candidate_oof_metrics_ / get_training_summary()
    for xgboost-alone/gpr-alone/ensemble OOF metrics if you want to sanity-check
    that on new data.

    beta is fit by bounded scalar optimization (0 <= beta <= 1.5) minimizing
    MSE against observed MDD on fine-grained rows -- data_analysis.ipynb's
    evaluate_specialist() procedure, unchanged.

    Parameters
    ----------
    general_model:
        An already-fitted model exposing `.predict(X)` and
        `.get_oof_results(y, index)` returning a DataFrame with
        `blend_oof_prediction`/`remaining_residual` columns -- e.g.
        general_model.WeightedBlendRegressor or
        general_model_impute.WeightedBlendRegressor. Never refit here. If
        stratified_split=True, get_oof_results() must also expose a
        `validated` column (i.e. general_model itself must have been fit
        with splits=... -- see general_model_impute.WeightedBlendRegressor).

    xgb_model:
        Override for the XGBoost candidate. If None (default), see
        optuna_trials. Passing an explicit model skips Optuna tuning
        entirely, same convention as general_model_impute's WeightedBlendRegressor.

    gpr_model:
        Override for the GPR candidate. Defaults to
        general_model_impute.make_default_gpr_model().

    optuna_trials:
        Optuna trials for tuning the XGBoost candidate on the fine-grained
        residual target (0 to skip tuning and use
        make_default_specialist_xgb() instead). Ignored if xgb_model is
        given explicitly.

    n_splits, random_state:
        CV folds / seed used to build out-of-fold residual predictions for
        Optuna tuning and beta-fitting when stratified_split=False (KFold,
        full coverage -- the default). random_state is also used for
        stratified_split=True's own splitting.

    beta_bounds:
        Lower/upper limits for the learned correction strength.

    stratified_split:
        False (default): draw a fresh plain KFold(n_splits) internally,
        exactly as before this parameter existed -- every fine-grained row
        gets an out-of-fold residual prediction.

        True: quantile-stratified K-fold instead of plain KFold. Requires
        general_model to have been fit with its own externally-provided
        splits (so get_oof_results() has a `validated` column). This
        corrector then: (1) restricts to `usable_mask = fine_mask &
        general_model_oof["validated"]` -- only fine-grained rows the
        general model itself validated have a legitimate (non-leaked) OOF
        prediction to use as a specialist feature (with the general model
        also on StratifiedKFold, this is effectively all of fine_mask, since
        StratifiedKFold covers every row -- unlike a single-holdout scheme,
        which would starve this down to a handful of rows); (2) draws its
        OWN stratified-K-fold split (make_stratified_kfold_splits, using
        folds/val_strata below) on that subset, so residual_oof_r2_/
        corrected_oof_r2_/etc. are honest out-of-fold metrics over
        (typically) the full fine-grained set, not a tiny holdout; (3)
        refits the final xgb_residual_model_/gpr_residual_model_ on ALL of
        usable_mask, same "validate small, deploy on everything" pattern
        used throughout this codebase.

        See module docstring for why this replaced an earlier single-holdout
        (StratifiedShuffleSplit) version of stratified_split -- it beat plain
        KFold on Kaggle but its tiny validated sample (4 rows) caused visible
        overfitting in this correction stage specifically.

    val_strata, folds:
        Only used when stratified_split=True -- passed straight to
        make_stratified_kfold_splits/make_stratified_shuffle_splits
        (val_strata, n_splits) for this corrector's own internal split.

    split_kind:
        Only used when stratified_split=True. "kfold" (default): this
        corrector's own internal split is make_stratified_kfold_splits
        (full coverage, folds foldscount). "shuffle": the original
        scripts/v15.py-literal single-holdout scheme instead
        (make_stratified_shuffle_splits, test_size=val_frac) -- kept only
        for comparing the two schemes against a held-out set (see
        scripts/evaluate_validation_schemes.py); "kfold" is what
        model2i.py actually uses, for the reasons in the module docstring.

    val_frac:
        Only used when stratified_split=True and split_kind="shuffle" --
        test_size passed to make_stratified_shuffle_splits.
    """

    def __init__(
        self,
        general_model: Any,
        xgb_model: Any | None = None,
        gpr_model: Any | None = None,
        optuna_trials: int = 50,
        n_splits: int = 5,
        random_state: int = 42,
        beta_bounds: tuple[float, float] = (0.0, 1.5),
        stratified_split: bool = False,
        val_strata: int = 5,
        folds: int = 5,
        split_kind: str = "kfold",
        val_frac: float = 0.2,
    ) -> None:
        self.general_model = general_model
        self.xgb_model = xgb_model
        self.gpr_model = gpr_model
        self.optuna_trials = optuna_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.beta_bounds = beta_bounds
        self.stratified_split = stratified_split
        self.val_strata = val_strata
        self.folds = folds
        self.split_kind = split_kind
        self.val_frac = val_frac

    def _get_gpr(self, n_features: int) -> Any:
        if self.gpr_model is None:
            return make_default_gpr_model(
                n_features=n_features, random_state=self.random_state,
            )

        return clone(self.gpr_model)

    def _make_cv(self) -> KFold:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

        return KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

    def fit(
        self,
        X_general: pd.DataFrame,
        X_specialist: pd.DataFrame,
        y: Any,
        fine_mask: Any,
    ) -> "MDDFineGrainedResidualCorrector":
        """
        X_general:
            The general model's own feature matrix -- the same one it was
            already `.fit()` on. Used only to call `get_oof_results()`.

        X_specialist:
            DataFrame with SPECIALIST_FEATURES_C's columns except
            `blend_oof_prediction` (see add_group_c_features), row-aligned
            with X_general.

        y:
            Full training target (MDD), row-aligned with X_general.

        fine_mask:
            Boolean array/Series, row-aligned with X_general, True for
            fine-grained rows. Only these rows are used to fit the
            correction -- coarse-grained rows are untouched at predict
            time. (With stratified_split=True, only the subset of these
            rows the general model itself validated is actually usable --
            see usable_mask below.)
        """
        if not getattr(self.general_model, "is_fitted_", False):
            raise ValueError(
                "general_model must already be fitted before fitting a "
                "correction on top of it."
            )

        fine_mask = np.asarray(fine_mask, dtype=bool)
        y_array = np.asarray(y, dtype=float)

        if len(fine_mask) != len(X_general) or len(y_array) != len(X_general):
            raise ValueError(
                "X_general, y, and fine_mask must all have the same length."
            )

        if not fine_mask.any():
            raise ValueError("fine_mask selects zero rows -- nothing to fit.")

        oof = self.general_model.get_oof_results(y=y_array, index=X_general.index)

        if self.stratified_split:
            # -----------------------------------------------------
            # Only fine-grained rows the general model itself validated
            # have a legitimate (non-leaked) OOF prediction to build a
            # specialist feature from. With the general model also on
            # StratifiedKFold, this is effectively all of fine_mask.
            # -----------------------------------------------------
            if "validated" not in oof.columns:
                raise ValueError(
                    "stratified_split=True requires general_model to have "
                    "been fit with its own externally-provided splits (so "
                    "get_oof_results() has a 'validated' column) -- e.g. "
                    "WeightedBlendRegressor(splits=make_stratified_kfold_splits(...))."
                )

            usable_mask = fine_mask & oof["validated"].to_numpy()

            if usable_mask.sum() < 10:
                raise ValueError(
                    f"Only {int(usable_mask.sum())} fine-grained rows were "
                    "validated by the general model's own split -- too few "
                    "to fit a correction on. Increase folds/val_strata at "
                    "the general-model level, or use stratified_split=False "
                    "(plain KFold, full coverage)."
                )

            specialist_usable = X_specialist.loc[usable_mask].copy()
            specialist_usable.insert(
                0, "blend_oof_prediction", oof.loc[usable_mask, "blend_oof_prediction"].to_numpy()
            )
            X_fit = specialist_usable[SPECIALIST_FEATURES_C]
            y_fit = oof.loc[usable_mask, "remaining_residual"].to_numpy()

            if X_fit.isna().any().any():
                missing_cols = X_fit.columns[X_fit.isna().any()].tolist()
                raise ValueError(
                    f"Specialist features contain missing values in columns: {missing_cols}"
                )

            if self.split_kind == "shuffle":
                splits = make_stratified_shuffle_splits(
                    y_fit, n_splits=self.folds, test_size=self.val_frac,
                    val_strata=self.val_strata, random_state=self.random_state,
                )
            elif self.split_kind == "kfold":
                splits = make_stratified_kfold_splits(
                    y_fit, n_splits=self.folds,
                    val_strata=self.val_strata, random_state=self.random_state,
                )
            else:
                raise ValueError(
                    f"Unknown split_kind={self.split_kind!r} -- must be 'kfold' or 'shuffle'."
                )
            general_prediction_fine = oof.loc[usable_mask, "blend_oof_prediction"].to_numpy()
            observed_fine = oof.loc[usable_mask, "observed"].to_numpy()
            self.n_usable_ = int(usable_mask.sum())
        else:
            # -----------------------------------------------------
            # Original behavior: KFold gives every fine-grained row an
            # out-of-fold prediction, so all of fine_mask is usable.
            # -----------------------------------------------------
            oof_fine = oof.loc[fine_mask]

            specialist_fine = X_specialist.loc[fine_mask].copy()
            specialist_fine.insert(
                0, "blend_oof_prediction", oof_fine["blend_oof_prediction"].to_numpy()
            )
            X_fit = specialist_fine[SPECIALIST_FEATURES_C]
            y_fit = oof_fine["remaining_residual"].to_numpy()

            if X_fit.isna().any().any():
                missing_cols = X_fit.columns[X_fit.isna().any()].tolist()
                raise ValueError(
                    f"Specialist features contain missing values in columns: {missing_cols}"
                )

            splits = list(self._make_cv().split(X_fit))
            general_prediction_fine = oof_fine["blend_oof_prediction"].to_numpy()
            observed_fine = oof_fine["observed"].to_numpy()
            self.n_usable_ = int(fine_mask.sum())

        # ---------------------------------------------------------
        # Residual model candidates: XGBoost (optionally Optuna-tuned)
        # and GPR, always blended as an ensemble -- see class docstring.
        # ---------------------------------------------------------
        self.optuna_study_ = None

        if self.xgb_model is not None:
            xgb_template = clone(self.xgb_model)
        elif self.optuna_trials > 0:
            xgb_template, self.optuna_study_ = tune_xgb_with_optuna(
                X_fit, y_fit, n_trials=self.optuna_trials,
                n_splits=self.n_splits, random_state=self.random_state,
                splits=splits if self.stratified_split else None,
            )
        else:
            xgb_template = make_default_specialist_xgb(self.random_state)

        gpr_template = self._get_gpr(n_features=X_fit.shape[1])

        if self.stratified_split:
            xgb_oof, validated_within_fit = _oof_predict_from_splits(
                lambda: clone(xgb_template), X_fit, y_fit, splits,
            )
            gpr_oof, _ = _oof_predict_from_splits(
                lambda: clone(gpr_template), X_fit, y_fit, splits,
            )
            if validated_within_fit.sum() < 2:
                raise ValueError(
                    f"Only {int(validated_within_fit.sum())} rows were held "
                    "out by this corrector's own split -- too few to report "
                    "metrics on. Check folds/val_strata."
                )
        else:
            cv = self._make_cv()
            xgb_oof = cross_val_predict(clone(xgb_template), X_fit, y_fit, cv=cv)
            gpr_oof = cross_val_predict(clone(gpr_template), X_fit, y_fit, cv=cv)
            validated_within_fit = np.ones(len(y_fit), dtype=bool)

        xgb_oof_v = xgb_oof[validated_within_fit]
        gpr_oof_v = gpr_oof[validated_within_fit]
        y_fit_v = y_fit[validated_within_fit]

        def blend_objective(w: float) -> float:
            blended = w * xgb_oof_v + (1.0 - w) * gpr_oof_v
            return mean_squared_error(y_fit_v, blended)

        blend_opt = minimize_scalar(blend_objective, bounds=(0.0, 1.0), method="bounded")
        self.ensemble_weight_ = float(blend_opt.x)
        ensemble_oof_v = self.ensemble_weight_ * xgb_oof_v + (1.0 - self.ensemble_weight_) * gpr_oof_v

        # Diagnostic-only: xgboost-alone/gpr-alone OOF metrics, kept for
        # get_training_summary() so a caller can sanity-check the ensemble is
        # still earning its keep on new data. Not used to pick a different
        # residual model -- see class docstring for why that comparison was
        # dropped in favor of always using the ensemble.
        candidate_oof = {
            "xgboost": xgb_oof_v,
            "gpr": gpr_oof_v,
            "ensemble": ensemble_oof_v,
        }
        self.candidate_oof_metrics_ = {
            name: {
                "r2": r2_score(y_fit_v, oof_pred),
                "rmse": mean_squared_error(y_fit_v, oof_pred) ** 0.5,
            }
            for name, oof_pred in candidate_oof.items()
        }
        self.residual_model_type_ = "ensemble"

        residual_oof_v = ensemble_oof_v

        general_prediction_fine_v = general_prediction_fine[validated_within_fit]
        observed_fine_v = observed_fine[validated_within_fit]

        def objective(beta: float) -> float:
            corrected = general_prediction_fine_v + beta * residual_oof_v
            return mean_squared_error(observed_fine_v, corrected)

        opt = minimize_scalar(objective, bounds=self.beta_bounds, method="bounded")
        self.beta_ = float(opt.x)

        # Refit both residual models on ALL usable rows (not just this
        # split's held-out slice) for later predict() calls -- same
        # "validate small, deploy on everything" pattern used throughout
        # this codebase (e.g. WeightedBlendRegressor refits its 3 base
        # models on all rows after weight-fitting on a validated subset).
        self.xgb_residual_model_ = clone(xgb_template)
        self.xgb_residual_model_.fit(X_fit, y_fit)
        self.gpr_residual_model_ = clone(gpr_template)
        self.gpr_residual_model_.fit(X_fit, y_fit)

        corrected_oof_v = general_prediction_fine_v + self.beta_ * residual_oof_v

        self.residual_oof_r2_ = r2_score(y_fit_v, residual_oof_v)
        self.general_oof_r2_ = r2_score(observed_fine_v, general_prediction_fine_v)
        self.corrected_oof_r2_ = r2_score(observed_fine_v, corrected_oof_v)
        self.general_oof_rmse_ = mean_squared_error(observed_fine_v, general_prediction_fine_v) ** 0.5
        self.corrected_oof_rmse_ = mean_squared_error(observed_fine_v, corrected_oof_v) ** 0.5
        self.n_fine_ = int(fine_mask.sum())
        self.n_validated_ = int(validated_within_fit.sum())

        self.is_fitted_ = True

        return self

    def _predict_residual(self, X_specialist_fine: pd.DataFrame) -> np.ndarray:
        """Always the XGBoost+GPR ensemble blend -- see class docstring."""
        xgb_pred = self.xgb_residual_model_.predict(X_specialist_fine)
        gpr_pred = self.gpr_residual_model_.predict(X_specialist_fine)
        return self.ensemble_weight_ * xgb_pred + (1.0 - self.ensemble_weight_) * gpr_pred

    def predict(
        self,
        X_general: pd.DataFrame,
        X_specialist: pd.DataFrame,
        fine_mask: Any,
    ) -> np.ndarray:
        """
        X_general:
            Features to pass to general_model.predict().

        X_specialist:
            DataFrame with SPECIALIST_FEATURES_C's columns except
            `blend_oof_prediction`, row-aligned with X_general.

        fine_mask:
            Boolean array/Series, row-aligned with X_general. Coarse-grained
            rows (False) get the general model's raw prediction; fine-grained
            rows (True) get the beta-corrected prediction.
        """
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError(
                f"{type(self).__name__} must be fitted before calling predict."
            )

        fine_mask = np.asarray(fine_mask, dtype=bool)

        general_prediction = np.asarray(self.general_model.predict(X_general), dtype=float)
        corrected_prediction = general_prediction.copy()

        if fine_mask.any():
            specialist_fine = X_specialist.loc[fine_mask].copy()
            specialist_fine.insert(
                0, "blend_oof_prediction", general_prediction[fine_mask]
            )

            residual_prediction = self._predict_residual(
                specialist_fine[SPECIALIST_FEATURES_C]
            )

            corrected_prediction[fine_mask] = (
                general_prediction[fine_mask] + self.beta_ * residual_prediction
            )

        return corrected_prediction

    def get_training_summary(self) -> dict[str, Any]:
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError(
                f"{type(self).__name__} must be fitted before calling get_training_summary."
            )

        summary = {
            "beta": self.beta_,
            "n_fine_grained": self.n_fine_,
            "n_usable": self.n_usable_,
            "n_validated": self.n_validated_,
            "residual_model_type": self.residual_model_type_,
            "candidate_oof_metrics": self.candidate_oof_metrics_,
            "ensemble_weight": self.ensemble_weight_,
            "residual_oof_r2": self.residual_oof_r2_,
            "general_oof_r2": self.general_oof_r2_,
            "corrected_oof_r2": self.corrected_oof_r2_,
            "general_oof_rmse": self.general_oof_rmse_,
            "corrected_oof_rmse": self.corrected_oof_rmse_,
        }

        if self.optuna_study_ is not None:
            summary["optuna_best_params"] = self.optuna_study_.best_params
            summary["optuna_best_value"] = self.optuna_study_.best_value

        return summary
