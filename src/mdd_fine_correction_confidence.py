"""
mdd_fine_correction_confidence.py
==================================
Confidence-weighted variant of mdd_fine_correction.MDDFineGrainedResidualCorrector,
used by model2ic.py (model2i.py + a confidence layer) in place of model2i.py's
MDDFineGrainedResidualCorrector/OWCFineGrainedResidualCorrector.

Same as model2i.py's Group C correction in every other respect: fine-grained
rows only (coarse rows get the general model's raw prediction, unchanged),
same SPECIALIST_FEATURES_C feature set, same always-ensemble (XGBoost + GPR)
residual model, same beta calibration by bounded scalar optimization on
out-of-fold predictions.

THE ADDITION: model2i.py's beta is a single global number applied uniformly
to every fine-grained row's correction, regardless of how confident the
residual model actually is at that specific row -- a row where the GPR half
is very unsure about its own residual estimate gets exactly as much
correction weight as a row where it's confident. That's exactly how a
correction stage can end up over-influencing a prediction it genuinely has
little basis for. This class adds a SECOND, per-row scaling factor on top of
beta, derived from the GPR half's own predictive standard deviation
(return_std=True): confidence = 1 / predictive_std, normalized against the
most-confident row seen during fitting so it's a shrinkage-only factor in
(0, 1] -- it can only pull the correction toward the general model's raw
prediction, never amplify it beyond what beta alone would give:

    residual_prediction  = ensemble_weight * xgb_pred + (1 - ensemble_weight) * gpr_pred   (unchanged)
    confidence            = min(1, (1 / gpr_std) / (1 / gpr_std_at_most_confident_training_row))
    corrected_prediction  = general_prediction + beta * confidence * residual_prediction

beta is fit the same bounded-scalar-optimization way as mdd_fine_correction.py,
but against this confidence-scaled correction instead of the raw one, so it
calibrates correctly given the now-varying per-row scaling.

general_model_impute.make_default_gpr_model() wraps its GaussianProcessRegressor
in a TransformedTargetRegressor for automatic target scaling -- convenient for
point predictions, but TransformedTargetRegressor.predict(return_std=True)
breaks (it tries to inverse-transform a (mean, std) tuple as a single array).
This module builds the identical GPR (same kernel/alpha/restarts) without that
wrapper, scaling X and y manually instead, so return_std=True works and both
the mean and the std can be inverse-transformed correctly (std scales by the
target scaler's std-dev factor only, unaffected by its mean shift).

Only supports the externally-provided-stratified-split mode model2i.py/
model2ic.py actually use -- not mdd_fine_correction.py's plain-KFold fallback
(stratified_split=False), since nothing here needs it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.general_model_impute import (
    make_stratified_kfold_splits,
    make_stratified_shuffle_splits,
    tune_xgb_with_optuna,
)
from src.mdd_fine_correction import (
    SPECIALIST_FEATURES_C,
    SPECIALIST_RAW_FEATURES,
    add_group_c_features,
    add_specialist_derived_features,
    make_default_specialist_xgb,
)

__all__ = [
    "ConfidenceWeightedFineGrainedResidualCorrector",
    "SPECIALIST_FEATURES_C",
    "SPECIALIST_RAW_FEATURES",
    "add_group_c_features",
    "add_specialist_derived_features",
    "make_default_specialist_xgb",
]


def _make_gpr(n_features: int, random_state: int) -> GaussianProcessRegressor:
    """Same kernel/alpha/restarts as general_model_impute.make_default_gpr_model(),
    returned bare (no TransformedTargetRegressor wrapper) so return_std=True
    works -- the caller scales X/y manually instead."""
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )
    return GaussianProcessRegressor(
        kernel=kernel, alpha=1e-8, normalize_y=False,
        n_restarts_optimizer=3, random_state=random_state,
    )


class ConfidenceWeightedFineGrainedResidualCorrector(BaseEstimator, RegressorMixin):
    """
    Fits a residual model on SPECIALIST_FEATURES_C to predict the residual a
    general model leaves on fine-grained rows, then blends it back in at a
    learned strength that ALSO shrinks per-row by how confident the GPR half
    is at that specific point:

        remaining_residual   = observed - general_model.predict(X)
        residual_prediction  = residual_model(specialist_features_c)
        confidence            = 1/gpr_std, normalized to (0, 1] against the
                                 most-confident training row
        corrected_prediction = general_prediction + beta * confidence * residual_prediction

    See module docstring for the full rationale. Target-agnostic (works for
    MDD or OWC, whichever y it's fit on), same as
    mdd_fine_correction.MDDFineGrainedResidualCorrector.

    Parameters
    ----------
    general_model:
        An already-fitted model exposing `.predict(X)` and
        `.get_oof_results(y, index)` returning a DataFrame with
        `blend_oof_prediction`/`remaining_residual`/`validated` columns --
        e.g. general_model_impute.WeightedBlendRegressor, or
        held_out_general_model.GeneralModelWithHeldOutRows wrapping one.
        Never refit here.

    xgb_model:
        Override for the XGBoost candidate. If None (default), see
        optuna_trials.

    optuna_trials:
        Optuna trials for tuning the XGBoost candidate on the fine-grained
        residual target (0 to skip tuning and use
        make_default_specialist_xgb() instead). Ignored if xgb_model is
        given explicitly.

    n_splits, random_state:
        Optuna's own internal CV folds / seed. random_state is also used for
        this corrector's own stratified split below.

    beta_bounds:
        Lower/upper limits for the learned correction strength.

    val_strata, folds, split_kind, val_frac:
        This corrector's own internal stratified split -- same
        make_stratified_kfold_splits/make_stratified_shuffle_splits
        machinery model2i.py's general models use, drawn on the fine-grained
        residual target. "kfold" (default): full coverage. "shuffle": single
        StratifiedShuffleSplit holdout.

    confidence_eps:
        Added to predictive std before inverting to a raw confidence score,
        to avoid dividing by ~0 wherever the GPR is extremely certain.
    """

    def __init__(
        self,
        general_model: Any,
        xgb_model: Any | None = None,
        optuna_trials: int = 50,
        n_splits: int = 5,
        random_state: int = 42,
        beta_bounds: tuple[float, float] = (0.0, 1.5),
        val_strata: int = 5,
        folds: int = 5,
        split_kind: str = "kfold",
        val_frac: float = 0.2,
        confidence_eps: float = 1e-3,
    ) -> None:
        self.general_model = general_model
        self.xgb_model = xgb_model
        self.optuna_trials = optuna_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.beta_bounds = beta_bounds
        self.val_strata = val_strata
        self.folds = folds
        self.split_kind = split_kind
        self.val_frac = val_frac
        self.confidence_eps = confidence_eps

    def fit(
        self,
        X_general: pd.DataFrame,
        X_specialist: pd.DataFrame,
        y: Any,
        fine_mask: Any,
    ) -> "ConfidenceWeightedFineGrainedResidualCorrector":
        """
        X_general:
            The general model's own feature matrix. Used only to call
            get_oof_results().

        X_specialist:
            DataFrame with SPECIALIST_FEATURES_C's columns except
            `blend_oof_prediction`, row-aligned with X_general.

        y:
            Full training target, row-aligned with X_general.

        fine_mask:
            Boolean array/Series, row-aligned with X_general. Only rows the
            general model itself validated (get_oof_results()'s `validated`
            column) AND fine_mask selects are usable for fitting.
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

        if "validated" not in oof.columns:
            raise ValueError(
                "general_model must have been fit with its own externally-"
                "provided splits (so get_oof_results() has a 'validated' "
                "column) -- e.g. WeightedBlendRegressor(splits=make_stratified_kfold_splits(...))."
            )

        usable_mask = fine_mask & oof["validated"].to_numpy(dtype=bool)
        if usable_mask.sum() < 10:
            raise ValueError(
                f"Only {int(usable_mask.sum())} fine-grained rows were "
                "validated by the general model's own split -- too few to "
                "fit a correction on."
            )

        specialist_usable = X_specialist.loc[usable_mask].copy()
        specialist_usable.insert(
            0, "blend_oof_prediction", oof.loc[usable_mask, "blend_oof_prediction"].to_numpy()
        )
        X_fit = specialist_usable[SPECIALIST_FEATURES_C].reset_index(drop=True)
        y_fit = oof.loc[usable_mask, "remaining_residual"].to_numpy()
        general_prediction_fine = oof.loc[usable_mask, "blend_oof_prediction"].to_numpy()
        observed_fine = oof.loc[usable_mask, "observed"].to_numpy()
        self.n_usable_ = int(usable_mask.sum())
        self.n_fine_ = int(fine_mask.sum())

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
                y_fit, n_splits=self.folds, val_strata=self.val_strata, random_state=self.random_state,
            )
        else:
            raise ValueError(
                f"Unknown split_kind={self.split_kind!r} -- must be 'kfold' or 'shuffle'."
            )

        # ---------------------------------------------------------
        # Residual model candidates: XGBoost (optionally Optuna-tuned) and a
        # bare GPR (return_std-capable). Always blended as an ensemble --
        # same architecture as mdd_fine_correction.MDDFineGrainedResidualCorrector.
        # ---------------------------------------------------------
        self.optuna_study_ = None
        if self.xgb_model is not None:
            xgb_template = clone(self.xgb_model)
        elif self.optuna_trials > 0:
            xgb_template, self.optuna_study_ = tune_xgb_with_optuna(
                X_fit, y_fit, n_trials=self.optuna_trials,
                n_splits=self.n_splits, random_state=self.random_state,
                splits=splits,
            )
        else:
            xgb_template = make_default_specialist_xgb(self.random_state)

        n_features = X_fit.shape[1]
        x_scaler = StandardScaler().fit(X_fit)
        X_scaled = x_scaler.transform(X_fit)
        y_scaler = StandardScaler().fit(y_fit.reshape(-1, 1))
        y_scaled = y_scaler.transform(y_fit.reshape(-1, 1)).ravel()

        n = len(y_fit)
        pred_sum_xgb = np.zeros(n)
        pred_sum_gpr = np.zeros(n)
        std_sum_gpr = np.zeros(n)
        cnt = np.zeros(n)

        for train_idx, val_idx in splits:
            xgb_fold = clone(xgb_template)
            xgb_fold.fit(X_fit.iloc[train_idx], y_fit[train_idx])
            pred_sum_xgb[val_idx] += xgb_fold.predict(X_fit.iloc[val_idx])

            gpr_fold = _make_gpr(n_features, self.random_state)
            gpr_fold.fit(X_scaled[train_idx], y_scaled[train_idx])
            mean_scaled, std_scaled = gpr_fold.predict(X_scaled[val_idx], return_std=True)
            pred_sum_gpr[val_idx] += mean_scaled * y_scaler.scale_[0] + y_scaler.mean_[0]
            std_sum_gpr[val_idx] += std_scaled * y_scaler.scale_[0]

            cnt[val_idx] += 1

        validated_within_fit = cnt > 0
        if validated_within_fit.sum() < 2:
            raise ValueError(
                f"Only {int(validated_within_fit.sum())} rows were held out "
                "by this corrector's own split -- too few to report metrics "
                "on. Check folds/val_strata."
            )

        v = validated_within_fit
        xgb_oof_v = pred_sum_xgb[v] / cnt[v]
        gpr_oof_v = pred_sum_gpr[v] / cnt[v]
        gpr_oof_std_v = std_sum_gpr[v] / cnt[v]
        y_fit_v = y_fit[v]

        def blend_objective(w: float) -> float:
            blended = w * xgb_oof_v + (1.0 - w) * gpr_oof_v
            return mean_squared_error(y_fit_v, blended)

        blend_opt = minimize_scalar(blend_objective, bounds=(0.0, 1.0), method="bounded")
        self.ensemble_weight_ = float(blend_opt.x)
        residual_oof_v = self.ensemble_weight_ * xgb_oof_v + (1.0 - self.ensemble_weight_) * gpr_oof_v

        candidate_oof = {"xgboost": xgb_oof_v, "gpr": gpr_oof_v, "ensemble": residual_oof_v}
        self.candidate_oof_metrics_ = {
            name: {
                "r2": float(r2_score(y_fit_v, pred)),
                "rmse": float(mean_squared_error(y_fit_v, pred) ** 0.5),
            }
            for name, pred in candidate_oof.items()
        }
        self.residual_model_type_ = "ensemble"

        # ---------------------------------------------------------
        # Confidence: inverse GPR predictive std, normalized against the
        # most-confident (smallest-std) validated row -- a shrinkage-only
        # factor in (0, 1], never an amplifier for a new row more confident
        # than anything seen during fitting.
        # ---------------------------------------------------------
        raw_confidence_v = 1.0 / (gpr_oof_std_v + self.confidence_eps)
        self.confidence_scale_ = float(raw_confidence_v.max())
        confidence_v = np.minimum(raw_confidence_v / self.confidence_scale_, 1.0)

        general_prediction_fine_v = general_prediction_fine[v]
        observed_fine_v = observed_fine[v]

        def objective(beta: float) -> float:
            corrected = general_prediction_fine_v + beta * confidence_v * residual_oof_v
            return mean_squared_error(observed_fine_v, corrected)

        opt = minimize_scalar(objective, bounds=self.beta_bounds, method="bounded")
        self.beta_ = float(opt.x)

        # Refit both residual models on ALL usable rows for later predict()
        # calls -- same "validate small, deploy on everything" pattern used
        # throughout this codebase.
        self.xgb_residual_model_ = clone(xgb_template)
        self.xgb_residual_model_.fit(X_fit, y_fit)
        self.x_scaler_ = x_scaler
        self.y_scaler_ = y_scaler
        self.gpr_residual_model_ = _make_gpr(n_features, self.random_state)
        self.gpr_residual_model_.fit(X_scaled, y_scaled)

        corrected_oof_v = general_prediction_fine_v + self.beta_ * confidence_v * residual_oof_v

        self.residual_oof_r2_ = float(r2_score(y_fit_v, residual_oof_v))
        self.general_oof_r2_ = float(r2_score(observed_fine_v, general_prediction_fine_v))
        self.corrected_oof_r2_ = float(r2_score(observed_fine_v, corrected_oof_v))
        self.general_oof_rmse_ = float(mean_squared_error(observed_fine_v, general_prediction_fine_v) ** 0.5)
        self.corrected_oof_rmse_ = float(mean_squared_error(observed_fine_v, corrected_oof_v) ** 0.5)
        self.n_validated_ = int(v.sum())
        self.mean_confidence_ = float(confidence_v.mean())
        self.min_confidence_ = float(confidence_v.min())

        self.is_fitted_ = True
        return self

    def _predict_residual_and_confidence(self, X_specialist_fine: pd.DataFrame):
        xgb_pred = self.xgb_residual_model_.predict(X_specialist_fine)
        X_scaled = self.x_scaler_.transform(X_specialist_fine)
        mean_scaled, std_scaled = self.gpr_residual_model_.predict(X_scaled, return_std=True)
        gpr_pred = mean_scaled * self.y_scaler_.scale_[0] + self.y_scaler_.mean_[0]
        gpr_std = std_scaled * self.y_scaler_.scale_[0]

        residual_prediction = self.ensemble_weight_ * xgb_pred + (1.0 - self.ensemble_weight_) * gpr_pred
        raw_confidence = 1.0 / (gpr_std + self.confidence_eps)
        confidence = np.minimum(raw_confidence / self.confidence_scale_, 1.0)
        return residual_prediction, confidence

    def predict(
        self,
        X_general: pd.DataFrame,
        X_specialist: pd.DataFrame,
        fine_mask: Any,
    ) -> np.ndarray:
        """
        fine_mask:
            Boolean array/Series, row-aligned with X_general. Coarse-grained
            rows (False) get the general model's raw prediction; fine-
            grained rows (True) get the beta-and-confidence-scaled
            correction.
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
            specialist_fine.insert(0, "blend_oof_prediction", general_prediction[fine_mask])

            residual_prediction, confidence = self._predict_residual_and_confidence(
                specialist_fine[SPECIALIST_FEATURES_C]
            )

            corrected_prediction[fine_mask] = (
                general_prediction[fine_mask] + self.beta_ * confidence * residual_prediction
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
            "confidence_scale": self.confidence_scale_,
            "mean_confidence": self.mean_confidence_,
            "min_confidence": self.min_confidence_,
        }

        if self.optuna_study_ is not None:
            summary["optuna_best_params"] = self.optuna_study_.best_params
            summary["optuna_best_value"] = self.optuna_study_.best_value

        return summary
