"""
coarse_specialist.py
=====================
Coarse-grained counterpart to fine_specialist.py -- see that module's
docstring for the full "specialist instead of correction" rationale:

    data -> coarse specialist model (fit on coarse-grained rows only, n=112)

instead of

    data -> general model (fit on everyone) -> specialist correction

Same architecture as the general model and the fine specialist
(WeightedBlendRegressor: Ridge + GPR + XGBoost, SLSQP-fit blend weights,
unchanged from general_model_impute.py) -- only the training population and
feature set differ.

Unlike fine_specialist.py, COARSE_SPECIALIST_FEATURES is NOT
IMPUTED_FEATURES plus something extra. There is no coarse-grained analogue
of the Atterberg-limit specialist columns to add: Atterberg limits are never
physically measured for coarse-grained soils in this dataset, so
add_imputed_features never imputes them for coarse rows either (see its own
docstring) -- there is nothing there to complete, real or imputed.
Loss-on-ignition and kf ARE imputed for coarse rows and are already part of
IMPUTED_FEATURES, so they're not missing from this specialist either. This
means COARSE_SPECIALIST_FEATURES == IMPUTED_FEATURES exactly -- not an
oversight, the coarse specialist simply doesn't get an enrichment step
because this dataset has no coarse-only lab measurement equivalent to what
fine_specialist.py gains from the Atterberg limits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.general_model_impute import (
    IMPUTED_FEATURES,
    WeightedBlendRegressor,
    add_imputed_features,
)

# See module docstring: no coarse-only enrichment exists in this dataset, so
# this is IMPUTED_FEATURES unchanged, kept as its own name for symmetry with
# fine_specialist.FINE_SPECIALIST_FEATURES and so a caller can import one
# name per specialist without knowing they happen to be equal.
COARSE_SPECIALIST_FEATURES: list[str] = list(IMPUTED_FEATURES)


def prepare_coarse_specialist_features(
    df: pd.DataFrame,
    fine_imputer: Any | None = None,
    coarse_imputer: Any | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Any, Any]:
    """
    Produce every column COARSE_SPECIALIST_FEATURES needs.

    Runs general_model_impute.add_imputed_features on the FULL dataframe
    (fine AND coarse rows) -- not just the coarse-grained subset -- so the
    MICE equations fitting loss-on-ignition/kf are fit with as much data as
    possible, even though only coarse-grained rows are used to train/predict
    with the result (see select_coarse_grained).

    Pass previously-fitted fine_imputer/coarse_imputer (as returned from an
    earlier call on training data) to transform new data (e.g. test.csv)
    without refitting/leaking -- same convention as add_imputed_features.
    """
    return add_imputed_features(
        df, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer,
        random_state=random_state,
    )


def select_coarse_grained(
    df: pd.DataFrame,
    target_col: str | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """
    Filter to coarse-grained rows and COARSE_SPECIALIST_FEATURES columns.

    `df` must already have been through prepare_coarse_specialist_features
    (or equivalently add_imputed_features). Raises if any coarse-grained row
    still has a NaN in COARSE_SPECIALIST_FEATURES -- catches a caller
    forgetting the imputation step, or an imputer fit on the wrong data,
    rather than silently feeding NaNs into the blend.

    If target_col is given, also returns the coarse-grained rows' target as
    a numpy array (X, y); otherwise returns X alone -- for use at predict
    time, when there's no target to select.
    """
    if "fine-grained" not in df.columns:
        raise KeyError("select_coarse_grained requires a 'fine-grained' column.")

    coarse_mask = ~df["fine-grained"].astype(bool)

    if not coarse_mask.any():
        raise ValueError("No coarse-grained rows in df.")

    X = df.loc[coarse_mask, COARSE_SPECIALIST_FEATURES]

    if X.isna().any().any():
        missing_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"COARSE_SPECIALIST_FEATURES has NaNs in columns: {missing_cols} -- "
            "run prepare_coarse_specialist_features first."
        )

    if target_col is None:
        return X

    y = df.loc[coarse_mask, target_col].to_numpy(dtype=float)
    return X, y


def make_coarse_specialist_model(
    random_state: int = 42,
    **kwargs: Any,
) -> WeightedBlendRegressor:
    """
    The coarse specialist's model is architecturally identical to the
    general model -- WeightedBlendRegressor, unmodified. This factory exists
    only so a caller doesn't need to import WeightedBlendRegressor directly
    and to match this module's make_default_*-style naming convention; every
    keyword argument WeightedBlendRegressor accepts (xgb_model, splits,
    n_splits, ...) can be passed straight through.
    """
    return WeightedBlendRegressor(random_state=random_state, **kwargs)
