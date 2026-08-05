"""
fine_specialist.py
===================
A different pipeline from mdd_fine_correction.py / owc_fine_correction.py.

Those wrap an already-fitted GENERAL model (WeightedBlendRegressor fit on
all 201 rows, both soil types, on general_model_impute.IMPUTED_FEATURES)
with a second-stage correction that only touches fine-grained residuals:

    data -> general model (fit on everyone) -> specialist correction (fine-grained residual only)

This module instead fits a STANDALONE model directly on fine-grained rows
only (n=89) -- no general model underneath it, no residual, no correction
stage:

    data -> fine specialist model (fit on fine-grained rows only)

Same underlying architecture as the general model (WeightedBlendRegressor:
Ridge + GPR + XGBoost, SLSQP-fit blend weights, unchanged from
general_model_impute.py) -- what's different is the training population and
the feature set. Restricting to fine-grained rows removes the reason
IMPUTED_FEATURES excludes the Atterberg limits in the first place (they're
not physically measured for coarse soils, so the general model -- which
sees both soil types -- can't use them at all without leaving ~56% of rows
with missing predictors). A model that only ever sees fine-grained rows has
no such constraint, so FINE_SPECIALIST_FEATURES adds
mdd_fine_correction.py's SPECIALIST_RAW_FEATURES (MICE-completed Atterberg
limits/PI/LOI/kf, plus their missingness flags) on top of IMPUTED_FEATURES's
universal columns.

This is a genuinely different bet from the correction approach, not just a
different implementation of the same idea: a correction model only has to
explain whatever residual signal the general model's blend weights (fit
across BOTH soil types) left behind on fine-grained rows specifically --
Section 7 of docs/260804.md documents that residual turning out to be thin
(nested OOF R^2 0.04-0.08) and the correction's Kaggle score coming in worse
than the uncorrected general model. A standalone specialist instead lets
Ridge/GPR/XGBoost's own blend-weight search run fresh on the fine-grained
population from scratch, with the extra Atterberg/PI features available
from the start rather than bolted on as a second stage -- whether that beats
the general model (or the correction) on Kaggle is exactly the open
question this pipeline exists to test, not something assumed here.

See coarse_specialist.py for the coarse-grained counterpart -- a caller
combining both specialists is responsible for routing each test row to the
right one via its own `fine-grained` value (this module never sees or
predicts on coarse-grained rows).

ONE MORE DEVIATION FROM THE GENERAL MODEL: `hyd_cond_hyd_gradient` is
MICE-imputed and included here, even though every other model in this
project (v15/v16, model1/model1i/model2i, general_model_impute.py's own
IMPUTED_FEATURES) drops it outright. That project-wide decision was tested
on the GENERAL model, which is fit across both soil types at once -- v10 vs.
v15's controlled A/B on Kaggle (0.2384 keeping it vs. 0.2383 dropping it)
found the difference indistinguishable from noise (docs/260804.md Section
6). But that same v10 ARD-GP run found `hyd_cond_hyd_gradient` DID survive
OWC's feature-relevance selection specifically (length-scale 7.14, top-10
for OWC) -- so the "doesn't matter" finding was never isolated to
fine-grained rows or to this WeightedBlendRegressor architecture. Since this
is a different population (fine-grained only) and a different model
context, it's re-included here as its own open question rather than
inheriting the general model's answer by default.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.general_model_impute import (
    IMPUTED_FEATURES,
    MICE_PREDICTORS,
    WeightedBlendRegressor,
    add_imputed_features,
    impute_features,
)
from src.mdd_fine_correction import (
    SPECIALIST_RAW_FEATURES,
    add_specialist_derived_features,
)

# IMPUTED_FEATURES (universal columns, safe for either soil type) plus the
# Atterberg/PI/LOI/kf specialist columns -- only guaranteed non-NaN for
# fine-grained rows, which is exactly the population this module trains on --
# plus the fine-grained-imputed hyd_cond_hyd_gradient columns (see module
# docstring for why this one column is re-included here against the rest of
# the project's default).
#
# SPECIALIST_RAW_FEATURES overlaps IMPUTED_FEATURES on the LOI/kf
# completed+missingness columns (both sets already include them, since LOI
# and kf are imputed for every row regardless of soil type) -- deduplicated
# here (order-preserving) so those four columns aren't passed to the model
# twice under the same name. The genuinely new columns this adds on top of
# IMPUTED_FEATURES are the Atterberg limits, feat_pi_completed, the two
# Atterberg missingness flags, and hyd_cond_hyd_gradient's completed/
# missingness pair.
FINE_SPECIALIST_FEATURES: list[str] = list(
    dict.fromkeys(
        IMPUTED_FEATURES
        + SPECIALIST_RAW_FEATURES
        + [
            "hyd_cond_hyd_gradient_completed",
            "hyd_cond_hyd_gradient_was_missing",
        ]
    )
)


def prepare_fine_specialist_features(
    df: pd.DataFrame,
    fine_imputer: Any | None = None,
    coarse_imputer: Any | None = None,
    hydgrad_imputer: Any | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Any, Any, Any]:
    """
    Produce every column FINE_SPECIALIST_FEATURES needs.

    Runs general_model_impute.add_imputed_features on the FULL dataframe
    (fine AND coarse rows) -- not just the fine-grained subset -- so the
    MICE equations are fit with as much data as possible, even though only
    fine-grained rows are used to train/predict with the result (see
    select_fine_grained). Then derives feat_pi_completed via
    mdd_fine_correction.add_specialist_derived_features.

    Separately, MICE-imputes `hyd_cond_hyd_gradient` -- fit and applied on
    fine-grained rows only (applicable_mask), matching how Atterberg limits
    are handled by add_imputed_features. Kept as its own impute_features()
    call rather than folded into add_imputed_features, since that function
    is shared by every other script in this project (model1i.py, model2i.py,
    ...) and changing its output would silently change their feature sets
    too. No log-transform: unlike kf (spans ~7 orders of magnitude),
    hyd_cond_hyd_gradient ranges 1.6-30.25 in train.csv, a normal-enough
    scale for plain-space MICE.

    Pass previously-fitted fine_imputer/coarse_imputer/hydgrad_imputer (as
    returned from an earlier call on training data) to transform new data
    (e.g. test.csv) without refitting/leaking -- same convention as
    add_imputed_features.
    """
    df_imputed, fine_imputer, coarse_imputer = add_imputed_features(
        df, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer,
        random_state=random_state,
    )
    df_imputed = add_specialist_derived_features(df_imputed)

    fine_mask = df_imputed["fine-grained"].astype(bool)
    df_imputed, hydgrad_imputer = impute_features(
        input_df=df_imputed,
        impute_cols=["hyd_cond_hyd_gradient"],
        predictors=MICE_PREDICTORS,
        applicable_mask=fine_mask,
        imputer=hydgrad_imputer,
        random_state=random_state,
    )

    return df_imputed, fine_imputer, coarse_imputer, hydgrad_imputer


def select_fine_grained(
    df: pd.DataFrame,
    target_col: str | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """
    Filter to fine-grained rows and FINE_SPECIALIST_FEATURES columns.

    `df` must already have been through prepare_fine_specialist_features (or
    equivalently add_imputed_features + add_specialist_derived_features).
    Raises if any fine-grained row still has a NaN in FINE_SPECIALIST_FEATURES
    -- catches a caller forgetting the imputation step, or an imputer fit on
    the wrong data, rather than silently feeding NaNs into the blend.

    If target_col is given, also returns the fine-grained rows' target as a
    numpy array (X, y); otherwise returns X alone -- for use at predict time,
    when there's no target to select.
    """
    if "fine-grained" not in df.columns:
        raise KeyError("select_fine_grained requires a 'fine-grained' column.")

    fine_mask = df["fine-grained"].astype(bool)

    if not fine_mask.any():
        raise ValueError("No fine-grained rows in df.")

    X = df.loc[fine_mask, FINE_SPECIALIST_FEATURES]

    if X.isna().any().any():
        missing_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"FINE_SPECIALIST_FEATURES has NaNs in columns: {missing_cols} -- "
            "run prepare_fine_specialist_features first."
        )

    if target_col is None:
        return X

    y = df.loc[fine_mask, target_col].to_numpy(dtype=float)
    return X, y


def make_fine_specialist_model(
    random_state: int = 42,
    **kwargs: Any,
) -> WeightedBlendRegressor:
    """
    The fine specialist's model is architecturally identical to the general
    model -- WeightedBlendRegressor, unmodified. This factory exists only so
    a caller doesn't need to import WeightedBlendRegressor directly and to
    match this module's make_default_*-style naming convention; every keyword
    argument WeightedBlendRegressor accepts (xgb_model, splits, n_splits,
    ...) can be passed straight through.
    """
    return WeightedBlendRegressor(random_state=random_state, **kwargs)
