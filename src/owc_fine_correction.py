"""
owc_fine_correction.py
=======================
Stage-2 residual correction for OWC on fine-grained soils only, using the
same "Group C: Full specialist" feature set (mdd_fine_correction.py's
SPECIALIST_FEATURES_C) and the same always-ensemble (XGBoost + GPR, weight
fit by bounded scalar optimization on OOF predictions) residual model that
MDDFineGrainedResidualCorrector uses for MDD.

data_analysis.ipynb never tried this for OWC: the Group C specialist
correction there (cells 129-139) was only ever evaluated against MDD's
residual ("the notebook never fit an analogous specialist correction for
OWC", per mdd_fine_correction.py's docstring). This module has no notebook
cell to cite -- it's validated directly against this repo's own OOF numbers,
reusing model2i.py's already-fitted general OWC model
(general_model_impute.WeightedBlendRegressor on IMPUTED_FEATURES):

    general model (fine-grained rows only, n=89): R^2 0.767  RMSE 1.624
    + Group C correction:                         R^2 0.779  RMSE 1.581   (beta ~0.93)

CAVEAT, unlike MDD: this gain is real but fragile. Re-running across 7
different CV seeds (fixed hyperparameters, isolating fold-randomness) always
showed a small *positive* delta (+0.003 to +0.012 R^2), so it isn't pure
noise -- but the residual model's own OOF R^2 against the fine-grained OWC
residual (not the final corrected prediction) was *negative* in 5 of those 7
seeds, meaning at full strength (beta=1) it would often make predictions
worse than no correction at all; the gain survives only because beta gets
fit small enough to mostly cancel the residual model's noise. MDD's
equivalent residual R^2 is consistently positive across seeds -- a real,
if modest, predictive signal, not shrinkage rescuing a weak one. Keep this
in mind before trusting OWC's correction on a genuinely new holdout set.

MDDFineGrainedResidualCorrector's fit()/predict() never actually reference
MDD by name -- they operate entirely on whatever X_general/X_specialist/y/
fine_mask they're given, duck-typing the general model exactly the same way
regardless of target. Rather than duplicating ~300 lines of identical
ensemble-fitting/beta-fitting logic, this module reuses that class directly
and re-exports it as OWCFineGrainedResidualCorrector, for callers
(model2i.py) that want a correctly-named class for this purpose.
"""

from __future__ import annotations

from src.mdd_fine_correction import (
    MDDFineGrainedResidualCorrector as OWCFineGrainedResidualCorrector,
    SPECIALIST_FEATURES_C,
    SPECIALIST_RAW_FEATURES,
    add_group_c_features,
    add_specialist_derived_features,
    make_default_specialist_xgb,
)

__all__ = [
    "OWCFineGrainedResidualCorrector",
    "SPECIALIST_FEATURES_C",
    "SPECIALIST_RAW_FEATURES",
    "add_group_c_features",
    "add_specialist_derived_features",
    "make_default_specialist_xgb",
]
