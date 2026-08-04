"""
owc_fine_correction.py
=======================
Stage-2 residual correction for OWC on fine-grained soils only, using the
same "Group C: Full specialist" feature set (mdd_fine_correction.py's
SPECIALIST_FEATURES_C) and the same candidate-comparison procedure --
XGBoost (Optuna-tuned) vs GPR vs a weighted ensemble of the two, chosen by
out-of-fold MSE -- that MDDFineGrainedResidualCorrector uses for MDD.

data_analysis.ipynb never tried this for OWC: the Group C specialist
correction there (cells 129-139) was only ever evaluated against MDD's
residual ("the notebook never fit an analogous specialist correction for
OWC", per mdd_fine_correction.py's docstring). This module has no notebook
cell to cite -- it's validated directly against this repo's own OOF numbers,
reusing model2i.py's already-fitted general OWC model
(general_model_impute.WeightedBlendRegressor on IMPUTED_FEATURES):

    general model (fine-grained rows only, n=89): R^2 0.774  RMSE 1.600
    + Group C correction:                         R^2 0.796  RMSE 1.520
    (beta ~0.79, residual_model=ensemble, ~96% XGBoost weight)

Same direction and rough magnitude as MDD's Group C result (R^2 0.798 ->
0.821), though the absolute gain is smaller -- OWC's fine-grained residual is
noisier and GPR alone is *worse* than the general model here too (OOF R^2
-0.29), same pattern as MDD.

MDDFineGrainedResidualCorrector's fit()/predict() never actually reference
MDD by name -- they operate entirely on whatever X_general/X_specialist/y/
fine_mask they're given, duck-typing the general model exactly the same way
regardless of target. Rather than duplicating ~300 lines of identical
candidate-comparison/beta-fitting logic, this module reuses that class
directly and re-exports it as OWCFineGrainedResidualCorrector, for callers
(model3i.py) that want a correctly-named class for this purpose.
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
