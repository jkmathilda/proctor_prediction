"""
held_out_general_model.py
==========================
GeneralModelWithHeldOutRows -- lets a correction stage (e.g.
mdd_fine_correction.MDDFineGrainedResidualCorrector) see rows that were
deliberately excluded from a general model's own fit (see model2i.py's
--exclude_ids), by substituting the general model's raw predict() output
for those rows wherever an out-of-fold prediction would normally be used.

Kept in src/ rather than defined inline in scripts/model2i.py so cached
correctors (joblib.dump'd holding this class as their `general_model`) can
be unpickled from any context, not just by re-running model2i.py itself as
__main__ -- joblib pickles by module path, and a class defined inside a
script executed as __main__ only resolves back to __main__.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class GeneralModelWithHeldOutRows:
    """
    Wraps an already-fitted general model (fit WITHOUT extra_index's rows at
    all) so a downstream correction stage that calls get_oof_results() can
    still see those rows. For every row the wrapped model actually
    fit/validated on, its real cross-validated OOF prediction is passed
    through unchanged. For extra_index's rows, which the wrapped model never
    saw during fitting, get_oof_results() substitutes the model's raw
    predict() output in place of an OOF prediction -- an honest, non-leaked
    estimate (the model genuinely never trained on these rows, unlike a row
    it WAS fit on), just not literally a cross-validation fold prediction.

    Only get_oof_results()/predict()/is_fitted_ are implemented -- the only
    parts of the general_model duck-typed contract
    MDDFineGrainedResidualCorrector (and anything built the same way) uses.
    """

    def __init__(self, general_model: Any, extra_X: pd.DataFrame, extra_index: Any) -> None:
        self.general_model = general_model
        self.extra_X = extra_X
        self.extra_index = pd.Index(extra_index)
        self.is_fitted_ = getattr(general_model, "is_fitted_", False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.general_model.predict(X)

    def get_oof_results(self, y: Any, index: Any) -> pd.DataFrame:
        index = pd.Index(index)
        y_arr = np.asarray(y, dtype=float)
        is_extra = index.isin(self.extra_index)

        base_index = index[~is_extra]
        base_y = y_arr[~is_extra]
        base_oof = self.general_model.get_oof_results(y=base_y, index=base_index)

        if is_extra.any():
            extra_idx = index[is_extra]
            extra_y = y_arr[is_extra]
            extra_pred = np.asarray(self.general_model.predict(self.extra_X.loc[extra_idx]), dtype=float)
            extra_oof = pd.DataFrame(
                {
                    "observed": extra_y,
                    "blend_oof_prediction": extra_pred,
                    "remaining_residual": extra_y - extra_pred,
                    "validated": True,
                },
                index=extra_idx,
            )
            combined = pd.concat([base_oof, extra_oof], sort=False)
        else:
            combined = base_oof

        return combined.loc[index]
