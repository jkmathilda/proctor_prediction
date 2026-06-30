# Gap Analysis: proctor-prediction

**Phase**: Check  
**Date**: 2026-06-30  
**Design**: `docs/02-design/features/proctor-prediction.design.md`  
**Implementation**: `models/claudev1.ipynb` + `models/claudev2.ipynb`  
**Submission**: `submissions/submission_20260630_1254_v1.csv`

---

## Overall Match Rate: 93% ✅

| Category | Score |
|---|:---:|
| Preprocessing pipeline (design §3) | 4/4 |
| Feature engineering (design §4) | 11/11 |
| Model tiers — v1 (design §5) | 5/5 |
| Post-processing / saturation constraint | ✅ |
| Notebook structure — v1 (design §6) | 10/10 sections |
| Submission format & validation (design §7) | ✅ |
| v2 SHAP/SHAPIQ additions (user-requested) | 8/8 |
| **Overall** | **93%** |

> v1 and v2 scored jointly. v2 is an explainability extension that intentionally omits model-tier duplication from v1.

---

## ✅ Implemented (matched)

### Preprocessing (§3) — complete
- Boolean coercion of `psd_has_sedimentation` (handles both bool and string keys)
- `pd.to_numeric(errors='coerce')` on all PSD / fraction / density / diameter columns
- Missingness indicator flags for all 5 high-missing features (`{col}_missing`)
- Median imputation fit-on-train-only — no leakage into CV folds

### Feature Engineering (§4) — complete
CU, CC, PI, saturation MDD proxy, D50, clay-to-fines ratio, sand%, gravel%, log-PSD D-cols (11), log(kf), soil-type flag. Feature count ~49 vs design's ~47 — within tolerance (+2: `feat_log_cu`, proxy naming).

### Model Tiers — v1 (§5) — complete
- Ridge baseline with `StandardScaler` pipeline (separate per target, preferred over `MultiOutputRegressor`)
- XGBoost separate per target — params match design exactly
- LightGBM separate per target — params match design exactly
- Weighted ensemble with **optimized** blend weight via CV grid search (not fixed 0.5)
- Optuna tuning for all 4 model×target combos (50 trials each)
- NMAE with fixed IQR (0.198 / 3.860), 5-fold `KFold(shuffle=True, random_state=42)`

### Post-processing — matched
`apply_saturation_constraint` with zero-air-voids line and 0.99 safety margin, applied to all test predictions.

### Submission — matched
Columns `[id, proctor_owc_pct, proctor_mdd_g_cm3]`, range assertions, file generated.

### v2 SHAP/SHAPIQ additions — all 8 present
1. SHAP beeswarm + mean |SHAP| bar chart (§6)
2. SHAP dependence plots — top-6 per target (§7)
3. SHAP-based feature correlation heatmap (§8)
4. SHAP interaction values tensor via `shap_interaction_values` (§8b)
5. SHAPIQ k-SII pairwise analysis via `TabularExplainer` (§9)
6. SHAP-based feature selection + CV comparison (§11)
7. SHAP-interaction product features + CV comparison (§12)
8. Local waterfall + force-plot HTML (§13)

---

## 🔴/🟡 Gaps

| # | Severity | Item | Detail |
|---|:---:|---|---|
| G1 | 🟡 Low | Submission assertions | Design specifies hardcoded `len == 87` and `id == range(201, 288)`. Both notebooks use `len(test_raw)` and `(submission['id'] == test_raw['id']).all()` — functionally equivalent but weaker contract. |
| G2 | 🟡 Med | v2 drops baseline/ensemble/Optuna | v2 is explainability-only; Ridge, ensemble, and Optuna are absent. Combined v1+v2 coverage is fine, but v2 standalone regresses from design's model-tier spec. |
| G3 | 🟡 Low | EDA completeness | Design specifies Pearson correlation matrix and PSD D-col log-scale distributions. v1 omits both; v2 has no EDA section. |
| G4 | 🔵 Cosmetic | NMAE signature | Design defines `nmae(y_true, y_pred, ...)` over 2-D arrays; implementation uses 4-arg form. Same math, different API. |

---

## 🟢 Extras (beyond design)

| Item | Location |
|---|---|
| `feat_log_cu` engineered feature | v1, v2 |
| Physical range assertions on submission (MDD 1.4–2.5, OWC 2.0–35) | v1, v2 |
| SHAPIQ interaction network graph (networkx) | v2 §10 |
| Force-plot HTML export (`docs/shap_force_mdd.html`) | v2 §13 |
| Entire v2 SHAP/SHAPIQ explainability suite | v2 |

---

## ⚠️ Operational Issues Fixed

| # | Issue | Fix applied |
|---|---|---|
| O1 | v2 imports `optuna` but never uses it | Removed from v2 §0 |
| O2 | Doc names deliverable `solution.ipynb`; actual files are `claudev1/v2.ipynb` | Design §2/§6 should be updated on next design revision |

---

## Recommendations

1. **Done (O1)**: Removed unused `import optuna` from v2 — v2 can now run without Optuna installed.
2. **Low priority (G1)**: Add `assert len(submission) == 87` and `assert submission['id'].tolist() == list(range(201, 288))` to harden the submission contract.
3. **Low priority (G3)**: Add Pearson correlation matrix and log-scale PSD D-col distribution plots to satisfy full EDA spec.
4. **Design sync (G2/G4)**: On next design revision, note that v1 owns model selection and v2 is explainability-only; update NMAE signature and deliverable filenames.

---

## PDCA Status

```
[Plan] ✅ → [Design] ✅ → [Do] ✅ → [Check] ✅ (93%) → [Act] — → [Report] ⏳
```

Match rate ≥ 90% → **Ready for `/pdca report proctor-prediction`**
