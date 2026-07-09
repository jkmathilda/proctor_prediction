# Plan: proctor-prediction

**Feature**: LeiGS 2026 Proctor Prediction Challenge  
**Phase**: Plan  
**Created**: 2026-06-30  
**Deadline**: 2026-08-16 (Final leaderboard published)

---

## 1. Objective

Build a robust, generalizable ML model to predict two Proctor compaction parameters from soil classification properties:

| Target Variable | Column | Unit | Approx. Range |
|---|---|---|---|
| Maximum Dry Density | `proctor_mdd_g_cm3` | g/cm³ | 1.6 – 2.2 |
| Optimum Water Content | `proctor_owc_pct` | % | 5.0 – 25.0 |

**Success metric**: Minimize NMAE (Normalized Mean Absolute Error) on the private leaderboard. NMAE normalizes each target's MAE by its IQR so both targets contribute equally (50/50).

---

## 2. Dataset Overview

| File | Rows | Notes |
|---|---|---|
| `train.csv` | 201 | Includes target variables |
| `test.csv` | 87 | Targets hidden; 15% public / 15% private split |
| `sample_submission.csv` | 87 | Format: `id, proctor_owc_pct, proctor_mdd_g_cm3` |

**Total dataset**: 288 observations × 24 columns (22 features + 2 targets)

---

## 3. Feature Inventory

### Particle Size Distribution (PSD)
| Column | Unit | Notes |
|---|---|---|
| `psd_size_at_d10_mm` … `psd_size_at_d98_mm` | mm | Particle diameter at 10/20/…/98% passing (11 columns) |
| `psd_has_sedimentation` | bool | True = hydrometer analysis done; False = extrapolated |
| `psd_passing_at_0_002mm_pct` | % | Clay fraction (<0.002 mm) |
| `psd_passing_at_0_063mm_pct` | % | Fine-grained fraction (<0.063 mm) |
| `psd_passing_at_2mm_pct` | % | Fine + medium fraction (<2 mm) |

### Atterberg Limits (only for fine-grained soils)
| Column | Unit | Notes |
|---|---|---|
| `atterberg_liquid_limit_pct` | % | Liquid Limit (LL) |
| `atterberg_plastic_limit_pct` | % | Plastic Limit (PL) |

### Physical / Hydraulic Properties
| Column | Unit | Notes |
|---|---|---|
| `grain_density_g_cm3` | g/cm³ | Particle density ρs (avg ~2.65) |
| `hyd_cond_kf_m_s` | m/s | Hydraulic conductivity kf |
| `hyd_cond_hyd_gradient` | – | Hydraulic gradient i |
| `loss_on_ignition_pct` | % | Organic content proxy |
| `proctor_diam_mm` | mm | Mold diameter: 100 mm or 150 mm |

### Missing Values
Many features have gaps: Atterberg limits are NaN for purely granular soils; hydraulic tests may not have been run for all samples. Imputation strategy is required.

---

## 4. Domain Knowledge Key Insights

1. **CU (Coefficient of Uniformity)** = D60/D10 — wider gradation → higher MDD
2. **CC (Coefficient of Curvature)** = D30² / (D60 × D10) — shape of PSD curve
3. **Plasticity Index (PI)** = LL – PL — strongly correlated with wopt
4. **Saturation line**: ρd = ρs / (1 + w · ρs/ρw) — physical upper bound for MDD
5. **Loss on ignition**: higher organic content → lower MDD, higher wopt
6. **Mold diameter**: 150 mm used for coarser soils; may shift Proctor results
7. Fine-grained soils have higher wopt and lower MDD than coarse-grained soils

---

## 5. Modeling Strategy (Phased Approach)

### Phase A — Baseline
- Simple imputation (median/mean for numerics, mode for boolean)
- Baseline models: Linear Regression, Ridge, Random Forest
- Internal CV: 5-fold cross-validation on train.csv
- Target: Establish NMAE baseline to beat

### Phase B — Feature Engineering
Derived geotechnical features:
- `cu` = D60 / D10
- `cc` = D30² / (D60 × D10)
- `pi` = LL – PL (Plasticity Index)
- `saturation_line_mdd` = ρs / (1 + wopt_est × ρs) — physical constraint proxy
- Log-transform of PSD diameters and kf (log-normally distributed)
- Soil type classification (coarse / mixed / fine) based on psd_passing_at_0_063mm_pct

### Phase C — Advanced Models
- Gradient Boosting: XGBoost, LightGBM, CatBoost
- Ensemble/stacking of top performers
- Bayesian hyperparameter optimization (Optuna)
- Physics-informed constraints (predictions near saturation line)

### Phase D — Generalization & Robustness
- Repeated k-fold CV to estimate variance
- Adversarial validation (check train/test distribution shift)
- Final model selection: pick submission with best public LB + stable CV

---

## 6. Evaluation Metric Details

```
NMAE = 0.5 × (MAE_mdd / IQR_mdd) + 0.5 × (MAE_owc / IQR_owc)
```

- IQR values are computed from the **full dataset** (train + test combined targets)
- Both targets weighted equally regardless of absolute scale difference
- Goal: minimize NMAE → 0.0 is perfect

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Overfitting to small dataset (201 train rows) | High | High | Strict CV, regularization, simple models first |
| Missing value leakage | Medium | High | Impute train/test separately after split |
| Public LB overfit (only 15% of test) | Medium | Medium | Trust CV score; select 2 diverse submissions for final |
| PSD extrapolation noise | Medium | Medium | Use `psd_has_sedimentation` flag as feature |
| Physical constraint violation | Low | Medium | Post-process predictions against saturation line |

---

## 8. Deliverables

| Deliverable | Description |
|---|---|
| `solution.ipynb` | Main analysis + modeling notebook |
| `submission.csv` | Final prediction file (format: id, proctor_owc_pct, proctor_mdd_g_cm3) |
| `docs/02-design/features/proctor-prediction.design.md` | Technical design document |

---

## 9. Timeline

| Date | Milestone |
|---|---|
| 2026-06-30 | Project setup, EDA, baseline model |
| 2026-07-07 | Feature engineering + first competitive submission |
| 2026-07-21 | Advanced models (XGBoost, LightGBM, ensemble) |
| 2026-08-10 | Final model selection (submit top 2) |
| 2026-08-16 | Competition ends, private leaderboard revealed |

---

## 10. Acceptance Criteria

- [ ] Internal CV NMAE < 0.15 (competitive with public leaderboard)
- [ ] Public leaderboard NMAE roughly matches internal CV (within ±0.03)
- [ ] No data leakage between train/test during feature engineering
- [ ] Predictions physically plausible (MDD < saturation line)
- [ ] Submission CSV matches required format exactly

---

## 11. v5 Plan — SINDy Feature Discovery + SHAP Refinement + Stacked Ensemble

**Added**: 2026-07-02
**Baseline to beat**: v1 tuned LightGBM, CV NMAE **0.2431 ± 0.0243** (see `model_description.md`)

### 11.1 Where v1–v4 left off

| Version | Approach | CV NMAE | Verdict |
|---|---|---|---|
| v1 | XGBoost/LightGBM + Optuna | **0.2431** | Best so far — trees win on 201 rows |
| v2 | SHAP/SHAPIQ analysis on v1 | 0.2548 (feature-pruned) | Pruning alone *hurt* — need better features, not fewer |
| v3 | Twin-head PINN, physics loss | 0.2726 | NN underpowered on this dataset size; 0 ZAV violations |
| v4 | Single-head PINN script | N/A (no CV) | Lighter features, hold-out only |

**Reading**: gradient boosting is the stronger predictor at n=201; PINNs so far only pay off on physical-plausibility, not accuracy. v2's SHAP work already surfaced two concrete, unused leads: (a) raw-vs-log PSD columns are redundant (|r| > 0.87 in SHAP space), and (b) `feat_sat_mdd_proxy × psd_passing_at_0_063mm_pct` is the strongest SHAPIQ interaction for MDD but was never materialized as an explicit engineered feature. v5 acts on both, and adds SINDy as a new feature-discovery step rather than a standalone model.

### 11.2 Method 1 — SINDy-style sparse symbolic feature discovery

PySINDy is built for `dx/dt = f(x)`, but this dataset has no time axis. We use its underlying mechanism — a large candidate function **library** (polynomial, ratio, and interaction terms of the PSD/Atterberg/hydraulic variables) reduced by **STLSQ** (sequential thresholded least squares) — as **static sparse symbolic regression**: `target ≈ Ξ · Θ(features)`, keeping only the handful of terms with non-negligible coefficients. This mirrors what SINDy does for dynamical systems, just with samples in place of time steps.

- **Library**: degree-2 polynomial terms over `{D10, D30, D60, D90, fines%, clay%, LL, PL, LOI, grain_density}` plus known geotechnical ratios (Cu, Cc, PI) as seed terms — keeps the library physically grounded instead of a blind degree-3 polynomial blow-up.
- **Fit**: `pysindy.optimizers.STLSQ` (or plain scikit-learn `Lasso`/`SR3` if pysindy's API friction with static regression proves too high) per target, threshold swept via CV.
- **Output**: a short symbolic formula per target (e.g., `MDD ≈ a·grain_density − b·(fines%·D10) + c`), reported alongside its own CV NMAE as a standalone interpretable baseline.
- **Primary use**: the surviving nonzero terms become **new engineered features** fed into the v1 LightGBM/XGBoost pipeline (not a replacement for it) — a data-driven complement to the hand-picked `feat_cu`/`feat_cc`/`feat_sat_mdd_proxy` terms already in `design.md §4.1`.
- **New dependency**: `pysindy` is not yet installed (`requirements.txt` doesn't list it) — add and pin a version before implementation.

### 11.3 Method 2 — SHAP-guided refinement (acting on v2's findings)

- Drop one side of each raw/log PSD pair per v2's redundancy finding (keep whichever has higher mean |SHAP|, not both).
- Materialize the top 2–3 SHAPIQ pairwise interactions from v2 §9 (`feat_sat_mdd_proxy × fines%`, plus the top OWC pair) as explicit product features — v2 found them but never fed them back into a retrained model.
- Re-run SHAP on the v5 feature set (SINDy terms + interaction products) to confirm the new features actually carry signal before locking the feature list — don't repeat v2's mistake of pruning without validating the replacement.

### 11.4 Method 3 — Stacked ensemble (GBM + PINN as diversity, not primary)

Given PINNs underperform individually, v5 does **not** try to make PINN win outright. Instead:

- Retrain v1 LightGBM/XGBoost on the v5 feature set (SINDy + SHAP-refined) → primary predictor.
- Retrain a v3-style PINN on the same v5 feature set, physics loss unchanged (ZAV bound + Sr constraint) → diversity predictor.
- Out-of-fold predictions from both feed a simple Ridge meta-learner (or CV-optimized weighted blend, matching v1's existing blend-weight approach) — same pattern already used for the XGB+LGB blend in v1.
- Ship the stack only if it beats plain tuned-LightGBM-on-v5-features; otherwise ship the tuned GBM alone. PINN's job here is error diversity, not a solo submission.

### 11.5 Validation & Guardrails

- Same 5-fold `KFold(shuffle=True, random_state=42)` and fixed-IQR NMAE as v1–v4 — results must stay comparable to `model_description.md`.
- SINDy terms and imputers still fit on train folds only (no leakage), per existing plan §7 risk register.
- Saturation-line clip (`design.md §5.5`) still applied post-hoc as a safety net regardless of which model wins.

### 11.6 Deliverables

| Deliverable | Description |
|---|---|
| `scripts/v5.ipynb` | SINDy feature discovery + SHAP refinement + GBM/PINN stacking |
| `model_description.md` (v5 section) | Symbolic formulas discovered, CV NMAE table, comparison vs v1–v4 |
| `submissions/submission_YYYYMMDD_v5.csv` | Final v5 prediction file |
| `requirements.txt` | Add `pysindy` |

### 11.7 Acceptance Criteria

- [x] CV NMAE ≤ 0.2431 (beats v1, the current best) — achieved 0.2410-0.2420 across runs (fixed blend, 0.8 GBM + 0.2 PINN; Optuna's sampler is unseeded)
- [x] At least one SINDy-discovered term shown (via re-run SHAP) to carry non-trivial importance in the GBM model — 9/31 new SINDy/interaction terms rank in the top 20 features
- [x] No raw/log PSD duplicate pair both retained in final feature set — 11 redundant pairs found, weaker side dropped
- [x] Stack (if shipped) beats standalone tuned GBM-on-v5-features in CV — fixed blend (0.2420) beats GBM alone (0.2450); Ridge meta-learner (0.2572) did not and was not shipped
- [x] `model_description.md` updated with v5 section following the existing v1–v4 format

**Result**: `scripts/v5.ipynb` executed end-to-end, CV NMAE 0.2410-0.2420 across runs (new best), submission
`submissions/submission_20260702_1515_v5.csv`. Also integrates `helper_functions.py`
(`calc_satline`, `get_missing_data_report`) as behavior-neutral drop-ins per follow-up request.
Full writeup in `model_description.md` §v5.

---

## 12. v7 Plan — KNN Diversity Model

**Added**: 2026-07-06
**Baseline to beat**: v5 fixed blend, CV NMAE **0.2410–0.2420** (current best, see `model_description.md` §v5)

### 12.1 Where v1–v6 left off

| Version | Approach | CV NMAE | Verdict |
|---|---|---|---|
| v1 | XGBoost/LightGBM + Optuna | **0.2431** | Best tree-only baseline |
| v2 | SHAP/SHAPIQ analysis on v1 | 0.2548 (feature-pruned) | Pruning alone hurt; surfaced unused feature leads |
| v3 | Twin-head PINN, physics loss | 0.2726 | Underpowered alone; 0 ZAV violations |
| v4 | Single-head PINN script | N/A (hold-out only) | Lighter features, no CV |
| v5 | SINDy + SHAP refinement + GBM/PINN fixed blend | **0.2410–0.2420** | Current best; PINN useful only as diversity term |
| v6 | `scripts/v6.py` — organizers' `helper_functions.py` wired into a full leak-free CV pipeline (MICE imputation, ExtraTrees/HistGB/Ridge/PINN registry, random/Optuna tuning, ensemble prediction, saturation clip) | not yet logged in `model_description.md` | Infrastructure/reusability upgrade — no new model family introduced, still no KNN candidate |

**Reading**: every version so far is either gradient boosting (global, additive) or a neural net (global, smooth). No version has tried a purely local, distance-based method. v3/v5 showed that a weaker standalone model (PINN) can still add value as an ensemble diversity term. v7 asks whether KNN — a different inductive bias again (local neighborhood averaging vs. global function approximation) — can play the same diversity role, independent of whether it's competitive alone.

### 12.2 Method — KNN as a standalone diversity model

- **Model**: `sklearn.neighbors.KNeighborsRegressor` wrapped for the two targets (`MultiOutputRegressor`, or two independent single-target KNN models if the optimal `k`/metric differs meaningfully between MDD and OWC).
- **Feature set**: reuse v5's SINDy-discovered + SHAP-refined feature set (§11.3) rather than re-deriving features — isolates this experiment to a model-family comparison, not a feature-engineering one.
- **Preprocessing**: KNN is scale-sensitive, unlike the tree models used so far — features must be scaled (reuse the `PowerTransformer`/`RobustScaler`/`StandardScaler` group assignments already computed in v6's `get_scaler_assignments`). This is a new requirement v1–v6's tree-based paths didn't need.
- **Hyperparameters to sweep**: `n_neighbors` (3–15), `weights` (uniform vs. distance), distance metric (Euclidean vs. Manhattan vs. a whitened/Mahalanobis-style metric), swept independently per target.
- **Missing values**: KNN cannot fit through NaNs the way `HistGradientBoostingRegressor` can — rely on the same fold-isolated MICE imputation already used in v5/v6 before fitting.
- **Dimensionality risk**: after v5's SINDy/interaction additions the feature set is 30+ columns on 201 training rows — likely too high-dimensional for distance metrics to stay meaningful. Also test a small hand-picked or PCA-reduced feature subset as a fallback.

### 12.3 Evaluation & ensembling

- Same 5-fold `KFold(shuffle=True, random_state=42)` and fixed-IQR NMAE as v1–v6 — results must stay comparable to `model_description.md`.
- Report standalone KNN CV NMAE first. A negative result (KNN alone is not competitive) is an acceptable, documented outcome — v2 already established that "not the best solo model" doesn't mean "not useful."
- If standalone KNN underperforms both GBM and PINN (expected), test its ensembling value: OOF-blend KNN into v5's GBM+PINN mix via a fixed-weight blend or Ridge meta-learner (same pattern as v5 §11.4). Ship only if the 3-way blend beats v5's 0.2410 fixed blend.

### 12.4 Validation & Guardrails

- KNN's imputer and scaler fit on train folds only — no leakage, per existing plan §7 risk register.
- Saturation-line clip (`design.md` §5.5) still applied post-hoc regardless of which model wins.
- Explicitly log whether KNN helps only as an ensemble term or not at all — don't force a shipped submission if it adds nothing.

### 12.5 Deliverables

| Deliverable | Description |
|---|---|
| `scripts/v7.ipynb` | Standalone KNN CV evaluation + optional blend with v5's GBM/PINN |
| `model_description.md` (v7 section) | KNN CV NMAE (standalone + blended), hyperparameter choices, comparison vs. v1–v6 |
| `submissions/submission_YYYYMMDD_v7.csv` | Final v7 prediction file (only if it beats or matches v5) |

### 12.6 Acceptance Criteria

- [x] Standalone KNN CV NMAE reported and compared against v1 (0.2431) and v5 (0.2410–0.2420) — best KNN config 0.3502 OOF NMAE, not competitive with either
- [x] Feature scaling applied correctly (KNN requires it; tree models so far did not) — KNN candidates run with `scaled=True` through `get_column_preprocessor`
- [x] If blended: blend CV NMAE ≤ v5's 0.2410 fixed blend, else ship v5 as-is and document KNN as a negative result — blend reached only 0.2493 (doesn't beat v5); documented as negative result, v5 ships as-is
- [x] No leakage: imputer/scaler fit on train folds only — reused v6's fold-isolated MICE + per-fold scaler fitting unchanged
- [x] `model_description.md` updated with a v7 section following the existing v1–v6 format

**Result**: `scripts/v7.py` executed end-to-end (5-fold CV, folds stratified on dominant soil-composition
bucket — Sand/Gravel/Fine — per follow-up request, instead of v1–v6's plain shuffled `KFold`). KNN is not
competitive standalone (0.3502 vs. ExtraTrees' 0.2497 OOF NMAE) and adds no real ensemble value (best
blend 0.2493, within noise of the blend-weight search itself). Also surfaced and worked around a
pre-existing bug in `scripts/v6.py`'s per-fold NMAE table (used per-fold-local IQR instead of the fixed
global IQR §6 specifies) without modifying `v6.py` itself. v7 does not beat v5 — **v5 (0.2410–0.2420)
remains the project's best**; `submissions/submission_20260706_v7.csv` is kept for reference per
follow-up request, but should not be used as the competition entry. Full writeup in
`model_description.md` §v7.
