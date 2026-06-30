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
