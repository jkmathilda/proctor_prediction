# Data Analysis

**Dataset**: 201 training samples, 25 raw columns (`train.csv`). Targets: `proctor_mdd_g_cm3` (MDD, mean 1.97 g/cm³, range 1.61–2.21) and `proctor_owc_pct` (OWC, mean 10.1%, range 3.5–20.7%).

**Engineered features added**: `clay`/`silt`/`sand`/`gravel` (from PSD passing %), `fine-grained` flag (clay+silt > 12%), `feat_cu`/`feat_cc`/`feat_log_cu` (coefficient of uniformity/curvature), `feat_pi` (plasticity index).

---

## 1. Correlation

- **Pearson**: only one feature pair exceeds |r| ≥ 0.9 — `atterberg_liquid_limit_pct` vs `atterberg_plastic_limit_pct` (expected, since PI is derived from both). No other severe multicollinearity among the checked high-correlation candidates.
- **Spearman vs Pearson**: broadly consistent rankings against both targets, no evidence of strong nonlinear-but-monotonic relationships that Pearson would miss.
- **Mutual information**: used to cross-check correlation-based rankings; no feature stands out as having high MI but near-zero linear correlation, i.e. no major nonlinear/non-monotonic signal hiding from the linear methods.

## 2. Linearity (Linear Regression vs Ridge)

5-fold CV, `params_av` feature set (drops Atterberg limits, hydraulic conductivity, LOI, `feat_pi`, target leakage columns):

| Model | MDD train R² | MDD val R² | OWC train R² | OWC val R² |
|---|---|---|---|---|
| Linear Regression | 0.813 | 0.744 ± 0.082 | 0.827 | **0.302 ± 0.879** |
| Ridge (best α ≈ 17.9 for MDD, 26.0 for OWC) | 0.798 | 0.735 ± 0.035 | 0.812 | **0.388 ± 0.736** |

- MDD is reasonably well explained by a linear model (~0.74 val R²).
- OWC is **not** well explained linearly — Fold 4 val R² is strongly negative for both plain Linear (-1.45) and Ridge (-1.08), and the huge std (0.74–0.88) shows the linear OWC fit is unstable/non-generalizing on at least one fold.
- Ridge trades a little mean R² for much lower variance across folds (regularization helps stability, especially for OWC), so Ridge > plain Linear Regression overall for both targets.
- Largest linear coefficients for OWC come from the PSD size features (`d60`, `d30`, `d80`, `d40`, `d70`, `d20`), several with coefficients of magnitude ~2–3 (after scaling) — much larger than any MDD coefficient, consistent with OWC being harder to fit linearly (unstable, high-magnitude coefficients typically shrink a lot under Ridge, which is exactly what's observed in the Linear-vs-Ridge coefficient comparison table).

## 3. Latent Structure (PCA)

- PCA fit on the standardized `params_av` feature set (25 components).
- Spearman correlation between each PC score and the **Ridge residuals**: almost all PCs show negligible/non-significant correlation with residuals (most p-values > 0.05); a few borderline significant ones (PC4, PC6, PC11 for MDD/OWC, p ≈ 0.03–0.09) but with very small ρ (~0.10–0.20).
- **Conclusion (per notebook annotation)**: "Linear latent structure exists and is captured well by Ridge. Residual analysis shows no remaining dependence on individual principal components." — i.e., the leftover error is not explained by any single linear latent direction, pointing toward nonlinear effects instead.

## 4. Nonlinearity

- Residual-vs-feature scatterplots (Ridge residuals against every raw feature) were inspected visually for both targets — used to motivate testing nonlinear models rather than yielding a single quantified finding.

### XGBoost (5-fold CV)

| Target | Train R² | Val R² |
|---|---|---|
| MDD | 0.983 | **0.835 ± 0.035** |
| OWC | 0.979 | **0.797 ± 0.047** |

Large jump over linear/Ridge, especially for OWC (0.39 → 0.80 val R², and much lower fold-to-fold variance). Confirms OWC's relationship to the raw features is substantially nonlinear.

**XGBoost gain importance (top features)**:
- MDD: dominated by coarse PSD size features — `psd_size_at_d70_mm` (17.3%), `d90` (10.8%), `d60` (10.1%), `d95` (9.0%), `d98` (8.7%), `d50` (8.1%), then engineered `feat_cc`/`feat_cu` (~4.8% each).
- OWC: dominated by mid-range PSD size features — `psd_size_at_d50_mm` (26.9%), `d40` (18.8%), `psd_passing_at_0_063mm_pct` (14.1%), `d70` (7.2%), `d60` (6.3%).
- Engineered features (`feat_cu`, `feat_cc`, `feat_log_cu`) contribute meaningfully to MDD (combined ~12% gain) but much less to OWC (combined ~2.4%).

**SHAP**: beeswarm/bar plots confirm the same top PSD-size features drive predictions for both targets, with directionally interpretable effects (used to sanity-check the gain-importance ranking rather than surfacing new features).

### Gaussian Process Regression (5-fold CV, RBF + White kernel)

| Target | Train R² | Val R² | Val RMSE |
|---|---|---|---|
| MDD | 0.923 | 0.815 ± 0.042 | 0.059 |
| OWC | 0.884 | 0.755 ± 0.072 | ~1.74 |

GPR performs close to but slightly below XGBoost on both targets, and with somewhat higher fold variance for OWC — another confirmation of nonlinearity, with XGBoost as the strongest model tested.

## 5. Linear vs Nonlinear Model Comparison

| Model | MDD mean val R² | OWC mean val R² | MDD val std | OWC val std |
|---|---|---|---|---|
| Linear | 0.744 | 0.302 | 0.082 | 0.879 |
| Ridge | 0.735 | 0.388 | 0.035 | 0.736 |
| **XGBoost** | **0.835** | **0.797** | **0.035** | **0.047** |
| GPR | 0.815 | 0.755 | 0.042 | 0.072 |

**Takeaway**: MDD is moderately linear (linear models already capture ~0.73–0.74 R², nonlinear models add ~0.08–0.09). OWC is strongly nonlinear — linear models are unstable and barely better than a mean predictor on some folds, while tree/kernel methods roughly triple the explained variance and dramatically cut the fold-to-fold variance. **XGBoost is the best-performing single model for both targets.**

## 6. Missing Data

Missing counts (out of 201 rows), consistent before/after feature engineering:

| Column | Missing |
|---|---|
| `hyd_cond_kf_m_s` | 140 |
| `hyd_cond_hyd_gradient` | 140 |
| `atterberg_liquid_limit_pct` | 175 |
| `atterberg_plastic_limit_pct` | 175 |
| `loss_on_ignition_pct` | 140 |
| `feat_pi` (derived from Atterberg limits) | 175 |

- Missingness is **not random with respect to soil type**: of the 175 rows missing `atterberg_liquid_limit_pct`, 63 are `fine-grained` and 112 are coarse-grained — Atterberg limits are typically only measured/meaningful for fine-grained soils, so this is a structural (not haphazard) missingness pattern that should probably be modeled as a category rather than imputed away.
- Rows missing `hyd_cond_hyd_gradient` don't show a systematic shift in any model's relative residual distribution — dropping/imputing these columns doesn't appear to bias predictions for the remaining models.
- Rows missing both `loss_on_ignition_pct` and `atterberg_plastic_limit_pct` (but with hydraulic conductivity present) split roughly evenly between fine-grained (13) and coarse-grained (10) — no strong pattern there.
- Largest OWC GPR errors (top of `owc_gpr_res_pct_abs`) are concentrated in rows with missing or very low `loss_on_ignition_pct`, suggesting organic content data gaps may be linked to the hardest-to-predict OWC samples.
- Follow-up: §7 tests whether the two most-missing lab measurements (`hyd_cond_kf_m_s`, `loss_on_ignition_pct`) can actually be recovered from the universal features, rather than assuming they can only be encoded as a missingness category.

## 7. Imputation Reliability

Before trusting any imputed values, `hyd_cond_kf_m_s` and `loss_on_ignition_pct` were each tested for whether they can be reconstructed from the universal (`params_av`) feature set — 5-fold OOF XGBoost, trained/evaluated only on the ~61 rows where each is actually observed (log10-transformed target, since both span multiple orders of magnitude):

| Target | OOF R² (log10) | OOF RMSE (log10) | Median error factor | 75th %ile error factor |
|---|---|---|---|---|
| `hyd_cond_kf_m_s` | 0.754 | 1.145 | ~3.5× | ~14.7× |
| `loss_on_ignition_pct` | 0.807 | 0.171 | ~1.26× | ~1.4× |

- Both are recoverable in the sense of explaining most of the variance, but **not equally trustworthy as point estimates**. LOI reconstructs tightly (most predictions within ~1.4× of the true value). kf's reconstruction is heavy-tailed — a typical prediction is within ~3.5×, but a meaningful minority miss by an order of magnitude or more (worst observed error factor ≈ 3637×), since kf itself spans ~7 orders of magnitude in this dataset.
- This nuances the §6 missing-data conclusion: rather than uniformly "model as a category, don't impute," LOI is a reasonable imputation candidate, while kf imputation should be paired with a missingness indicator and treated as a coarse/approximate signal rather than a precise measurement.
- An `df_impute` frame was built with XGBoost-completed `hyd_cond_kf_m_s` and `loss_on_ignition_pct` columns (`impute_feature`, log-space regression), but as of this analysis these completed columns are **not yet fed into the blended models below** — the "three blend with imputed data" comparison is a planned but unimplemented follow-up.

## 8. Model Blending (Ridge + GPR + XGBoost)

A constrained-weight blend (SLSQP, weights ≥ 0 summing to 1, minimizing OOF MSE) was tested against single models and a residual-correction hybrid.

**Ridge + GPR + XGBoost (optimized weights):**

| Target | Ridge weight | GPR weight | XGBoost weight | OOF R² | OOF RMSE |
|---|---|---|---|---|---|
| MDD | 0.089 | 0.366 | 0.545 | **0.850** | 0.054 |
| OWC | 0.000 | 0.273 | 0.728 | **0.805** | 1.484 |

- This is the best result found for **both** targets in this analysis — better than any single model (XGBoost alone: MDD 0.835/OWC 0.797; GPR alone: MDD 0.815/OWC 0.755).
- For OWC, Ridge's optimized weight is exactly **0** — consistent with §2's finding that OWC is poorly explained linearly; Ridge contributes nothing once GPR/XGBoost are available.

**GPR + XGBoost only (Ridge dropped)**: MDD R² = 0.848 (weights 50/50), OWC R² = 0.805 (weights 27/73, identical to the 3-way blend). Dropping Ridge costs almost nothing for MDD and nothing for OWC — confirms Ridge's marginal-at-best value once GPR and XGBoost are blended.

**Ridge + weighted-GPR-residual-correction** (alternative hybrid: prediction = Ridge + α·GPR(Ridge residual)):

| Target | Ridge-only R² | Unweighted hybrid R² | Weighted hybrid R² (best α) |
|---|---|---|---|
| MDD | 0.740 | 0.764 | 0.781 (α = 0.612) |
| OWC | 0.282 | 0.298 | 0.302 (α = 0.687) |

- The residual-correction approach improves on Ridge alone but falls well short of the direct constrained blend (0.850 / 0.805) for both targets — confirms the notebook's own conclusion: **"blend works better than residual correction."**

## 9. Target Dependence & Alternative Architectures

- **Targets are strongly (negatively) correlated**: raw MDD vs. OWC r = -0.768 (physically expected — denser optimum compaction goes with lower optimum water content).
- **Best-blend residual correlation**: r = -0.474 — still meaningfully negative, meaning each target's independently-tuned model leaves correlated error behind. This motivated testing whether chaining one target's prediction into the other's inputs would help.

**Architectures compared (5-fold OOF, same `params_av` feature set):**

| Architecture | MDD R² | OWC R² |
|---|---|---|
| **Separate Ridge+GPR+XGBoost blend** | **0.850** | **0.805** |
| OWC → MDD chain (predicted OWC as feature) | 0.830 | — |
| MDD → OWC chain (predicted MDD as feature) | — | 0.787 |
| Independent XGBoost (no chaining) | 0.827 | 0.788 |
| Separate MLP (2 independent networks) | 0.179 | 0.709 |
| Shared multi-output MLP (1 network, 2 outputs) | 0.097 | **-0.151** |

- **Chaining doesn't help**: feeding a predicted MDD into the OWC model (or vice versa) performs about the same as, or slightly worse than, not chaining at all — the plain independent-XGBoost baseline is competitive with both chained variants.
- **MLPs substantially underperform** tree/kernel methods on this ~200-row dataset, consistent with the small-data rationale already used elsewhere in this project for preferring GPR/XGBoost over neural nets.
- **The shared multi-output MLP is the worst model tested** — OWC R² is *negative* (worse than predicting the mean), and its cross-output residual correlation is r = -0.888 (even more negative than the blend's -0.474), suggesting the shared network mostly learns the MDD/OWC trade-off axis rather than each target's independent signal.
- **Conclusion (matches notebook annotation): independent per-target models with the Ridge+GPR+XGBoost blend is the best architecture found.** Chaining and shared-network multi-output learning both underperform treating MDD and OWC as separate, independently-blended problems.

## Overall Conclusions

1. **MDD** is reasonably linear and well-behaved; Ridge/XGBoost both perform well, XGBoost only modestly ahead.
2. **OWC** is the harder target — plain linear models are unstable (even producing negative fold R²), and the gain from switching to nonlinear models (XGBoost/GPR) is large (~0.3–0.5 R² improvement).
3. PSD size features (especially d40–d90 range) and the engineered `feat_cu`/`feat_cc` gradation indices are the dominant predictors for both targets, more so for MDD than OWC.
4. Missingness in Atterberg limits, LOI, and hydraulic conductivity is structural (tied to soil coarseness), not random.
5. XGBoost is the strongest **single** model tested for both targets, motivating its use as a core component in later pipeline versions (see `model_description.md`).
6. A constrained Ridge+GPR+XGBoost blend beats every single model and the residual-correction hybrid, reaching the best OOF R² found in this analysis for both MDD (0.850) and OWC (0.805).
7. Chaining one target's prediction into the other's features, and multi-output/shared-network architectures, do **not** improve on independent per-target blended models — the shared MLP is in fact the worst model tested (negative OWC R²).
8. `hyd_cond_kf_m_s` and `loss_on_ignition_pct` can be reconstructed from universal features with usable OOF R² (0.75 / 0.81), though kf's reconstruction error is heavy-tailed; LOI is the safer imputation candidate. This imputed data has not yet been incorporated into the blended models above.
