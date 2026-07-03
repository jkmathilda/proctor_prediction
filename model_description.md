# Model Description — LeiGS 2026 Proctor Prediction Challenge

**Task**: Predict `proctor_mdd_g_cm3` (MDD) and `proctor_owc_pct` (OWC) from soil classification features.  
**Metric**: NMAE = 0.5 × (MAE_mdd / IQR_mdd) + 0.5 × (MAE_owc / IQR_owc), lower is better.  
**Dataset**: 201 train / 87 test samples, 25 raw features.

---

## Summary Comparison

| Version | Approach | CV NMAE | Submission |
|---------|----------|---------|------------|
| [v5](#v5--sindy-feature-discovery--shap-refinement--gbmpinn-blend) | SINDy features + SHAP refinement + GBM/PINN blend | **0.2410** | submission_20260702_1200_v5.csv |
| [v4](#v4--standalone-pinn-script) | PINN (single-head script, hold-out only) | N/A (no full CV) | submission20250630_1544_v4.csv |
| [v3](#v3--physics-informed-neural-network-twin-head-pinn) | PINN (twin-head, physics loss in training) | 0.2726 ± 0.0257 | 
submission_20260630_1630_v3_pinn.csv |
| [v2](#v2--shap--shapiq-interpretability--feature-selection) | SHAP/SHAPIQ analysis + feature selection | 0.2548 | submission_20260630_1449_v2.csv |
| [v1](#v1--gradient-boosting-ensemble-with-optuna-tuning) | Gradient boosting ensemble + Optuna | 0.2431 ± 0.0243 | submission_20260702_1004_v1.csv |

---

## v1 — Gradient Boosting Ensemble with Optuna Tuning
**File**: `scripts/v1.ipynb` | **Saved models**: `models/v1_lgb_mdd.pt`, `models/v1_lgb_owc.pt`

### Pipeline
1. **Preprocessing** — boolean coercion, numeric cast, median imputation for 5 high-missing columns (Atterberg limits, hydraulic conductivity, LOI), missingness indicator flags.
2. **Feature engineering** — 49 total features: raw PSD D-cols (D10–D98) + log transforms, Atterberg-derived PI, geotechnical indices (Cu, Cc), grain-size fractions (sand%, gravel%, clay-to-fines ratio), soil type classification, physics-informed saturation MDD proxy (ρd_sat at estimated wopt).
3. **Models** — Ridge baseline, XGBoost, LightGBM. Separate model per target (MDD and OWC trained independently).
4. **Tuning** — Optuna (50 trials, 5-fold CV inner loop, minimizing per-target MAE).
5. **Physical constraint** — post-prediction saturation line clip (ρd ≤ 0.99 × ρd_sat). 2 test samples clipped.
6. **Final model** — best CV model (tuned LightGBM) retrained on full training set.

### Key Points
- Two independent models per target allows each to specialize on its own signal.
- Saturation constraint is enforced as a post-hoc clip, not during training.
- Optuna tuning improved NMAE by ~0.02 over default hyperparameters.
- Ensemble (50/50 XGB+LGB blend) was weaker than either tuned model individually.

### Results (5-fold CV)
| Model | NMAE (mean) | NMAE (std) |
|-------|-------------|------------|
| Baseline (Ridge) | 0.3347 | 0.0711 |
| XGBoost (default) | 0.2612 | 0.0258 |
| LightGBM (default) | 0.2628 | 0.0239 |
| Ensemble (XGB 0.5 + LGB 0.5) | 0.2562 | — |
| XGBoost (tuned) | 0.2494 | 0.0259 |
| **LightGBM (tuned)** | **0.2431** | **0.0243** |

---

## v2 — SHAP + SHAPIQ Interpretability & Feature Selection
**File**: `scripts/v2.ipynb` | **Saved models**: none (analysis only)

### Pipeline
1. Same preprocessing and feature engineering as v1 (49 features).
2. XGBoost and LightGBM fitted on full training set (default params, no Optuna).
3. **SHAP global importance** — TreeExplainer beeswarm plots and mean |SHAP| bar charts across all 4 model×target combinations.
4. **SHAP dependence plots** — scatter of SHAP value vs feature value for top-6 features per target.
5. **SHAP correlation analysis** — pairwise Pearson correlation of SHAP value vectors to identify redundant features.
6. **SHAPIQ k-SII** — pairwise interaction indices via `TabularExplainer` on 30 representative samples to find synergistic feature pairs.
7. **Feature selection** — retain features where mean |SHAP| ≥ 1% of max; 41/49 kept.
8. Final predictions using SHAP-selected feature set.

### Key Points
- Raw PSD values and their log transforms are highly redundant (|r| > 0.87–0.98 in SHAP space); only one from each pair is needed.
- Atterberg limit features (liquid limit, plastic limit) have near-zero SHAP importance despite high missingness making them noise-heavy — confirmed safe to drop.
- Top OWC drivers: `psd_size_at_d40_mm`, `feat_sat_mdd_proxy`, `psd_size_at_d70_mm`.
- Top MDD drivers: `psd_size_at_d70_mm`, `feat_cu`, `feat_cc`.
- Strongest SHAPIQ interaction for MDD: `feat_sat_mdd_proxy × psd_passing_at_0_063mm_pct` (mean |SII| = 2.19).
- Feature selection (41 features) yielded NMAE 0.2548, slightly worse than v1's full 49-feature result — suggests the dropped features carry marginal but non-zero signal.

### Results (5-fold CV)
| Feature set | NMAE |
|-------------|------|
| All 49 features (XGBoost) | 0.2612 |
| SHAP-selected 41 features (XGBoost) | ~0.2606 |
| SHAP-selected features (best config) | 0.2548 |

---

## v3 — Physics-Informed Neural Network (Twin-Head PINN)
**File**: `scripts/v3.ipynb` | **Saved model**: `models/pinn_v3_model.pt`

### Pipeline
1. Same preprocessing and feature engineering as v1 (49 features).
2. **Architecture** — `ProctorPINN`: shared trunk (49 → 256 → 128 → 64, LayerNorm + GELU + Dropout 0.15) with two separate heads (MDD head and OWC head). 59,074 parameters total.
3. **Physics loss** — two penalty terms added to Huber data loss:
   - ZAV bound: `L_bound = E[max(0, ρd − ρd_sat)²]`, weight 5.0
   - Sr constraint: degree of saturation penalised outside [0.7, 1.0], weight 0.5
4. **Training** — AdamW, lr=1e-3, batch=32, 1500 epochs per fold, best checkpoint by val NMAE.
5. **Cross-validation** — 5-fold; final model trained 2000 epochs on full dataset.
6. **Inference** — ensemble of 5 fold models + physical saturation clip (safety 0.999).
7. Checkpoint is fully self-contained: stores model weights, feature list, and preprocessing stats (medians, x_mu/sd, y_mu/sd) for portable inference.

### Key Points
- Physics constraint is embedded in training loss, not just post-hoc clipping — the model learns to avoid saturation violations rather than having them corrected after the fact.
- Degree of saturation `Sr` in test predictions: 0 ZAV violations, 70.1% of samples within Sr [0.7, 1.0].
- Twin heads allow MDD and OWC to share a common soil-feature representation while learning target-specific final transformations.
- LayerNorm (vs BatchNorm) chosen for stability on the small dataset (160 train samples per fold).
- CV NMAE (0.2726) is higher than v1 (0.2431) — the neural network is underpowered relative to gradient boosting on this tabular dataset of ~200 samples.

### Results (5-fold CV)
| Fold | Best val NMAE |
|------|--------------|
| 1 | 0.2745 |
| 2 | ~0.27 |
| 3–5 | similar range |
| **Mean** | **0.2726 ± 0.0257** |

---

## v4 — Standalone PINN Script
**File**: `scripts/v4.py` | **Saved model**: `models/v4_model.pt`

### Pipeline
1. **Feature engineering** — lighter than v1–v3: only Cu, Cc, PI, log10(kf), log10(D10/D30/D60) + raw numeric columns present in both train and test. All missing features get missingness flags (`_isna` columns).
2. **Architecture** — `ProctorPINN`: single body (n_in → 128 → 128 → 64, SiLU + Dropout 0.1) with one shared head outputting [OWC, MDD] jointly. Simpler than v3 (no twin heads, no LayerNorm).
3. **Physics loss** — same two terms as v3 but Sr range relaxed to [0.6, 1.0] (wider band).
4. **Training** — AdamW + CosineAnnealingLR (T_max = epochs), 2000 epochs, lr=2e-3.
5. **Validation** — simple 80/20 random hold-out split (no cross-validation).
6. **CLI** — fully parameterised via `argparse`; usable as `python v4.py --epochs 2000 --out submission.csv`.
7. Checkpoint stores weights + full preprocessing state for portable `predict_df()` inference.

### Key Points
- Script form makes it easy to run headlessly (no Jupyter required) and integrate into pipelines.
- Fewer engineered features than v1–v3 (no saturation proxy, no log PSD D-cols beyond D10/D30/D60) — simpler but potentially loses predictive signal.
- Single shared output head (vs twin heads in v3) means MDD and OWC compete for the same final representation.
- No cross-validation — hold-out NMAE is less reliable than 5-fold CV on a dataset of only 201 samples.
- Cosine annealing provides implicit learning rate warmdown, useful when training for more epochs.
- Best validation NMAE not recorded in this document (reported at runtime only).

---

## v5 — SINDy Feature Discovery + SHAP Refinement + GBM/PINN Blend
**File**: `scripts/v5.ipynb` | **Saved models**: `models/v5_xgb_mdd.pt`, `models/v5_xgb_owc.pt`, `models/v5_pinn_model.pt`
**Plan**: `docs/01-plan/features/proctor-prediction.plan.md` §11

### Pipeline
1. **Preprocessing / baseline features** — identical to v1 (49 features: PSD raw + log, Cu/Cc/PI, missingness flags, median imputation).
2. **SINDy sparse symbolic feature discovery** — PySINDy's STLSQ (built for `dx/dt=f(x)`) repurposed as *static* sparse regression: a degree-2 polynomial library over 11 physically-grounded base variables (D10/D30/D60, fines%, clay%, grain density, LOI, PI, log(kf), Cu, Cc) is reduced from 77 candidate terms to a compact surviving set per target via a 2D grid search over STLSQ's `(alpha, threshold)`. Default `alpha=0.05` (tuned for dynamical systems) was far too weak here — it let a collinear 77-term library on ~160 rows/fold blow up to a non-generalizing dense solution (NMAE-equivalent ~8); grid search found `alpha∈{1,2}, threshold=0.15` gives a stable fit. Surviving terms: 19 for MDD, 6 for OWC (21 unique, union). Standalone SINDy-formula NMAE-equivalent: 0.276 (MDD), 0.355 (OWC) — comparable to v1's Ridge baseline (0.335) despite being a handful of interpretable terms.
3. **SHAP-guided refinement** — acts on two v2 findings that were surfaced but never applied: (a) drops the weaker (lower mean|SHAP|) side of each raw/log PSD pair (11 dropped, all `log_psd_size_at_dXX_mm`), (b) materializes the top-5 SHAP interaction pairs per target (via exact `shap_interaction_values`, restricted to base features to avoid interaction-of-interaction bloat) as explicit product features (10 unique pairs, e.g. `feat_cc × psd_size_at_d70_mm`). Net v5 feature set: 69 (70 v5-raw − 11 redundant + 10 interaction). Re-validated with SHAP afterward: 9/31 new SINDy/interaction terms rank in the top 20 features by importance — the guardrail v2 skipped.
4. **GBM (primary)** — Optuna (50 trials × 4 model×target combos) tuning XGBoost/LightGBM on the 69-feature v5 set; XGBoost selected as primary (0.2441 CV NMAE vs LightGBM's 0.2455).
5. **PINN (diversity, not primary)** — v3's twin-head architecture and physics loss (ZAV bound + Sr constraint) retrained unchanged on the v5 feature set. Confirmed again that PINN underperforms GBM at n=201 (0.2776 vs GBM's 0.2441) — consistent with v3/v4.
6. **Ensembling** — out-of-fold GBM and PINN predictions combined two ways: a Ridge meta-learner (0.2575, worse than GBM alone — too little data for a reliable 2-feature meta-fit at n=201) and a CV-grid-searched fixed blend weight (same pattern as v1's XGB+LGB blend), which won: `0.8·GBM + 0.2·PINN` → **0.2410**, beating both GBM alone and v1's previous best.
7. **Physical constraint** — same zero-air-voids saturation clip as v1/v3 (0 predictions clipped on the final blended test predictions).

### Key Points
- SINDy's `dx/dt=f(x)` machinery is legitimately reusable as generic sparse symbolic regression by treating samples as if they were time steps — the STLSQ solver itself doesn't know the difference, but its default regularization strength assumes cleaner, lower-dimensional dynamical-systems libraries, so it needed retuning for this collinear polynomial library.
- SHAP work is only valuable if acted on: v2 computed the same redundancy/interaction findings months earlier but never engineered them back into a model; doing so here moved default-XGBoost NMAE from 0.2612 (v1 feature set) to 0.2560 (v5-refined, pre-tuning) — a "free" ~0.005 improvement before any hyperparameter search.
- PINN's role changed from "candidate primary model" (v3/v4) to "diversity source for ensembling" — it still underperforms solo, but a small (20%) blend weight extracts a genuine ~0.003 NMAE improvement over GBM alone, more than the Ridge meta-learner could recover from the same two prediction streams.
- **Environment note**: mixing XGBoost/LightGBM/SHAP (OpenMP-threaded) with PyTorch in one process deadlocks on this machine the instant `DataLoader` starts iterating. Fixed by pinning `OMP_NUM_THREADS=1` (and `OPENBLAS_/MKL_/VECLIB_MAXIMUM_/NUMEXPR_NUM_THREADS`) plus `torch.set_num_threads(1)` before any of these libraries are imported — necessary because v5 is the first version to combine GBM and PINN in a single kernel.

### Results (5-fold CV)
| Model | CV NMAE |
|-------|---------|
| **v5 fixed blend (0.8·GBM + 0.2·PINN)** | **0.2410** |
| v1 tuned LightGBM (paper baseline, 49 feats) | 0.2431 |
| v5 tuned XGBoost (69 feats) | 0.2441 |
| v5 tuned LightGBM (69 feats) | 0.2455 |
| v5 Ridge stack (GBM+PINN meta-learner) | 0.2575 |
| v5 PINN alone (69 feats) | 0.2776 |
