# Model Description — LeiGS 2026 Proctor Prediction Challenge

**Task**: Predict `proctor_mdd_g_cm3` (MDD) and `proctor_owc_pct` (OWC) from soil classification features.  
**Metric**: NMAE = 0.5 × (MAE_mdd / IQR_mdd) + 0.5 × (MAE_owc / IQR_owc), lower is better.  
**Dataset**: 201 train / 87 test samples, 25 raw features.

---

## Contents

- [Summary Comparison](#summary-comparison) — every model tried, one row each, newest first
- [General-Model Family — `model1` → `model6i`](#general-model-family--model1--model6i) — individual writeups for the Ridge+GPR+XGBoost line
- [Final Submissions](#final-submissions)
  - [v15 — ARD-GP + HistGBT Blend (100-Seed Mean)](#v15--ard-gp--histgbt-blend-100-seed-mean)
  - [model1i — Ridge + GPR + XGBoost Blend on MICE-Imputed Features (Precalibration)](#model1i--ridge--gpr--xgboost-blend-on-mice-imputed-features-precalibration)

---

## Summary Comparison

Every model tried in this project, newest first. `v1`–`v14` were the first documented wave (SINDy/SHAP/GBM/PINN,
scored by 5-fold CV NMAE only — no Kaggle account was being used for submission yet). Starting with `v6`/`model1`,
every version below was scored against the real Kaggle leaderboard at least once, a directly comparable,
apples-to-apples number across all of them (unlike the mix of CV protocols used upstream — see
[`docs/260804.md`](../docs/260804.md) §1 for the full internal-vs-Kaggle reconciliation). Two lines of development
ran in parallel: a **GPR/GBT line** (`v6`→`v17`, Gaussian Process + gradient-boosted trees) and a **general-model
line** (`model1`→`model6i`, Ridge+GPR+XGBoost blend + correction stages), plus a handful of blends and one-off
architecture/baseline experiments. **The two bolded rows are this project's actual final Kaggle submissions** —
see [Final Submissions](#final-submissions--what-we-actually-submitted) at the bottom for the long-form writeup of
both.

| Model | Approach | Score | Submission / Status |
|---|---|---|---|
| v17 | Joint (multi-output) version of v15: Intrinsic Coregionalization Model GP (learns the MDD↔OWC correlation instead of fitting it independently) + `RegressorChain` GBT | no numeric result ever logged | submitted but unscored/untracked in project notes |
| v16 | v15's exact architecture, evaluated over 3 independent stratified splits instead of 1 (variance check) | 0.2515 ± 0.034 (internal, range 0.206–0.288) | not submitted — evaluation-only |
| v15test | v15 with the CV split removed entirely — no OOF data exists, so the GP/GBT blend silently collapses to pure-GP | **0.2930 (Kaggle) — worst score in the project** | submitted, negative result |
| v15_excl155 | v15 + Cook's-distance-motivated id-155 exclusion (the lever that helped model1i a lot) | 0.2305 single-seed (lucky) / 0.2473 (100-seed mean, honest — *worse* than plain v15) | submitted, negative result |
| **v15** | ARD-GP + HistGBT blend, same as v10 but **drops** `hyd_cond_hyd_gradient` | 0.2383 single-seed (lucky draw) / **0.2439 (100-seed mean, honest) — FINAL SUBMISSION #1** | **submitted (final)** |
| v13 | Same GP+HistGBT architecture as v10, formalized with proper stratified/repeated k-fold and saved OOF artifacts | 0.2484 (recomputed from saved OOF predictions) | not submitted |
| v12 | Physics-consistent chain: predict wopt directly → predict logit(Sr at optimum) with wopt as a chained feature → reconstruct ρPr analytically (guarantees ρPr ≤ ZAV line) → blend with a direct ρPr model | not logged | not submitted |
| v10 | ARD-kernel GP + HistGBT blend, MICE-imputed engineered features, **keeps** `hyd_cond_hyd_gradient` | **0.2384 (Kaggle)** | submitted |
| v9 | First standalone GPR entry (no GBT ensemble yet), post-hoc ZAV clip | not logged separately from v10 | not submitted |
| v8 | Twin-head PINN v2 — full physics-loss suite (ZAV bound, Sr band, ρd<ρs, range boxes, air-voids bound, optional monotonicity priors) | not logged | not submitted |
| v6 | PINN + first leak-free CV infra (`helper_functions.py`, ExtraTrees/HistGB/Ridge/PINN candidate registry) | ExtraTrees 0.2497 / HistGB 0.2501 (OOF) | infra groundwork, not itself submitted |
| model6i | MDD is plain model1i; OWC gets an augmented feature set (Atterberg PI/Skempton/Hazen/Casagrande terms model1i already imputes but never uses, plus log-diff PSD ratios) | 0.2327 (internal OOF) | not submitted to Kaggle |
| model5i | Asymmetric two-stage model1i: MDD stays pure stage-1; OWC's stage-2 adds MDD's stage-1 OOF prediction as one extra feature | 0.2617 (Kaggle) | submitted, negative result |
| model4i | XGBoost + tiny feed-forward NN blend on hand-built PSD/gradation features (structurally distinct from model1i) | 0.2472 (internal nested OOF) | submitted, underperformed model1i |
| model3i | model2i architecture, but β becomes a per-sample confidence-weighted `beta_i` (bounded 2-parameter fit, can't do worse than model2i's own OOF objective) | 0.2560 (Kaggle, k-fold) / 0.2688 (Kaggle, `--split_kind shuffle`) | submitted — best-scoring correction/blend attempt in the project, still lost to plain model1i |
| model3 (hierarchical NN) | `SharedEncoderTargetPopulationHead`: shared trunk → MDD/OWC branches → fine/coarse leaf heads, PyTorch, ~1,876 params | 0.3296 ± 0.0075 (internal, `RepeatedStratifiedKFold`) | not submitted — weakest model tested |
| model2ic | model2i, but the correction's blend strength β becomes a per-row confidence factor (1/GPR-predictive-std) instead of a flat scalar | internal only | not submitted |
| model2i_kfold | model2i variant using a plain (non-exclusion) k-fold configuration | internal only (MDD R²=0.863, OWC R²=0.808) | not submitted |
| model2i | model1i + Group-C fine-grained residual correction (`MDDFineGrainedResidualCorrector` / `OWCFineGrainedResidualCorrector`) on fine-grained rows (n=89) for both targets | 0.2667 (Kaggle) | submitted, negative result |
| model1i_v15features_model1iimpute | v15's feature set, re-imputed with model1i's own fine/coarse-split discipline (removes an imputation confound — see below) | 0.2355 (internal OOF) | not submitted to Kaggle |
| model1i_v15features | model1i's exact fitting recipe applied to v15's 46-col feature set instead of its own 28 | 0.2532 (Kaggle) despite the largest internal gain of any single change tried (confound: v15's imputer fabricates Atterberg values for coarse rows) | submitted, negative result |
| model1i (`--split_kind shuffle`) | model1i-precalibration re-validated with v15's single stratified holdout instead of k-fold | 0.2574 (Kaggle) | submitted, negative result |
| model1i (OWC-calibrated) | model1i + post-hoc OWC isotonic calibration | worse than 0.2465 on Kaggle (exact score not recorded) despite a small honest internal gain | submitted, negative result — calibration reverted |
| **model1i** | Ridge+GPR+XGBoost blend, MICE-imputed `IMPUTED_FEATURES` (28 cols, fine/coarse-split imputation), id-155 excluded | **0.2465 (Kaggle, precalibration) — FINAL SUBMISSION #2**; was 0.2610 before excluding id-155 | **submitted (final)** |
| model1h | model1i's exact architecture with HistGBT swapped in for XGBoost (isolates the boosting-library variable) | 0.2495 (internal OOF, HistGBT beats XGBoost head-to-head) | not submitted — comparison only |
| model1 | Ridge+GPR+XGBoost blend, no-imputation feature set (25 cols, drops Atterberg/kf/LOI outright) | 0.2663 (Kaggle) | submitted |
| blend_v15_model1i | Prediction-level blend of v15 + model1i (most architecturally distinct pair in the project — no shared base learner, feature set, or imputer) | 0.2343 (internal, honest leave-one-seed-out) | not submitted to Kaggle |
| blend_model1i_model4i | Prediction-level blend of model1i + model4i, per-target weights from leave-one-seed-out nested CV | 0.2582 (Kaggle) | submitted, negative result (loses to model1i's true 0.2465 baseline) |
| model1i_mdd_blend_owc | Manual splice of model1i's MDD column with an alternate OWC source (no dedicated pipeline script survives in `scripts/`) | not recorded | status unclear |
| tabicl | TabICL-v2 — gradient-fine-tuned pretrained tabular foundation model, `IMPUTED_FEATURES`/CV scheme matching model1i | 0.2479 (internal OOF) | not clearly submitted to Kaggle |
| tabpfn | TabPFN-v2 pretrained tabular foundation model — in-context learning, zero gradient training, same `IMPUTED_FEATURES`/CV scheme as model1i | 0.2513 (internal OOF) | not clearly submitted to Kaggle |
| curve_lstm | Shared-LSTM encoder over the 11-point PSD curve (percentile, log-diameter) with sensitivity-weighted per-target pooling heads, plus a scalar-augmented variant | never validated — no R²/NMAE ever recorded, ablation cell never run | submission file written but not scored |
| blr | Early linear/Bayesian baseline from the project's first week | no surviving script or recorded score | superseded |
| [v14](#v14--self-contained-three-stage-pipeline-ridge--xgb-residual--finecoarse-correction) | Ridge (global linear) + XGBoost-on-residuals + fine/coarse-grained correction, stratified CV | 0.2585 (Stage 1+2; Stage 3 doesn't help) | submission_20260722_v14.csv (kept for reference — worse than v5, do not submit) |
| [v7](#v7--knn-diversity-model--uscs-particle-class-features) | KNN (standalone + blend) on v6's helper-based pipeline, soil-composition-stratified CV | 0.2493 (0.3502 standalone) | submission_20260706_v7.csv (kept for reference — worse than v5, do not submit) |
| [v5](#v5--sindy-feature-discovery--shap-refinement--gbmpinn-blend) | SINDy features + SHAP refinement + GBM/PINN blend | **0.2410** | submission_20260702_1200_v5.csv |
| [v4](#v4--standalone-pinn-script) | PINN (single-head script, hold-out only) | N/A (no full CV) | submission20250630_1544_v4.csv |
| [v3](#v3--physics-informed-neural-network-twin-head-pinn) | PINN (twin-head, physics loss in training) | 0.2726 ± 0.0257 | submission_20260630_1630_v3_pinn.csv |
| [v2](#v2--shap--shapiq-interpretability--feature-selection) | SHAP/SHAPIQ analysis + feature selection | 0.2548 | submission_20260630_1449_v2.csv |
| [v1](#v1--gradient-boosting-ensemble-with-optuna-tuning) | Gradient boosting ensemble + Optuna | 0.2431 ± 0.0243 | submission_20260702_1004_v1.csv |

---

## General-Model Family — `model1` → `model6i`

The table above gives each of these one line; this section gives each its own short writeup. Every model
here shares the same bones — `general_model.py` / `general_model_impute.py`'s `WeightedBlendRegressor`
(Ridge + Gaussian Process Regression + XGBoost, blend weights fit by constrained optimization against
5-fold `StratifiedKFold`-on-MDD-quantile OOF predictions), the same saturation-line physics clip, and the
same joblib model-caching/CLI conventions — so each entry below only calls out what's different from
`model1i` (the family's best and final-submitted member — full writeup in
[Final Submissions](#final-submissions--what-we-actually-submitted)).

**model1** — the family's starting point. Uses `NO_MISSING_FEATURES` (25 columns): no MICE imputation at
all, Atterberg limits / `hyd_cond_kf_m_s` / `loss_on_ignition_pct` are dropped outright because they have
missing values. **0.2663 (Kaggle)** — worse than every imputed model in the family, the clearest evidence
in this project that MICE recovers real signal rather than just adding noise.

**model1h** — identical to model1i except the XGBoost blend component is swapped for a
`HistGradientBoostingRegressor` (Optuna-tuned the same way, 50 trials). A direct, equal-effort comparison:
MDD is a near-tie (R²=0.8533 XGB vs. 0.8522 HistGBT), OWC is a real win for HistGBT (R²=0.7888 vs. 0.7983),
and combined it's **0.2495 (internal OOF)** vs. XGBoost's 0.2549 — HistGBT wins the isolated comparison.
Never submitted to Kaggle on its own; its purpose was to test whether v15's HistGBT-based blend beating
model1i's XGBoost-based blend was really about the boosting library, and the internal numbers say it's part
of the story but not all of it (v15 still separately wins on ARD feature pruning and a simpler blend).

**model1i** — MICE-imputes the columns model1 drops (`IMPUTED_FEATURES`, 28 cols), fit separately for
fine- vs. coarse-grained rows so Atterberg limits are never fabricated for coarse (non-plastic) soils, plus
id-155 excluded from training. **0.2465 (Kaggle) — this project's second final submission.** See
[Final Submissions](#final-submissions--what-we-actually-submitted) for the full writeup, including the
id-155 outlier finding and the OWC-calibration attempt that was tried and reverted.

**model2i** — model1i plus a second-stage residual correction on fine-grained rows only (n=89):
`MDDFineGrainedResidualCorrector` / `OWCFineGrainedResidualCorrector`, using MICE-completed Atterberg
limits/PI/LOI/kf plus the general model's own OOF prediction as specialist features, blended back in with a
single global weight β fit by bounded scalar optimization. Nested internal result looks like a real win
(fine-grained MDD R² 0.807→0.826, OWC R² 0.767→0.779), but **0.2667 (Kaggle) — worse than plain model1i**.
The residual the correctors fit is thin to begin with (nested OOF R² on the residual itself: 0.05–0.08 for
MDD, 0.04 to −0.38 for OWC, on an already-small n=89 subset) — a second model layered on that weak a signal
overfits rather than generalizes. Two variants exist: `model2i_kfold` (plain k-fold, id-155 included,
internal only) and `model2ic` (β becomes a per-row confidence factor derived from the GPR residual model's
predictive std, shrinking the correction wherever the specialist is unsure — internal only, never
submitted).

**model3i** — same idea as model2i, but β becomes a per-*sample* confidence-weighted `beta_i` instead of
one global scalar, driven by the specialist's own out-of-fold predictive uncertainty at that row (low
confidence → lean on the general model; high confidence → lean on the correction). Bounded 2-parameter fit
initialized at model2i's own optimal scalar β, so it can never do worse on the OOF objective it's fit to.
**0.2560 (Kaggle, default k-fold)** — the best-scoring correction/blend attempt in the whole project, and a
clear ~4% relative improvement over model2i (0.2667), but still loses to plain model1i (0.2465). Re-run with
`--split_kind shuffle` (v15's single stratified holdout instead of k-fold) scored **0.2688** — worse, the
third independent confirmation in this project that the shuffle-split scheme underperforms k-fold here.

**model4i** — structurally distinct from the rest of the family: an XGBoost + tiny feed-forward-NN blend on
hand-built PSD/gradation features, rather than model1i's Ridge+GPR+XGBoost architecture. **0.2472 (internal
nested-CV OOF: MDD R²=0.8333, OWC R²=0.8251)**; submitted to Kaggle but underperformed model1i. Its main
role in the project ended up being a diversity source for `blend_model1i_model4i` rather than a standalone
contender.

**model5i** — asymmetric two-stage extension of model1i, motivated by a real residual correlation (r=−0.32)
left over between MDD and OWC even after `IMPUTED_FEATURES` explains most of their raw −0.76 correlation.
MDD stays pure stage-1 model1i (an earlier symmetric version made MDD worse by feeding it OWC's noisier OOF
prediction). OWC's stage-2 adds one extra feature — MDD's stage-1 OOF prediction (never the true value) —
and refits. Internal NMAE: OWC 0.2776→0.2753 (small real gain), MDD unchanged. **0.2617 (Kaggle) — worse
than plain model1i**, the third correction-stage attempt in the family to lose despite an honestly-validated
internal gain.

**model6i** — MDD is exactly model1i's stage; OWC gets a separately-tuned blend on an augmented feature set
adding Atterberg-derived plasticity/Skempton/Hazen/Casagrande terms and log-diff PSD ratios that model1i's
own imputer already computes but `IMPUTED_FEATURES` never includes. An earlier symmetric version (both
targets augmented) helped OWC but hurt MDD, so only OWC gets the extra columns here. **0.2327 (internal
OOF: MDD NMAE 0.1936, OWC NMAE 0.2717)** — the best internal combined number in the entire general-model
family, but never submitted to Kaggle, so it's untested against the pattern (seen in model2i/model5i/
model1i_v15features) of honest internal gains failing to survive a real submission.

**blend_model1i_model4i** — prediction-level blend of model1i and model4i, per-target weights from
leave-one-seed-out nested CV (5 seeds × 10-fold): MDD 0.88·model1i + 0.12·model4i (honest NMAE 0.1937,
model4i adds almost nothing), OWC 0.31·model1i + 0.69·model4i (honest NMAE 0.2731, a real ~1.5% relative
gain over either alone in the nested estimate). **0.2582 (Kaggle) — still worse than model1i's true 0.2465
baseline**; the weights were validated against model1i's *stale* pre-id-155-exclusion score, so what looked
like a milestone at the time was actually a loss once compared against model1i's real strength.

**blend_v15_model1i** — the project's most architecturally distinct pairing (v15 and model1i share no base
learner, feature set, or imputer), blended the same leave-one-seed-out way. **0.2343 (internal, honest)** —
this project's best honest internal estimate for any blend, but it was never submitted to Kaggle, so it
remains this project's single most promising untested lead for beating v15's 0.2439.

---

## Final Submissions

Two very different models were chosen as this project's final Kaggle entries: **v15** (the GPR/GBT line's
best model) and **model1i** (the general-model line's best model). Every other model documented above,
including several with *better-looking internal validation numbers*, either lost to one of these two on
the actual leaderboard or was never trusted enough to submit as a final entry — see
[`docs/260804.md`](../docs/260804.md) for the full experimental trail behind that filtering. Both final
picks share one important property that most of the negative-result models above lack: their internal
cross-validation estimate turned out to be an honest, close predictor of their real Kaggle score, once
seed/split noise was accounted for.

### v15 — ARD-GP + HistGBT Blend (100-Seed Mean)
**File**: `scripts/v15.py` | **Submission**: 100-seed-mean of `scripts/v15.py` runs (see
`notebooks/analysis_paper.ipynb`'s "Seed sensitivity" cells) | **Kaggle score**: **0.2439** (honest,
100-seed mean) — a single lucky seed (45) of the same script scored 0.2383, which is the number most often
quoted informally as "the best score," but is not the trustworthy estimate of this architecture's real
strength.

**Architecture.** Two independent per-target models, each an ensemble of:
- An **ARD-kernel Gaussian Process** (`Matern(ν=2.5)` + `WhiteKernel`, one length-scale per feature) fit on
  ~47 MICE-imputed engineered features (DIN grain-size fractions, log-ratio PSD shape terms, imputed
  kf/LOI with missingness flags). After fitting, features whose learned length-scale exceeds 100 (the GP
  found them irrelevant) are dropped, with a floor of 5 kept, and the GP is refit on the survivors — this
  narrows the ~47-column input down to roughly 13–17 features per target, entirely automatically. MDD leans
  on gradation/PSD-size features and Atterberg plasticity; OWC leans more on fines content
  (`psd_passing_at_0_063mm_pct`), `atterberg_plastic_limit_pct`, and `loss_on_ignition_pct` — matching the
  project's independently-derived XGBoost gain-importance findings for the same two targets.
- A **`HistGradientBoostingRegressor`** on the same feature set (no ARD pruning — trees get the full column
  set).
- A **fixed per-target blend weight** (GP share 0.9 for MDD, 0.35 for OWC) rather than an optimized one —
  simpler than the general-model line's SLSQP-fit 3-way blend, and it still wins. Internally, GP-alone
  scores 0.2166 (MDD-heavy) and GBT-alone scores 0.2359; the blend (0.2037 on the single-split internal
  estimate) beats both, the same "two different error profiles cancel out" pattern seen throughout this
  project (v5's GBM/PINN blend, the general-model line's Ridge+GPR+XGBoost blend).
- `hyd_cond_hyd_gradient` is **dropped outright** (not imputed, not used) — 70% missing, always co-missing
  with `hyd_cond_kf_m_s`, and empirically a wash: architecturally-identical v10 (which keeps this column)
  scored 0.2384 on Kaggle, v15 (which drops it) scored 0.2383 — a 0.0001 difference, indistinguishable from
  noise.
- Predicted MDD is clipped to the Zero-Air-Voids (saturation) line as a final physics guardrail, same
  convention as every other model in this project.

**Why the submission is a 100-seed mean, not the single best-looking run.** v15's own validation scheme is
a single 80/20 `StratifiedShuffleSplit` (density-quantile stratified), and this project's own v16 experiment
— the identical architecture evaluated over 3 independent splits — showed that single-split NMAE ranges from
0.206 to 0.288 by chance alone (mean 0.2515 ± 0.034). A single Kaggle submission is exactly one more draw
from that same distribution: seed 45's 0.2383 looked like a milestone, but running the full pipeline (fresh
MICE imputation, fresh GP kernel fit, fresh GBT fit, fresh blend weight) across 100 different seeds and
averaging the 100 resulting test-set predictions gives a submission whose score (0.2439) lands almost
exactly on an independently-computed 5-fold internal aggregate for the same architecture (also 0.2439) —
two unrelated estimation methods agreeing this closely is strong evidence that ~0.244 is v15's real expected
performance, and 0.2383 was a favorable draw. The same de-lucking check was later run with id-155 excluded
from training (`v15_excl155`): single-seed scored an even more striking 0.2305, but its own 100-seed mean
came back at 0.2473 — *worse* than plain v15's 0.2439 — confirming that dropping id-155 (a large, real,
reproducible win for model1i) does not transfer to v15's architecture, and that chasing single-seed lucky
scores in this family is a dead end. **The submitted entry is the honest 100-seed-mean prediction, not the
lucky single-seed one**, specifically because it's the only version of v15's score that has been
cross-checked against an independent estimate and found trustworthy.

### model1i — Ridge + GPR + XGBoost Blend on MICE-Imputed Features (Precalibration)
**File**: `scripts/model1i.py` | **Submission**: `submissions/submission_model1i.csv` (default k-fold run,
`--optuna_trials 50`, id 155 excluded, no OWC calibration applied) | **Kaggle score**: **0.2465** — the
best-scoring member of the entire general-model family, and the closest any model in this project came to
beating v15.

**Architecture.** A `WeightedBlendRegressor` (Ridge + Gaussian Process Regression + Optuna-tuned XGBoost,
blend weights fit by constrained optimization against 5-fold `StratifiedKFold`-on-MDD-quantile OOF
predictions) trained per target on `IMPUTED_FEATURES` — 28 columns built from `model1.py`'s no-imputation
25-column baseline, with Atterberg limits, `hyd_cond_kf_m_s`, and `loss_on_ignition_pct` MICE-imputed back
in (rather than dropped) using an imputer fit **separately for fine- vs. coarse-grained rows**, since
Atterberg limits are only physically measured for fine-grained soils in the first place — imputing them for
a coarse row would fabricate a lab measurement that structurally doesn't apply. Missingness-indicator flags
ride alongside every imputed column so the model can discount a fabricated point estimate rather than trust
it as precise. The fitted blend weights land very differently per target: MDD blends `ridge=0.000
gpr=0.503 xgboost=0.497` (roughly even GPR/XGBoost split), while OWC blends `ridge=0.225 gpr=0.000
xgboost=0.775` (XGBoost-dominated, Ridge picking up a small linear residual, GPR contributing nothing) —
consistent with `docs/03-analysis/proctor-prediction.analysis.md`'s finding that OWC is poorly explained by
a purely linear or purely GP fit. Predicted OWC is clipped to ≥ 0 and predicted MDD is clipped to the
Zero-Air-Voids line, the same physics guardrail v15 uses.

**The single largest, most-confirmed lever in this entire project: dropping training id 155.** Row 155 is an
extreme leverage point — the highest Cook's distance in the dataset by a wide margin (leverage 0.985, OWC
Cook's D=12.4 vs. ~0.1 for the next-highest row; see `figures/owc_leverage_cooksd_id155.png` and
`figures/mdd_owc_scatter_id155.png`). Excluding it from model1i's training set moved the Kaggle score from
**0.2610 to 0.2465** — a ~5.6% relative improvement, and unlike almost every other intervention tried in
this project (corrections, blends, calibration, alternate feature sets), it isn't a fitted parameter or a
tuned weight, it's the removal of one bad data point. It is also, notably, an effect that did *not*
transfer to v15 when the same check was tried there (see above) — a reminder that "outlier hurts model X"
is an empirical, per-architecture question in this project, not a general rule.

**Why this is the "precalibration" version, specifically.** A post-hoc OWC isotonic calibration step
(`fit_owc_calibrator`) was built and validated with honest nested 10-fold CV, which showed a small positive
internal NMAE delta (~-0.0015 raw, ~-0.005 with damping) — i.e., it looked like a real, if modest,
improvement by every internal measure available. It was submitted to Kaggle anyway to check, and it scored
**worse** than the uncalibrated 0.2465. This is the same pattern seen repeatedly elsewhere in this project
(model2i, model5i, blend_model1i_model4i, model1i_v15features all had positive-looking honest internal
deltas that didn't survive contact with the real test set) — at n≈200 training rows, a small internal gain
is not reliable evidence of a real one. The calibration step was reverted, and the version actually
submitted as this project's second final entry is the plain, uncalibrated `model1i.py` output — hence
"precalibration" in every score reference above, even though no separately-calibrated version was ever kept
as the shipped default.
