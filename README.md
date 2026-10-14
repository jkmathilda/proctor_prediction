# Proctor Prediction Challenge — LeiGS 2026

Our submission for the "Proctor Prediction Challenge," part of the 2nd Leipzig Geotechnical Symposium
(LeiGS 2026), hosted on Kaggle. The challenge provides an exclusive dataset from the soil mechanics
research laboratory of HTWK Leipzig and asks participants to predict Proctor compaction behavior from
classic soil classification properties.

---

## Contents

- [Objective](#objective)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Reproducing the Final Submissions](#reproducing-the-final-submissions)
- [Data](#data)
- [Further Reading](#further-reading)

---

## Objective

Predict the two Proctor compaction targets for each soil sample from its classification properties
(grain-size distribution, consistency limits, hydraulic conductivity, loss on ignition, etc.):

- **`proctor_mdd_g_cm3`** — maximum dry density, ρPr (g/cm³)
- **`proctor_owc_pct`** — optimum water content, wopt (%)

A strong correlation is known to exist between these compaction parameters and a soil's classification
properties — grain-size distribution and consistency limits (Atterberg limits) in particular — but existing
empirical formulations from the geotechnical literature are fit to narrow, specific soil groups. The goal
here is a single, robust, data-driven model that bridges classification properties to compaction parameters
across fine-grained, mixed-grained, **and** coarse-grained soils.

**Metric**: NMAE = 0.5 × (MAE<sub>MDD</sub> / IQR<sub>MDD</sub>) + 0.5 × (MAE<sub>OWC</sub> / IQR<sub>OWC</sub>),
lower is better. Scored on Kaggle in two stages — a public leaderboard (15% of the held-out data, visible
during the competition) and a private leaderboard (the remaining 15%, revealed only at the end) — so the
real test is generalization to soil samples the model never saw, not just fitting the known data.

---

## Results

Two architecturally distinct models were submitted as our final entries. Every other model we tried —
over 30 of them, spanning PINNs, Gaussian Processes, gradient-boosted trees, residual-correction stages,
blends, and pretrained tabular foundation models — either lost to one of these two on the real leaderboard
or was never trusted enough to submit. See **[`model_description.md`](model_description.md)** for the full
model-by-model comparison table, long-form writeups, outlier analysis, and seed-sensitivity checks.

| Submission | Approach | Kaggle Score (NMAE) |
|---|---|---|
| **v15** (100-seed mean) | ARD-kernel Gaussian Process + HistGradientBoostingRegressor blend on MICE-imputed engineered features | **0.2439** |
| **model1i** (precalibration) | Ridge + GPR + XGBoost blend on MICE-imputed features, fine/coarse-split imputation, id-155 outlier excluded | **0.2465** |

Both scores are the *honest* estimate for each model — not a lucky single-seed draw. A single-seed run of
v15 scored 0.2383 and is often quoted informally as "the best score," but a 100-seed sweep showed that
number was favorable variance, not a stable estimate of the architecture's real strength (see
`model_description.md`'s Final Submissions section for the full de-lucking analysis).

---

## Repository Structure

```
data/            Competition data (train.csv, test.csv) + supplementary soil-type CSVs
scripts/         One runnable pipeline per model version (v1–v17, model1–model6i, blends, baselines)
src/             Shared library code: feature engineering, imputation, blend regressors, correction stages
models/          Cached fitted models (joblib / torch checkpoints), reused across runs unless --force_retrain
submissions/     Every submission.csv generated, one per model/ablation tried
notebooks/       EDA and analysis notebooks (data_analysis.ipynb, eda.ipynb, starter.ipynb)
docs/            Plan/design/analysis documents
logs/            Per-run text logs (OOF metrics, blend weights, Optuna trials) for every script above
figures/         Diagnostic plots (residuals, leverage/Cook's distance, PCA, public/private comparisons)
resources/       Reference PDFs (MICE methodology, empirical Proctor-parameter literature)
model_description.md   Model-by-model comparison table + long-form writeups (start here)
```

---

## Setup

Requires **Python 3.10** (see `.python-version`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `scikit-learn`, `xgboost`, `lightgbm`, `torch`, `optuna`, `shap`/`shapiq`, `pysindy`,
`tabpfn`, `tabicl[finetune]`.

---

## Reproducing the Final Submissions

Both scripts read `train.csv`/`test.csv` from `data/` by default and write predictions to `submissions/`.

```bash
# v15 — single-seed run (fast; scores ~0.2383, the lucky single-seed number, not the honest estimate)
python scripts/v15.py

# v15 — honest 100-seed-mean submission requires sweeping --seed and averaging the outputs
# (see notebooks/analysis_paper.ipynb's "Seed sensitivity" cells for the exact procedure)

# model1i — precalibration (default: id 155 excluded, 50 Optuna trials, k-fold split, no OWC calibration)
python scripts/model1i.py
```

Every other model in `scripts/` follows the same CLI convention — run with `--help` for its full flag list,
or see `model_description.md` for what each script's defaults represent.

---

## Data

- **288 total observations**, split into `train.csv` (70%, targets included) and `test.csv` (30%, targets
  hidden — the prediction target of this challenge).
- **22 predictor features** per sample from classification and hydraulic laboratory tests: particle-size
  distribution (PSD) percentiles, Atterberg (consistency) limits, hydraulic conductivity, loss on ignition,
  grain density, and related derived quantities. Several columns have substantial missingness (Atterberg
  limits, hydraulic conductivity, loss on ignition) — every model here either imputes these (MICE) or drops
  them outright; see `model_description.md` for which choice each model makes.
- **Submission format**:
  ```
  id,proctor_owc_pct,proctor_mdd_g_cm3
  201,10.09,1.972
  202,11.25,1.872
  ....
  287,13.54,1.842
  ```

---

## Further Reading

- **[`model_description.md`](model_description.md)** — every model tried, one-line summary + score each,
  plus long-form writeups of the two final submissions and the general-model family.
- **[`docs/03-analysis/proctor-prediction.analysis.md`](docs/03-analysis/proctor-prediction.analysis.md)** —
  exploratory data analysis backing the feature-engineering decisions.
