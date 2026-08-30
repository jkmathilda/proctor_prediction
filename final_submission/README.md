Files behind this project's two Kaggle-submitted models. See `docs/README.md` and
`docs/model_description.md` ("Final Submissions — What We Actually Submitted") for the full writeup.

- **Submission #1 — v15 (100-seed mean)**, Kaggle score **0.2439**
- **Submission #2 — model1i (precalibration)**, Kaggle score **0.2465**

Below is every file in this folder, grouped by subfolder.

---

## `data/`
Only the two files either final script actually reads (no supplementary soil-type CSVs are used).
- `train.csv` — competition training data (targets included), read by both `scripts/v15.py` and
  `scripts/model1i.py`.
- `test.csv` — competition test data (ids 201–287, features only) — the rows predicted in both
  submission CSVs.

## `src/`
Shared library code imported by the scripts below.
- `helper_functions.py` — feature-engineering / preprocessing helpers used by both final scripts,
  including `calc_satline` (the Zero-Air-Voids physics clip applied to predicted MDD in both models).
- `general_model_impute.py` — `WeightedBlendRegressor`, the fine/coarse-split MICE imputation pipeline,
  and related utilities; imported by `scripts/model1i.py` (`from src.general_model_impute import ...`).

## `scripts/`
- `model1i.py` — trains/evaluates **model1i**: a Ridge + Gaussian Process + Optuna-tuned XGBoost blend
  on MICE-imputed features (id 155 excluded, k-fold blend-weight fitting). Run with defaults to
  reproduce `submissions/submission_model1i.csv` exactly.
- `v15.py` — trains/evaluates **v15** for a single seed: an ARD-kernel Gaussian Process +
  `HistGradientBoostingRegressor` blend. `--seed` (default 45) controls the run; one run alone does
  *not* reproduce the submitted entry (see below).
- `v15_seed_sweep.py` — runs `v15.py`'s `main()` across 100 seeds in parallel (CPU multiprocessing),
  averages the per-row test predictions, and writes `submissions/submission_v15_100seed_mean.csv` plus
  `models/v15_100seed_sweep_checkpoint.pkl`. This *is* the procedure that produces submission #1 — written
  to stand in for `notebooks/analysis_paper.ipynb`'s "Seed sensitivity" cells (referenced in
  `docs/model_description.md`), which were not present in the repo. Actually re-run end-to-end
  (100/100 seeds, ~34 min wall time, completed 2026-08-29); its output matches the original
  `submissions/v15_mean.csv` closely (mean abs diff 0.13 OWC / 0.0097 MDD — consistent with the GP
  kernel optimizer's restarts not being fully seed-locked run-to-run).
- `v15_excl155_seed_sweep.py` — the same sweep pattern with training id 155 excluded. A negative-result
  variant (its 100-seed mean scored *worse* than plain v15) — **not** part of either final submission,
  kept only as the in-repo reference for the sweep methodology.

## `models/`
Cached artifacts from the runs above.
- `model1i.joblib` — the fitted `model1i` blend (Ridge+GPR+XGBoost) from the exact run that produced
  `submissions/submission_model1i.csv`. Reload with `joblib.load(...)` to skip retraining.
- `v15_100seed_sweep_checkpoint.pkl` — a pickled `{"owc_wide": DataFrame, "mdd_wide": DataFrame}`, each
  with 100 columns (one per seed) of per-row test predictions from `v15_seed_sweep.py`. Averaging these
  columns is exactly how `submissions/submission_v15_100seed_mean.csv` was produced. This is the closest
  thing to "the model" the 100-seed-mean approach has — there is no single fitted GP/GBT object behind
  it, since the submission is a prediction-level average across 100 independently-fit models.
- `v15_seed45.pt` — the cached fitted GP+GBT model from one `v15.py` run (seed 45, the script's default
  `--model_out`). **Caveat**: this is only the seed-45 run's model — the same run that scored the
  "lucky" 0.2383 single-seed number — and is not equivalent to the 100-seed-mean submission.

## `submissions/`
- `submission_model1i.csv` — the exact file submitted for entry #2 (Kaggle 0.2465).
- `submission_v15_100seed_mean.csv` — the 100-seed-mean file submitted for entry #1 (Kaggle 0.2439),
  regenerated via `scripts/v15_seed_sweep.py`.

## `docs/`
Copied from the repo root / `docs/` for context.
- `README.md` — the main repo's README (objective, results table, repo structure, setup instructions).
- `model_description.md` — every model tried in the project, one-line summary + score each, plus the
  long-form "Final Submissions" writeups for v15 and model1i excerpted above.
- `260804.md` — the detailed experimental log behind the model comparisons (id-155 outlier analysis,
  seed-sensitivity de-lucking, internal-vs-Kaggle validation reconciliation).

## Root
- `README.md` — this file.
- `requirements.txt` — Python dependencies, copied from the main repo (`scikit-learn`, `xgboost`,
  `optuna`, `torch`, etc.).
