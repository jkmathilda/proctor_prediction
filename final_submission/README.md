Files behind this project's two Kaggle-submitted models. See `docs/README.md` and
`docs/model_description.md` ("Final Submissions — What We Actually Submitted") for the full writeup.

## Submission #1 — v15 (100-seed mean), Kaggle 0.2439
- `submissions/submission_v15_100seed_mean.csv` — the exact file submitted (copied from
  `submissions/v15_mean.csv` in the main repo).
- `scripts/v15.py` — the model. Run with `--seed <n>` (default 45) to reproduce one seed; the honest
  submission averages predictions from 100 seed runs. The averaging notebook
  (`notebooks/analysis_paper.ipynb`, "Seed sensitivity" cells) referenced in `docs/model_description.md`
  was not present in the repo at copy time — `scripts/v15_excl155_seed_sweep.py` is included as the closest
  in-repo reference for the seed-sweep methodology (it sweeps the `--exclude_ids 155` variant, not plain
  v15, but shows the sweep/aggregate pattern used).

## Submission #2 — model1i (precalibration), Kaggle 0.2465
- `submissions/submission_model1i.csv` — the exact file submitted.
- `scripts/model1i.py` — reproduce with defaults (`--optuna_trials 50`, `--exclude_ids 155`,
  `--split_kind kfold`, no OWC calibration — all defaults already).
- `models/model1i.joblib` — cached fitted model from the submitted run.

## Shared dependencies
- `src/helper_functions.py`, `src/general_model_impute.py` — imported by both scripts.
- `data/train.csv`, `data/test.csv` — only data files either script reads.
- `requirements.txt` — from the main repo.

## Docs
- `docs/README.md`, `docs/model_description.md`, `docs/260804.md` — copied from the repo root / `docs/`
  for context (results table, architecture writeups, experimental log).
