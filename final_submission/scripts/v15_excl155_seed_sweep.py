"""
v15_excl155_seed_sweep.py
==========================
Same 100-seed sweep methodology as the "Seed sensitivity" section of
notebooks/analysis_paper.ipynb (which produced v15's 0.2439 100-seed-mean
Kaggle submission, de-lucking the 0.2383 single-seed result) -- but with
--exclude_ids 155, matching the v15_excl155 run that scored 0.2305 on
Kaggle (single-seed=42). This checks whether 0.2305 is itself a lucky
single-seed draw or a genuine improvement, the same way the original
100-seed sweep checked v15 plain.

Runs scripts/v15.py's main() end-to-end for N_SEEDS different --seed
values (own MICE imputation, own StratifiedShuffleSplit CV, own GP kernel
optimization, own GBT fit each time, id 155 excluded from training every
run), averages the per-row test predictions across seeds, and writes the
mean submission. Checkpoints every 10 completed seeds to --checkpoint_out
so a crash mid-sweep doesn't lose completed work.

GPU is not applicable here: v15.py's actual compute is scikit-learn's
GaussianProcessRegressor + HistGradientBoostingRegressor, both CPU-only
(torch is used only to save/load the .pt artifact, not for computation).
Instead this parallelizes across CPU processes (--n_workers, one process
per seed at a time) since the 100 seed runs are fully independent -- each
worker process pins its BLAS/OpenMP thread count to 1 (set before numpy
is imported) to avoid oversubscribing cores across workers.

Run:
    python scripts/v15_excl155_seed_sweep.py                  # all cores-ish
    python scripts/v15_excl155_seed_sweep.py --n_seeds 20 --n_workers 4
    python scripts/v15_excl155_seed_sweep.py --n_workers 1     # sequential
"""

import os

# Must happen before numpy/sklearn are imported anywhere in this process
# (main or worker) -- otherwise each of the N_WORKERS processes tries to
# grab all cores for BLAS and they fight each other.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import logging
import sys
import tempfile
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def _quiet_setup_logger(path):
    logger = logging.getLogger("gpr")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def make_args(seed, scratch_dir, exclude_ids):
    """Same defaults as v15.parse_args(), just built as a Namespace (matches
    the notebook's seed-sweep cell), with --exclude_ids overridden."""
    return Namespace(
        data_dir=str(repo_root / "data"),
        helpers_dir=str(repo_root / "src"),
        out=str(scratch_dir / f"submission_seed{seed}.csv"),
        model_out="",
        uncertainty_out="",
        report_dir=str(scratch_dir),
        log=str(scratch_dir / f"v15_seed{seed}.log"),
        folds=1,
        val_frac=0.2,
        val_strata=5,
        restarts=6,
        kernel="matern",
        ard=True,
        relevance_top=10,
        select=True,
        select_thresh=100.0,
        select_min=5,
        linear=False,
        ensemble=True,
        gbt_iter=400,
        gbt_lr=0.05,
        gbt_leaves=31,
        seed=seed,
        exclude_ids=exclude_ids,
    )


def _run_one_seed(payload):
    """Runs in a worker process. Re-imports scripts.v15 fresh (spawn start
    method), patches its logger/plotting the same way the notebook did, and
    returns (seed, owc_series, mdd_series) as plain (index, values) tuples
    so the result is cheaply picklable back to the main process.
    """
    seed, scratch_dir_str, exclude_ids = payload
    scratch_dir = Path(scratch_dir_str)

    from scripts import v15  # imported inside the worker on purpose

    v15.setup_logger = _quiet_setup_logger
    v15.plot_pred_vs_actual = lambda *a, **k: None

    t0 = time.time()
    out_df = v15.main(make_args(seed, scratch_dir, exclude_ids))
    elapsed = time.time() - t0

    df = out_df.set_index("id")[["proctor_owc_pct", "proctor_mdd_g_cm3"]]
    return seed, df["proctor_owc_pct"], df["proctor_mdd_g_cm3"], elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds", type=int, default=100)
    p.add_argument("--n_workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--exclude_ids", type=int, nargs="*", default=[155])
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_v15_excl155_mean.csv"))
    p.add_argument("--checkpoint_out", default=str(repo_root / "docs" / "v15_logs" / "v15_excl155_seed_sweep_checkpoint.pkl"))
    p.add_argument("--progress_log", default=str(repo_root / "logs" / "v15_excl155_seed_sweep_progress.log"))
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.progress_log) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    progress = open(args.progress_log, "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        progress.write(line + "\n")
        progress.flush()

    scratch_dir = Path(tempfile.mkdtemp(prefix="v15_excl155_seed_sweep_"))
    log(f"scratch dir: {scratch_dir}")
    log(f"n_seeds={args.n_seeds}  n_workers={args.n_workers}  exclude_ids={args.exclude_ids}")

    seed_owc = {}
    seed_mdd = {}
    t0 = time.time()
    n_done = 0

    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {
            ex.submit(_run_one_seed, (seed, str(scratch_dir), args.exclude_ids)): seed
            for seed in range(args.n_seeds)
        }
        for fut in as_completed(futures):
            seed, owc, mdd, elapsed = fut.result()
            seed_owc[seed] = owc
            seed_mdd[seed] = mdd
            n_done += 1
            total_elapsed = time.time() - t0
            log(f"seed {seed} done in {elapsed:.1f}s "
                f"({n_done}/{args.n_seeds} complete, total {total_elapsed/60:.1f}min)")

            if n_done % 10 == 0 or n_done == args.n_seeds:
                owc_wide = pd.concat(seed_owc, axis=1)
                mdd_wide = pd.concat(seed_mdd, axis=1)
                pd.to_pickle({"owc_wide": owc_wide, "mdd_wide": mdd_wide}, args.checkpoint_out)
                log(f"checkpoint saved ({n_done} seeds) -> {args.checkpoint_out}")

    owc_wide = pd.concat(seed_owc, axis=1)
    mdd_wide = pd.concat(seed_mdd, axis=1)

    out = pd.DataFrame({
        "id": owc_wide.index,
        "proctor_owc_pct": owc_wide.mean(axis=1).round(3).values,
        "proctor_mdd_g_cm3": mdd_wide.mean(axis=1).round(4).values,
    })
    out.to_csv(args.out, index=False)
    log(f"wrote {args.n_seeds}-seed mean submission ({len(out)} rows) -> {args.out}")
    log(f"total wall time: {(time.time()-t0)/60:.1f} min")
    progress.close()


if __name__ == "__main__":
    main()
