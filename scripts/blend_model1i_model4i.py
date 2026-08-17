"""
blend_model1i_model4i.py
=========================
Prediction-level blend of model1i (Ridge + GPR + XGBoost on MICE-imputed
features) and model4i (XGBoost + Tiny FFN on hand-built PSD/gradation
features) -- two structurally different pipelines, blended per-target with
weights fit via leave-one-seed-out nested CV (5 seeds x 10-fold
StratifiedKFold, model1i Optuna-tuned each time):

    MDD: 0.88 * model1i + 0.12 * model4i   (honest nested NMAE 0.1937 --
         barely different from model1i alone at 0.1938; model4i adds
         essentially nothing on MDD)
    OWC: 0.31 * model1i + 0.69 * model4i   (honest nested NMAE 0.2731 vs
         model1i alone 0.2773 and model4i alone 0.2742 -- a real,
         consistently-replicated ~1.5% relative gain, weight landed in the
         0.29-0.39 range across every held-out seed)

CAVEAT: the weights above were fit against model1i's RAW (uncalibrated) OWC
OOF predictions. This script blends them against model1i's CURRENT
(isotonic-calibrated) OWC predictions instead, since that's the actual
deployed model1i output -- the weight itself was not re-validated against
the calibrated OOF, so treat the OWC blend weight as a reasonable carry-over
rather than a freshly re-optimized number.

ALSO UNVALIDATED ON THE ACTUAL LEADERBOARD: this blend's gain was only ever
checked against our own nested OOF CV. model4i itself already underperformed
model1i on Kaggle, so whether the OWC blend gain here survives contact with
the real test set is genuinely unknown until submitted.

Run:
    python scripts/blend_model1i_model4i.py
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

MDD_WEIGHT_MODEL1I = 0.88
MDD_WEIGHT_MODEL4I = 1.0 - MDD_WEIGHT_MODEL1I

OWC_WEIGHT_MODEL1I = 0.31
OWC_WEIGHT_MODEL4I = 1.0 - OWC_WEIGHT_MODEL1I


def setup_logger(path):
    logger = logging.getLogger("blend_model1i_model4i")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def main(args):
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logger = setup_logger(args.log)
    logger.info("blend_model1i_model4i -- prediction-level blend, weights from leave-one-seed-out nested CV")
    logger.info(
        "MDD: %.2f model1i + %.2f model4i   OWC: %.2f model1i + %.2f model4i",
        MDD_WEIGHT_MODEL1I, MDD_WEIGHT_MODEL4I, OWC_WEIGHT_MODEL1I, OWC_WEIGHT_MODEL4I,
    )

    m1i = pd.read_csv(args.model1i_submission)
    m4i = pd.read_csv(args.model4i_submission)
    logger.info("loaded model1i submission: %s (%d rows)", args.model1i_submission, len(m1i))
    logger.info("loaded model4i submission: %s (%d rows)", args.model4i_submission, len(m4i))

    if len(m1i) != len(m4i):
        raise ValueError(f"Row count mismatch: model1i={len(m1i)} model4i={len(m4i)}")

    m1i = m1i.sort_values("id").reset_index(drop=True)
    m4i = m4i.sort_values("id").reset_index(drop=True)

    if not (m1i["id"].to_numpy() == m4i["id"].to_numpy()).all():
        raise ValueError("model1i and model4i submissions do not cover the same ids.")

    blended = pd.DataFrame({
        "id": m1i["id"],
        "proctor_mdd_g_cm3": (
            MDD_WEIGHT_MODEL1I * m1i["proctor_mdd_g_cm3"]
            + MDD_WEIGHT_MODEL4I * m4i["proctor_mdd_g_cm3"]
        ),
        "proctor_owc_pct": (
            OWC_WEIGHT_MODEL1I * m1i["proctor_owc_pct"]
            + OWC_WEIGHT_MODEL4I * m4i["proctor_owc_pct"]
        ),
    })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    blended.to_csv(args.out, index=False)
    logger.info("Wrote %d blended predictions -> %s", len(blended), args.out)
    logger.info("Run log -> %s", args.log)

    return blended


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--model1i_submission", default=str(repo_root / "submissions" / "submission_model1i.csv"))
    p.add_argument("--model4i_submission", default=str(repo_root / "submissions" / "submission_model4i.csv"))
    p.add_argument("--out", default=str(repo_root / "submissions" / "submission_blend_model1i_model4i.csv"))
    p.add_argument("--log", default=str(repo_root / "logs" / "blend_model1i_model4i_run.log"))
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
