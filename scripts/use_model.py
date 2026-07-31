"""
use_model_v15.py
================
Load a saved .pt model from v15.py and use it -- no retraining.
(The v16 counterpart is use_model.py; the two differ because the artifacts
dicts differ. v15 stores a LIST of per-target GPs, a LIST of per-target GBTs,
and a LIST of per-target selected-feature indices, since v15 models MDD and OWC
separately. v16 stores one joint model plus a task covariance.)

Three things it does:

  1. PREDICT on a raw CSV (same schema as the competition's test.csv). The saved
     artifacts carry their own MICE imputer, scaler, per-target feature
     selection and blend weights, so the input needs no preprocessing at all --
     hand it the raw file exactly as the organizers ship it.

  2. SCORE, automatically, if the input CSV happens to contain the target
     columns. Point it at train.csv and it reports NMAE / MAE / RMSE / R^2.
     NOTE this is *in-sample* performance for a model fit on that same file --
     a check that the artifacts loaded correctly, NOT a generalization estimate.
     Use v15's own cross-validated NMAE for that.

  3. INSPECT the model (--inspect): the two learned kernels, blend weights, and
     which features each target's ARD kept -- including how much the two targets
     disagreed about feature relevance, which is the interesting part of a
     separate-models setup and the thing a joint model gives up.

Examples
--------
    # default paths, mirroring v15.py's own defaults
    python use_model_v15.py

    # explicit
    python use_model_v15.py --model models/v15_ensemble+mice_nohydgrad.pt \
                            --input data/test.csv --out submissions/from_saved.csv

    # include per-sample predictive uncertainty
    python use_model_v15.py --full

    # just look inside the model
    python use_model_v15.py --inspect

    # sanity-check against labelled data
    python use_model_v15.py --input data/train.csv --out /tmp/train_pred.csv

Requires v15.py and helper_functions.py to be importable/locatable.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# v15.py normally sits next to this script; make sure it is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v15  # noqa: E402


def find_helpers_dir(explicit, model_path):
    """Locate the directory holding helper_functions.py.

    Checked in order: --helpers_dir, this script's folder, its parent (the usual
    repo root), and the model file's parent chain.
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve().parent
    candidates += [here, here.parent]
    if model_path:
        p = Path(model_path).resolve().parent
        candidates += [p, p.parent]
    for c in candidates:
        if (Path(c) / "helper_functions.py").exists():
            return str(c)
    raise FileNotFoundError(
        "helper_functions.py not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + "\nPass --helpers_dir explicitly.")


def describe(artifacts, relevance_top=8):
    """Print what the saved model learned, per target."""
    targets = artifacts.get("targets", v15.TARGETS)
    feat = artifacts["feature_cols"]
    sels = artifacts.get("selected", [np.arange(len(feat))] * len(targets))
    gps = artifacts["gps"]
    gbts = artifacts.get("gbts", [None] * len(targets))
    w = np.asarray(artifacts.get("weights", np.ones(len(targets))))
    kernels = artifacts.get("kernels", ["n/a"] * len(targets))

    print("=" * 68)
    print("MODEL SUMMARY  (v15 -- separate model per target)")
    print("=" * 68)
    print(f"  targets            : {list(targets)}")
    print(f"  features at fit    : {len(feat)}")
    print(f"  ensemble           : "
          f"{'GP + gradient-boosted trees' if gbts[0] is not None else 'GP only'}")

    print("\n  blend weights (GP share vs trees):")
    for name, wi in zip(targets, w):
        print(f"      {name:<22s} GP {wi:.2f} / trees {1 - wi:.2f}")

    for j, name in enumerate(targets):
        sel = np.atleast_1d(sels[j])
        sel_names = [feat[i] for i in sel]
        print(f"\n  --- {name} ---")
        print(f"      features kept : {len(sel)}/{len(feat)} after ARD selection")
        print(f"      kernel        : {kernels[j]}")
        ls = v15.learned_length_scales(gps[j], len(sel))
        if ls is None:
            print("      (isotropic kernel -- no per-feature relevance)")
        else:
            print("      most informative features (shortest length-scale):")
            for i in np.argsort(ls)[:relevance_top]:
                print(f"          {sel_names[i]:<32s} {float(ls[i]):.3g}")

    # what the two targets disagreed about -- the payoff of modelling separately
    if len(targets) == 2:
        a, b = set(np.atleast_1d(sels[0]).tolist()), set(np.atleast_1d(sels[1]).tolist())
        only_a, only_b = sorted(a - b), sorted(b - a)
        print(f"\n  feature-selection overlap: {len(a & b)} shared, "
              f"{len(only_a)} only for {targets[0]}, {len(only_b)} only for {targets[1]}")
        if only_a:
            print(f"      only {targets[0]:<22s}: " + ", ".join(feat[i] for i in only_a))
        if only_b:
            print(f"      only {targets[1]:<22s}: " + ", ".join(feat[i] for i in only_b))
        if not only_a and not only_b:
            print("      (identical subsets -- the two targets wanted the same "
                  "features, so a joint model would give up little here)")
    print("=" * 68)


def main(args):
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"model not found at {args.model}\n"
            "Train one first with:  python v15.py --model_out <path>")

    try:
        artifacts = v15.load_model(args.model)
    except (AttributeError, ModuleNotFoundError) as e:
        # v15 models contain only sklearn objects, so unpickling never needs
        # anything custom. A missing-class error therefore means this file was
        # written by a different version -- v16 pickles ICMKernel / JointGP.
        if any(n in str(e) for n in ("JointGP", "ICMKernel", "TaskWhiteKernel",
                                     "IndependentGPs")):
            raise SystemExit(
                f"{args.model} is a v16 model (it contains a joint ICM Gaussian "
                f"Process).\nUse:  python use_model.py --model {args.model}") from e
        raise

    if not isinstance(artifacts, dict) or "feature_cols" not in artifacts:
        raise ValueError(f"{args.model} does not look like a v15 artifacts dict.")
    if "gps" not in artifacts:
        hint = ("This looks like a v16 model (one joint ICM model). "
                "Use use_model.py instead."
                if "gp" in artifacts else
                "Expected a 'gps' list of per-target Gaussian Processes.")
        raise ValueError(f"{args.model} is not a v15 model. {hint}")

    print(f"loaded {args.model}")
    if args.inspect or args.verbose:
        describe(artifacts, args.relevance_top)
    if args.inspect:
        return None

    H = v15.load_helpers(find_helpers_dir(args.helpers_dir, args.model))

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"input CSV not found at {args.input}")
    raw = pd.read_csv(args.input)
    print(f"read {args.input}  shape={raw.shape}")

    targets = artifacts.get("targets", v15.TARGETS)
    has_truth = all(t in raw.columns for t in targets)
    y_true = raw[targets].values.astype(float) if has_truth else None

    # the saved imputer/scaler/selection do all the preprocessing; we only need
    # the organizers' base feature engineering first
    fe = v15.base_feature_engineering(raw.copy(), H)

    # predict unclipped, then apply the Zero-Air-Voids guardrail ourselves so we
    # can report how often the physical bound actually bound
    means, stds = v15.gp_predict(artifacts, fe, H, clip=False)
    rho_s = fe["grain_density_g_cm3"].fillna(2.65).values
    owc = np.clip(means[:, 1], 0.0, None)
    zav = H.calc_satline(owc, rho_s) * 0.999
    n_clipped = int(np.sum(means[:, 0] > zav))
    mdd = np.minimum(means[:, 0], zav)

    out = pd.DataFrame({
        "id": raw["id"].values if "id" in raw.columns else np.arange(len(raw)),
        "proctor_owc_pct": np.round(owc, 3),
        "proctor_mdd_g_cm3": np.round(mdd, 4),
    })

    if args.full:
        out["owc_std"] = np.round(stds[:, 1], 3)
        out["mdd_std"] = np.round(stds[:, 0], 4)
        # No cross-target correlation column here: v15 fits the two GPs
        # independently, so its posterior covariance between MDD and OWC is
        # zero by construction. Reporting it would be meaningless.

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\npredicted {len(out)} samples")
    print(f"  OWC  mean {owc.mean():7.3f}  range [{owc.min():.3f}, {owc.max():.3f}]"
          f"   mean σ {stds[:, 1].mean():.3f}")
    print(f"  MDD  mean {mdd.mean():7.4f}  range [{mdd.min():.4f}, {mdd.max():.4f}]"
          f"   mean σ {stds[:, 0].mean():.4f}")
    print(f"  saturation-line clip applied to {n_clipped}/{len(out)} samples")

    if has_truth:
        pred = np.column_stack([mdd, owc])
        mae, rmse, r2 = v15.regression_metrics(y_true, pred)
        print("\n  input contained the true targets -- scoring against them.")
        print("  NOTE: if this is the file the model was trained on, these are")
        print("        in-sample numbers, not a generalization estimate.")
        print(f"  {'target':<24s}{'MAE':>10s}{'RMSE':>10s}{'R2':>8s}")
        for i, name in enumerate(targets):
            print(f"  {name:<24s}{mae[i]:10.4f}{rmse[i]:10.4f}{r2[i]:8.3f}")
        print(f"  NMAE (competition metric): {v15.nmae(y_true, pred):.4f}")
        # residual coupling: how much MDD/OWC error is still correlated. Large
        # magnitude here is the signal that a joint model (v16) may pay off.
        res = y_true - pred
        print(f"  residual corr(MDD, OWC): "
              f"{float(np.corrcoef(res[:, 0], res[:, 1])[0, 1]):+.3f}  "
              f"(structure two independent models cannot exploit)")

    print(f"\nwrote -> {args.out}")
    return out


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Run a saved v15 .pt model on a raw CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model",
                   default=str(repo_root / "models" / "v15_ensemble+mice_nohydgrad.pt"),
                   help="saved .pt model from v15.py")
    p.add_argument("--input", default=str(repo_root / "data" / "test.csv"),
                   help="raw CSV to predict on (unpreprocessed)")
    p.add_argument("--out",
                   default=str(repo_root / "submissions" / "from_saved_model_v15.csv"),
                   help="where to write predictions")
    p.add_argument("--helpers_dir", default=None,
                   help="folder containing helper_functions.py (auto-detected if omitted)")
    p.add_argument("--full", action="store_true",
                   help="also write per-sample predictive standard deviations")
    p.add_argument("--inspect", action="store_true",
                   help="print what the model learned and exit (no input CSV needed)")
    p.add_argument("--verbose", action="store_true",
                   help="print the model summary before predicting")
    p.add_argument("--relevance_top", type=int, default=8,
                   help="how many top ARD features to list per target")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())