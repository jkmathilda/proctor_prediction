# Plan: model3

**Feature**: `scripts/model3.py` — hierarchical shared-encoder architecture (target branch, then population branch)
**Phase**: Plan
**Created**: 2026-08-05
**Parent feature**: [proctor-prediction](proctor-prediction.plan.md)

**Scope note**: this is a from-scratch architecture exploration, not judged against `model1`/`model1i`/`model2i`
or required to beat them before being built. Prior work (cited below) informs *design decisions* — capacity,
where to branch, what not to repeat — it is not a gate this plan's acceptance criteria are conditioned on.

---

## 1. Objective

Build and evaluate a three-level hierarchical shared-encoder architecture:

```
Universal Features (28 cols, all 201 rows)
        |
        v
Lower shared encoder          <- fit on ALL rows, both targets' gradients flow through it
        |
   +----+----+
   v         v
MDD encoder  OWC encoder      <- each target gets its own nonlinear sub-encoder
   |             |
 +-+-+         +-+-+
 v   v         v   v
Fine Coarse   Fine Coarse     <- population split happens last, nested inside each target's encoder
head  head    head  head
```

Four leaf outputs (MDD-fine, MDD-coarse, OWC-fine, OWC-coarse), three levels of parameter sharing: full
sharing at the trunk, target-specific sharing in the middle, zero sharing at the leaves.

---

## 2. Why this ordering (target branches before population branches)

This was worked out in conversation before writing anything down; recorded here because it's load-bearing for
the design, not because it needs re-litigating later.

**Population split cost real data; target split doesn't.** MDD and OWC are two labels on the *same* 201 rows
— an MDD encoder and an OWC encoder each still see all 201 rows' features no matter where the branch sits,
just fit against a different label. Fine/coarse is a genuine row split (89 vs. 112) — anything downstream of
that branch point only ever sees its own subset from then on. `model_reasoning.ipynb`'s Regime A/B tests
found pooling fine+coarse rows measurably helps (e.g. MDD fine-group R² 0.744 → 0.786 under XGBoost when the
model also sees coarse rows). So the population split should sit as **late** as possible — at the leaves —
to keep as much of the network's depth benefiting from the full pooled 201 rows for as long as possible.

**Target split doesn't cost data, but needs real capacity relatively early, not late.** The cautionary
precedent is `docs/260804.md` Section 8's shared multi-output MLP (`hidden_layer_sizes=(128,64,32)`, shared
all the way to the output layer, diverging only at the final neuron) — R² 0.097 MDD / **-0.151** OWC, the
worst architecture tested on this dataset. The likely failure mode: with almost the entire network shared,
MDD's easier signal dominated the shared representation at OWC's expense. Giving each target a real encoder
of its own — not just a final linear readout — is the direct fix, which is what the "MDD encoder"/"OWC
encoder" boxes in the diagram are for.

Net: the axis that costs data (population) goes last; the axis that risks gradient interference if left
unstructured (target) gets dedicated capacity early.

---

## 3. Related prior work (context, not baseline)

| Prior work | Relevance |
|---|---|
| `docs/260804.md` Section 8 (shared multi-output MLP, chaining, separate MLPs) | The only prior test of MDD/OWC sharing on this dataset — informs why target encoders need real capacity, not a warning against sharing per se |
| `eda.ipynb`'s `SharedEncoderTwoHead` (curve-LSTM) | Same "shared encoder, target heads" shape as this plan's upper half, applied to curve-shaped PSD inputs instead of scalar `IMPUTED_FEATURES` — unvalidated, no R²/NMAE ever recorded |
| `model_reasoning.ipynb` Sections 1-4.1 | Fine/coarse near-orthogonal standardized-Ridge-coefficient finding (justifies zero sharing at the population leaves); pooling-helps finding under XGBoost (justifies delaying the population split); the flag-conditioning-adds-nothing finding is specific to tree splits and does **not** directly transfer here — this architecture's "conditioning" is structural (separate weight branches), not a concatenated scalar feature, so it isn't invalidated by that result |
| `coarse_specialist.py` / `fine_specialist.py` | Zero-pooling extreme; not used as a required baseline here, but their `IMPUTED_FEATURES`/imputation pipeline is reused directly (Section 5) |

---

## 4. Scope

- **Both targets, always** — MDD and OWC are both produced by one `model3` forward pass (this is different
  from every other model in this project, which fits them as fully separate models/scripts).
- **All four population×target leaves** — no leaf is optional or gated behind the others clearing some bar.
- **No comparison-based go/no-go.** Results get reported (Section 8) for documentation, same as every other
  model in this project's `docs/260804.md` — but nothing here is contingent on beating model1i, v15, or the
  specialists.

---

## 5. Data Pipeline

Reuse existing project infrastructure — no new preprocessing:

```
1. add_no_missing_features(df)                              # src/general_model_impute.py
2. add_imputed_features(df, fine_imputer, coarse_imputer)    # MICE, fold-isolated
3. X = df[IMPUTED_FEATURES].astype(float)                    # 28 universal columns
4. fine_flag = df["fine-grained"].astype(bool)                # ROUTING signal only, not concatenated as an
                                                               #   input feature (see note below)
5. Y = df[["proctor_mdd_g_cm3", "proctor_owc_pct"]].astype(float)
6. StandardScaler fit on X; separate StandardScaler fit on Y (both columns), matching eda.ipynb's
   shared_mlp_multioutput_oof convention — fold-isolated
```

**Why `fine_flag` isn't also concatenated as an input feature**: routing rows to entirely separate head
weights is a much stronger form of conditioning than handing a nonlinear model one more scalar column — the
architecture's branching *is* the conditioning mechanism. Concatenating the flag as well would be redundant
with no expected benefit (consistent with, though not identical to, `model_reasoning.ipynb` §4.1.3's finding
that the flag added nothing to a pooled XGBoost).

---

## 6. Capacity Budget

Starting point, sized well below the failed `(128,64,32)` MLP (~14,100 params) but larger than a flat
two-branch design, since this has three levels instead of two:

| Level | Shape | Params |
|---|---|---|
| Shared trunk | `28 → 24 → 16` (ReLU, dropout 0.2 between) | ~1,100 |
| MDD encoder / OWC encoder (each) | `16 → 12 → 8` (ReLU, dropout 0.2) | ~310 each, 620 total |
| 4 leaf heads (each) | `8 → 4 → 1` (ReLU, dropout 0.15) | ~40 each, 165 total |
| **Total** | | **~1,880** |

Still ~7x smaller than the architecture that failed catastrophically in Section 8. This is a starting point
for the Do phase to validate (train/val gap, not an external benchmark) — add capacity one level at a time if
underfitting, don't jump back to anything near `(128,64,32)`.

---

## 7. Training Procedure

| Setting | Value | Source/rationale |
|---|---|---|
| Loss | MSE, summed/averaged over both standardized targets | Matches `eda.ipynb`'s multi-output convention; revisit per-target loss weighting only if training curves show one target dominating (the Section 8 failure mode this architecture is designed to avoid) |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 | `eda.ipynb` precedent |
| Batch size | 16 | `eda.ipynb` precedent |
| Max epochs | 300, early stopping patience=30 | `eda.ipynb` precedent |
| CV | `RepeatedStratifiedKFold(n_splits=5, n_repeats=5)`, stratified on `fine-grained` | Ensures every fold's train/val split keeps both populations represented; simpler than full soil-composition stratification since the population axis is exactly what this architecture is structured around |
| Final model | Refit on all 201 rows, early-stopping slice used only for the patience mechanism | Matches every "final" model in this project |

---

## 8. Reporting (not gating)

Report, per leaf and combined, the same way `docs/260804.md` documents every other model — including if the
result is negative:

| Metric | Where |
|---|---|
| OOF R² / RMSE per leaf (MDD-fine, MDD-coarse, OWC-fine, OWC-coarse), mean ± std over repeated folds | `docs/260804.md` addendum |
| Combined NMAE (competition metric) | Same |
| Comparison row against model1i/v15/specialists | For context only — informative, not a pass/fail condition |

---

## 9. Physical Constraint

Reuse `helper_functions.calc_satline(w, Gs)`. Simpler than the previous MDD-only design: since `model3` now
natively produces both `mdd_pred` and `owc_pred` from one forward pass, the clip needs no cross-model
dependency on an external OWC estimate — `mdd = min(mdd_pred, calc_satline(owc_pred, rho_s) * 0.999)` using
this model's own OWC output.

---

## 10. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Every prior attempt at MDD/OWC sharing on this dataset (chaining, flat shared MLP) has underperformed independent modeling | Medium — this could too | Real, on record (§3), not hypothetical | This is exactly why target encoders get real dedicated capacity (§2) rather than a shared trunk all the way to output — the architecture is a direct response to the known failure mode, not a repeat of it; still needs empirical validation, not assumed fixed |
| Capacity at n=201, 3 levels deep | Medium — more levels than anything tried before at this n | Medium | Start at ~1,880 params (§6), watch train/val gap per fold, add capacity by one level at a time |
| One target's loss dominates training, starving the other (the Section 8 mechanism) | Medium | Medium, mitigated but not eliminated by dedicated encoders | Log per-target loss curves during training; move to weighted/alternating loss if one target's error stops improving while the other's does |
| Repeated-CV variance at n=89-112 per population leaf | Medium | Medium (same pattern flagged throughout this project) | Report mean ± std over the full repeated CV, not a point estimate; Kaggle score before calling anything a win |

---

## 11. Deliverables

| Deliverable | Description |
|---|---|
| `src/model3.py` | `SharedEncoderTargetPopulationHead` (`nn.Module`) + train/predict helpers |
| `scripts/model3.py` | CLI pipeline: data prep, repeated-CV training/eval, final refit, submission write |
| `docs/260804.md` addendum | OOF results per leaf + combined NMAE + Kaggle score, reported regardless of outcome |
| `submissions/submission_model3.csv` | Written after the final refit |

---

## 12. Next Steps

1. [ ] Design phase: finalize the exact `nn.Module` spec, file/CLI structure, dependencies
2. [ ] Implement `src/model3.py` + `scripts/model3.py` per Sections 5-9
3. [ ] Run repeated CV, log results to `docs/260804.md` regardless of how they compare to existing models
4. [ ] Submit to Kaggle, record the score

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-08-05 | Initial draft (fine/coarse-only, MDD-only, gated on beating model1i) | Claude |
| 0.2 | 2026-08-05 | Section 4.1 correction — Ridge proxy misleading for OWC; group-conditioning adds nothing to pooled XGBoost | Claude |
| 1.0 | 2026-08-05 | Full rewrite per direction from conversation — hierarchical target-then-population architecture, both targets natively, no baseline gate | Claude |
