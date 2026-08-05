# Design: model3

**Feature**: `scripts/model3.py` — hierarchical shared-encoder architecture
**Plan**: [model3.plan.md](../../01-plan/features/model3.plan.md)
**Phase**: Design
**Created**: 2026-08-05

---

## 1. Resolving the Plan's Open Questions

### 1.1 Branch ordering (Plan §2)

**Decision: target branches before population branches**, per the two reasons already recorded in Plan §2
(population split costs real data and should be delayed; target split doesn't cost data but needs early
dedicated capacity to avoid Section 8's gradient-domination failure mode). Not re-litigated here.

### 1.2 Does `fine-grained` also get concatenated as an input feature?

**Decision: no.** Routing to separate head weights is a structural, stronger form of conditioning than a
concatenated scalar column. See Plan §5's note.

### 1.3 Loss weighting between MDD and OWC

**Decision: start with unweighted summed MSE on standardized targets** (`(mdd_loss + owc_loss)`, both
targets already on comparable scale after `StandardScaler`). **Not fixed** — Section 8's flat-MLP failure is
exactly the scenario where equal nominal weighting let one target dominate in practice (through gradient
magnitude, not the loss formula's stated weights), so training must be watched for this, not assumed safe
because the loss function looks balanced on paper. If per-target loss curves diverge (one plateaus early while
the other keeps improving, or one dominates gradient norm), escalate to explicit loss weighting or an
alternating/uncertainty-weighted scheme — flagged as a concrete Do-phase checkpoint, not a hypothetical.

### 1.4 Capacity budget

**Decision**: per Plan §6's table (~1,880 params total: trunk `28→24→16`, two target encoders `16→12→8`
each, four leaf heads `8→4→1` each). Starting point for the Do phase to validate against its own train/val
gap — not benchmarked against another model's score.

---

## 2. Architecture Specification

### 2.1 Module (PyTorch, `nn.Module`)

```python
class SharedEncoderTargetPopulationHead(nn.Module):
    """Shared trunk -> per-target encoder -> per-population head, nested.

    Naming follows eda.ipynb's SharedEncoderTwoHead / SharedEncoderScalarTwoHead convention,
    extended one level: target-split in the middle, population-split at the leaves.
    """

    def __init__(
        self,
        n_features,
        trunk_units=(24, 16),
        target_units=(12, 8),
        head_units=(4,),
        trunk_dropout=0.2,
        head_dropout=0.15,
    ):
        super().__init__()

        def make_mlp(in_sz, units, dropout, final_relu=True):
            layers, sz = [], in_sz
            for i, u in enumerate(units):
                layers += [nn.Linear(sz, u), nn.ReLU()]
                if dropout > 0 and (final_relu or i < len(units) - 1):
                    layers.append(nn.Dropout(dropout))
                sz = u
            return nn.Sequential(*layers), sz

        self.trunk, trunk_out = make_mlp(n_features, trunk_units, trunk_dropout)
        self.mdd_encoder, mdd_out = make_mlp(trunk_out, target_units, trunk_dropout)
        self.owc_encoder, owc_out = make_mlp(trunk_out, target_units, trunk_dropout)

        def make_head(in_sz):
            body, sz = make_mlp(in_sz, head_units, head_dropout)
            return nn.Sequential(body, nn.Linear(sz, 1))

        self.mdd_fine_head = make_head(mdd_out)
        self.mdd_coarse_head = make_head(mdd_out)
        self.owc_fine_head = make_head(owc_out)
        self.owc_coarse_head = make_head(owc_out)

    def forward(self, x, fine_flag):
        h = self.trunk(x)
        h_mdd = self.mdd_encoder(h)
        h_owc = self.owc_encoder(h)

        is_fine = fine_flag.bool().unsqueeze(-1)
        mdd_pred = torch.where(is_fine, self.mdd_fine_head(h_mdd), self.mdd_coarse_head(h_mdd))
        owc_pred = torch.where(is_fine, self.owc_fine_head(h_owc), self.owc_coarse_head(h_owc))
        return torch.cat([mdd_pred, owc_pred], dim=1)  # (batch, 2) -- [mdd, owc]
```

### 2.2 Why `torch.where` at each leaf, not a hard row-split anywhere

Every row passes through the full trunk and both target encoders regardless of population — the population
split only ever affects which head's weights receive that row's gradient. This is what makes the trunk and
target encoders see the full pooled 201 rows (Plan §2's core rationale) while the leaf heads still train on
disjoint population-specific data, matching the diagram exactly.

---

## 3. Data Pipeline

```
1. add_no_missing_features(df)                              # src/general_model_impute.py
2. add_imputed_features(df, fine_imputer, coarse_imputer)    # MICE, fold-isolated
3. X = df[IMPUTED_FEATURES].astype(float)                    # 28 universal columns
4. fine_flag = df["fine-grained"].astype(bool).to_numpy()    # routing only (Decision 1.2)
5. Y = df[["proctor_mdd_g_cm3", "proctor_owc_pct"]].astype(float)
6. StandardScaler fit on X (fold-isolated); StandardScaler fit on Y (fold-isolated, both columns together,
   matching eda.ipynb's shared_mlp_multioutput_oof)
```

No new imputer or feature set — identical to `coarse_specialist.py`/`fine_specialist.py`'s `IMPUTED_FEATURES`
subset for `X`.

---

## 4. Training Procedure

| Setting | Value | Source/rationale |
|---|---|---|
| Loss | `MSELoss()` on concatenated `[mdd, owc]` standardized output vs. standardized `Y` | Decision 1.3 |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 | `eda.ipynb` precedent |
| Batch size | 16 | `eda.ipynb` precedent |
| Max epochs | 300, early stopping patience=30 on val-fold loss | `eda.ipynb` precedent |
| CV | `RepeatedStratifiedKFold(n_splits=5, n_repeats=5)` on `fine-grained` | Plan §7 |
| Per-target loss logging | Track `mdd_loss`/`owc_loss` separately every epoch even though the optimizer sees their sum | Decision 1.3's checkpoint — needed to detect one target dominating before it shows up as a bad final metric |
| Final model | Refit on all 201 rows; early-stopping slice used only for the patience mechanism | Matches every "final" model in this project |

---

## 5. Evaluation / Reporting Harness

Per Plan §8 — no pass/fail gate. Compute OOF predictions for all four leaves across the repeated CV, then:

```python
# per-leaf: mask by (fine_flag == target_group) within each target's OOF column
mdd_fine_r2   = r2_score(y_mdd[fine_mask],  oof_mdd[fine_mask])
mdd_coarse_r2 = r2_score(y_mdd[~fine_mask], oof_mdd[~fine_mask])
owc_fine_r2   = r2_score(y_owc[fine_mask],  oof_owc[fine_mask])
owc_coarse_r2 = r2_score(y_owc[~fine_mask], oof_owc[~fine_mask])
combined_nmae = calculate_nmae(...)   # reuse helper_functions.calculate_nmae, fixed global IQR (v14.design.md §4 precedent)
```

Log all four leaf R²s (mean ± std over repeats) and the combined NMAE to `docs/260804.md`, plus a context-only
row comparing against model1i/v15/specialists — informative, not a condition for shipping.

---

## 6. Physical Constraint

```python
owc_pred = np.clip(model3_output[:, 1], 0, None)
mdd_pred = np.minimum(model3_output[:, 0], calc_satline(owc_pred, rho_s) * 0.999)
```

Reuse `helper_functions.calc_satline`. No cross-model dependency — both targets come from this model's own
forward pass (simpler than the earlier MDD-only design, which needed an external OWC estimate to clip against).

---

## 7. File / CLI Structure

```
python scripts/model3.py --data_dir ./data --out ./submissions/submission_model3.csv
```

| Arg | Default | Purpose |
|---|---|---|
| `--data_dir` | `./data` | train.csv/test.csv location |
| `--out` | `./submissions/submission_model3.csv` | final predictions |
| `--repeats` | 5 | `RepeatedStratifiedKFold` repeat count |
| `--folds` | 5 | `RepeatedStratifiedKFold` fold count |
| `--seed` | 42 | reproducibility |
| `--trunk_units` | `24,16` | §1.4 |
| `--target_units` | `12,8` | §1.4 |
| `--head_units` | `4` | §1.4 |
| `--trunk_dropout` | 0.2 | §1.4 |
| `--head_dropout` | 0.15 | §1.4 |
| `--weight_decay` | 1e-4 | §4 |

New module: `src/model3.py` holding `SharedEncoderTargetPopulationHead` plus train/predict helpers, following
the `src/` (reusable model code) vs. `scripts/` (runnable CLI) split already used by
`coarse_specialist.py`/`fine_specialist.py`.

---

## 8. Dependencies

`torch` — already a project dependency (`eda.ipynb`). No `requirements.txt`/`pyproject.toml` change needed.

---

## 9. Acceptance Criteria

- [ ] `SharedEncoderTargetPopulationHead` implemented exactly per §2.1's structure (trunk → target encoders →
      population heads, `torch.where` routing at each leaf)
- [ ] Fold-isolated preprocessing (imputation, scaling) — no leakage between train/val fold or across repeats
- [ ] Per-target loss curves logged during training (Decision 1.3) — checked for one target dominating before
      results are reported
- [ ] Repeated CV (`RepeatedStratifiedKFold(5, 5)` on `fine-grained`), mean ± std reported for all four leaves
      and combined NMAE — no single-split point estimates (`docs/260804.md` Section 3 precedent)
- [ ] Saturation clip applied via `helper_functions.calc_satline`, using this model's own OWC output (§6)
- [ ] `docs/260804.md` updated with results — reported regardless of whether they beat existing models
- [ ] Kaggle score recorded

---

## 10. PDCA Status

Plan: complete (`docs/01-plan/features/model3.plan.md`, v1.0). Design: this document. Next: Do phase —
implement `src/model3.py` + `scripts/model3.py` per Sections 2-7.
