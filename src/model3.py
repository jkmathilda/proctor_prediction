"""
model3.py
=========
Hierarchical shared-encoder architecture -- see docs/01-plan/features/model3.plan.md
and docs/02-design/features/model3.design.md for the full design rationale. Summary:

    Universal Features (28 cols, IMPUTED_FEATURES, all rows)
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
    Fine Coarse   Fine Coarse     <- population split happens last, nested inside each
    head  head    head  head        target's encoder

Target (MDD/OWC) branches before population (fine/coarse) branches -- deliberately, not
arbitrarily: MDD and OWC are two labels on the SAME rows, so splitting on target costs no
data density wherever it happens, but the flat shared MLP that shares almost everything
down to the output layer already failed badly on this dataset (docs/260804.md Section 8:
R^2 0.097 MDD / -0.151 OWC), so target-specific capacity is given early to avoid one
target's gradient dominating the shared representation. Fine/coarse IS a genuine row split
(89 vs 112 of 201), and model_reasoning.ipynb's Regime A/B tests found pooling those rows
measurably helps -- so that split is delayed to the leaves, keeping as much of the network
as possible benefiting from the full pooled data for as long as possible.

This is a from-scratch architecture exploration (model3.plan.md's scope note) -- it is not
benchmarked against model1/model1i/model2i as a gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.general_model_impute import IMPUTED_FEATURES, add_imputed_features

# Same 28 universal columns coarse_specialist.py/fine_specialist.py use -- kept as its own
# name (rather than importing IMPUTED_FEATURES directly everywhere) so callers don't need
# to know it's currently identical, matching the convention in those two modules.
MODEL3_FEATURES: list[str] = list(IMPUTED_FEATURES)

TARGET_COLS: list[str] = ["proctor_mdd_g_cm3", "proctor_owc_pct"]


def prepare_model3_features(
    df: pd.DataFrame,
    fine_imputer: Any | None = None,
    coarse_imputer: Any | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Any, Any]:
    """
    Produce every column MODEL3_FEATURES needs, plus the `fine-grained` routing column.

    Thin wrapper around general_model_impute.add_imputed_features -- model3 needs no
    feature engineering beyond what every other model in this project already uses (see
    module docstring: routing to separate head weights is the conditioning mechanism, not
    a concatenated feature, so no Atterberg/PI enrichment is needed here either).

    Pass previously-fitted fine_imputer/coarse_imputer (as returned from an earlier call on
    training data) to transform new data (e.g. test.csv) without refitting/leaking.
    """
    return add_imputed_features(
        df, fine_imputer=fine_imputer, coarse_imputer=coarse_imputer,
        random_state=random_state,
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SharedEncoderTargetPopulationHead(nn.Module):
    """Shared trunk -> per-target encoder -> per-population head, nested.

    Naming follows eda.ipynb's SharedEncoderTwoHead / SharedEncoderScalarTwoHead
    convention, extended one level: target-split in the middle, population-split at the
    leaves (see module docstring for why that ordering, not the reverse).
    """

    def __init__(
        self,
        n_features: int,
        trunk_units: tuple[int, ...] = (24, 16),
        target_units: tuple[int, ...] = (12, 8),
        head_units: tuple[int, ...] = (4,),
        trunk_dropout: float = 0.2,
        head_dropout: float = 0.15,
    ):
        super().__init__()

        def make_mlp(in_sz, units, dropout):
            layers, sz = [], in_sz
            for u in units:
                layers += [nn.Linear(sz, u), nn.ReLU()]
                if dropout > 0:
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

    def forward(self, x: torch.Tensor, fine_flag: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_features). fine_flag: (batch,), 1.0 for fine-grained rows.

        Every row passes through the full trunk and BOTH target encoders regardless of
        population -- torch.where only selects which head's weights receive that row's
        gradient. This is what keeps the trunk/target encoders seeing the full pooled
        batch while the leaf heads still train on disjoint population-specific rows.
        """
        h = self.trunk(x)
        h_mdd = self.mdd_encoder(h)
        h_owc = self.owc_encoder(h)

        is_fine = fine_flag.bool().unsqueeze(-1)
        mdd_pred = torch.where(is_fine, self.mdd_fine_head(h_mdd), self.mdd_coarse_head(h_mdd))
        owc_pred = torch.where(is_fine, self.owc_fine_head(h_owc), self.owc_coarse_head(h_owc))
        return torch.cat([mdd_pred, owc_pred], dim=1)  # (batch, 2) -- [mdd, owc]


def make_default_model3(n_features: int = len(MODEL3_FEATURES), **kwargs: Any) -> SharedEncoderTargetPopulationHead:
    """Factory matching coarse_specialist.py/fine_specialist.py's make_default_*_model
    naming convention -- every SharedEncoderTargetPopulationHead kwarg can be passed
    through."""
    return SharedEncoderTargetPopulationHead(n_features=n_features, **kwargs)


def train_model3(
    model: SharedEncoderTargetPopulationHead,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 300,
    patience: int = 30,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Train with early stopping on val_loader's combined loss.

    Also tracks per-target (MDD/OWC) validation loss every epoch even though the
    optimizer only ever sees their sum -- design.md Decision 1.3's checkpoint against the
    docs/260804.md Section 8 failure mode (one target's gradient dominating a shared
    representation while the summed loss still looks fine). val_loader should be an inner
    slice of the current fold's TRAINING rows only -- never the outer CV's held-out test
    fold, to avoid leaking test-fold information into the early-stopping decision.
    """
    device = device or get_device()
    model.to(device)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "val_mdd_loss": [], "val_owc_loss": [],
    }
    best_val, best_state, counter = float("inf"), None, 0

    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, fb, yb in train_loader:
            xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, fb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        history["train_loss"].append(train_loss / len(train_loader))

        if val_loader is None:
            continue

        model.eval()
        val_loss = val_mdd_loss = val_owc_loss = 0.0
        with torch.no_grad():
            for xb, fb, yb in val_loader:
                xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
                pred = model(xb, fb)
                val_loss += criterion(pred, yb).item()
                val_mdd_loss += criterion(pred[:, 0], yb[:, 0]).item()
                val_owc_loss += criterion(pred[:, 1], yb[:, 1]).item()
        n_val_batches = len(val_loader)
        history["val_loss"].append(val_loss / n_val_batches)
        history["val_mdd_loss"].append(val_mdd_loss / n_val_batches)
        history["val_owc_loss"].append(val_owc_loss / n_val_batches)

        if val_loss < best_val:
            best_val, counter = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def predict_model3(
    model: SharedEncoderTargetPopulationHead,
    X: torch.Tensor,
    fine_flag: torch.Tensor,
    device: torch.device | None = None,
) -> np.ndarray:
    """Returns standardized-scale predictions, shape (n, 2) -- [mdd, owc]. Caller is
    responsible for inverse-transforming with the same StandardScaler fit on the training
    fold's targets."""
    device = device or get_device()
    model.eval()
    model.to(device)
    with torch.no_grad():
        pred = model(X.to(device), fine_flag.to(device))
    return pred.cpu().numpy()
