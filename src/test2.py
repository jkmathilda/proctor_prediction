import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

class TinyMLPRegressor(nn.Module):
    """Small MLP for low-sample tabular regression."""

    def __init__(
        self,
        n_features: int,
        hidden_1: int = 12,
        hidden_2: int = 6,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_1),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),

            nn.Linear(hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

def count_trainable_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_1: int = 12,
    hidden_2: int = 6,
    dropout: float = 0.05,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 1000,
    patience: int = 80,
    min_delta: float = 1e-5,
    random_state: int = 42,
    device: str | None = None,
) -> dict:
    set_seed(random_state)

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # Fit preprocessing on training fold only.
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    feature_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(
        X_train_imputed
    )
    X_val_scaled = feature_scaler.transform(
        X_val_imputed
    )

    # Standardize target using training fold only.
    target_scaler = StandardScaler()

    y_train_scaled = target_scaler.fit_transform(
        y_train.reshape(-1, 1)
    ).ravel()

    y_val_scaled = target_scaler.transform(
        y_val.reshape(-1, 1)
    ).ravel()

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train_scaled,
        dtype=torch.float32,
    )

    X_val_tensor = torch.tensor(
        X_val_scaled,
        dtype=torch.float32,
        device=device,
    )

    y_val_tensor = torch.tensor(
        y_val_scaled,
        dtype=torch.float32,
        device=device,
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=False,
    )

    model = TinyMLPRegressor(
        n_features=X_train_scaled.shape[1],
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loss_fn = nn.MSELoss()

    best_state = None
    best_val_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(max_epochs):
        model.train()

        batch_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = loss_fn(
                predictions,
                y_batch,
            )

            loss.backward()

            # Useful for stability on very small datasets.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            batch_losses.append(loss.item())

        mean_train_loss = float(
            np.mean(batch_losses)
        )

        model.eval()

        with torch.no_grad():
            val_predictions = model(X_val_tensor)

            val_loss = loss_fn(
                val_predictions,
                y_val_tensor,
            ).item()

        history["train_loss"].append(
            mean_train_loss
        )
        history["val_loss"].append(
            val_loss
        )

        improved = (
            val_loss
            < best_val_loss - min_delta
        )

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            "Training failed to produce a valid model state."
        )

    model.load_state_dict(best_state)
    model.eval()

    X_train_device = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        train_pred_scaled = (
            model(X_train_device)
            .cpu()
            .numpy()
        )

        val_pred_scaled = (
            model(X_val_tensor)
            .cpu()
            .numpy()
        )

    train_predictions = target_scaler.inverse_transform(
        train_pred_scaled.reshape(-1, 1)
    ).ravel()

    val_predictions = target_scaler.inverse_transform(
        val_pred_scaled.reshape(-1, 1)
    ).ravel()

    return {
        "model": model,
        "imputer": imputer,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "train_predictions": train_predictions,
        "val_predictions": val_predictions,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "n_parameters": count_trainable_parameters(
            model
        ),
    }

def evaluate_tiny_mlp_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    hidden_1: int = 12,
    hidden_2: int = 6,
    dropout: float = 0.05,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 1000,
    patience: int = 80,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = X.copy()
    y = pd.to_numeric(
        y,
        errors="coerce",
    ).copy()

    valid_mask = (
        y.notna()
        & np.isfinite(y)
    )

    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_predictions = np.full(
        len(y),
        np.nan,
        dtype=float,
    )

    train_predictions_by_fold = []
    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X),
        start=1,
    ):
        X_train = X.iloc[train_idx].to_numpy(
            dtype=float
        )

        X_val = X.iloc[val_idx].to_numpy(
            dtype=float
        )

        y_train = y.iloc[train_idx].to_numpy(
            dtype=float
        )

        y_val = y.iloc[val_idx].to_numpy(
            dtype=float
        )

        result = train_one_fold(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state + fold,
        )

        val_predictions = result[
            "val_predictions"
        ]

        train_predictions = result[
            "train_predictions"
        ]

        oof_predictions[val_idx] = val_predictions

        train_predictions_by_fold.append(
            {
                "fold": fold,
                "r2": r2_score(
                    y_train,
                    train_predictions,
                ),
                "rmse": mean_squared_error(
                    y_train,
                    train_predictions,
                ) ** 0.5,
                "mae": mean_absolute_error(
                    y_train,
                    train_predictions,
                ),
            }
        )

        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_parameters": result[
                    "n_parameters"
                ],
                "best_epoch": result[
                    "best_epoch"
                ],
                "train_r2": r2_score(
                    y_train,
                    train_predictions,
                ),
                "val_r2": r2_score(
                    y_val,
                    val_predictions,
                ),
                "train_rmse": (
                    mean_squared_error(
                        y_train,
                        train_predictions,
                    ) ** 0.5
                ),
                "val_rmse": (
                    mean_squared_error(
                        y_val,
                        val_predictions,
                    ) ** 0.5
                ),
                "train_mae": mean_absolute_error(
                    y_train,
                    train_predictions,
                ),
                "val_mae": mean_absolute_error(
                    y_val,
                    val_predictions,
                ),
            }
        )

    if np.isnan(oof_predictions).any():
        raise RuntimeError(
            "Some observations did not receive an "
            "OOF prediction."
        )

    oof_results = pd.DataFrame(
        {
            "observed": y,
            "tiny_mlp_oof_prediction": (
                oof_predictions
            ),
            "residual": (
                y.to_numpy()
                - oof_predictions
            ),
        }
    )

    overall_metrics = {
        "overall_oof_r2": r2_score(
            y,
            oof_predictions,
        ),
        "overall_oof_rmse": (
            mean_squared_error(
                y,
                oof_predictions,
            ) ** 0.5
        ),
        "overall_oof_mae": mean_absolute_error(
            y,
            oof_predictions,
        ),
    }

    print("Overall OOF metrics")
    print(
        f"R²:   "
        f"{overall_metrics['overall_oof_r2']:.4f}"
    )
    print(
        f"RMSE: "
        f"{overall_metrics['overall_oof_rmse']:.4f}"
    )
    print(
        f"MAE:  "
        f"{overall_metrics['overall_oof_mae']:.4f}"
    )

    fold_results = pd.DataFrame(fold_rows)

    return oof_results, fold_results

