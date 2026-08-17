# ============================================================
# Validated architecture:
#
#   MDD:
#       0.650 * XGBoost
#       0.350 * Tiny FFN
#
#   OWC:
#       0.656 * XGBoost
#       0.344 * Tiny FFN
#
# Validation:
#   Nested-CV final NMAE = 0.247181 (StratifiedKFold folds=5 val_strata=5,
#   stratified on MDD, seed=42 -- see run_oof_cv()/logs/model4i_run.log)
#
# IMPORTANT:
#   - ID 155 is removed from TRAINING ONLY.
#   - Test rows are NEVER removed.
#   - Preprocessing is fit using training data only.
#   - XGB and NN are trained on all retained training data.
# ============================================================


# ============================================================
# 0. Native thread protection
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ============================================================
# 1. Imports
# ============================================================

import logging
from pathlib import Path
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
from sklearn.preprocessing import StandardScaler

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from xgboost import XGBRegressor

import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

from src.general_model_impute import make_stratified_kfold_splits


torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# 2. Configuration
# ============================================================

RANDOM_STATE = 42

EXCLUDED_ID = 155

FINE_THRESHOLD = 15.0


# ------------------------------------------------------------
# Nested-CV validation scheme
#
# Same StratifiedKFold-on-MDD-quantiles scheme as model1i.py's
# make_stratified_kfold_splits, so the two pipelines' OOF NMAE
# figures are computed the same way and are comparable.
# ------------------------------------------------------------

CV_FOLDS = 5
CV_VAL_STRATA = 5


# ------------------------------------------------------------
# Validated blend weights from nested CV
# ------------------------------------------------------------

MDD_XGB_WEIGHT = 0.650
MDD_NN_WEIGHT = 1.0 - MDD_XGB_WEIGHT

OWC_XGB_WEIGHT = 0.656
OWC_NN_WEIGHT = 1.0 - OWC_XGB_WEIGHT


# ------------------------------------------------------------
# Neural network architecture
# ------------------------------------------------------------

HIDDEN_1 = 12
HIDDEN_2 = 6

DROPOUT = 0.05

NN_LEARNING_RATE = 5e-4
NN_WEIGHT_DECAY = 1e-3

NN_BATCH_SIZE = 32


# ------------------------------------------------------------
# Final epoch counts
#
# Median nested-CV epochs:
#
# MDD:
#   394, 423, 652, 637, 446
#   median = 446
#
# OWC:
#   410, 211, 352, 146, 440
#   median = 352
# ------------------------------------------------------------

MDD_NN_EPOCHS = 446
OWC_NN_EPOCHS = 352


# ============================================================
# 3. Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "train.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "test.csv"
)

SUBMISSION_PATH = (
    PROJECT_ROOT
    / "submissions"
    / "submission_model4i.csv"
)

LOG_PATH = (
    PROJECT_ROOT
    / "logs"
    / "model4i_run.log"
)


# ============================================================
# 4. Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# 4a. Logging
#
# Mirrors model1i.py's setup_logger so both pipelines leave a
# comparable run log under logs/.
# ============================================================

def setup_logger(path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger("model4i")

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    fh = logging.FileHandler(
        path,
        mode="w",
    )

    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(message)s"
        )
    )

    logger.addHandler(fh)

    sh = logging.StreamHandler()

    sh.setFormatter(
        logging.Formatter(
            "%(message)s"
        )
    )

    logger.addHandler(sh)

    logger.propagate = False

    return logger


# ============================================================
# 5. Load train/test
# ============================================================

def load_data():

    if not TRAIN_PATH.exists():

        raise FileNotFoundError(
            f"Train file not found:\n"
            f"{TRAIN_PATH}"
        )

    if not TEST_PATH.exists():

        raise FileNotFoundError(
            f"Test file not found:\n"
            f"{TEST_PATH}"
        )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    print(
        f"Loaded train: "
        f"{train_df.shape[0]} rows × "
        f"{train_df.shape[1]} columns"
    )

    print(
        f"Loaded test:  "
        f"{test_df.shape[0]} rows × "
        f"{test_df.shape[1]} columns"
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if "id" not in train_df.columns:

        raise KeyError(
            "'id' missing from train data."
        )

    if "id" not in test_df.columns:

        raise KeyError(
            "'id' missing from test data."
        )

    # --------------------------------------------------------
    # Remove ID 155 from TRAIN ONLY
    # --------------------------------------------------------

    remove_mask = (
        train_df["id"]
        ==
        EXCLUDED_ID
    )

    n_removed = int(
        remove_mask.sum()
    )

    print(
        f"\nRemoving id={EXCLUDED_ID} "
        f"from training data: "
        f"{n_removed} row(s)"
    )

    train_df = (
        train_df.loc[
            ~remove_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"Final training samples: "
        f"{len(train_df)}"
    )

    return (
        train_df,
        test_df,
    )


# ============================================================
# 6. Feature engineering
# ============================================================

def add_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Soil fractions
    # --------------------------------------------------------

    df["clay"] = (
        df[
            "psd_passing_at_0_002mm_pct"
        ]
    )

    df["silt"] = (
        df[
            "psd_passing_at_0_063mm_pct"
        ]
        -
        df[
            "psd_passing_at_0_002mm_pct"
        ]
    )

    df["sand"] = (
        df[
            "psd_passing_at_2mm_pct"
        ]
        -
        df[
            "psd_passing_at_0_063mm_pct"
        ]
    )

    df["gravel"] = (
        100.0
        -
        df[
            "psd_passing_at_2mm_pct"
        ]
    )

    # --------------------------------------------------------
    # Fine/coarse indicator
    # --------------------------------------------------------

    total_fines = (
        df["clay"]
        +
        df["silt"]
    )

    df["fine-grained"] = (
        total_fines
        >
        FINE_THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # Gradation features
    # --------------------------------------------------------

    d10 = (
        df[
            "psd_size_at_d10_mm"
        ]
        .replace(
            0,
            np.nan,
        )
    )

    d30 = (
        df[
            "psd_size_at_d30_mm"
        ]
    )

    d60 = (
        df[
            "psd_size_at_d60_mm"
        ]
        .replace(
            0,
            np.nan,
        )
    )

    df["feat_cu"] = (
        d60
        /
        d10
    )

    df["feat_cc"] = (
        d30 ** 2
        /
        (
            d10
            *
            d60
        )
    )

    df["feat_log_cu"] = (
        np.log1p(
            df["feat_cu"]
        )
    )

    return df


# ============================================================
# 7. Features
# ============================================================

FEATURE_COLS = [

    "psd_size_at_d10_mm",
    "psd_size_at_d20_mm",
    "psd_size_at_d30_mm",
    "psd_size_at_d40_mm",
    "psd_size_at_d50_mm",
    "psd_size_at_d60_mm",
    "psd_size_at_d70_mm",
    "psd_size_at_d80_mm",
    "psd_size_at_d90_mm",
    "psd_size_at_d95_mm",
    "psd_size_at_d98_mm",

    "psd_has_sedimentation",

    "psd_passing_at_0_002mm_pct",
    "psd_passing_at_0_063mm_pct",
    "psd_passing_at_2mm_pct",

    "proctor_diam_mm",
    "grain_density_g_cm3",

    "clay",
    "silt",
    "sand",
    "gravel",

    "fine-grained",

    "feat_cu",
    "feat_cc",
    "feat_log_cu",
]


# ============================================================
# 8. Validate columns
# ============================================================

def validate_features(
    train_df,
    test_df,
):

    missing_train = [

        col

        for col
        in FEATURE_COLS

        if col
        not in train_df.columns
    ]

    missing_test = [

        col

        for col
        in FEATURE_COLS

        if col
        not in test_df.columns
    ]

    if missing_train:

        raise KeyError(
            "Missing training features:\n"
            f"{missing_train}"
        )

    if missing_test:

        raise KeyError(
            "Missing test features:\n"
            f"{missing_test}"
        )

    print(
        f"\nNumber of model features: "
        f"{len(FEATURE_COLS)}"
    )


# ============================================================
# 9. XGBoost
# ============================================================

def make_xgb(seed):

    return XGBRegressor(

        objective="reg:squarederror",

        n_estimators=500,

        learning_rate=0.03,

        max_depth=3,

        min_child_weight=3,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=0.0,

        reg_lambda=1.0,

        random_state=seed,

        n_jobs=1,
    )


def train_xgb(
    X_train,
    y_train,
    X_test,
    *,
    seed,
):

    model = make_xgb(
        seed
    )

    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(
        X_test
    )

    return (
        model,
        prediction,
    )


# ============================================================
# 10. Tiny FFN
# ============================================================

class TinyMLP(nn.Module):

    def __init__(
        self,
        n_features,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                n_features,
                HIDDEN_1,
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_1,
                HIDDEN_2,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_2,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):

        return (
            self.network(
                x
            )
            .squeeze(-1)
        )


# ============================================================
# 11. Final FFN training
# ============================================================

def train_nn(
    X_train_df,
    y_train,
    X_test_df,
    *,
    epochs,
    seed,
    verbose=True,
):

    set_seed(
        seed
    )

    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    X_train = (
        X_train_df
        .to_numpy(
            dtype=float
        )
    )

    X_test = (
        X_test_df
        .to_numpy(
            dtype=float
        )
    )

    y_train = np.asarray(
        y_train,
        dtype=float,
    )

    # --------------------------------------------------------
    # Imputation
    #
    # Fit ONLY on training data.
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = (
        imputer.fit_transform(
            X_train
        )
    )

    X_test = (
        imputer.transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # Feature scaling
    #
    # Again fit ONLY on training data.
    # --------------------------------------------------------

    x_scaler = StandardScaler()

    X_train = (
        x_scaler.fit_transform(
            X_train
        )
    )

    X_test = (
        x_scaler.transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # Target scaling
    # --------------------------------------------------------

    y_scaler = StandardScaler()

    y_train_scaled = (
        y_scaler
        .fit_transform(
            y_train.reshape(
                -1,
                1,
            )
        )
        .ravel()
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TinyMLP(
        X_train.shape[1]
    ).to(device)

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=NN_LEARNING_RATE,

        weight_decay=(
            NN_WEIGHT_DECAY
        ),
    )

    loss_fn = nn.MSELoss()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = TensorDataset(

        torch.tensor(
            X_train,
            dtype=torch.float32,
        ),

        torch.tensor(
            y_train_scaled,
            dtype=torch.float32,
        ),
    )

    loader = DataLoader(

        dataset,

        batch_size=min(
            NN_BATCH_SIZE,
            len(dataset),
        ),

        shuffle=True,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    if verbose:

        print(
            f"Training FFN for "
            f"{epochs} epochs..."
        )

    for epoch in range(
        epochs
    ):

        model.train()

        epoch_loss = 0.0

        for X_batch, y_batch in loader:

            X_batch = (
                X_batch.to(
                    device
                )
            )

            y_batch = (
                y_batch.to(
                    device
                )
            )

            optimizer.zero_grad()

            prediction = model(
                X_batch
            )

            loss = loss_fn(
                prediction,
                y_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            epoch_loss += (
                loss.item()
                *
                len(X_batch)
            )

        # Occasional progress
        if (
            verbose
            and
            (
                epoch == 0
                or
                (epoch + 1) % 100 == 0
                or
                epoch + 1 == epochs
            )
        ):

            mean_loss = (
                epoch_loss
                /
                len(dataset)
            )

            print(
                f"  epoch "
                f"{epoch + 1:4d}/"
                f"{epochs}: "
                f"loss="
                f"{mean_loss:.6f}"
            )

    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    model.eval()

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        prediction_scaled = (
            model(
                X_test_tensor
            )
            .cpu()
            .numpy()
        )

    prediction = (
        y_scaler
        .inverse_transform(

            prediction_scaled.reshape(
                -1,
                1,
            )
        )
        .ravel()
    )

    return {

        "model":
            model,

        "prediction":
            prediction,

        "imputer":
            imputer,

        "x_scaler":
            x_scaler,

        "y_scaler":
            y_scaler,
    }


# ============================================================
# 12. Prediction diagnostics
# ============================================================

def prediction_summary(
    name,
    prediction,
):

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    print(
        f"\n{name}"
    )

    print(
        f"  min:    "
        f"{prediction.min():.6f}"
    )

    print(
        f"  Q25:    "
        f"{np.percentile(prediction, 25):.6f}"
    )

    print(
        f"  median: "
        f"{np.median(prediction):.6f}"
    )

    print(
        f"  mean:   "
        f"{prediction.mean():.6f}"
    )

    print(
        f"  Q75:    "
        f"{np.percentile(prediction, 75):.6f}"
    )

    print(
        f"  max:    "
        f"{prediction.max():.6f}"
    )


# ============================================================
# 12a. Competition metric
# ============================================================

def nmae(y_true, y_pred):
    """Official competition metric: IQR-normalized MAE. Matches
    model1i.py's nmae() so the two pipelines' figures are comparable."""

    y_true, y_pred = (
        np.asarray(y_true, dtype=float),
        np.asarray(y_pred, dtype=float),
    )

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    q75, q25 = np.percentile(
        y_true,
        [75, 25],
    )

    iqr = (q75 - q25) if (q75 - q25) != 0 else 1e-8

    return float(mae / iqr)


# ============================================================
# 12b. Nested cross-validation
#
# model4i.py trains its final XGB+FFN blend on ALL retained
# training rows -- there is no held-out data left to score it
# against. This reproduces an honest OOF estimate of the blend's
# NMAE using the same StratifiedKFold-on-MDD scheme as
# model1i.py, instead of relying on a hardcoded figure in a
# comment.
# ============================================================

def run_oof_cv(
    X_df,
    y,
    *,
    splits,
    xgb_weight,
    nn_weight,
    nn_epochs,
    xgb_seed,
    nn_seed,
    target_name,
    logger,
):

    n = len(y)

    oof_xgb = np.full(n, np.nan)
    oof_nn = np.full(n, np.nan)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):

        X_tr = X_df.iloc[train_idx]
        X_val = X_df.iloc[val_idx]

        y_tr = y[train_idx]

        _, xgb_val_pred = train_xgb(
            X_tr,
            y_tr,
            X_val,
            seed=xgb_seed,
        )

        oof_xgb[val_idx] = xgb_val_pred

        nn_result = train_nn(
            X_tr,
            y_tr,
            X_val,
            epochs=nn_epochs,
            seed=nn_seed,
            verbose=False,
        )

        oof_nn[val_idx] = nn_result["prediction"]

        logger.info(
            f"  [{target_name}] fold {fold_idx + 1}/{len(splits)} done "
            f"(train={len(train_idx)}, val={len(val_idx)})"
        )

    validated = ~np.isnan(oof_xgb)

    oof_blend = (
        xgb_weight * oof_xgb
        +
        nn_weight * oof_nn
    )

    y_val = y[validated]
    pred_val = oof_blend[validated]

    metrics = {
        "r2": r2_score(y_val, pred_val),
        "mae": mean_absolute_error(y_val, pred_val),
        "rmse": np.sqrt(mean_squared_error(y_val, pred_val)),
        "nmae": nmae(y_val, pred_val),
    }

    logger.info(
        f"[{target_name}] nested-CV OOF (n_validated={int(validated.sum())}/{n}): "
        f"R2={metrics['r2']:.4f} MAE={metrics['mae']:.4f} "
        f"RMSE={metrics['rmse']:.4f} NMAE={metrics['nmae']:.6f}"
    )

    return {
        "oof_blend": oof_blend,
        "validated": validated,
        **metrics,
    }


# ============================================================
# 13. Main
# ============================================================

def main():

    logger = setup_logger(
        LOG_PATH
    )

    logger.info(
        "model4i -- final XGBoost + Tiny FFN blend "
        "(0.650/0.350 MDD, 0.656/0.344 OWC)"
    )

    logger.info(
        f"seed={RANDOM_STATE}  "
        f"cv_folds={CV_FOLDS}  "
        f"cv_val_strata={CV_VAL_STRATA}"
    )

    print(
        "=" * 90
    )

    print(
        "FINAL XGBOOST + FFN SUBMISSION"
    )

    print(
        "=" * 90
    )

    # ========================================================
    # Load
    # ========================================================

    train_df, test_df = (
        load_data()
    )

    # ========================================================
    # Feature engineering
    # ========================================================

    train_df = add_features(
        train_df
    )

    test_df = add_features(
        test_df
    )

    validate_features(
        train_df,
        test_df,
    )

    # ========================================================
    # Matrices
    # ========================================================

    X_train = (
        train_df[
            FEATURE_COLS
        ]
        .copy()
    )

    X_test = (
        test_df[
            FEATURE_COLS
        ]
        .copy()
    )

    y_mdd = (
        train_df[
            "proctor_mdd_g_cm3"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_owc = (
        train_df[
            "proctor_owc_pct"
        ]
        .to_numpy(
            dtype=float
        )
    )

    print(
        "\nTraining matrix:"
    )

    print(
        X_train.shape
    )

    print(
        "Test matrix:"
    )

    print(
        X_test.shape
    )

    logger.info(
        f"train shape={X_train.shape} (id={EXCLUDED_ID} excluded)  "
        f"test shape={X_test.shape}"
    )

    # ========================================================
    # Nested-CV validation
    #
    # Honest OOF estimate of the fixed 0.650/0.350 (MDD) and
    # 0.656/0.344 (OWC) XGB/FFN blend, using the same
    # StratifiedKFold-on-MDD scheme as model1i.py -- replaces
    # the hardcoded NMAE figure in this file's header comment
    # with a number this run actually computed.
    # ========================================================

    logger.info(
        "\n"
        + "=" * 90
    )

    logger.info(
        "NESTED-CV VALIDATION"
    )

    logger.info(
        "=" * 90
    )

    cv_splits = make_stratified_kfold_splits(
        y_mdd,
        n_splits=CV_FOLDS,
        val_strata=CV_VAL_STRATA,
        random_state=RANDOM_STATE,
    )

    mdd_cv = run_oof_cv(
        X_train,
        y_mdd,
        splits=cv_splits,
        xgb_weight=MDD_XGB_WEIGHT,
        nn_weight=MDD_NN_WEIGHT,
        nn_epochs=MDD_NN_EPOCHS,
        xgb_seed=RANDOM_STATE,
        nn_seed=RANDOM_STATE + 100,
        target_name="MDD",
        logger=logger,
    )

    owc_cv = run_oof_cv(
        X_train,
        y_owc,
        splits=cv_splits,
        xgb_weight=OWC_XGB_WEIGHT,
        nn_weight=OWC_NN_WEIGHT,
        nn_epochs=OWC_NN_EPOCHS,
        xgb_seed=RANDOM_STATE + 1,
        nn_seed=RANDOM_STATE + 200,
        target_name="OWC",
        logger=logger,
    )

    overall_nmae = (
        mdd_cv["nmae"]
        +
        owc_cv["nmae"]
    ) / 2.0

    logger.info(
        f"\nNested-CV overall NMAE (mean of MDD+OWC): {overall_nmae:.6f}"
    )

    # ========================================================
    # MDD XGBoost
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "MDD — XGBOOST"
    )

    print(
        "=" * 90
    )

    (
        mdd_xgb_model,
        mdd_xgb_pred,
    ) = train_xgb(

        X_train,

        y_mdd,

        X_test,

        seed=(
            RANDOM_STATE
        ),
    )

    # ========================================================
    # MDD FFN
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "MDD — FFN"
    )

    print(
        "=" * 90
    )

    mdd_nn_result = train_nn(

        X_train,

        y_mdd,

        X_test,

        epochs=(
            MDD_NN_EPOCHS
        ),

        seed=(
            RANDOM_STATE
            +
            100
        ),
    )

    mdd_nn_pred = (
        mdd_nn_result[
            "prediction"
        ]
    )

    # ========================================================
    # MDD blend
    # ========================================================

    mdd_final = (

        MDD_XGB_WEIGHT
        *
        mdd_xgb_pred

        +

        MDD_NN_WEIGHT
        *
        mdd_nn_pred
    )

    # ========================================================
    # OWC XGBoost
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "OWC — XGBOOST"
    )

    print(
        "=" * 90
    )

    (
        owc_xgb_model,
        owc_xgb_pred,
    ) = train_xgb(

        X_train,

        y_owc,

        X_test,

        seed=(
            RANDOM_STATE
            +
            1
        ),
    )

    # ========================================================
    # OWC FFN
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "OWC — FFN"
    )

    print(
        "=" * 90
    )

    owc_nn_result = train_nn(

        X_train,

        y_owc,

        X_test,

        epochs=(
            OWC_NN_EPOCHS
        ),

        seed=(
            RANDOM_STATE
            +
            200
        ),
    )

    owc_nn_pred = (
        owc_nn_result[
            "prediction"
        ]
    )

    # ========================================================
    # OWC blend
    # ========================================================

    owc_final = (

        OWC_XGB_WEIGHT
        *
        owc_xgb_pred

        +

        OWC_NN_WEIGHT
        *
        owc_nn_pred
    )

    # ========================================================
    # Diagnostics
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL PREDICTION DIAGNOSTICS"
    )

    print(
        "=" * 90
    )

    prediction_summary(
        "MDD — XGB",
        mdd_xgb_pred,
    )

    prediction_summary(
        "MDD — NN",
        mdd_nn_pred,
    )

    prediction_summary(
        "MDD — BLEND",
        mdd_final,
    )

    prediction_summary(
        "OWC — XGB",
        owc_xgb_pred,
    )

    prediction_summary(
        "OWC — NN",
        owc_nn_pred,
    )

    prediction_summary(
        "OWC — BLEND",
        owc_final,
    )

    # ========================================================
    # Check predictions
    # ========================================================

    for name, pred in [

        (
            "MDD",
            mdd_final,
        ),

        (
            "OWC",
            owc_final,
        ),

    ]:

        if not np.all(
            np.isfinite(
                pred
            )
        ):

            raise RuntimeError(
                f"{name} predictions contain "
                f"NaN or infinity."
            )

    # ========================================================
    # Submission
    # ========================================================

    submission = pd.DataFrame(
        {

            "id":
                test_df[
                    "id"
                ]
                .to_numpy(),

            "proctor_mdd_g_cm3":
                mdd_final,

            "proctor_owc_pct":
                owc_final,
        }
    )

    # --------------------------------------------------------
    # Preserve test row count/order
    # --------------------------------------------------------

    if (
        len(submission)
        !=
        len(test_df)
    ):

        raise RuntimeError(
            "Submission row count "
            "does not match test data."
        )

    if not np.array_equal(

        submission[
            "id"
        ].to_numpy(),

        test_df[
            "id"
        ].to_numpy(),

    ):

        raise RuntimeError(
            "Submission IDs are not "
            "in original test order."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    submission.to_csv(
        SUBMISSION_PATH,
        index=False,
    )

    # ========================================================
    # Also save component predictions
    #
    # Very useful later if we want to change blend weights
    # WITHOUT retraining everything.
    # ========================================================

    component_path = (
        PROJECT_ROOT
        /
        "final_component_predictions.csv"
    )

    components = pd.DataFrame(
        {

            "id":
                test_df[
                    "id"
                ]
                .to_numpy(),

            "mdd_xgb":
                mdd_xgb_pred,

            "mdd_nn":
                mdd_nn_pred,

            "mdd_blend":
                mdd_final,

            "owc_xgb":
                owc_xgb_pred,

            "owc_nn":
                owc_nn_pred,

            "owc_blend":
                owc_final,
        }
    )

    components.to_csv(
        component_path,
        index=False,
    )

    # ========================================================
    # Final output
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "SUBMISSION CREATED"
    )

    print(
        "=" * 90
    )

    print(
        "\nBlend weights:"
    )

    print(
        f"MDD: "
        f"{MDD_XGB_WEIGHT:.3f} XGB + "
        f"{MDD_NN_WEIGHT:.3f} NN"
    )

    print(
        f"OWC: "
        f"{OWC_XGB_WEIGHT:.3f} XGB + "
        f"{OWC_NN_WEIGHT:.3f} NN"
    )

    print(
        "\nFFN epochs:"
    )

    print(
        f"MDD: {MDD_NN_EPOCHS}"
    )

    print(
        f"OWC: {OWC_NN_EPOCHS}"
    )

    print(
        "\nSubmission preview:"
    )

    print(
        submission.head(
            10
        ).to_string(
            index=False
        )
    )

    print(
        "\nSubmission shape:"
    )

    print(
        submission.shape
    )

    print(
        "\nSaved submission:"
    )

    print(
        SUBMISSION_PATH
    )

    print(
        "\nSaved component predictions:"
    )

    print(
        component_path
    )

    logger.info(
        "\n"
        + "=" * 90
    )

    logger.info(
        "SUBMISSION CREATED"
    )

    logger.info(
        "=" * 90
    )

    logger.info(
        f"MDD blend weights: xgboost={MDD_XGB_WEIGHT:.3f} nn={MDD_NN_WEIGHT:.3f}  "
        f"(FFN epochs={MDD_NN_EPOCHS})"
    )

    logger.info(
        f"OWC blend weights: xgboost={OWC_XGB_WEIGHT:.3f} nn={OWC_NN_WEIGHT:.3f}  "
        f"(FFN epochs={OWC_NN_EPOCHS})"
    )

    logger.info(
        f"Nested-CV overall NMAE (mean of MDD+OWC): {overall_nmae:.6f}"
    )

    logger.info(
        f"Wrote {len(submission)} predictions -> {SUBMISSION_PATH}"
    )

    logger.info(
        f"Saved component predictions -> {component_path}"
    )

    logger.info(
        f"Run log -> {LOG_PATH}"
    )


# ============================================================
# 14. Entry point
# ============================================================

if __name__ == "__main__":

    main()