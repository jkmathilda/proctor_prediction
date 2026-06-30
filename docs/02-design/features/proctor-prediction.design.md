# Design: proctor-prediction

**Feature**: LeiGS 2026 Proctor Prediction Challenge  
**Phase**: Design  
**Created**: 2026-06-30  
**References**: `docs/01-plan/features/proctor-prediction.plan.md`

---

## 1. Data Findings (EDA Summary)

Key statistics derived from `train.csv` (201 rows):

| Stat | MDD (g/cm³) | OWC (%) |
|---|---|---|
| Min | 1.609 | 3.54 |
| Max | 2.209 | 20.71 |
| Mean | 1.972 | 10.09 |
| Std | 0.140 | 3.365 |
| Q1 | 1.882 | 7.73 |
| Q3 | 2.080 | 11.59 |
| **IQR** | **0.198** | **3.860** |

### Missing Values Profile

| Feature | Missing | % | Interpretation |
|---|---|---|---|
| `atterberg_liquid_limit_pct` | 175/201 | 87% | Only fine-grained soils have plasticity |
| `atterberg_plastic_limit_pct` | 175/201 | 87% | Same as LL — same samples |
| `hyd_cond_kf_m_s` | 140/201 | 70% | Not tested for all samples |
| `hyd_cond_hyd_gradient` | 140/201 | 70% | Always paired with kf |
| `loss_on_ignition_pct` | 140/201 | 70% | Not tested for all samples |

**Key insight**: All 201 rows have valid PSD D10–D98 → CU and CC are computable for every sample without imputation.

### Feature Correlation with Targets

| Feature | Corr with MDD | Corr with OWC |
|---|---|---|
| `psd_passing_at_0_063mm_pct` (fines%) | -0.492 | **+0.810** |
| `atterberg_liquid_limit_pct` (n=26) | -0.739 | — |
| `atterberg_plastic_limit_pct` (n=26) | — | +0.737 |

**Key insight**: Fine fraction % is the single strongest predictor for OWC. Atterberg limits are highly predictive but sparse (only 26 samples have them).

### Other Data Notes
- Grain density range: 2.622–2.780 g/cm³ (mean 2.658, quartzitic soils)
- Proctor mold: 134 samples at 100 mm, 67 samples at 150 mm
- LOI (when available): 0.2–5.2%, mean 1.96%

---

## 2. Project File Structure

```
leigs-2026-proctor-prediction-challenge/
├── train.csv
├── test.csv
├── sample_submission.csv
├── solution.ipynb              # Main notebook (EDA + modeling + submission)
├── docs/
│   ├── 01-plan/features/proctor-prediction.plan.md
│   ├── 02-design/features/proctor-prediction.design.md  ← this file
│   └── ...
└── submissions/
    └── submission_YYYYMMDD_vN.csv
```

The entire solution lives in `solution.ipynb` with clear section headers. No separate Python modules needed at this scale.

---

## 3. Preprocessing Pipeline

### 3.1 Column Definitions

```python
TARGET_COLS = ['proctor_mdd_g_cm3', 'proctor_owc_pct']

PSD_D_COLS = [
    'psd_size_at_d10_mm', 'psd_size_at_d20_mm', 'psd_size_at_d30_mm',
    'psd_size_at_d40_mm', 'psd_size_at_d50_mm', 'psd_size_at_d60_mm',
    'psd_size_at_d70_mm', 'psd_size_at_d80_mm', 'psd_size_at_d90_mm',
    'psd_size_at_d95_mm', 'psd_size_at_d98_mm'
]

NUMERIC_COLS = PSD_D_COLS + [
    'psd_passing_at_0_002mm_pct', 'psd_passing_at_0_063mm_pct',
    'psd_passing_at_2mm_pct', 'grain_density_g_cm3',
    'hyd_cond_kf_m_s', 'hyd_cond_hyd_gradient',
    'atterberg_liquid_limit_pct', 'atterberg_plastic_limit_pct',
    'loss_on_ignition_pct', 'proctor_diam_mm'
]

BOOL_COLS = ['psd_has_sedimentation']
```

### 3.2 Type Coercion

```python
# Boolean: "True"/"False" strings → int (1/0)
df['psd_has_sedimentation'] = df['psd_has_sedimentation'].map({'True': 1, 'False': 0})

# Numeric: empty strings → NaN
for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
```

### 3.3 Missing Value Strategy

**Rule: fit imputers on train only, apply to both train and test.**

| Feature Group | Strategy | Rationale |
|---|---|---|
| PSD D-cols | No imputation needed (complete) | All 201 rows valid |
| PSD fractions | No imputation needed (complete) | All 201 rows valid |
| `grain_density_g_cm3` | No imputation (complete) | All rows valid |
| `proctor_diam_mm` | No imputation (complete) | All rows valid |
| `psd_has_sedimentation` | No imputation (complete) | All rows valid |
| `atterberg_liquid_limit_pct` | Median imputation | 87% missing — median of available 26 |
| `atterberg_plastic_limit_pct` | Median imputation | Same as LL |
| `hyd_cond_kf_m_s` | Median imputation | 70% missing |
| `hyd_cond_hyd_gradient` | Median imputation | 70% missing |
| `loss_on_ignition_pct` | Median imputation | 70% missing |

**Additionally**: Add binary missingness indicator flags for the 5 high-missing features:
```python
HIGH_MISSING = ['atterberg_liquid_limit_pct', 'atterberg_plastic_limit_pct',
                'hyd_cond_kf_m_s', 'hyd_cond_hyd_gradient', 'loss_on_ignition_pct']

for col in HIGH_MISSING:
    df[f'{col}_missing'] = df[col].isna().astype(int)
```
This lets the model learn that "Atterberg limit = missing" implies a coarse soil type.

---

## 4. Feature Engineering

All engineered features are computed **after** imputation. Use `np.log1p` for log transforms (safe for zero values).

### 4.1 Geotechnical Derived Features

```python
D10 = df['psd_size_at_d10_mm']
D30 = df['psd_size_at_d30_mm']
D60 = df['psd_size_at_d60_mm']

# Coefficient of uniformity — grading width
df['feat_cu'] = D60 / D10.replace(0, np.nan)

# Coefficient of curvature — curve symmetry
df['feat_cc'] = (D30 ** 2) / (D60 * D10).replace(0, np.nan)

# Plasticity Index (only meaningful where Atterberg limits exist)
df['feat_pi'] = df['atterberg_liquid_limit_pct'] - df['atterberg_plastic_limit_pct']

# Saturation line constraint at mean grain density:
# ρd_sat = ρs / (1 + w * ρs/ρw) → rearranged for expected MDD upper bound
# Useful as a physics-informed feature
RHO_W = 1.0
df['feat_sat_mdd_at_owc_est'] = df['grain_density_g_cm3'] / (
    1 + (df['psd_passing_at_0_063mm_pct'] / 100 * 10) * df['grain_density_g_cm3'] / RHO_W
)
# ^ rough proxy: fine-grained soils (higher fines%) have higher wopt

# Median grain size (D50)
df['feat_d50'] = df['psd_size_at_d50_mm']

# Fine fraction ratio (clay to total fines)
df['feat_clay_to_fines_ratio'] = (
    df['psd_passing_at_0_002mm_pct'] / df['psd_passing_at_0_063mm_pct'].replace(0, np.nan)
)

# Sand fraction (between 0.063mm and 2mm)
df['feat_sand_pct'] = df['psd_passing_at_2mm_pct'] - df['psd_passing_at_0_063mm_pct']

# Gravel fraction (above 2mm)
df['feat_gravel_pct'] = 100 - df['psd_passing_at_2mm_pct']
```

### 4.2 Log-Transformed PSD Diameters

PSD diameters span orders of magnitude (0.0003 mm to ~30 mm) — log scale is the natural space.

```python
for col in PSD_D_COLS:
    df[f'log_{col}'] = np.log1p(df[col])

# Also log-transform hydraulic conductivity (spans many orders of magnitude)
df['log_hyd_cond_kf_m_s'] = np.log1p(df['hyd_cond_kf_m_s'])
```

### 4.3 Soil Classification Flag

Classify each sample into broad soil type based on fines content:
```python
def soil_type(fines_pct):
    if fines_pct < 10:
        return 0   # coarse (gravel/sand)
    elif fines_pct < 40:
        return 1   # mixed
    else:
        return 2   # fine-grained (silt/clay)

df['feat_soil_type'] = df['psd_passing_at_0_063mm_pct'].apply(soil_type)
```

### 4.4 Final Feature Set

After engineering, the feature matrix includes:
- 11 raw PSD D-cols
- 11 log-transformed PSD D-cols
- 3 PSD passing fractions
- 5 high-missing features (after imputation)
- 5 missingness indicator flags
- 1 boolean (psd_has_sedimentation)
- 1 mold diameter
- 1 grain density
- 7 engineered geotechnical features (CU, CC, PI, D50, clay/fines ratio, sand%, gravel%)
- 1 log(kf)
- 1 soil type flag

**Total: ~47 features**

---

## 5. Model Architecture

### 5.1 Validation Strategy

```python
from sklearn.model_selection import KFold

CV = KFold(n_splits=5, shuffle=True, random_state=42)
```

Use 5-fold CV on train.csv. Report mean ± std NMAE across folds. This is the primary signal for model comparison.

### 5.2 NMAE Metric Implementation

```python
import numpy as np

def nmae(y_true, y_pred, iqr_mdd=0.198, iqr_owc=3.860):
    """
    IQR values computed from full training set targets.
    Both targets weighted 50/50.
    """
    mdd_true, owc_true = y_true[:, 0], y_true[:, 1]
    mdd_pred, owc_pred = y_pred[:, 0], y_pred[:, 1]
    
    mae_mdd = np.mean(np.abs(mdd_true - mdd_pred))
    mae_owc = np.mean(np.abs(owc_true - owc_pred))
    
    return 0.5 * (mae_mdd / iqr_mdd) + 0.5 * (mae_owc / iqr_owc)
```

**Note**: IQR is computed once from all available training targets and kept fixed. Do not recompute per fold.

### 5.3 Multi-output Strategy

Train separate models for MDD and OWC, or use `MultiOutputRegressor` wrapper. Separate models preferred because:
- MDD and OWC have different dominant features (fines content has different correlation sign)
- Separate hyperparameter tuning is possible per target

### 5.4 Model Tiers

#### Tier 1: Baseline (always run first)
```python
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

baseline = MultiOutputRegressor(Ridge(alpha=1.0))
```

#### Tier 2: Tree-based (main approach)
```python
import xgboost as xgb
import lightgbm as lgb

# XGBoost — separate for MDD and OWC
xgb_mdd = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1
)

# LightGBM
lgb_mdd = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1
)
```

#### Tier 3: Ensemble
Simple weighted average of Tier 2 models:
```python
pred_mdd = 0.5 * xgb_pred_mdd + 0.5 * lgb_pred_mdd
pred_owc = 0.5 * xgb_pred_owc + 0.5 * lgb_pred_owc
```

Weights optimized by CV score, not fixed at 0.5.

#### Tier 4: Hyperparameter Optimization (Optuna)
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
    }
    # ... run CV and return mean NMAE
```

### 5.5 Post-processing: Physical Constraint

MDD predictions must lie below the saturation line. Apply a hard cap:
```python
def apply_saturation_constraint(mdd_pred, owc_pred, grain_density, rho_w=1.0):
    sat_limit = grain_density / (1 + owc_pred / 100 * grain_density / rho_w)
    return np.minimum(mdd_pred, sat_limit * 0.99)  # 1% safety margin
```

---

## 6. Notebook Structure (`solution.ipynb`)

```
Section 0: Setup & Imports
  - pandas, numpy, sklearn, xgboost, lightgbm, optuna, matplotlib, seaborn

Section 1: Data Loading
  - Load train.csv, test.csv
  - Quick shape/dtypes check

Section 2: EDA
  - Target distributions (histograms)
  - Missing value heatmap
  - Correlation matrix (Pearson)
  - PSD D-col distributions (log scale)
  - Fines% vs MDD and OWC scatter plots

Section 3: Preprocessing
  - Type coercion
  - Missingness flags
  - Median imputation (fit on train, transform both)

Section 4: Feature Engineering
  - CU, CC, PI, soil type, log-PSD, etc.

Section 5: Baseline Model
  - Ridge regression CV
  - Report NMAE per fold + mean

Section 6: XGBoost & LightGBM
  - Separate models per target
  - 5-fold CV with NMAE metric
  - Feature importance plots

Section 7: Ensemble
  - Weighted average (optimize weights)
  - Final CV score

Section 8: Hyperparameter Tuning (optional, time-permitting)
  - Optuna study for best model

Section 9: Final Prediction
  - Fit best model on full train set
  - Predict on test.csv
  - Apply saturation constraint
  - Write submission CSV

Section 10: Submission Validation
  - Assert id matches sample_submission.csv
  - Assert column names correct
  - Check prediction ranges are physically plausible
```

---

## 7. Submission Output

```python
submission = pd.DataFrame({
    'id': test_df['id'],
    'proctor_owc_pct': owc_pred,
    'proctor_mdd_g_cm3': mdd_pred
})

# Validation
assert list(submission.columns) == ['id', 'proctor_owc_pct', 'proctor_mdd_g_cm3']
assert len(submission) == 87
assert submission['id'].tolist() == list(range(201, 288))

submission.to_csv('submissions/submission_YYYYMMDD_v1.csv', index=False)
```

---

## 8. Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
optuna>=3.0
matplotlib>=3.7
seaborn>=0.12
```

Install: `pip install pandas numpy scikit-learn xgboost lightgbm optuna matplotlib seaborn`

---

## 9. Acceptance Criteria (from Plan)

- [ ] Internal CV NMAE < 0.15
- [ ] Public LB NMAE within ±0.03 of internal CV (no overfitting)
- [ ] No data leakage (imputers fit on train fold only inside CV)
- [ ] MDD predictions ≤ saturation line for all test samples
- [ ] Submission CSV: 87 rows, columns `[id, proctor_owc_pct, proctor_mdd_g_cm3]`, id 201–287
- [ ] Feature importances reviewed — physically interpretable top features

---

## 10. PDCA Status

```
[Plan] ✅ → [Design] ✅ → [Do] ⏳ → [Check] ⏳ → [Act] ⏳
```

**Next**: `/pdca do proctor-prediction` → implement `solution.ipynb` following Sections 0–10 above.
