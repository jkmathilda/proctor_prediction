"""
Monte Carlo synthetic-data generator for geotechnical soil (PSD) dataset.

Method: Gaussian copula.
 - Each numeric column is mapped to a uniform via its empirical CDF (rank-based),
   then to a standard normal. This captures arbitrary skewed marginals exactly.
 - A correlation matrix is estimated from the normal-scores (pairwise complete),
   nudged to the nearest positive-definite matrix, and used to draw correlated
   latent normals for the synthetic rows.
 - Latents are pushed back through each column's inverse empirical CDF to recover
   realistic marginal values while preserving cross-column correlation.

Post-processing:
 - PSD percentile sizes (d10..d98) are sorted per row so they stay monotonically
   increasing (physical constraint).
 - psd_passing_* percentages clipped to [0, 100] and kept monotone by grain size.
 - proctor_diam_mm snapped to the observed discrete set (100 / 150).
 - psd_has_sedimentation drawn as a correlated Bernoulli.
 - Sparse lab columns (hyd cond, atterberg, loss on ignition): the joint
   missingness pattern is bootstrapped from real rows so co-occurrence of gaps is
   preserved ("keep realistic gaps").
"""

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(42)
N_SYNTH = 5000
SRC = "train.csv"
OUT = f"synthetic_soil_{N_SYNTH}.csv"

df = pd.read_csv(SRC)

id_col = "id"
bool_col = "psd_has_sedimentation"
diam_col = "proctor_diam_mm"

psd_size_cols = [c for c in df.columns if c.startswith("psd_size_at_d")]
passing_cols = [
    "psd_passing_at_0_002mm_pct",
    "psd_passing_at_0_063mm_pct",
    "psd_passing_at_2mm_pct",
]
sparse_cols = [
    "hyd_cond_kf_m_s",
    "hyd_cond_hyd_gradient",
    "atterberg_liquid_limit_pct",
    "atterberg_plastic_limit_pct",
    "loss_on_ignition_pct",
]

# Every column we model through the copula (numeric + bool encoded as 0/1)
model_cols = [c for c in df.columns if c not in (id_col,)]

# ---------------------------------------------------------------------------
# 1. Build normal-score representation of each modeled column
# ---------------------------------------------------------------------------
work = df.copy()
work[bool_col] = work[bool_col].astype(float)  # True/False -> 1/0

normal_scores = pd.DataFrame(index=work.index, columns=model_cols, dtype=float)
sorted_vals = {}  # column -> sorted non-null values, for inverse transform

for c in model_cols:
    vals = work[c].to_numpy(dtype=float)
    mask = ~np.isnan(vals)
    v = vals[mask]
    sorted_vals[c] = np.sort(v)
    # rank -> uniform in (0,1) using rank/(n+1) plotting position
    ranks = stats.rankdata(v, method="average")
    u = ranks / (len(v) + 1.0)
    z = stats.norm.ppf(u)
    ns = np.full(len(vals), np.nan)
    ns[mask] = z
    normal_scores[c] = ns

# ---------------------------------------------------------------------------
# 2. Correlation of normal scores (pairwise complete) + nearest PD
# ---------------------------------------------------------------------------
corr = normal_scores.corr(method="pearson").to_numpy()
corr = np.nan_to_num(corr, nan=0.0)
np.fill_diagonal(corr, 1.0)


def nearest_pd(A):
    B = (A + A.T) / 2
    w, V = np.linalg.eigh(B)
    w = np.clip(w, 1e-6, None)
    B = (V * w) @ V.T
    d = np.sqrt(np.diag(B))
    B = B / np.outer(d, d)
    np.fill_diagonal(B, 1.0)
    return B


corr_pd = nearest_pd(corr)

# ---------------------------------------------------------------------------
# 3. Draw correlated latent normals and invert through empirical CDFs
# ---------------------------------------------------------------------------
L = np.linalg.cholesky(corr_pd)
Z = RNG.standard_normal((N_SYNTH, len(model_cols))) @ L.T
U = stats.norm.cdf(Z)

synth = pd.DataFrame(index=range(N_SYNTH), columns=model_cols, dtype=float)
for j, c in enumerate(model_cols):
    sv = sorted_vals[c]
    # inverse empirical CDF via quantile interpolation
    synth[c] = np.quantile(sv, U[:, j], method="linear")

# ---------------------------------------------------------------------------
# 4. Post-processing / physical constraints
# ---------------------------------------------------------------------------
# PSD percentile sizes must be non-decreasing across d10..d98
synth[psd_size_cols] = np.sort(synth[psd_size_cols].to_numpy(), axis=1)
synth[psd_size_cols] = synth[psd_size_cols].clip(lower=0)

# passing percentages: clip to [0,100] and enforce monotone with grain size
synth[passing_cols] = synth[passing_cols].clip(0, 100)
synth[passing_cols] = np.sort(synth[passing_cols].to_numpy(), axis=1)

# bool column -> Bernoulli from the correlated latent uniform, thresholded at
# the observed True-rate (this keeps its copula correlation with other columns)
p_true = df[bool_col].mean()
bool_u = U[:, model_cols.index(bool_col)]
synth[bool_col] = (bool_u >= (1 - p_true)).astype(bool)

# proctor diameter -> snap to observed discrete categories
diam_choices = np.sort(df[diam_col].dropna().unique())
snapped = diam_choices[np.abs(synth[diam_col].to_numpy()[:, None] - diam_choices[None, :]).argmin(axis=1)]
synth[diam_col] = snapped.astype(int)

# grain density is physically bounded; clip to observed range
gd = "grain_density_g_cm3"
synth[gd] = synth[gd].clip(df[gd].min(), df[gd].max())

# proctor water content / density non-negative
for c in ["proctor_mdd_g_cm3", "proctor_owc_pct"]:
    synth[c] = synth[c].clip(lower=0)

# ---------------------------------------------------------------------------
# 5. Realistic missingness: bootstrap observed gap patterns for sparse cols
# ---------------------------------------------------------------------------
observed_patterns = df[sparse_cols].isna().to_numpy()
pick = RNG.integers(0, len(observed_patterns), size=N_SYNTH)
gap_mask = observed_patterns[pick]  # (N_SYNTH, len(sparse_cols))
for k, c in enumerate(sparse_cols):
    synth.loc[gap_mask[:, k], c] = np.nan

# atterberg: plastic limit should not exceed liquid limit where both present
ll, pl = "atterberg_liquid_limit_pct", "atterberg_plastic_limit_pct"
both = synth[[ll, pl]].notna().all(axis=1)
swap = both & (synth[pl] > synth[ll])
synth.loc[swap, [ll, pl]] = synth.loc[swap, [pl, ll]].to_numpy()

# ---------------------------------------------------------------------------
# 6. Assemble, order columns, write
# ---------------------------------------------------------------------------
synth.insert(0, id_col, range(len(df), len(df) + N_SYNTH))
synth = synth[df.columns]  # match original column order
synth.to_csv(OUT, index=False)
print("wrote", OUT, synth.shape)