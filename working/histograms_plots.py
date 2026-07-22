
# from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from scipy import stats
from itertools import combinations
cur_dirc = os.getcwd()
df = pd.read_csv(f'{cur_dirc}/final_training_data.csv')


# # Descriptive stats
# desc = df.describe()

# # Skew and Kurtosis
# agg_dict = {}
# for col in list(df.columns)[1:]:
#     agg_dict[col] = ["skew", "kurtosis"]
# agg = df.agg(agg_dict)

# df_combined = pd.concat([desc, agg])

# print(df_combined)
# df_combined.to_excel("data_analysis_basic_stats.xlsx")



# Histograms
numeric_cols = df.select_dtypes(include='number').columns
n = len(numeric_cols)

fig, axes = plt.subplots(10, 5, figsize=(6 * 5, 5 * 10))
axes = np.array(axes).flatten()

numeric_cols = numeric_cols.drop("id")
for col, ax in zip(numeric_cols, axes):

    ax.hist(df[col], edgecolor='black', alpha=0.7, bins=30)
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    if "pct" in col:
        ax.set_xlim(0, 100)

    if ("gravel" in col) or ("sand" in col) or ("silt" in col) or ("clay" in col):
        ax.set_ylim(0,130)

# for col, ax in zip(numeric_cols, axes): 
#     if col != "id":
#         print(col, ax)
#         ax.hist(df[col], edgecolor='black', alpha=0.7,   bins= 30 )
#         ax.title(col)
#         ax.xlabel(col)
#         ax.ylabel('Frequency')
#         if "pct" in col: 
#             ax.set_xlim(0,100)
plt.savefig(f'{cur_dirc}/histogram.png')
plt.clf()



# Monotonic relationships graphs
pairs = list(combinations(numeric_cols, 2))
n_pairs = len(pairs)
n_cols = 3
n_rows = int(np.ceil(n_pairs / n_cols))

# Folder for the individual monotonic relationship graphs
mono_dir = os.path.join(cur_dirc, "monotonic_new_2")
os.makedirs(mono_dir, exist_ok=True)


def plot_monotonic_pair(ax, x_col, y_col):
    """Draw the scatter + regression line for one pair onto `ax`."""
    common = df[[x_col, y_col]].dropna()
    x, y = common[x_col], common[y_col]

    # Spearman for monotonic, Pearson for linear
    spearman_r, spearman_p = stats.spearmanr(x, y)
    pearson_r, pearson_p = stats.pearsonr(x, y)

    # Scatter
    ax.scatter(x, y, alpha=0.6, color='steelblue', edgecolors='white', linewidths=0.4, s=40)

    # Regression line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, m * x_line + b, color='tomato', linewidth=1.5, linestyle='--')

    # Color-code title by strength of monotonic relationship
    spearman_abs = abs(spearman_r)
    if spearman_abs >= 0.7:
        title_color = 'green'
        strength = 'strong'
    elif spearman_abs >= 0.4:
        title_color = 'darkorange'
        strength = 'moderate'
    else:
        title_color = 'gray'
        strength = 'weak'

    ax.set_title(
        f'{x_col} vs {y_col}\n'
        f'Spearman r={spearman_r:.2f} (p={spearman_p:.3f}) [{strength}]\n'
        f'Pearson  r={pearson_r:.2f} (p={pearson_p:.3f})',
        fontsize=9, color=title_color
    )
    ax.set_xlabel(x_col, fontsize=8)
    ax.set_ylabel(y_col, fontsize=8)
    ax.tick_params(labelsize=7)


ah = [['psd_passing_at_0_002mm_pct', 'psd_passing_at_0_063mm_pct'],
 ['psd_passing_at_0_063mm_pct', 'psd_passing_at_2mm_pct'],
 ['psd_passing_at_0_002mm_pct', 'atterberg_liquid_limit_pct'],
 ['atterberg_liquid_limit_pct', 'atterberg_plastic_limit_pct'],
 ['psd_passing_at_0_002mm_pct', 'loss_on_ignition_pct'],
 ['psd_passing_at_0_063mm_pct', 'loss_on_ignition_pct'],
 ['atterberg_liquid_limit_pct', 'loss_on_ignition_pct'],
 ['atterberg_plastic_limit_pct', 'loss_on_ignition_pct']]
ah2 = []
for a in ah: 
    x = a[0]
    y = a[1]
    ah2.append(x + " vs "+ y)
    ah2.append(y + " vs "+ x)


# --- Individual graphs: one file per pair, saved to "monotonic relations" ---
for x_col, y_col in pairs:
    if f"{x_col} vs {y_col}" in ah2:
        fig_i, ax_i = plt.subplots(figsize=(6, 5))
        plot_monotonic_pair(ax_i, x_col, y_col)
        fig_i.tight_layout()
        safe = f"{x_col} vs {y_col}".replace("/", "_")
        fig_i.savefig(os.path.join(mono_dir, f"{safe}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig_i)

# --- Big picture: all pairs on one figure ---
fig, axes = plt.subplots(4, 2, figsize=(2 * 5, 4 * 5))
axes = np.array(axes).flatten()

for ax2, a in zip(axes,ah):
    x_col = a[0]
    y_col = a[1]
    plot_monotonic_pair(ax2, x_col, y_col)


# Summary legend
legend_text = (
    "Title colour:  green = strong (|r|≥0.7)  "
    "orange = moderate (|r|≥0.4)  "
    "gray = weak (|r|<0.4)"
)
fig.text(0.5, 0.01, legend_text, ha='center', fontsize=9, color='dimgray')

plt.suptitle('Pairwise monotonic relationship analysis', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("Pairwise monotonic relationship analysis.png")
plt.close(fig)



# # """
# # SHAP analysis for the ICSMGE cofferdam dataset.

# # Target:
# #     max |u_x|  -- maximum absolute horizontal wall displacement per
# #                   simulation (engineering design value).

# # Features (7):
# #     _H, _W, -B, -EI, -za, -t, -Id      design / soil parameters

# # Pipeline:
# #     1. Load the tab-separated file.
# #     2. Sanitize column names.
# #     3. Aggregate to ONE row per simulation (File Name) by taking
# #        max(|u_x|) over all y-coordinates. y_coordinate is NOT a feature.
# #     4. Random train/val/test split over simulations.
# #     5. Fit an XGBoost regressor with early-stopping on the val set.
# #     6. Evaluate on the held-out simulations (R^2, MAE, RMSE).
# #     7. Run SHAP TreeExplainer and save:
# #          - SHAP summary (beeswarm) plot
# #          - SHAP mean-|value| bar plot
# #          - SHAP dependence plot for each feature
# #     8. Export mean-|SHAP| feature importance to CSV.

# # Usage:
# #     pip install pandas numpy scikit-learn xgboost shap matplotlib
# #     python shap_analysis_cofferdams.py \
# #         --data ICSMGE_cofferdams_consolidated_results_splitted_updated.txt \
# #         --outdir shap_outputs
# # """
# # import argparse
# # from pathlib import Path

# # import matplotlib
# # matplotlib.use("Agg")  # headless backend; safe for scripts
# # import matplotlib.pyplot as plt

# # import numpy as np
# # import pandas as pd
# # import shap
# # import xgboost as xgb
# # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # from sklearn.model_selection import train_test_split


# # # ---------------------------------------------------------------------------
# # # Config
# # # ---------------------------------------------------------------------------

# # # Original column header in the file -> clean name we use in code/plots.
# # COLUMN_RENAME = {
# #     "_H": "H",
# #     "_W": "W",
# #     "-B": "B",
# #     "-EI": "EI",
# #     "-za": "za",
# #     "-t": "t",
# #     "-Id": "Id",
# #     "y_coordinate[m]": "y",
# #     "u_x[m]": "u_x",
# # }

# # FEATURES = ["H", "W", "B", "EI", "za", "t", "Id"]
# # TARGET = "max_abs_u_x"            # aggregated target, see aggregate_per_simulation
# # RAW_TARGET = "u_x"                # per-row column we aggregate over
# # GROUP_COL = "File Name"           # one "group" == one simulation


# # # ---------------------------------------------------------------------------
# # # Data
# # # ---------------------------------------------------------------------------

# # def load_data(path: Path) -> pd.DataFrame:
# #     """Read the tab-separated cofferdam results file and normalize columns."""
# #     df = pd.read_csv(path, sep="\t")

# #     missing = [c for c in COLUMN_RENAME if c not in df.columns]
# #     if missing:
# #         raise ValueError(
# #             f"Expected columns {missing} not found in {path}. "
# #             f"Got columns: {list(df.columns)}"
# #         )

# #     df = df.rename(columns=COLUMN_RENAME)

# #     # Drop any rows with NaNs in features / raw target (defensive).
# #     df = df.dropna(subset=FEATURES + [RAW_TARGET]).reset_index(drop=True)
# #     return df


# # def aggregate_per_simulation(df: pd.DataFrame) -> pd.DataFrame:
# #     """
# #     Collapse the multi-row-per-simulation dataframe into one row per File Name.

# #     Target = max(|u_x|) across all y-coordinates for that simulation.
# #     Features (H, W, B, EI, za, t, Id) are constant within a simulation, so
# #     `first` is a safe aggregator.
# #     """
# #     agg = (
# #         df.assign(_abs_u_x=df[RAW_TARGET].abs())
# #           .groupby(GROUP_COL, as_index=False)
# #           .agg(**{
# #               **{f: (f, "first") for f in FEATURES},
# #               TARGET: ("_abs_u_x", "max"),
# #           })
# #     )
# #     return agg


# # # ---------------------------------------------------------------------------
# # # Model
# # # ---------------------------------------------------------------------------

# # def train_xgb(X_train, y_train, X_val, y_val, seed: int) -> xgb.XGBRegressor:
# #     """Fit an XGBoost regressor with reasonable defaults + early stopping."""
# #     model = xgb.XGBRegressor(
# #         n_estimators=2000,
# #         learning_rate=0.05,
# #         max_depth=6,
# #         subsample=0.8,
# #         colsample_bytree=0.8,
# #         reg_lambda=1.0,
# #         objective="reg:squarederror",
# #         tree_method="hist",
# #         random_state=seed,
# #         early_stopping_rounds=50,
# #         eval_metric="rmse",
# #         n_jobs=-1,
# #     )
# #     model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
# #     return model


# # def evaluate(model, X, y, label: str) -> dict:
# #     """Print and return basic regression metrics."""
# #     pred = model.predict(X)
# #     rmse = float(np.sqrt(mean_squared_error(y, pred)))
# #     mae = float(mean_absolute_error(y, pred))
# #     r2 = float(r2_score(y, pred))
# #     print(f"  [{label}]  R^2 = {r2:.4f}   MAE = {mae:.3e}   RMSE = {rmse:.3e}")
# #     return {"split": label, "r2": r2, "mae": mae, "rmse": rmse}


# # # ---------------------------------------------------------------------------
# # # SHAP
# # # ---------------------------------------------------------------------------

# # def run_shap(model, X_explain: pd.DataFrame, outdir: Path,
# #              max_dependence_points: int = 5000) -> None:
# #     """Compute SHAP values and save the standard set of plots + an importance CSV."""
# #     outdir.mkdir(parents=True, exist_ok=True)

# #     # For very large datasets, subsample for plotting to keep things responsive.
# #     if len(X_explain) > max_dependence_points:
# #         X_plot = X_explain.sample(max_dependence_points, random_state=0).reset_index(drop=True)
# #     else:
# #         X_plot = X_explain.reset_index(drop=True)

# #     explainer = shap.TreeExplainer(model)
# #     shap_values = explainer.shap_values(X_plot)

# #     # --- Beeswarm summary ---------------------------------------------------
# #     plt.figure()
# #     shap.summary_plot(shap_values, X_plot, show=False)
# #     plt.tight_layout()
# #     plt.savefig(outdir / "shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
# #     plt.close()

# #     # --- Mean |SHAP| bar plot ----------------------------------------------
# #     plt.figure()
# #     shap.summary_plot(shap_values, X_plot, plot_type="bar", show=False)
# #     plt.tight_layout()
# #     plt.savefig(outdir / "shap_summary_bar.png", dpi=200, bbox_inches="tight")
# #     plt.close()

# #     # --- Per-feature dependence plots --------------------------------------
# #     dep_dir = outdir / "dependence"
# #     dep_dir.mkdir(exist_ok=True)
# #     for feat in X_plot.columns:
# #         plt.figure()
# #         shap.dependence_plot(
# #             feat, shap_values, X_plot,
# #             interaction_index="auto", show=False,
# #         )
# #         plt.tight_layout()
# #         safe = feat.replace("/", "_")
# #         plt.savefig(dep_dir / f"dep_{safe}.png", dpi=200, bbox_inches="tight")
# #         plt.close()

# #     # --- Importance CSV -----------------------------------------------------
# #     importance = (
# #         pd.DataFrame({
# #             "feature": X_plot.columns,
# #             "mean_abs_shap": np.abs(shap_values).mean(axis=0),
# #         })
# #         .sort_values("mean_abs_shap", ascending=False)
# #         .reset_index(drop=True)
# #     )
# #     importance.to_csv(outdir / "shap_feature_importance.csv", index=False)
# #     print("\nMean |SHAP| feature importance:")
# #     print(importance.to_string(index=False))


# # # ---------------------------------------------------------------------------
# # # Main
# # # ---------------------------------------------------------------------------

# # def main() -> None:
# #     parser = argparse.ArgumentParser(
# #         description="SHAP analysis for the ICSMGE cofferdam dataset."
# #     )
# #     parser.add_argument(
# #         "--data",
# #         type=Path,
# #         default=Path("ICSMGE_cofferdams_consolidated_results_splitted_updated.txt"),
# #         help="Path to the tab-separated results file.",
# #     )
# #     parser.add_argument(
# #         "--outdir",
# #         type=Path,
# #         default=Path("shap_outputs_1"),
# #         help="Directory where plots and CSVs are written.",
# #     )
# #     parser.add_argument("--test-size", type=float, default=0.2,
# #                         help="Fraction of simulations held out for testing.")
# #     parser.add_argument("--val-size", type=float, default=0.2,
# #                         help="Fraction of TRAIN simulations used as XGBoost early-stopping set.")
# #     parser.add_argument("--seed", type=int, default=42)
# #     args = parser.parse_args()

# #     print(f"Loading data: {args.data}")
# #     df_raw = load_data(args.data)
# #     print(f"  raw rows: {len(df_raw):,}   simulations: {df_raw[GROUP_COL].nunique():,}")

# #     df = aggregate_per_simulation(df_raw)
# #     print(f"  aggregated rows (one per simulation): {len(df):,}")

# #     # One row per simulation -> a plain random split is safe; no leakage possible.
# #     train_full_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed)
# #     train_df, val_df = train_test_split(train_full_df, test_size=args.val_size, random_state=args.seed)
# #     print(f"  train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

# #     X_train, y_train = train_df[FEATURES], train_df[TARGET]
# #     X_val,   y_val   = val_df[FEATURES],   val_df[TARGET]
# #     X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

# #     print("\nTraining XGBoost regressor (early stopping on val)...")
# #     model = train_xgb(X_train, y_train, X_val, y_val, args.seed)
# #     print(f"  best_iteration: {getattr(model, 'best_iteration', 'n/a')}")

# #     print("\nEvaluation:")
# #     metrics = [
# #         evaluate(model, X_train, y_train, "train"),
# #         evaluate(model, X_val,   y_val,   "val"),
# #         evaluate(model, X_test,  y_test,  "test"),
# #     ]
# #     args.outdir.mkdir(parents=True, exist_ok=True)
# #     pd.DataFrame(metrics).to_csv(args.outdir / "metrics.csv", index=False)

# #     # Explain on the TEST set so SHAP describes generalization behavior.
# #     print("\nComputing SHAP values on the held-out test set...")
# #     run_shap(model, X_test, args.outdir)

# #     # Save the trained model for reuse.
# #     model.save_model(args.outdir / "xgb_model.json")
# #     print(f"\nAll outputs written to: {args.outdir.resolve()}")


# # if __name__ == "__main__":
# #     main()