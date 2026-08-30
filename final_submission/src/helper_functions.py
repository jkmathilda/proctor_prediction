"""
helper_functions.py
===================
Project-specific helper functions for the LeiGS 2026 Proctor Prediction Challenge.
All imports are consolidated at the top. Functions are grouped by topic and
sorted alphabetically within each section:

  1. Metrics
  2. Data I/O
  3. Geotechnical Helpers
  4. EDA & Visualization
  5. Feature Engineering & Preprocessing
  6. Scaling Analysis
  7. ML Pipeline

> Author: Hermann Busse, M.Sc., HTWK Leipzig - Institut für Geotechnik, 2026-05
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import random
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Numerics & data
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import seaborn as sns

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# ---------------------------------------------------------------------------
# Machine learning
# NOTE: enable_iterative_imputer must be imported before IterativeImputer
# ---------------------------------------------------------------------------
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
mpl.rcParams['mathtext.default'] = 'regular'
warnings.filterwarnings("ignore")


class CFG:
    """Central configuration for the Proctor Prediction notebook."""

    seed: int = 42
    lang: str = 'de'

    @staticmethod
    def _resolve_data_dir() -> Path:
        """
        resolves the data directory based on the execution environment (Kaggle vs. local).
        """
        # Systemvariable KAGGLE_KERNEL_RUN_TYPE ist in allen Kaggle-Notebooks gesetzt
        is_kaggle_env = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
        
        if is_kaggle_env:
            kaggle_base = Path("/kaggle/input")
            if not kaggle_base.exists():
                raise FileNotFoundError("Kaggle-Umgebung erkannt, aber Basisverzeichnis '/kaggle/input' fehlt.")
            
            # Nutzung von next() zur effizienten Ermittlung des ersten Treffers
            try:
                target_file = next(kaggle_base.rglob("train.csv"))
                return target_file.parent
            except StopIteration:
                raise FileNotFoundError(
                    f"Die Datei 'train.csv' wurde im Verzeichnisbaum unter {kaggle_base} nicht gefunden. "
                    "Bitte Dataset-Anbindung und Dateinamen prüfen (Case-Sensitivity)."
                )
        else:
            # Lokale Entwicklungsumgebung
            local_dir = Path(".") / "kaggle" / "input"
            if not local_dir.exists():
                print(f"Warnung: Lokales Fallback-Verzeichnis '{local_dir.resolve()}' existiert nicht.")
            return local_dir

    # Zuweisung des evaluierten Pfades
    data_dir: Path = _resolve_data_dir.__func__()

    # (Dein bestehendes labels-Dictionary bleibt hier unverändert)
    labels: dict = {

        'de': {

            'target_mdd': 'Max. Trockendichte (MDD)',

            'target_owc': 'Opt. Wassergehalt (OWC)',

            'dist_title': 'Verteilung der Zielgröße',

            'box_title': 'Boxplot der Zielgröße',

            'freq': 'Häufigkeit',

            'value': 'Wert',

            'stat_summary_targets': '--- Statistische Kennzahlen der Zielgrößen ---',

            'scatter_title': 'Plausibilität: Proctor-Optima vs. Sättigungslinien',

            'sat_line': 'Sättigungslinie',

            'density_title': 'Verteilung der Korndichte (Daten-Artefakte)',

            'density_label': 'Korndichte [g/cm³]',

            'density_adv_title': 'Verteilung der Korndichte $\\rho_s$ inkl. empirischer Referenzbereiche',

            'density_x': 'Korndichte $\\rho_s$ [g/cm³]',

            'density_y': 'Anzahl der Proben $n$',

            'measured_data': 'Messdaten',

            'quartz_dom': 'Quarz-Dominanz',

            'zone_sand': 'Sand (Sa)',

            'zone_silt': 'Schluff (Si)',

            'zone_clay': 'Ton (Cl)',

            'min_feldspar': 'Feldspat',

            'min_mica': 'Glimmer',

            'min_gypsum': 'Gips',

            'min_calcite': 'Kalzit',

            'min_quartz': 'Quarz',

            'stat_summary_features': '--- Statistische Kennzahlen der Features ---',

            'count': 'Anzahl',

            'sed_false': 'Ohne Schlämmanalyse (False)',

            'sed_true': 'Mit Schlämmanalyse (True)',

            'psd_title': 'Korngrößenverteilung (KGV)',

            'psd_x': 'Korndurchmesser d [mm]',

            'psd_y': 'Massenanteil (< d) [%]',

            'with_sed': 'Mit Schlämmanalyse',

            'without_sed': 'Extrapoliert (ohne Schlämmanalyse)',

            'psd_standard': 'Zuverlässig / Mit Schlämmanalyse',

            'psd_extrapolated_high_fines': 'Kritisch: Extrapoliert & Feinkorn > 10%',

            'heatmap_title': 'Korrelation (Multikollinearität) der KGV-Features',

            'd50_dist': 'Verteilung: $d_{50}$ (Mittlere Korngröße)',

            'cu_dist': 'Verteilung: $C_U$ (Ungleichförmigkeitszahl)',

            'cc_dist': 'Verteilung: $C_C$ (Krümmungszahl)',

            'cu_cc_scatter': 'Form- & Abstufungsanalyse ($C_U$ vs. $C_C$)',

            'soil_clay': 'Ton',

            'soil_silt': 'Schluff',

            'soil_sand': 'Sand',

            'soil_gravel': 'Kies',

            'psd_C_U': 'Ungleichförmigkeitszahl $C_U$',

            'psd_C_C': 'Krümmungszahl $C_C$',

            'bivariate_title': 'Bivariate Regressionsanalyse',

            'influence_mdd': 'Einfluss auf die Max. Trockendichte',

            'influence_owc': 'Einfluss auf den Opt. Wassergehalt',

            'outlier_title': 'Anomalieanalyse: Hohes C_U vs. Geringe Dichte',

            'outlier_regular': 'Reguläre Proben',

            'outlier_anomalies': 'Kritische Anomalien (Kieslücke? Organisch?)',

            'count_diam': 'Verteilung der Topfdurchmesser:',

            'cat_title_mdd': 'MDD-Verteilung nach Topfgröße',

            'cat_title_owc': 'OWC-Verteilung nach Topfgröße',

            'proctor_diam_mm': 'Topfdurchmesser [mm]',

            'stat_summary_airvoids': '--- Statistik: Luftporenanteil im Proctor-Optimum ---',

            'airvoids_title_left': 'Optima vs. Luftporen-Isobaren ($\\rho_s={rho_s_ref}$ g/cm³)',

            'airvoids_title_right': 'Verteilung des Luftporenanteils ($n_a$)',

            'n_a_pct': 'Luftporenanteil $n_a$ [%]',

            'log10_kf': 'Wasserdurchlässigkeit $\\log_{10}(k_f)$ [m/s]',

            'kf_zones_title': 'Durchlässigkeit ($k_f$) über geotechnische Zonen',

            'zone_gravel': 'Kies',

            'interaction_title': 'Interaktion: Feinkornanteil vs. Durchlässigkeit (Farbe: Dichte)',

            'fines_pct': 'Feinkornanteil (< 0.063 mm) [%]',

            'casagrande_title': 'Casagrande-Diagramm nach DIN 18196',

            'liquid_limit': 'Fließgrenze $w_L$ [%]',

            'plasticity_index': 'Plastizitätszahl $I_p$ [%]',

            'test_data': 'Testdaten (Unbekannte Zielwerte)',

            'fines_legend': 'Feinkorn < 0.063mm [%]',

            'u_line_lbl': 'U-Linie $I_p = 0.9(w_L - 8)$',

            'a_line_lbl': 'A-Linie $I_p = 0.73(w_L - 20)$',

            'limit_high_plas': 'Grenze Hochplastizität (50%)',

            'zone_tl': 'TL\n(schwach plastisch)',

            'zone_tm': 'TM\n(mittelplastisch)',

            'zone_ta': 'TA\n(ausgeprägt plastisch)',

            'zone_ul': 'UL\n(schwach plastisch)',

            'zone_um': 'UM / OU\n(mittelplastisch)',

            'zone_ua': 'UA / OT\n(ausgeprägt plastisch)',

            'zone_st_su': 'Übergangszone\n(ST/SU)',

            'feat_eng_title': 'Einfluss des Feature Engineerings auf den Modellfehler (Niedriger = Besser)',

            'mae_mdd_title': 'MAE: Max. Trockendichte (MDD)',

            'mae_owc_title': 'MAE: Opt. Wassergehalt (OWC)',

            'error_mdd_ylabel': 'Fehler [g/cm³]',

            'error_owc_ylabel': 'Fehler [%]',

            'vif_analysis': '--- VIF-Analyse ---',

            'vif_mean_raw': 'Mittlerer VIF (Originale KGV):',

            'vif_mean_diff': 'Mittlerer VIF (Log-Diff):',

            'comp_impact': '--- Vergleich: Impact der Feature-Kombinationen ---',

            'comp_base': '1. Basis-Modell (Nur Kumulativwerte)',

            'comp_frac': '2. Fraktions-Modell (Isolierte DIN-Fraktionen)',

            'comp_comb': '3. Kombi-Modell (Fraktionen + Kumulativwerte)',

            'imp_mice_done': '-> MICE-Imputation unkorrumpiert abgeschlossen.',

            'imp_coarse_zero': '-> Danach {count} Zeilen (Grobkorn) physikalisch begründet auf 0.0 gesetzt.',

            'diag_imp_title': 'Diagnose: Korrigierte 2-Stufen-Imputation für {feature}',

            'diag_ref_label': 'Physikalisches Referenzmerkmal: {ref}',

            'diag_lbl_orig': 'Originale Messwerte',

            'diag_lbl_mice': 'Imputiert (MICE - Unverzerrt)',

            'diag_lbl_rule': 'Physikalisch begründet (Grobkorn = 0.0)',

            'scale_success': '✅ Datensatz erfolgreich transformiert! Dimensionen der Feature-Matrix: {shape}',

            'scale_summary': '--- Zuweisung für ColumnTransformer abgeschlossen ---',

            'scale_pt': '🔹 PowerTransformer (Yeo-Johnson) : {features}',

            'scale_robust': '🔸 RobustScaler (Ausreißerschutz)  : {features}',

            'scale_std': '⚙️ StandardScaler (Standardisierung): {count} Features zugewiesen.',

            'imp_eval_title': '📊 Vergleich: Einfluss der Imputation auf den Modellfehler (MAE)',

            'imp_base': '  1. Basis (NaN nativ, HistGB):    MAE MDD = {mdd:.4f} g/cm³  |  MAE OWC = {owc:.4f} %',

            'imp_mice': '  2. MICE (ohne neue Features):    MAE MDD = {mdd:.4f} g/cm³  |  MAE OWC = {owc:.4f} %',

            'imp_feat': '  3. MICE + neue Features:          MAE MDD = {mdd:.4f} g/cm³  |  MAE OWC = {owc:.4f} %',

            'loi_title': 'Verteilung des Glühverlusts (LOI)',

            'loi_main_title': 'Verteilung und Klassifizierung des Glühverlusts (LOI)',

            'loi_hist_title': 'Häufigkeitsverteilung',

            'loi_bar_title': 'Organik-Klassen nach DIN 18196',

            'loi_label': 'Glühverlust $V_{gl}$ [Gew.%]',

            'loi_min': 'Mineralisch (< 2%)',

            'loi_sl_org': 'Schwach organisch (2–6%)',

            'loi_org': 'Organisch (6–15%)',

            'loi_highly_org': 'Stark organisch (> 15%)',

            'top_corr_title': 'Top {top_n} Prädiktoren basierend auf maximaler Zielvariablen-Korrelation',

            'top_corr_xlabel': 'Spearman-Korrelationskoeffizient $\\rho$',

            'top_corr_mdd_label': 'MDD (Dichte)',

            'top_corr_owc_label': 'OWC (Feuchte)',

        },

        'en': {

            'target_mdd': 'Max. Dry Density (MDD)',

            'target_owc': 'Opt. Water Content (OWC)',

            'dist_title': 'Distribution of Target',

            'box_title': 'Boxplot of Target',

            'freq': 'Frequency',

            'value': 'Value',

            'stat_summary_targets': '--- Statistical Summary of Targets ---',

            'scatter_title': 'Plausibility: Proctor Optima vs. Zero Air Voids',

            'sat_line': 'ZAV Curve',

            'density_title': 'Grain Density Distribution (Data Artifacts)',

            'density_label': 'Grain Density [g/cm³]',

            'density_adv_title': 'Grain Density $\\rho_s$ Distribution incl. Empirical References',

            'density_x': 'Grain Density $\\rho_s$ [g/cm³]',

            'density_y': 'Number of Samples $n$',

            'measured_data': 'Measured Data',

            'quartz_dom': 'Quartz Dominance',

            'zone_sand': 'Sand (Sa)',

            'zone_silt': 'Silt (Si)',

            'zone_clay': 'Clay (Cl)',

            'min_feldspar': 'Feldspar',

            'min_mica': 'Mica',

            'min_gypsum': 'Gypsum',

            'min_calcite': 'Calcite',

            'min_quartz': 'Quartz',

            'stat_summary_features': '--- Statistical Summary of Features ---',

            'count': 'Count',

            'sed_false': 'Without Sedimentation (False)',

            'sed_true': 'With Sedimentation (True)',

            'psd_title': 'Particle Size Distribution (PSD)',

            'psd_x': 'Particle Diameter d [mm]',

            'psd_y': 'Percent Passing (< d) [%]',

            'with_sed': 'With Sedimentation',

            'without_sed': 'Extrapolated (no Sedimentation)',

            'psd_standard': 'Reliable / With Sedimentation',

            'psd_extrapolated_high_fines': 'Critical: Extrapolated & Fines > 10%',

            'heatmap_title': 'Correlation (Multicollinearity) of PSD Features',

            'd50_dist': 'Distribution: $d_{50}$ (Mean Grain Size)',

            'cu_dist': 'Distribution: $C_U$ (Coefficient of Uniformity)',

            'cc_dist': 'Distribution: $C_C$ (Coefficient of Curvature)',

            'cu_cc_scatter': 'Shape & Grading Analysis ($C_U$ vs. $C_C$)',

            'soil_clay': 'Clay',

            'soil_silt': 'Silt',

            'soil_sand': 'Sand',

            'soil_gravel': 'Gravel',

            'psd_C_U': 'Uniformity Coefficient $C_U$',

            'psd_C_C': 'Coefficient of Curvature $C_C$',

            'bivariate_title': 'Bivariate Regression Analysis',

            'influence_mdd': 'Influence on Max. Dry Density',

            'influence_owc': 'Influence on Opt. Water Content',

            'outlier_title': 'Anomaly Analysis: High Uniformity vs. Low Density',

            'outlier_regular': 'Regular Samples',

            'outlier_anomalies': 'Critical Anomalies (Gap-Graded? Organic?)',

            'count_diam': 'Distribution of Mold Diameters:',

            'cat_title_mdd': 'MDD Distribution by Mold Size',

            'cat_title_owc': 'OWC Distribution by Mold Size',

            'proctor_diam_mm': 'Mold Diameter [mm]',

            'stat_summary_airvoids': '--- Statistics: Air Void Ratio at Proctor Optimum ---',

            'airvoids_title_left': 'Optima vs. Air Void Isobars ($\\rho_s={rho_s_ref}$ g/cm³)',

            'airvoids_title_right': 'Distribution of Air Void Ratio ($n_a$)',

            'n_a_pct': 'Air Void Ratio $n_a$ [%]',

            'log10_kf': 'Hydraulic Conductivity $\\log_{10}(k_f)$ [m/s]',

            'kf_zones_title': 'Permeability ($k_f$) Distribution across Geotechnical Zones',

            'zone_gravel': 'Gravel',

            'interaction_title': 'Interaction: Fines Content vs. Permeability (Colored by Density)',

            'fines_pct': 'Fines Content (< 0.063 mm) [%]',

            'casagrande_title': 'Casagrande Plasticity Chart - DIN 18196',

            'liquid_limit': 'Liquid Limit $w_L$ [%]',

            'plasticity_index': 'Plasticity Index $I_p$ [%]',

            'test_data': 'Test Data (Unknown Targets)',

            'fines_legend': 'Fines < 0.063mm [%]',

            'u_line_lbl': 'U-Line $I_p = 0.9(w_L - 8)$',

            'a_line_lbl': 'A-Line $I_p = 0.73(w_L - 20)$',

            'limit_high_plas': 'High Plasticity Limit (50%)',

            'zone_tl': 'TL\n(low plasticity)',

            'zone_tm': 'TM\n(medium plasticity)',

            'zone_ta': 'TA\n(high plasticity)',

            'zone_ul': 'UL\n(low plasticity)',

            'zone_um': 'UM / OU\n(medium plasticity)',

            'zone_ua': 'UA / OT\n(high plasticity)',

            'zone_st_su': 'Intermediate Zone\n(ST/SU)',

            'feat_eng_title': 'Impact of Feature Engineering on Model Error (Lower = Better)',

            'mae_mdd_title': 'MAE: Maximum Dry Density (MDD)',

            'mae_owc_title': 'MAE: Optimum Water Content (OWC)',

            'error_mdd_ylabel': 'Error [g/cm³]',

            'error_owc_ylabel': 'Error [%]',

            'vif_analysis': '--- VIF Analysis ---',

            'vif_mean_raw': 'Mean VIF (Original PSD):',

            'vif_mean_diff': 'Mean VIF (Log-Diff):',

            'comp_impact': '--- Comparison: Impact of Feature Combinations ---',

            'comp_base': '1. Base Model (Cumulative passing only)',

            'comp_frac': '2. Fraction Model (Isolated DIN fractions)',

            'comp_comb': '3. Combined Model (Fractions + Cumulative)',

            'imp_mice_done': '-> MICE imputation completed without data corruption.',

            'imp_coarse_zero': '-> Afterwards, {count} rows (coarse-grained soils) were physically justified and set to 0.0.',

            'diag_imp_title': 'Diagnostic: Corrected 2-Step Imputation for {feature}',

            'diag_ref_label': 'Physical Reference Feature: {ref}',

            'diag_lbl_orig': 'Original Measured Values',

            'diag_lbl_mice': 'Imputed (MICE - Unbiased)',

            'diag_lbl_rule': 'Physically Justified (Coarse Soil = 0.0)',

            'scale_success': '✅ Dataset successfully transformed! Feature matrix dimensions: {shape}',

            'scale_summary': '--- ColumnTransformer Assignment Completed ---',

            'scale_pt': '🔹 PowerTransformer (Yeo-Johnson) : {features}',

            'scale_robust': '🔸 RobustScaler (Outlier Protection) : {features}',

            'scale_std': '⚙️ StandardScaler (Standardization): {count} Features assigned.',

            'imp_eval_title': '📊 Comparison: Impact of Imputation on Model Error (MAE)',

            'imp_base': '  1. Baseline (Native NaN, HistGB):  MAE MDD = {mdd:.4f} g/cm³  |  MAE OWC = {owc:.4f} %',

            'imp_mice': '  2. MICE (no new features):          MAE MDD = {mdd:.4f} g/cm³  |  MAE OWC = {owc:.4f} %',

            'imp_feat': '  3. MICE + new features:             MAE MDD = {mdd:.4f} g/cm³  |  MAE OWC = {owc:.4f} %',

            'loi_title': 'Distribution of Loss on Ignition (LOI)',

            'loi_main_title': 'Distribution and Classification of Loss on Ignition (LOI)',

            'loi_hist_title': 'Frequency Distribution',

            'loi_bar_title': 'DIN 18196 Organic Classes',

            'loi_label': 'Loss on Ignition $V_{gl}$ [wt.%]',

            'loi_min': 'Mineral (< 2%)',

            'loi_sl_org': 'Slightly Organic (2–6%)',

            'loi_org': 'Organic (6–15%)',

            'loi_highly_org': 'Highly Organic (> 15%)',

            'top_corr_title': 'Top {top_n} Predictors Based on Maximum Target Correlation',

            'top_corr_xlabel': 'Spearman Correlation Coefficient $\\rho$',

            'top_corr_mdd_label': 'MDD (Density)',

            'top_corr_owc_label': 'OWC (Moisture)',

        },

    }



    @classmethod
    def get_label(cls, key: str) -> str:
        """
        Returnt den lokalisierten String für *key* in der aktuellen Sprache.
        Gibt den Key selbst zurück, falls dieser nicht definiert ist.
        """
        return cls.labels[cls.lang].get(key, key)

    @staticmethod
    def seed_everything(seed: int = 42) -> None:
        """
        Setzt alle Random Seeds für vollständige Reproduzierbarkeit.
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        CFG.seed = seed
        messages = {
            "de": f"Zufallsseed auf {seed} gesetzt für Reproduzierbarkeit.",
            "en": f"Random seed set to {seed} for reproducibility.",
        }
        print(messages.get(CFG.lang, messages["en"]))


# ===========================================================================
# 1. METRICS
# ===========================================================================

def calculate_nmae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate the mean column-wise Normalised Mean Absolute Error (NMAE).

    Normalisation is performed per column using the interquartile range (IQR),
    which makes the metric robust to outliers and scale differences between
    the two Proctor targets (MDD and OWC).

    Args:
        y_true: Ground-truth target matrix of shape ``(n_samples, n_targets)``.
        y_pred: Predicted target matrix of the same shape.

    Returns:
        Macro-averaged NMAE across all target columns.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different shapes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    # Mean absolute error per target column
    mae_per_col = np.mean(np.abs(y_true - y_pred), axis=0)

    # IQR per column in a single vectorised call
    q75, q25 = np.percentile(y_true, [75, 25], axis=0)
    iqr_per_col = q75 - q25

    # Guard against division by zero
    iqr_per_col = np.where(iqr_per_col == 0, 1e-8, iqr_per_col)

    return float(np.mean(mae_per_col / iqr_per_col))


# ===========================================================================
# 2. DATA I/O
# ===========================================================================

def get_missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a report of columns with missing values.

    Args:
        df: Input DataFrame to analyse.

    Returns:
        DataFrame indexed by column name with columns
        ``missing_count``, ``missing_pct``, and ``dtype``.
        Only columns that have at least one missing value are included.
    """
    missing = df.isnull().sum()
    report = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': (missing / len(df)) * 100,
        'dtype': df.dtypes,
    })
    # Exclude columns with no missing values
    return report[report['missing_count'] > 0]


def load_and_summarize_data(
    data_dir: Path,
    lang: str = 'de',
) -> tuple:
    """Load the competition CSV files and print a concise summary.

    Args:
        data_dir: Directory containing ``train.csv``, ``test.csv``, and
            ``sample_submission.csv``.
        lang: Language code for the output message (``'de'`` or ``'en'``).

    Returns:
        Tuple of ``(train, test, sample_submission)`` DataFrames.
    """
    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / "train.csv")
    test  = pd.read_csv(data_dir / "test.csv")
    sub   = pd.read_csv(data_dir / "sample_submission.csv")

    messages = {
        'de': (
            f"DataFrames erfolgreich geladen:\n"
            f"  - Train: {train.shape[0]} Zeilen, {train.shape[1]} Spalten\n"
            f"  - Test:  {test.shape[0]} Zeilen, {test.shape[1]} Spalten\n"
            f"  - Sub:   {sub.shape[0]} Zeilen, {sub.shape[1]} Spalten"
        ),
        'en': (
            f"DataFrames successfully loaded:\n"
            f"  - Train: {train.shape[0]} rows, {train.shape[1]} columns\n"
            f"  - Test:  {test.shape[0]} rows, {test.shape[1]} columns\n"
            f"  - Sub:   {sub.shape[0]} rows, {sub.shape[1]} columns"
        ),
    }
    print(messages.get(lang, messages['de']))
    return train, test, sub


# ===========================================================================
# 3. GEOTECHNICAL HELPERS
# ===========================================================================

def calc_satline(w: np.ndarray, Gs: float) -> np.ndarray:
    """Compute the theoretical maximum dry density at full saturation (S_r = 100 %).

    Args:
        w: Water content values in percent [%].
        Gs: Specific gravity of soil solids [-].

    Returns:
        Array of dry density values in g/cm³.
    """
    return Gs / (1.0 + (w / 100.0) * Gs)


# ===========================================================================
# 4. EDA & VISUALIZATION
# ===========================================================================

def add_gradation_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the uniformity coefficient C_U and coefficient of curvature C_C.

    Division by zero is avoided by masking invalid d10 and d60 values.

    Args:
        df: DataFrame containing ``psd_size_at_d10_mm``, ``psd_size_at_d30_mm``,
            and ``psd_size_at_d60_mm`` columns.

    Returns:
        Copy of *df* with appended ``psd_C_U`` and ``psd_C_C`` columns.
    """
    df = df.copy()
    d10 = df["psd_size_at_d10_mm"]
    d30 = df["psd_size_at_d30_mm"]
    d60 = df["psd_size_at_d60_mm"]

    df["psd_C_U"] = np.where(d10 > 0, d60 / d10, np.nan)
    df["psd_C_C"] = np.where((d10 > 0) & (d60 > 0), (d30 ** 2) / (d60 * d10), np.nan)
    return df


def analyze_and_plot_ignition_loss(df: pd.DataFrame, cfg: CFG) -> None:
    """Plot the loss-on-ignition (LOI) distribution with DIN 18196 classification.

    Creates a 1x2 subplot figure:
      - Left: Histogram with KDE, mean, and median lines.
      - Right: Horizontal bar chart of DIN 18196 organic content classes.

    Args:
        df: DataFrame containing the ``loss_on_ignition_pct`` column.
        cfg: Configuration object providing localised labels.
    """
    col = "loss_on_ignition_pct"

    if col not in df.columns:
        print(f"Column '{col}' not found.")
        return

    valid_data = df[col].dropna()
    if valid_data.empty:
        print("No valid LOI values available.")
        return

    # Classify samples according to DIN 18196 organic content thresholds
    bins   = [-np.inf, 2.0, 6.0, 15.0, np.inf]
    labels = [
        cfg.get_label('loi_min'),
        cfg.get_label('loi_sl_org'),
        cfg.get_label('loi_org'),
        cfg.get_label('loi_highly_org'),
    ]
    categories = pd.cut(valid_data, bins=bins, labels=labels)
    cat_counts  = categories.value_counts().reindex(labels)

    sns.set_theme(style="ticks", rc={"axes.grid": True, "grid.color": "#E5E7E9", "grid.linestyle": "--"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120, gridspec_kw={'width_ratios': [1.5, 1]})

    # Left panel: histogram & KDE with mean and median lines
    ax_hist = axes[0]
    sns.histplot(valid_data, bins=25, kde=True, color="#8E44AD", edgecolor="black", alpha=0.6, ax=ax_hist)
    ax_hist.axvline(valid_data.mean(),   color="#C0392B", ls="--", lw=2.5, label=f"Mean: {valid_data.mean():.2f}%")
    ax_hist.axvline(valid_data.median(), color="#2980B9", ls="-.", lw=2.5, label=f"Median: {valid_data.median():.2f}%")
    ax_hist.set_title(cfg.get_label('loi_hist_title'), fontsize=13, fontweight="bold", pad=12)
    ax_hist.set_xlabel(cfg.get_label('loi_label'), fontsize=11, fontweight="bold")
    ax_hist.set_ylabel(cfg.get_label('freq'), fontsize=11)
    ax_hist.set_xlim(0, max(valid_data.max() * 1.1, 20))
    ax_hist.legend(framealpha=0.9)

    # Right panel: horizontal bar chart of DIN classes
    ax_bar = axes[1]
    colors = ["#BDC3C7", "#F1C40F", "#E67E22", "#C0392B"]
    ax_bar.barh(labels, cat_counts.values, color=colors, edgecolor="#34495E", lw=1.5, alpha=0.85)

    # Annotate each bar with count and percentage
    total = len(valid_data)
    for i, count in enumerate(cat_counts.values):
        ax_bar.text(
            count + (max(cat_counts.values) * 0.05), i,
            f"{count} ({(count / total) * 100:.1f}%)",
            va='center', fontsize=11, fontweight='bold', color="#2C3E50",
        )

    ax_bar.set_title(cfg.get_label('loi_bar_title'), fontsize=13, fontweight="bold", pad=12)
    ax_bar.set_xlabel(cfg.get_label('count'), fontsize=11, fontweight="bold")
    ax_bar.set_xlim(0, max(cat_counts.values) * 1.3)
    ax_bar.invert_yaxis()  # Mineral class at top

    plt.suptitle(cfg.get_label('loi_main_title'), fontsize=15, fontweight="bold", y=1.05, color="#2C3E50")
    sns.despine(fig)
    plt.tight_layout()
    plt.show()


def analyze_proctor_air_voids(
    df: pd.DataFrame,
    cfg: CFG,
    rho_w: float = 1.0,
) -> tuple:
    """Compute the air void ratio at the Proctor optimum and visualise its distribution.

    Missing grain densities are temporarily imputed with 2.65 g/cm³ for the
    volumetric calculation. Uses the three-phase soil model:
    ``n_a = 1 - (rho_d / rho_s) - w * (rho_d / rho_w)``.

    Args:
        df: DataFrame with columns ``proctor_owc_pct``, ``proctor_mdd_g_cm3``,
            and ``grain_density_g_cm3``.
        cfg: Configuration object providing localised labels.
        rho_w: Density of water in g/cm³. Default is 1.0.

    Returns:
        Tuple of ``(df_eval, stats_df, n_a_dec)`` where *df_eval* is the
        filtered working copy, *stats_df* is the descriptive statistics table,
        and *n_a_dec* is the air void ratio as a decimal series.
    """
    df_eval = df[["proctor_owc_pct", "proctor_mdd_g_cm3", "grain_density_g_cm3"]].dropna().copy()

    # Impute missing or invalid grain densities with a standard reference value
    df_eval["rho_s"] = df_eval["grain_density_g_cm3"].fillna(2.65)
    df_eval.loc[df_eval["rho_s"] <= 0, "rho_s"] = 2.65

    w     = df_eval["proctor_owc_pct"] / 100.0
    rho_d = df_eval["proctor_mdd_g_cm3"]
    rho_s = df_eval["rho_s"]

    n_a_dec = 1.0 - (rho_d / rho_s) - w * (rho_d / rho_w)
    df_eval["air_void_pct"] = n_a_dec * 100.0

    stats_df = df_eval[["air_void_pct"]].describe().T

    sns.set_theme(style="ticks", rc={"axes.grid": True, "grid.color": "#E5E7E9", "grid.linestyle": "--"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)

    # Left panel: Proctor optima overlaid with air void isobars
    rho_s_ref = 2.65
    w_range   = np.linspace(2, df_eval["proctor_owc_pct"].max() * 1.1, 200)

    for n_a_pct, color, ls in zip([0, 5, 10], ["#C0392B", "#E67E22", "#F39C12"], ["-", "--", ":"]):
        n_a_ref  = n_a_pct / 100.0
        # Isobar equation solved for rho_d: rho_d = (1 - n_a) / (1/rho_s + w/rho_w)
        rho_line = (1.0 - n_a_ref) / ((1.0 / rho_s_ref) + (w_range / 100.0) / rho_w)
        axes[0].plot(w_range, rho_line, ls=ls, color=color, lw=2.5, label=f"$n_a = {n_a_pct}$ %")

    axes[0].scatter(
        df_eval["proctor_owc_pct"], df_eval["proctor_mdd_g_cm3"],
        alpha=0.6, color="#2C3E50", s=40, edgecolor="white", zorder=3,
    )
    axes[0].set_title(cfg.get_label('airvoids_title_left').format(rho_s_ref=rho_s_ref), fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel(cfg.get_label('target_owc'), fontsize=11, fontweight="bold")
    axes[0].set_ylabel(cfg.get_label('target_mdd'), fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=10, framealpha=0.9)

    # Right panel: histogram of air void ratio with physical plausibility zones
    sns.histplot(df_eval["air_void_pct"], bins=30, kde=True, color="#2980B9", ax=axes[1], edgecolor="black")
    axes[1].axvspan(-5,  0,  color="#C0392B", alpha=0.15, label="Physically impossible (n_a < 0)")
    axes[1].axvspan(12, df_eval["air_void_pct"].max() + 5, color="#F1C40F", alpha=0.15, label="Outlier warning (n_a > 12)")
    axes[1].set_title(cfg.get_label('airvoids_title_right'), fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel(cfg.get_label('n_a_pct'), fontsize=11, fontweight="bold")
    axes[1].set_ylabel(cfg.get_label('freq'), fontsize=11)
    axes[1].legend(loc="upper right", fontsize=10, framealpha=0.9)

    sns.despine(fig)
    plt.tight_layout()
    plt.show()

    return df_eval, stats_df, n_a_dec


def get_psd_segmentation_counts(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    """Count samples by sedimentation flag and fines-content threshold.

    Args:
        df: DataFrame with ``psd_has_sedimentation`` (bool) and
            ``psd_passing_at_0_063mm_pct`` (float) columns.
        cfg: Configuration object providing localised labels (uses ``'count'``).

    Returns:
        DataFrame with a single count column indexed by descriptive segment labels.
    """
    segments = zip(df['psd_has_sedimentation'], df['psd_passing_at_0_063mm_pct'] > 10)

    mapping = {
        (True,  True):  "Mit Schlämmanalyse & Feinkorn > 10%",
        (True,  False): "Mit Schlämmanalyse & Feinkorn <= 10%",
        (False, True):  "Ohne Schlämmanalyse & Feinkorn > 10%",
        (False, False): "Ohne Schlämmanalyse & Feinkorn <= 10%",
    }

    counts = pd.Series(segments).value_counts().to_frame(cfg.get_label('count'))
    return counts.rename(index=mapping)


def plot_casagrande(
    df: pd.DataFrame,
    cfg: CFG,
    test_df: pd.DataFrame | None = None,
    hue_col: str = "psd_passing_at_0_063mm_pct",
) -> None:
    """Render a fully annotated Casagrande plasticity chart per DIN 18196.

    Includes colour-coded zone fills, A-line, U-line, SU/ST transition zones,
    text labels, continuous fines-content colour mapping for training data, and
    an optional test-data overlay for covariate shift assessment.

    Args:
        df: Training DataFrame with Atterberg limit columns.
        cfg: Configuration object providing localised labels.
        test_df: Optional test DataFrame for the overlay scatter.
        hue_col: Column used for continuous colour mapping. Default is fines %.
    """
    ll_col = "atterberg_liquid_limit_pct"
    pl_col = "atterberg_plastic_limit_pct"

    df_plot       = df.dropna(subset=[ll_col, pl_col]).copy()
    df_plot["Ip"] = df_plot[ll_col] - df_plot[pl_col]

    sns.set_theme(style="whitegrid", rc={"axes.grid": True, "grid.color": "#E5E7E9", "grid.linestyle": "--"})
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white", dpi=120)

    ax.set_xlim(0, 80)
    ax.set_ylim(0, 50)

    wl_range    = np.linspace(0, 80, 1000)
    a_line_vals = np.maximum(0, 0.73 * (wl_range - 20))
    u_line_vals = np.maximum(0, 0.9  * (wl_range - 8))

    # Coloured zone fills (clay above A-line, silt below)
    ax.fill_between(wl_range, a_line_vals, u_line_vals, where=(wl_range >= 8), color="#F8EFFA", alpha=0.7, zorder=0)
    ax.fill_between(wl_range, 0, a_line_vals, color="#F0FDF4", alpha=0.7, zorder=0)

    # Reference lines
    ax.plot(wl_range[wl_range >= 8],  u_line_vals[wl_range >= 8],  "k:", lw=1.5, alpha=0.6, label=cfg.get_label('u_line_lbl'), zorder=1)
    ax.plot(wl_range[wl_range >= 20], a_line_vals[wl_range >= 20], "k-", lw=3,                label=cfg.get_label('a_line_lbl'),  zorder=2)

    # Vertical classification boundaries (medium and high plasticity)
    ax.axvline(35, color="black", lw=1.5, ls="--", alpha=0.5, zorder=1)
    ax.axvline(50, color="black", lw=2,   ls="-",  label=cfg.get_label('limit_high_plas'), zorder=2)

    # ST/SU transition zone horizontal boundaries
    ax.axhline(y=7, xmin=0, xmax=35 / 80, color="black", lw=1.2, ls="--", zorder=1)
    ax.axhline(y=4, xmin=0, xmax=31 / 80, color="black", lw=1.2, ls="--", zorder=1)

    # Zone text labels (clays above A-line, silts below)
    text_kws = {"fontweight": "bold", "ha": "center", "va": "center", "zorder": 5}
    ax.text(28, 20, cfg.get_label('zone_tl'), color="#4B0082", fontsize=10, **text_kws)
    ax.text(42, 30, cfg.get_label('zone_tm'), color="#4B0082", fontsize=10, **text_kws)
    ax.text(65, 41, cfg.get_label('zone_ta'), color="#4B0082", fontsize=10, **text_kws)
    ax.text(30, 2,  cfg.get_label('zone_ul'), color="#006400", fontsize=10, **text_kws)
    ax.text(42, 11, cfg.get_label('zone_um'), color="#006400", fontsize=10, **text_kws)
    ax.text(65, 18, cfg.get_label('zone_ua'), color="#006400", fontsize=10, **text_kws)
    ax.text(12, 8.5, "ST", color="#D32F2F", fontsize=9, fontweight="bold", ha="center")
    ax.text(12, 2.0, "SU", color="#E67E22", fontsize=9, fontweight="bold", ha="center")
    ax.text(17, 5.5, cfg.get_label('zone_st_su'), fontsize=8, fontstyle="italic", ha="center", va="center", color="#5D6D7E")

    # Training data scatter with continuous fines-content colour encoding
    scatter = ax.scatter(
        df_plot[ll_col], df_plot["Ip"], c=df_plot[hue_col],
        cmap="viridis", s=130, edgecolor="black", linewidth=1.0, alpha=0.85, zorder=10,
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(cfg.get_label('fines_legend'), fontsize=11, fontweight="bold")

    # Optional test-data overlay for covariate shift inspection
    if test_df is not None:
        test_plot       = test_df.dropna(subset=[ll_col, pl_col]).copy()
        test_plot["Ip"] = test_plot[ll_col] - test_plot[pl_col]
        ax.scatter(
            test_plot[ll_col], test_plot["Ip"],
            color='#E5E7E9', label=cfg.get_label('test_data'),
            s=160, alpha=0.9, edgecolor='#2C3E50', marker='X', linewidth=1.5, zorder=12,
        )

    ax.set_title(cfg.get_label('casagrande_title'),   fontsize=16, fontweight="bold", pad=15, color="#2C3E50")
    ax.set_xlabel(cfg.get_label('liquid_limit'),       fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel(cfg.get_label('plasticity_index'),   fontsize=12, fontweight="bold", labelpad=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.legend(loc="upper left", bbox_to_anchor=(1.22, 1.0), frameon=True, borderpad=1, fontsize=10)

    sns.despine(fig)
    plt.tight_layout()
    plt.show()


def plot_categorical_impact(
    df: pd.DataFrame,
    category_col: str,
    cfg: CFG,
) -> None:
    """Create violin plots showing the effect of a categorical variable on Proctor targets.

    Args:
        df: DataFrame with Proctor target columns and the specified category column.
        category_col: Name of the categorical column (e.g. mold diameter).
        cfg: Configuration object providing localised labels.
    """
    rho_col = "proctor_mdd_g_cm3"
    w_col   = "proctor_owc_pct"

    plot_df = df.dropna(subset=[category_col, rho_col, w_col]).copy()
    plot_df[category_col] = plot_df[category_col].astype(str)

    cat_label = cfg.get_label(category_col)

    sns.set_theme(style="ticks", rc={"axes.grid": True, "grid.color": "#E5E7E9", "grid.linestyle": "--"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    sns.violinplot(
        data=plot_df, x=category_col, y=rho_col, ax=axes[0],
        hue=category_col, palette="Blues", legend=False, inner="quartile",
    )
    axes[0].set_title(cfg.get_label('cat_title_mdd'), fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel(cfg.get_label('target_mdd'), fontsize=11)
    axes[0].set_xlabel(cat_label, fontsize=11, fontweight="bold")

    sns.violinplot(
        data=plot_df, x=category_col, y=w_col, ax=axes[1],
        hue=category_col, palette="Oranges", legend=False, inner="quartile",
    )
    axes[1].set_title(cfg.get_label('cat_title_owc'), fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel(cfg.get_label('target_owc'), fontsize=11)
    axes[1].set_xlabel(cat_label, fontsize=11, fontweight="bold")

    sns.despine(fig)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: list,
    cfg: CFG,
) -> None:
    """Render a lower-triangular Pearson correlation heatmap.

    Args:
        df: DataFrame containing the columns to correlate.
        cols: List of column names to include in the correlation matrix.
        cfg: Configuration object providing the heatmap title label.
    """
    corr_matrix = df[cols].corr()
    # Mask the upper triangle (symmetric matrix)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, mask=mask, cmap="coolwarm", center=0,
        vmax=1.0, vmin=-1.0, annot=True, fmt=".2f",
        square=True, linewidths=.5, cbar_kws={"shrink": .7},
    )

    # Strip the common "psd_size_at_" prefix for cleaner tick labels
    clean_labels = [c.replace("psd_size_at_", "") for c in cols]
    plt.xticks(ticks=np.arange(len(cols)) + 0.5, labels=clean_labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(cols)) + 0.5, labels=clean_labels, rotation=0)

    plt.title(cfg.get_label('heatmap_title'), fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()


def plot_feature_vs_proctor_optima(
    df: pd.DataFrame,
    feature_col: str,
    cfg: CFG,
) -> None:
    """Bivariate regression scatter plots for one feature against both Proctor targets.

    Pearson correlation and p-value are annotated on each panel. A log x-axis
    is automatically applied when the feature name suggests a ratio (C_U or cu).

    Args:
        df: DataFrame containing the feature and both Proctor target columns.
        feature_col: Name of the independent feature column.
        cfg: Configuration object providing localised labels.

    Raises:
        KeyError: If any required column is absent from *df*.
    """
    rho_col = "proctor_mdd_g_cm3"
    w_col   = "proctor_owc_pct"

    # Automatically use log scale for ratio-type features
    log_x_scale   = "C_U" in feature_col or "cu" in feature_col.lower()
    feature_label = cfg.get_label(feature_col)

    missing = [c for c in [feature_col, rho_col, w_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns not found in DataFrame: {missing}")

    if log_x_scale and (df[feature_col] <= 0).any():
        log_x_scale = False  # Fallback: non-positive values prevent log scale

    plot_df = df[[feature_col, rho_col, w_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(plot_df) < 3:
        print(f"Insufficient valid data points for {feature_col} (n = {len(plot_df)}).")
        return

    sns.set_theme(style="ticks", rc={"axes.grid": True, "grid.color": "#E5E7E9", "grid.linestyle": "--"})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=120)

    for ax, y_col, color, line_col, txt_position in [
        (axes[0], rho_col, "#2980B9", "#2C3E50", "upper_right"),
        (axes[1], w_col,   "#E67E22", "#C0392B", "upper_left"),
    ]:
        sns.regplot(
            data=plot_df, x=feature_col, y=y_col, ax=ax, color=color,
            scatter_kws={"s": 50, "alpha": 0.5, "edgecolor": "white", "linewidths": 0.8},
            line_kws={"lw": 2.5, "color": line_col},
        )

        r, p     = pearsonr(plot_df[feature_col], plot_df[y_col])
        x_pos    = 0.95 if txt_position == "upper_right" else 0.05
        ha_align = "right" if txt_position == "upper_right" else "left"

        ax.text(
            x_pos, 0.95, f"r = {r:.2f}\np = {p:.3f}\nn = {len(plot_df)}",
            transform=ax.transAxes, fontsize=10, va="top", ha=ha_align, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#BDC3C7"),
        )

        if log_x_scale:
            ax.set_xscale("log")

        ax.set_xlabel(feature_label, fontsize=11, fontweight="bold")

    axes[0].set_title(cfg.get_label('influence_mdd'), fontsize=13, fontweight="bold", pad=12, color="#2C3E50")
    axes[1].set_title(cfg.get_label('influence_owc'), fontsize=13, fontweight="bold", pad=12, color="#2C3E50")
    axes[0].set_ylabel(cfg.get_label('target_mdd'), fontsize=11)
    axes[1].set_ylabel(cfg.get_label('target_owc'), fontsize=11)

    plt.suptitle(
        f"{cfg.get_label('bivariate_title')}: {feature_label}",
        fontsize=15, fontweight="bold", y=1.02, color="#2C3E50",
    )
    sns.despine(fig)
    plt.tight_layout()
    plt.show()


def plot_grain_density_distribution(
    df: pd.DataFrame,
    cfg: CFG,
    density_col: str = "grain_density_g_cm3",
    x_min: float | None = None,
    x_max: float | None = None,
) -> None:
    """Plot grain density histogram with mineral reference ranges.

    Patches that exceed the histogram y-limit (quartz-dominated bin) are
    visually clipped and annotated to avoid axis compression.

    Args:
        df: DataFrame containing the grain density column.
        cfg: Configuration object providing localised labels.
        density_col: Name of the grain density column. Default is
            ``'grain_density_g_cm3'``.
        x_min: Left x-axis limit. Inferred from data if ``None``.
        x_max: Right x-axis limit. Inferred from data if ``None``.
    """
    sns.set_theme(style="ticks", rc={
        "axes.facecolor": "#FFFFFF",
        "axes.grid": True,
        "grid.color": "#E5E7E9",
        "grid.linestyle": "--",
    })

    fig, ax = plt.subplots(figsize=(15, 9), dpi=120)

    data       = df[density_col].dropna()
    median_val = data.median()

    dist_min, dist_max = data.min(), data.max()
    x_min = min(2.5,  dist_min - 0.01) if x_min is None else x_min
    x_max = max(2.82, dist_max + 0.02) if x_max is None else x_max

    num_bins = 50
    counts, bins = np.histogram(data, bins=num_bins, range=(x_min, x_max))
    outside_main_peak = np.concatenate([counts[bins[:-1] < 2.64], counts[bins[:-1] > 2.66]])

    y_limit_hist    = max(outside_main_peak) * 1.3 if len(outside_main_peak) > 0 else 20
    y_limit_hist    = max(y_limit_hist, 15)
    y_limit_display = y_limit_hist * 2.8  # Extra headroom for mineral reference tracks

    sns.histplot(
        data, bins=num_bins, shrink=0.9, kde=True,
        color="#95A5A6", edgecolor="none", alpha=0.5,
        label=cfg.get_label('measured_data'), ax=ax,
    )

    if ax.lines:
        ax.lines[0].set_color("#2C3E50")
        ax.lines[0].set_linewidth(2.5)

    ax.set_ylim(0, y_limit_display)
    ax.set_xlim(x_min, x_max)

    ax.axvline(median_val, color="#2C3E50", linestyle="-", linewidth=2, zorder=10)
    ax.text(
        median_val, y_limit_hist * 0.25, f"Median: {median_val:.3f}",
        rotation=90, va='center', ha='right', fontsize=10, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#2C3E50', boxstyle='round,pad=0.3'),
        zorder=11,
    )

    # Visually clip the dominant quartz bin and annotate it
    for patch in ax.patches:
        if patch.get_x() <= 2.65 <= (patch.get_x() + patch.get_width()):
            if patch.get_height() > y_limit_hist:
                patch.set_alpha(0.3)
                patch.set_hatch("////")
                patch.set_edgecolor("#34495E")
                ax.annotate(
                    f"{cfg.get_label('quartz_dom')}\n(n = {int(patch.get_height())})",
                    xy=(patch.get_x() + patch.get_width() / 2, y_limit_hist * 0.95),
                    xytext=(0, -35), textcoords="offset points",
                    ha="center", va="top", fontsize=9, fontweight="bold", color="#C0392B",
                    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5),
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.1),
                )

    # Soil-type zone shading
    zones = [
        (2.63, 2.67, cfg.get_label('zone_sand'), "#F9E79F"),
        (2.67, 2.70, cfg.get_label('zone_silt'), "#EDBB99"),
        (2.70, 2.80, cfg.get_label('zone_clay'), "#F5B7B1"),
    ]
    for start, end, label, color in zones:
        if start < x_max and end > x_min:
            plot_start, plot_end = max(start, x_min), min(end, x_max)
            ax.axvspan(plot_start, plot_end, color=color, alpha=0.15, zorder=0)
            ax.text(
                (plot_start + plot_end) / 2, y_limit_hist * 1.15, label,
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#566573",
            )

    # Mineral reference tracks (range bars above histogram)
    track_space = y_limit_display - (y_limit_hist * 1.3)
    step        = track_space / 7
    t_base      = y_limit_display - (0.5 * step)

    minerals_range = [
        ("Kaolinit",       2.60, 2.64, t_base,           "#2980B9"),
        ("Chlorit",        2.70, 2.80, t_base,           "#1ABC9C"),
        ("Illit",          2.66, 2.68, t_base - step,    "#8E44AD"),
        ("Montmorillonit", 2.75, 2.78, t_base - step,    "#D35400"),
        (cfg.get_label('min_feldspar'), 2.55, 2.76, t_base - 2 * step, "#F39C12"),
        (cfg.get_label('min_mica'),     2.60, 3.20, t_base - 3 * step, "#7F8C8D"),
    ]
    minerals_point = [
        (cfg.get_label('min_gypsum'),  2.32, t_base - 4 * step, "#27AE60"),
        (cfg.get_label('min_calcite'), 2.71, t_base - 4 * step, "#16A085"),
        ("Dolomit",                    2.85, t_base - 4 * step, "#E67E22"),
        (cfg.get_label('min_quartz'),  2.65, t_base - 5 * step, "#C0392B"),
    ]

    for name, start, end, y_pos, color in minerals_range:
        if start > x_max or end < x_min:
            continue
        plot_start, plot_end = max(start, x_min), min(end, x_max)
        ax.hlines(y=y_pos, xmin=plot_start, xmax=plot_end, color=color, linewidth=6, alpha=0.7, zorder=4)
        ax.text(
            plot_start + (plot_end - plot_start) / 2, y_pos + (step * 0.08), name,
            ha="center", va="bottom", fontsize=8, color=color, fontweight="bold",
        )

    for name, val, y_pos, color in minerals_point:
        if x_min <= val <= x_max:
            ax.scatter(val, y_pos, color=color, marker="D", s=40, zorder=5)
            ax.text(
                val, y_pos + (step * 0.08), name,
                ha="center", va="bottom", fontsize=8, color=color, fontweight="bold",
            )

    ax.axhline(y_limit_hist * 1.3, color="#BDC3C7", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_title(cfg.get_label('density_adv_title'), fontsize=14, pad=15, fontweight="bold", loc="left", color="#2C3E50")
    ax.set_xlabel(cfg.get_label('density_x'), fontsize=11, labelpad=10)
    ax.set_ylabel(cfg.get_label('density_y'), fontsize=11, labelpad=10)

    sns.despine(ax=ax, trim=False)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    plt.tight_layout()
    plt.show()


def plot_interaction(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    cfg: CFG,
) -> None:
    """Scatter plot of two features with a continuous colour map on a third variable.

    The y-axis is rendered on a log scale, as the function is designed for
    hydraulic conductivity (k_f) data.

    Args:
        df: DataFrame containing the three specified columns.
        x: Column name for the x-axis (e.g. fines content).
        y: Column name for the y-axis (must be > 0; e.g. k_f).
        hue: Column name for continuous colour encoding (e.g. MDD).
        cfg: Configuration object providing localised labels.
    """
    plot_df = df[[x, y, hue]].dropna().copy()
    plot_df = plot_df[plot_df[y] > 0]  # Guard against log(0) on y-axis

    plt.figure(figsize=(11, 7), dpi=120)
    sns.set_theme(style="ticks", rc={"axes.grid": True, "grid.color": "#E5E7E9", "grid.linestyle": "--"})

    sns.scatterplot(
        data=plot_df, x=x, y=y, hue=hue, palette="viridis",
        size=hue, sizes=(40, 150), alpha=0.8, edgecolor="black", linewidth=0.5,
    )

    plt.yscale("log")
    plt.title(cfg.get_label('interaction_title'), fontsize=13, fontweight="bold", pad=15)
    plt.xlabel(cfg.get_label('fines_pct'), fontsize=11, fontweight="bold")
    plt.ylabel(cfg.get_label('log10_kf').replace("\\log_{10}(k_f)", "k_f"), fontsize=11, fontweight="bold")
    plt.legend(title=cfg.get_label('target_mdd'), bbox_to_anchor=(1.05, 1), loc='upper left')

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_kde_and_scatter(df: pd.DataFrame, cfg: CFG) -> None:
    """Joint scatter plot of Proctor optima with marginal KDE distributions.

    Also overlays zero-air-voids saturation lines for G_s = 2.65 and 2.75.

    Args:
        df: DataFrame with ``proctor_owc_pct`` and ``proctor_mdd_g_cm3`` columns.
        cfg: Configuration object providing localised labels.
    """
    owc_col = 'proctor_owc_pct'
    mdd_col = 'proctor_mdd_g_cm3'

    x_label = cfg.get_label('target_owc')
    y_label = cfg.get_label('target_mdd')

    # Three-panel layout: top KDE | main scatter | right KDE
    fig = plt.figure(figsize=(10, 8))
    gs  = fig.add_gridspec(3, 3, wspace=0.15, hspace=0.15)

    ax_main  = fig.add_subplot(gs[1:, :-1])
    ax_top   = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    # Main scatter
    sns.scatterplot(x=df[owc_col], y=df[mdd_col], ax=ax_main, alpha=0.6, color='dodgerblue', edgecolor='k')

    # Saturation (zero-air-voids) curves for two reference grain densities
    w_range = np.linspace(2, df[owc_col].max() * 1.1, 200)
    for Gs, ls in zip([2.65, 2.75], ['--', ':']):
        ax_main.plot(
            w_range, calc_satline(w_range, Gs), color='red',
            linestyle=ls, lw=1.5, label=f"{cfg.get_label('sat_line')} ($G_s={Gs}$)",
        )

    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)
    ax_main.legend(fontsize=9, loc='upper right')
    ax_main.grid(True, linestyle=':', alpha=0.7)

    # Top marginal: OWC distribution
    sns.kdeplot(x=df[owc_col], ax=ax_top, fill=True, color="salmon", alpha=0.7)
    ax_top.set_title(cfg.get_label('scatter_title'), fontweight='bold', pad=15)
    ax_top.set_ylabel(cfg.get_label('freq'))
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, linestyle=':', alpha=0.4)

    # Right marginal: MDD distribution
    sns.kdeplot(y=df[mdd_col], ax=ax_right, fill=True, color="skyblue", alpha=0.7)
    ax_right.set_xlabel(cfg.get_label('freq'))
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, linestyle=':', alpha=0.4)

    plt.show()


def plot_permeability_zones(
    df: pd.DataFrame,
    cfg: CFG,
    col_k: str = "hyd_cond_kf_m_s",
) -> None:
    """Histogram of log10(k_f) with DIN geotechnical permeability zone shading.

    Args:
        df: DataFrame containing the hydraulic conductivity column.
        cfg: Configuration object providing localised labels.
        col_k: Name of the k_f column. Default is ``'hyd_cond_kf_m_s'``.
    """
    if col_k not in df.columns:
        print(f"Column '{col_k}' not found.")
        return

    valid_k = pd.to_numeric(df[col_k].astype(str).str.replace(",", "."), errors="coerce").dropna()
    log_k   = np.log10(valid_k[valid_k > 0])

    if log_k.empty:
        print("No valid k_f values available.")
        return

    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#FFFFFF"})
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

    sns.histplot(log_k, bins=25, color="#7F8C8D", kde=True, alpha=0.5, edgecolor="black", ax=ax, zorder=2)

    # DIN permeability zones: clay → silt → sand → gravel
    zones = [
        (-12, -8, cfg.get_label('zone_clay'),   "#E74C3C"),
        (-8,  -4, cfg.get_label('zone_silt'),   "#F1C40F"),
        (-4,  -2, cfg.get_label('zone_sand'),   "#3498DB"),
        (-2,   0, cfg.get_label('zone_gravel'), "#2ECC71"),
    ]

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for min_val, max_val, label, color in zones:
        if min_val < log_k.max() and max_val > log_k.min():
            ax.axvspan(min_val, max_val, color=color, alpha=0.15, zorder=0)
            ax.text(
                (min_val + max_val) / 2, 0.95, label,
                transform=trans, ha='center', va='top',
                fontsize=11, fontweight='bold', color="#2C3E50",
            )

    ax.set_title(cfg.get_label('kf_zones_title'), fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(cfg.get_label('log10_kf'), fontsize=12, fontweight="bold")
    ax.set_ylabel(cfg.get_label('freq'), fontsize=12)
    ax.set_xlim(-12, -1)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()


def plot_psd_curves(df: pd.DataFrame, cfg: CFG) -> None:
    """Plot all particle size distribution (PSD) curves with fraction zone background.

    Reliable curves (with sedimentation) are drawn in blue; critical
    extrapolated curves (no sedimentation AND fines > 10 %) are highlighted
    in red.

    Args:
        df: DataFrame containing ``psd_size_at_d*_mm`` columns,
            ``psd_has_sedimentation``, and ``psd_passing_at_0_063mm_pct``.
        cfg: Configuration object providing localised labels.
    """
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)

    psd_passing   = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98]
    psd_size_cols = [c for c in df.columns if c.startswith("psd_size_at_d")]

    # Grain-size fraction background zones (DIN EN ISO 14688 palette)
    fractions = [
        {"name": "Ton",    "min": 1e-4, "max": 0.002, "color": "#E5E7E9"},
        {"name": "Schluff","min": 0.002,"max": 0.063,  "color": "#D7DBDD"},
        {"name": "Sand",   "min": 0.063,"max": 2.0,    "color": "#CACFD2"},
        {"name": "Kies",   "min": 2.0,  "max": 63.0,   "color": "#BDC3C7"},
        {"name": "Steine", "min": 63.0, "max": 200.0,  "color": "#A6ACAF"},
    ]
    en_map  = {"Ton": "Clay", "Schluff": "Silt", "Sand": "Sand", "Kies": "Gravel", "Steine": "Cobbles"}
    x_min, x_max = 0.001, 100.0
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    for frac in fractions:
        vis_min = max(x_min, frac["min"])
        vis_max = min(x_max, frac["max"])
        if vis_min >= vis_max:
            continue
        ax.axvspan(vis_min, vis_max, color=frac["color"], alpha=0.8, zorder=0)
        mid_val   = 10 ** ((np.log10(vis_min) + np.log10(vis_max)) / 2)
        frac_name = en_map.get(frac["name"], frac["name"]) if cfg.lang == 'en' else frac["name"]
        ax.text(
            mid_val, 1.02, frac_name, transform=trans,
            fontsize=11, fontweight="bold", color="#2C3E50",
            ha="center", va="bottom", zorder=5, clip_on=False,
        )

    # Classify curves: critical = no sedimentation AND fines > 10 %
    col_fines     = "psd_passing_at_0_063mm_pct"
    mask_critical = (df["psd_has_sedimentation"] == False) & (df[col_fines] > 10)
    mask_standard = ~mask_critical

    for _, row in df[mask_standard].iterrows():
        ax.plot(row[psd_size_cols].values, psd_passing, color="steelblue", alpha=0.3, lw=1.0, zorder=2)

    for _, row in df[mask_critical].iterrows():
        ax.plot(row[psd_size_cols].values, psd_passing, color="crimson", alpha=0.8, lw=1.8, zorder=4)

    # Legend dummies for the two categories
    ax.plot([], [], color="steelblue", lw=2, label=cfg.get_label('psd_standard'))
    ax.plot([], [], color="crimson",   lw=2, label=cfg.get_label('psd_extrapolated_high_fines'))

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 100)
    ax.set_title(f"{cfg.get_label('psd_title')} (n = {df.shape[0]})", fontsize=14, fontweight="bold", pad=25, color="#2C3E50")
    ax.set_xlabel(cfg.get_label('psd_x'), fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(cfg.get_label('psd_y'), fontsize=12, fontweight="bold", labelpad=10)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, axis="both", which="major", color="#FFFFFF", linestyle="-", linewidth=1.2, alpha=0.9, zorder=1)
    ax.legend(loc="lower right", fontsize=11, framealpha=1.0, facecolor="white", edgecolor="#BDC3C7")
    sns.despine(ax=ax, trim=False)

    plt.tight_layout()
    plt.show()


def plot_psd_outliers(df: pd.DataFrame, cfg: CFG) -> pd.Series:
    """Identify and plot anomalies in the coarse-grained fraction.

    Anomalies are defined as samples with 15 < C_U < 30 and MDD < 2.0 g/cm³,
    which may indicate gap-graded or organic soils.

    Args:
        df: DataFrame with ``psd_C_U`` and ``proctor_mdd_g_cm3`` columns.
        cfg: Configuration object providing localised labels.

    Returns:
        Boolean Series aligned with *df* marking anomalous rows as ``True``.
    """
    # Outlier mask on the full DataFrame (preserves original index alignment)
    is_outlier = (df["psd_C_U"] < 30) & (df["psd_C_U"] > 15) & (df["proctor_mdd_g_cm3"] < 2.0)

    # Restrict visualisation to C_U < 30 for readability
    df_vis         = df[df["psd_C_U"] < 30]
    is_outlier_vis = (df_vis["psd_C_U"] > 15) & (df_vis["proctor_mdd_g_cm3"] < 2.0)

    plt.figure(figsize=(10, 6), dpi=120)
    plt.scatter(
        df_vis.loc[~is_outlier_vis, "psd_C_U"],
        df_vis.loc[~is_outlier_vis, "proctor_mdd_g_cm3"],
        color="tab:gray", alpha=0.4, label=cfg.get_label('outlier_regular'), edgecolor="none",
    )
    plt.scatter(
        df_vis.loc[is_outlier_vis, "psd_C_U"],
        df_vis.loc[is_outlier_vis, "proctor_mdd_g_cm3"],
        color="crimson", alpha=0.9, edgecolor="black", linewidths=1.2, s=60, zorder=5,
        label=cfg.get_label('outlier_anomalies'),
    )

    plt.xlabel(cfg.get_label('psd_C_U'), fontsize=11, fontweight="bold")
    plt.ylabel(cfg.get_label('target_mdd'), fontsize=11)
    plt.title(cfg.get_label('outlier_title'), fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower left", framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return is_outlier


def plot_psd_params(df: pd.DataFrame, cfg: CFG) -> None:
    """Visualise d50, C_U, and C_C distributions and their mutual scatter.

    Creates a 2×2 figure:
      - Top-left:  d50 histogram on log scale with soil-type zone shading.
      - Top-right: C_U vs. C_C scatter (log-log axes) with well-graded zones.
      - Bottom-left:  C_U histogram.
      - Bottom-right: C_C histogram (clipped at 10 for readability).

    Args:
        df: DataFrame with PSD characteristic diameter columns.
        cfg: Configuration object providing localised labels.
    """
    d10 = df['psd_size_at_d10_mm']
    d30 = df['psd_size_at_d30_mm']
    d50 = df['psd_size_at_d50_mm']
    d60 = df['psd_size_at_d60_mm']

    work = pd.DataFrame()
    work['d50'] = d50
    work['cu']  = np.where(d10 > 0, d60 / d10, np.nan)
    work['cc']  = np.where((d10 > 0) & (d60 > 0), (d30 ** 2) / (d60 * d10), np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f9f9f9"})

    # Top-left: d50 distribution with soil-type background zones
    ax1 = axes[0, 0]
    sns.histplot(work['d50'].dropna(), log_scale=True, kde=True, color="mediumseagreen", ax=ax1, edgecolor="black")

    zones = [
        (1e-4, 0.002, cfg.get_label('soil_clay'),   "#E5E7E9"),
        (0.002, 0.063, cfg.get_label('soil_silt'),  "#D7DBDD"),
        (0.063, 2.0,   cfg.get_label('soil_sand'),  "#CACFD2"),
        (2.0,   63.0,  cfg.get_label('soil_gravel'), "#BDC3C7"),
    ]
    trans1 = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    for min_val, max_val, label, color in zones:
        ax1.axvspan(min_val, max_val, color=color, alpha=0.3, zorder=0)
        ax1.text(
            10 ** ((np.log10(min_val) + np.log10(max_val)) / 2), 0.95, label,
            transform=trans1, ha='center', va='top', fontsize=9, fontweight='bold', color="#555555",
        )

    ax1.set_title(cfg.get_label('d50_dist'), fontweight="bold")
    ax1.set_xlabel("$d_{50}$ [mm]")
    ax1.set_ylabel(cfg.get_label('freq'))

    # Top-right: C_U vs. C_C scatter (log-log) with well-graded highlight zones
    ax2 = axes[0, 1]
    sns.scatterplot(data=work, x='cu', y='cc', alpha=0.6, color="steelblue", ax=ax2, edgecolor="black")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.axvspan(6, work['cu'].max() * 1.5, color="#F1C40F", alpha=0.15, label="Well graded (C_U > 6)")
    ax2.axhspan(1, 3, color="#2ECC71", alpha=0.15, label="Well graded (1 < C_C < 3)")
    ax2.set_title(cfg.get_label('cu_cc_scatter'), fontweight="bold")
    ax2.set_xlabel("$C_U$ (log)")
    ax2.set_ylabel("$C_C$ (log)")
    ax2.legend(fontsize=9, loc='upper left')

    # Bottom-left: C_U distribution
    ax3 = axes[1, 0]
    sns.histplot(work['cu'].dropna(), log_scale=True, kde=True, color="#E67E22", ax=ax3, edgecolor="black")
    ax3.axvline(6, color="#E74C3C", ls="--", lw=2, label=r"$C_U=6$")
    ax3.set_title(cfg.get_label('cu_dist'), fontweight="bold")
    ax3.set_xlabel("$C_U$")
    ax3.set_ylabel(cfg.get_label('freq'))
    ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3.legend()

    # Bottom-right: C_C distribution (extreme outliers clipped for readability)
    ax4 = axes[1, 1]
    cc_max  = 10.0
    cc_data = work[work['cc'] <= cc_max]['cc']
    sns.histplot(cc_data, log_scale=False, kde=True, color="#9B59B6", ax=ax4, edgecolor="black")
    ax4.axvspan(1, 3, color="#2ECC71", alpha=0.2)
    ax4.set_title(cfg.get_label('cc_dist'), fontweight="bold")
    ax4.set_xlabel("$C_C$")
    ax4.set_ylabel(cfg.get_label('freq'))
    ax4.set_xlim(0, cc_max)

    plt.show()


def plot_target_distributions(
    df: pd.DataFrame,
    target_cols: list,
    cfg: CFG,
) -> None:
    """Render a 2×2 grid of histograms and boxplots for the Proctor target variables.

    Args:
        df: DataFrame containing the target columns.
        target_cols: List of two target column names (MDD and OWC).
        cfg: Configuration object providing localised labels.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axes   = axs.flatten()
    colors = ["salmon", "skyblue"]

    for i, target in enumerate(target_cols):
        # Resolve human-readable label from CFG (fall back to column name)
        target_name = cfg.get_label('target_owc') if 'owc' in target else cfg.get_label('target_mdd')

        # Top row: histograms with KDE
        sns.histplot(df[target], kde=True, bins=30, ax=axes[i], color=colors[i])
        axes[i].set_title(f"{cfg.get_label('dist_title')}: {target_name}")
        axes[i].set_xlabel(target_name)
        axes[i].set_ylabel(cfg.get_label('freq'))
        axes[i].grid(True)

        # Bottom row: boxplots
        idx = i + len(target_cols)
        sns.boxplot(x=df[target], ax=axes[idx], color=colors[i])
        axes[idx].set_title(f"{cfg.get_label('box_title')}: {target_name}")
        axes[idx].set_xlabel(target_name)
        axes[idx].set_ylabel(cfg.get_label('value'))
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()


def plot_top_correlations(
    df: pd.DataFrame,
    cfg: CFG,
    target_cols: list | None = None,
    top_n: int = 15,
) -> None:
    """Cleveland Dot Plot of Spearman correlations for the top predictors.

    Computes Spearman correlation for all numeric features against both Proctor
    targets, ranks them by the maximum absolute correlation, and visualises the
    top *top_n* features with connected dots for MDD and OWC.

    Args:
        df: DataFrame with feature and target columns.
        cfg: Configuration object providing localised labels.
        target_cols: List of two target column names. Defaults to the standard
            Proctor columns ``['proctor_mdd_g_cm3', 'proctor_owc_pct']``.
        top_n: Number of top features to display. Default is 15.
    """
    if target_cols is None:
        target_cols = ["proctor_mdd_g_cm3", "proctor_owc_pct"]

    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "0.9"})

    numeric_cols = df.select_dtypes(include=[np.number, bool]).columns
    corr_matrix  = df[numeric_cols].corr(method='spearman')
    target_corr  = corr_matrix[target_cols].copy()

    # Remove target variables themselves (and any suffixed duplicates)
    filter_mask = target_corr.index.str.contains("proctor_mdd|proctor_owc|target", case=False)
    target_corr = target_corr[~filter_mask]

    # Sort by maximum absolute correlation across both targets
    target_corr['max_abs_corr'] = target_corr[target_cols].abs().max(axis=1)
    top_features = target_corr.sort_values('max_abs_corr', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=120)
    y = np.arange(len(top_features))

    color_mdd = '#2c5d88'
    color_owc = '#b84a39'

    # Connecting lines ("dumbbell bar")
    ax.hlines(
        y=y,
        xmin=top_features[target_cols[1]],   # owc column
        xmax=top_features[target_cols[0]],   # mdd column
        color='#cccccc', alpha=0.7, linewidth=2.5, zorder=1,
    )

    ax.scatter(
        top_features[target_cols[1]], y,
        color=color_owc, s=130, label=cfg.get_label('top_corr_owc_label'),
        edgecolor='white', linewidth=1.2, zorder=3,
    )
    ax.scatter(
        top_features[target_cols[0]], y,
        color=color_mdd, s=130, label=cfg.get_label('top_corr_mdd_label'),
        edgecolor='white', linewidth=1.2, zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(top_features.index, fontweight='bold', fontsize=10)
    ax.axvline(0, color='#555555', linewidth=1.2, linestyle='--', alpha=0.5, zorder=0)
    ax.set_xlabel(cfg.get_label('top_corr_xlabel'), fontweight='bold', fontsize=11, labelpad=10)
    ax.set_xlim(-1.05, 1.05)

    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#dddddd')
    sns.despine(left=True, bottom=True)

    plt.title(
        cfg.get_label('top_corr_title').format(top_n=top_n),
        pad=25, fontsize=13, fontweight='bold', color='#222222',
    )
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=11)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 5. FEATURE ENGINEERING & PREPROCESSING
# ===========================================================================

def add_plasticity_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute geotechnical index features from Atterberg limits and PSD.

    Adds four new feature columns derived from the (post-MICE) imputed data:

    - ``atterberg_plasticity_index``: Plasticity Index PI = w_L − w_P
    - ``skempton_index``: Skempton Activity A = PI / clay fraction (0 for coarse soils)
    - ``hazen_proxy``: d10² proportional to hydraulic conductivity k_f
    - ``psd_C_U_is_low``: Binary indicator — 1 if C_U < 5 (poorly graded)

    Args:
        df: DataFrame with imputed Atterberg limit and PSD columns.

    Returns:
        Copy of *df* with the four additional columns appended.
    """
    df = df.copy()

    # Plasticity Index (PI = w_L - w_P)
    df["atterberg_plasticity_index"] = (
        df["atterberg_liquid_limit_pct"] - df["atterberg_plastic_limit_pct"]
    )

    # Skempton Activity (A = PI / clay fraction; safe division — coarse soils get 0)
    clay_fraction = (
        df["psd_passing_at_0_063mm_pct"] - df["psd_passing_at_0_002mm_pct"]
    ).replace(0, np.nan)
    df["skempton_index"] = (df["atterberg_plasticity_index"] / clay_fraction).fillna(0)

    # Hazen Proxy (d10² ∝ k_f after Hazen)
    df["hazen_proxy"] = df["psd_size_at_d10_mm"] ** 2

    # Uniformity Indicator (binary: 1 = poorly graded)
    df["psd_C_U_is_low"] = (df["psd_C_U"] < 5).astype(int)

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add DIN grain-size fractions and log-ratio PSD features relative to d10.

    Call after ``add_gradation_parameters`` (C_U/C_C) and before imputation or
    scaling.  Mirrors the test-data preparation in the submission cell so that
    both datasets go through identical feature construction.

    Adds
    ----
    - ``psd_fraction_clay/silt/sand/gravel`` — DIN 18196 fractions [%]
    - ``log_diff_psd_size_at_d*_mm``          — log₁₀(d_i / d10) for d20 … d98,
      encoding PSD shape and absolute position without multicollinearity

    Args:
        df: DataFrame with raw PSD sieve columns (``psd_passing_at_*``,
            ``psd_size_at_d*_mm``).

    Returns:
        Copy of *df* with the derived columns appended.
    """
    df = df.copy()

    # DIN 18196 grain-size fractions
    df["psd_fraction_clay"]   = df["psd_passing_at_0_002mm_pct"]
    df["psd_fraction_silt"]   = (df["psd_passing_at_0_063mm_pct"]
                                  - df["psd_fraction_clay"])
    df["psd_fraction_sand"]   = (df["psd_passing_at_2mm_pct"]
                                  - df["psd_passing_at_0_063mm_pct"])
    df["psd_fraction_gravel"] = 100.0 - df["psd_passing_at_2mm_pct"]

    # Log-ratio PSD features relative to d10 (shape + scale without raw collinearity)
    _d10 = df["psd_size_at_d10_mm"].clip(lower=1e-10)
    for _pct in [20, 30, 40, 50, 60, 70, 80, 90, 95, 98]:
        _col_raw = f"psd_size_at_d{_pct}_mm"
        if _col_raw in df.columns:
            df[f"log_diff_psd_size_at_d{_pct}_mm"] = (
                np.log10(df[_col_raw].clip(lower=1e-10)) - np.log10(_d10)
            )

    return df


def apply_fold_feature_engineering(df: pd.DataFrame) -> None:
    """Apply deterministic post-imputation feature engineering in-place for CV folds.

    Must be called AFTER:
    1. Missing-indicator columns (``*_is_missing``) have been set.
    2. MICE imputation has filled ``hyd_cond_kf_m_s``, ``atterberg_liquid_limit_pct``,
       and ``atterberg_plastic_limit_pct``.

    Assumes ``psd_fraction_clay`` (= passing at 0.002 mm), ``psd_C_U``, and
    ``log_diff_psd_size_at_d20_mm`` are already present (pre-computed before the CV loop).

    Modifies *df* in-place — no copy is created, matching the ``for df_fold in [X_tr, X_va]``
    iteration pattern used in the cross-validation loop.

    Computed features
    -----------------
    - ``log10_kf`` — log₁₀(k_f) recalculated from the now-imputed hydraulic conductivity
    - ``atterberg_plasticity_index`` — PI = w_L − w_P (after rule-based coarse-soil override)
    - ``psd_C_U_is_low`` — binary flag: 1 if C_U < 5 (poorly graded)
    - ``skempton_index`` — simplified A = PI × clay_pct / 100
    - ``hazen_interaction`` — log(k_f) − 2·log_diff_d20 (Hazen coupling term)
    - ``casagrande_interaction`` — PI − 0.73·(w_L − 20) (offset from Casagrande A-line)
    """
    # ── log10(kf) from imputed kf ────────────────────────────────────────────
    df["log10_kf"] = np.log10(df["hyd_cond_kf_m_s"].clip(lower=1e-15))

    # ── Geotechnical rule: coarse soils have no plastic limits ───────────────
    mask_coarse = (
        (df["psd_passing_at_0_063mm_pct"] < 15) & (df["atterberg_is_missing"] == 1)
    )
    df.loc[mask_coarse, ["atterberg_liquid_limit_pct", "atterberg_plastic_limit_pct"]] = 0.0

    # ── Plasticity Index ──────────────────────────────────────────────────────
    df["atterberg_plasticity_index"] = (
        df["atterberg_liquid_limit_pct"] - df["atterberg_plastic_limit_pct"]
    )

    # ── Binary uniformity indicator ───────────────────────────────────────────
    df["psd_C_U_is_low"] = (df["psd_C_U"] < 5).astype(int)

    # ── Skempton Activity (simplified: PI × clay_pct / 100) ──────────────────
    df["skempton_index"] = (df["atterberg_plasticity_index"] * df["psd_fraction_clay"]) / 100.0

    # ── Hazen coupling term ───────────────────────────────────────────────────
    df["hazen_interaction"] = (
        np.log10(df["hyd_cond_kf_m_s"].clip(lower=1e-15))
        - 2 * df["log_diff_psd_size_at_d20_mm"].fillna(0)
    )

    # ── Casagrande A-line offset ──────────────────────────────────────────────
    df["casagrande_interaction"] = (
        df["atterberg_plasticity_index"] - 0.73 * (df["atterberg_liquid_limit_pct"] - 20)
    )


def calculate_vif(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Compute the Variance Inflation Factor (VIF) for the given features.

    A constant column is added before computing VIF, which is required for
    the intercept term and produces numerically correct results.

    Args:
        df: DataFrame containing the feature columns.
        columns: List of feature column names to evaluate.

    Returns:
        DataFrame with columns ``'Feature'`` and ``'VIF'``, sorted descending
        by VIF, with the added constant row excluded.
    """
    X = df[columns].dropna()
    X = add_constant(X)  # Required for a correct VIF calculation

    vif_data = pd.DataFrame({
        "Feature": X.columns,
        "VIF":     [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })

    # Remove the constant row from the output
    return vif_data[vif_data["Feature"] != "const"].sort_values(by="VIF", ascending=False)


def evaluate_imputation_impact(
    train_df_raw: pd.DataFrame,
    df_imputed_initial: pd.DataFrame,
    df_imputed_with_features: pd.DataFrame,
    target_cols: list,
    seed: int = 42,
    cfg_class: CFG | None = None,
) -> tuple:
    """Benchmark three imputation / feature-engineering stages with HistGradientBoosting.

    Runs 10-fold cross-validation on each stage and prints a comparison table.

    Args:
        train_df_raw: Original training data with native NaN values.
        df_imputed_initial: Data after MICE only (no extra engineered features).
        df_imputed_with_features: Data after MICE + domain feature engineering.
        target_cols: List of two Proctor target column names.
        seed: Random seed for reproducibility.
        cfg_class: Configuration object for localised output messages.

    Returns:
        Tuple of ``(mae_mdd_base, mae_owc_base, mae_mdd_imp, mae_owc_imp,
        mae_mdd_all, mae_owc_all)``.
    """
    sota_model = HistGradientBoostingRegressor(random_state=seed)

    features_raw = [c for c in train_df_raw.columns              if c not in target_cols]
    features_imp = [c for c in df_imputed_initial.columns        if c not in target_cols]
    features_all = [c for c in df_imputed_with_features.columns  if c not in target_cols]

    mae_mdd_base, mae_owc_base = test_with_model(
        train_df_raw[features_raw + target_cols], model=sota_model, seed=seed
    )
    mae_mdd_imp, mae_owc_imp = test_with_model(
        df_imputed_initial[features_imp + target_cols], model=sota_model, seed=seed
    )
    mae_mdd_all, mae_owc_all = test_with_model(
        df_imputed_with_features[features_all + target_cols], model=sota_model, seed=seed
    )

    if cfg_class is not None:
        print(cfg_class.get_label('imp_eval_title'))
        print(cfg_class.get_label('imp_base').format(mdd=mae_mdd_base, owc=mae_owc_base))
        print(cfg_class.get_label('imp_mice').format(mdd=mae_mdd_imp,  owc=mae_owc_imp))
        print(cfg_class.get_label('imp_feat').format(mdd=mae_mdd_all,  owc=mae_owc_all))

    return mae_mdd_base, mae_owc_base, mae_mdd_imp, mae_owc_imp, mae_mdd_all, mae_owc_all


def impute_missing_values(
    df: pd.DataFrame,
    columns_for_imputation: list | None = None,
    target_cols: list | None = None,
    seed: int = 42,
    cfg_class: CFG | None = None,
) -> pd.DataFrame:
    """Impute missing values using MICE, then apply physical domain rules.

    The two-step procedure:
      1. Run MICE (IterativeImputer with ExtraTrees) on the uncorrupted data.
      2. Override Atterberg limits of coarse-grained samples (fines < 15 %) with 0.0.

    Missing-indicator columns (``*_is_missing``) are appended before imputation
    to preserve the missingness signal for downstream models.

    Args:
        df: Input DataFrame (not modified in place).
        columns_for_imputation: Columns to impute. Defaults to all columns
            with missing values that are not target columns.
        target_cols: Target columns excluded from imputation.
        seed: Random seed for the IterativeImputer.
        cfg_class: Configuration object for localised status messages.

    Returns:
        Copy of *df* with imputed values and appended missing-indicator columns.
    """
    df_impute   = df.copy()
    target_cols = target_cols or []

    if columns_for_imputation is None:
        columns_for_imputation = [
            col for col in df.columns
            if df[col].isnull().sum() > 0 and col not in target_cols
        ]

    # Persist missing-indicator flags before imputation corrupts the signal
    df_impute['atterberg_is_missing'] = df_impute['atterberg_liquid_limit_pct'].isnull().astype(int)
    df_impute['kf_is_missing']        = df_impute['hyd_cond_kf_m_s'].isnull().astype(int)
    df_impute['loi_is_missing']       = df_impute['loss_on_ignition_pct'].isnull().astype(int)

    # Identify coarse-grained soils: fines < 15 % and Atterberg limits absent
    mask_coarse = (
        (df_impute['psd_passing_at_0_063mm_pct'] < 15) &
        df_impute['atterberg_liquid_limit_pct'].isnull()
    )

    # Step 1: MICE on uncorrupted data
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=50, random_state=seed, n_jobs=-1),
        max_iter=10,
        random_state=seed,
        min_value=0.0,
    )
    imputed_array    = imputer.fit_transform(df_impute[columns_for_imputation])
    imputed_features = pd.DataFrame(imputed_array, columns=columns_for_imputation, index=df_impute.index)

    for col in columns_for_imputation:
        df_impute[col] = imputed_features[col]

    # Step 2: Apply physical rule — coarse-grained soils are non-plastic
    cols_atterberg = ['atterberg_liquid_limit_pct', 'atterberg_plastic_limit_pct']
    df_impute.loc[mask_coarse, cols_atterberg] = 0.0

    if cfg_class is not None and hasattr(cfg_class, 'get_label'):
        print(cfg_class.get_label('imp_mice_done'))
        print(cfg_class.get_label('imp_coarse_zero').format(count=mask_coarse.sum()))
    else:
        print("-> MICE imputation completed without data corruption.")
        print(f"-> Afterwards, {mask_coarse.sum()} rows (coarse-grained soils) set to 0.0.")

    return df_impute


def run_distribution_diagnostics(
    df: pd.DataFrame,
    exclude_cols: list,
    redundant_cols: list,
) -> tuple:
    """Compute skewness and IQR-based outlier rates for all relevant numeric columns.

    Renders a 1×2 bar chart: left panel shows absolute skewness (threshold at
    1.5), right panel shows outlier percentage (threshold at 3 %).

    Args:
        df: DataFrame to analyse.
        exclude_cols: Columns to exclude (e.g. targets).
        redundant_cols: Additional columns to exclude (e.g. duplicates).

    Returns:
        Tuple of ``(df_metrics, num_cols)`` where *df_metrics* contains the
        diagnostic statistics and *num_cols* is the list of evaluated columns.
    """
    num_cols    = [c for c in df.columns if c not in exclude_cols + redundant_cols]
    diagnostics = []

    for col in num_cols:
        skew     = df[col].skew()
        q25, q75 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr      = q75 - q25
        outliers = (
            ((df[col] < (q25 - 1.5 * iqr)) | (df[col] > (q75 + 1.5 * iqr))).sum()
            if iqr > 0 else 0
        )
        diagnostics.append({
            'Feature':     col,
            'Abs_Skew':    abs(skew),
            'Outlier_Pct': (outliers / len(df)) * 100,
        })

    df_metrics = pd.DataFrame(diagnostics).sort_values(by='Abs_Skew', ascending=False)

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: skewness bar chart with threshold line at 1.5
    colors_skew = ['#b84a39' if x > 1.5 else '#2c5d88' for x in df_metrics['Abs_Skew']]
    sns.barplot(x='Abs_Skew', y='Feature', data=df_metrics, ax=ax1, palette=colors_skew, hue='Feature', legend=False)
    ax1.axvline(1.5, color='#333333', linestyle='--')
    ax1.set_title('Distribution Skewness (Red = Strongly Asymmetric)', fontsize=12, fontweight='bold')

    # Right: outlier share bar chart with threshold line at 3 %
    df_outliers = df_metrics.sort_values(by='Outlier_Pct', ascending=False)
    colors_out  = ['#e67e22' if x > 3.0 else '#27ae60' for x in df_outliers['Outlier_Pct']]
    sns.barplot(x='Outlier_Pct', y='Feature', data=df_outliers, ax=ax2, palette=colors_out, hue='Feature', legend=False)
    ax2.axvline(3.0, color='#333333', linestyle='--')
    ax2.set_title('Outlier Share (IQR Method, > 3% = Elevated)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return df_metrics, num_cols


def test_with_model(
    train_data: pd.DataFrame,
    target_columns: list | None = None,
    model=None,
    seed: int = 42,
    n_splits: int = 10,
) -> tuple:
    """Evaluate a model via K-fold cross-validation and return median MAEs.

    Args:
        train_data: DataFrame containing features and target columns.
        target_columns: Names of the two Proctor target columns. Defaults to
            ``['proctor_mdd_g_cm3', 'proctor_owc_pct']``.
        model: Scikit-learn estimator. Defaults to ``LinearRegression()``.
        seed: Random seed for the K-fold split.
        n_splits: Number of cross-validation folds.

    Returns:
        Tuple of ``(median_mae_mdd, median_mae_owc)``.
    """
    if target_columns is None:
        target_columns = ["proctor_mdd_g_cm3", "proctor_owc_pct"]

    if model is None:
        model = LinearRegression()

    X     = train_data.drop(columns=target_columns)
    y_mdd = train_data[target_columns[0]]
    y_owc = train_data[target_columns[1]]

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores_mdd = -cross_val_score(model, X, y_mdd, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
    scores_owc = -cross_val_score(model, X, y_owc, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)

    return float(np.median(scores_mdd)), float(np.median(scores_owc))


def visualize_imputation_strategies(
    df_raw: pd.DataFrame,
    df_imputed: pd.DataFrame,
    feature_missing: str = "atterberg_liquid_limit_pct",
    feature_ref: str = "proctor_owc_pct",
    cfg_class: CFG | None = None,
) -> None:
    """Diagnostic scatter plot comparing original, MICE-imputed, and rule-set values.

    Args:
        df_raw: Original DataFrame before imputation (used to identify missing rows).
        df_imputed: DataFrame after the two-step imputation procedure.
        feature_missing: Feature column that was imputed (y-axis).
        feature_ref: Reference feature (x-axis) for visual context.
        cfg_class: Configuration object for localised labels and titles.
    """
    plot_df = df_imputed.copy()

    # Derive masks from the original (pre-imputation) data to avoid contamination
    initial_missing_mask = df_raw[feature_missing].isna()
    mask_coarse = initial_missing_mask & (plot_df[feature_missing] == 0.0)   # rule-set to zero
    mask_mice   = initial_missing_mask & (plot_df[feature_missing] > 0.0)    # MICE estimate

    sns.set_theme(style="ticks", rc={"axes.grid": True, "grid.color": "#f0f0f0", "grid.linestyle": "-"})
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)

    lbl_orig = cfg_class.get_label('diag_lbl_orig') if cfg_class else "Original values"
    lbl_mice = cfg_class.get_label('diag_lbl_mice') if cfg_class else "Imputed (MICE)"
    lbl_rule = cfg_class.get_label('diag_lbl_rule') if cfg_class else "Rule-based (0.0)"

    # Original (non-missing) data points in background
    sns.scatterplot(
        data=plot_df[~initial_missing_mask], x=feature_ref, y=feature_missing,
        ax=ax, color='#2c5d88', alpha=0.5, s=45, label=lbl_orig,
    )

    # MICE-estimated values
    if mask_mice.sum() > 0:
        sns.scatterplot(
            data=plot_df[mask_mice], x=feature_ref, y=feature_missing,
            ax=ax, color='#e67e22', marker="X", s=100, alpha=0.9, label=lbl_mice,
        )

    # Rule-set zeros (coarse-grained, non-plastic soils)
    if mask_coarse.sum() > 0:
        sns.scatterplot(
            data=plot_df[mask_coarse], x=feature_ref, y=feature_missing,
            ax=ax, color='#b84a39', marker="o", s=55, alpha=0.8, label=lbl_rule,
        )

    title_str = (
        cfg_class.get_label('diag_imp_title').format(feature=feature_missing)
        if cfg_class else f"Imputation: {feature_missing}"
    )
    xlbl_str = (
        cfg_class.get_label('diag_ref_label').format(ref=feature_ref)
        if cfg_class else feature_ref
    )

    ax.set_title(title_str, fontsize=12, fontweight="bold", pad=15, loc='left')
    ax.set_xlabel(xlbl_str, fontsize=10, fontweight="bold")
    ax.set_ylabel(feature_missing, fontsize=10, fontweight="bold")

    ax.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0', loc='upper right')
    sns.despine()
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 6. SCALING ANALYSIS
# ===========================================================================

def get_scaler_assignments(
    df: pd.DataFrame,
    df_metrics: pd.DataFrame,
    num_cols: list,
    skew_threshold: float = 1.5,
) -> tuple:
    """Assign numeric features to the appropriate scaler based on domain knowledge.

    Three groups are returned for use with ``get_column_preprocessor``:
      - Highly skewed → PowerTransformer (Yeo-Johnson)
      - Physical outliers → RobustScaler
      - All others → StandardScaler

    Features that are already log-transformed (``log_diff_*`` prefix,
    or specific interaction terms) are protected from double transformation.

    Args:
        df: Training DataFrame (used to identify log-prefixed columns).
        df_metrics: Diagnostics table from ``run_distribution_diagnostics``
            with ``'Feature'`` and ``'Abs_Skew'`` columns.
        num_cols: Pool of candidate numeric columns.
        skew_threshold: Absolute skewness above which PowerTransformer is used.

    Returns:
        Tuple of ``(features_highly_skewed, features_with_outliers, features_standard)``.
    """
    # Fixed outlier-prone features based on domain expectations
    features_with_outliers = ['grain_density_g_cm3', 'loss_on_ignition_pct']

    # Protect already-transformed features from double transformation
    protected_features = (
        ['hazen_interaction', 'casagrande_interaction', 'psd_fraction_gravel'] +
        [col for col in df.columns if col.startswith('log_diff_')]
    )

    check_skew_pool = [
        col for col in num_cols
        if col not in features_with_outliers + protected_features
    ]

    features_highly_skewed = df_metrics[
        (df_metrics['Feature'].isin(check_skew_pool)) &
        (df_metrics['Abs_Skew'] > skew_threshold)
    ]['Feature'].tolist()

    features_standard = [
        col for col in num_cols
        if col not in features_highly_skewed + features_with_outliers
    ]

    return features_highly_skewed, features_with_outliers, features_standard


def build_scaled_dataframe(
    df_imputed: pd.DataFrame,
    exclude_cols: list,
    target_cols: list,
    redundant_cols: list | None = None,
    cfg_class: CFG | None = None,
) -> tuple:
    """Fit a ColumnTransformer on the imputed data and return the scaled DataFrame.

    Orchestrates the full scaling pipeline in one call:
    ``run_distribution_diagnostics`` → ``get_scaler_assignments`` →
    ``get_column_preprocessor`` → ``fit_transform``.

    Binary indicator columns (``*_is_missing``, ``psd_has_sedimentation``, etc.)
    stay in *exclude_cols* so they are excluded from diagnostics but are still
    passed through the ``ColumnTransformer`` via ``remainder='passthrough'``.

    Args:
        df_imputed: Feature DataFrame after imputation and engineering.
        exclude_cols: Columns excluded from distribution diagnostics
            (binary indicators, ``'id'``). Do **not** include target columns here.
        target_cols: Proctor target columns — excluded from both diagnostics
            and the fit input matrix.
        redundant_cols: Additional columns to exclude from the scaler pool
            (e.g. highly correlated cumulative PSD columns). Default is ``None``.
        cfg_class: Configuration object for localised log messages.

    Returns:
        Tuple of ``(df_scaled, preprocessor, features_highly_skewed,
        features_with_outliers, features_standard)`` where *df_scaled* is a
        DataFrame with the same index as *df_imputed*.
    """
    redundant_cols = redundant_cols or []
    all_diagnostic_excludes = exclude_cols + target_cols

    df_metrics, num_cols = run_distribution_diagnostics(
        df_imputed, all_diagnostic_excludes, redundant_cols
    )
    features_highly_skewed, features_with_outliers, features_standard = get_scaler_assignments(
        df_imputed, df_metrics, num_cols
    )

    # Guard: only keep columns present in df_imputed and not in exclude set
    features_standard = [
        c for c in features_standard
        if c in df_imputed.columns and c not in all_diagnostic_excludes
    ]

    if cfg_class is not None:
        print(cfg_class.get_label('scale_summary'))
        print(cfg_class.get_label('scale_pt').format(features=features_highly_skewed))
        print(cfg_class.get_label('scale_robust').format(features=features_with_outliers))
        print(cfg_class.get_label('scale_std').format(count=len(features_standard)))

    preprocessor = get_column_preprocessor(
        features_highly_skewed, features_with_outliers, features_standard
    )

    # Build feature matrix: drop targets + 'id'; binary indicators stay → passthrough
    drop_for_fit = [c for c in target_cols + ['id'] if c in df_imputed.columns]
    X_features   = df_imputed.drop(columns=drop_for_fit, errors='ignore')

    X_scaled_array       = preprocessor.fit_transform(X_features)
    scaled_feature_names = [name.split("__")[-1] for name in preprocessor.get_feature_names_out()]
    df_scaled = pd.DataFrame(X_scaled_array, columns=scaled_feature_names, index=df_imputed.index)

    if cfg_class is not None:
        print("\n" + cfg_class.get_label('scale_success').format(shape=df_scaled.shape))

    return df_scaled, preprocessor, features_highly_skewed, features_with_outliers, features_standard


# ===========================================================================
# 7. ML PIPELINE
# ===========================================================================

def get_column_preprocessor(
    highly_skewed: list,
    with_outliers: list,
    standard: list,
) -> ColumnTransformer:
    """Build a ColumnTransformer that applies the appropriate scaler to each group.

    Args:
        highly_skewed: Features to be transformed with PowerTransformer (Yeo-Johnson).
        with_outliers: Features to be scaled with RobustScaler.
        standard: Features to be scaled with StandardScaler.

    Returns:
        Unfitted ``ColumnTransformer`` instance.
    """
    return ColumnTransformer(
        transformers=[
            ('skewed',   PowerTransformer(method='yeo-johnson'), highly_skewed),
            ('outliers', RobustScaler(),                          with_outliers),
            ('standard', StandardScaler(),                        standard),
        ],
        remainder='passthrough',
    )


def get_default_mice_imputer(seed: int = 42) -> IterativeImputer:
    """Create a pre-configured MICE imputer backed by ExtraTreesRegressor.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Unfitted ``IterativeImputer`` instance.
    """
    return IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=50, random_state=seed, n_jobs=-1),
        max_iter=10,
        random_state=seed,
        min_value=0.0,
    )


def compute_shap_and_plot(
    X_base: pd.DataFrame,
    best_fold_artifacts: dict,
    cols_for_imputation: list,
) -> None:
    """Compute SHAP values with fold-1 artifacts and render beeswarm plots.

    Replicates the CV preprocessing pipeline (missing indicators → MICE
    transform → feature engineering → optional scaling) on the full training
    set before computing SHAP via ``shap.TreeExplainer``.

    Args:
        X_base: Full training feature matrix (before preprocessing).
        best_fold_artifacts: Dict with keys ``'fitted_imputer'``,
            ``'fitted_scaler'`` (or ``None``), and ``'fitted_model'``.
        cols_for_imputation: Columns passed to the fold-specific MICE imputer.
    """
    import shap

    print("⏳ Berechne SHAP-Werte für das Gewinner-Modell ...")
    final_ml_model     = best_fold_artifacts['fitted_model']
    preprocessor_final = best_fold_artifacts['fitted_scaler']
    imputer_final      = best_fold_artifacts['fitted_imputer']

    X_final = X_base.copy()
    X_final['atterberg_is_missing'] = X_final['atterberg_liquid_limit_pct'].isnull().astype(int)
    X_final['kf_is_missing']        = X_final['hyd_cond_kf_m_s'].isnull().astype(int)
    X_final['loi_is_missing']       = X_final['loss_on_ignition_pct'].isnull().astype(int)

    X_final[cols_for_imputation] = imputer_final.transform(X_final[cols_for_imputation])
    apply_fold_feature_engineering(X_final)

    if preprocessor_final is not None:
        X_final_proc        = preprocessor_final.transform(X_final)
        feature_names_clean = [n.split("__")[-1] for n in preprocessor_final.get_feature_names_out()]
    else:
        X_final_proc        = X_final.values
        feature_names_clean = X_final.columns.tolist()

    X_explain         = pd.DataFrame(X_final_proc, columns=feature_names_clean, index=X_final.index)
    explainer         = shap.TreeExplainer(final_ml_model)
    shap_values_multi = explainer(X_explain)

    for target_idx, target_label in [
        (0, "Maximale Trockendichte – MDD [g/cm³]"),
        (1, "Optimaler Wassergehalt – OWC [%]"),
    ]:
        plt.figure(figsize=(10, 5.5))
        plt.title(f"SHAP Feature Importance: {target_label}", fontsize=12, fontweight='bold', pad=15)
        shap.plots.beeswarm(shap_values_multi[..., target_idx], max_display=12, show=False)
        plt.xlabel("SHAP-Wert (Einfluss auf Vorhersage)", fontweight='bold')
        plt.tight_layout()
        plt.show()


def generate_oof_predictions(
    X_base: pd.DataFrame,
    y_base: pd.DataFrame,
    kf,
    best_model_template,
    best_is_scaled: bool,
    cols_for_imputation: list,
    features_highly_skewed: list,
    features_with_outliers: list,
    features_standard: list,
    seed: int = 42,
) -> np.ndarray:
    """Generate out-of-fold predictions for the winning CV configuration.

    Runs one pass of fold-isolated preprocessing (missing indicators → MICE →
    feature engineering → optional scaling) for the supplied model template,
    matching the logic of the main CV loop exactly.

    Args:
        X_base: Feature matrix (same as used in the main CV loop).
        y_base: Target DataFrame with MDD and OWC columns.
        kf: Configured ``KFold`` instance.
        best_model_template: Unfitted estimator for the winning configuration.
        best_is_scaled: Whether to apply the ``ColumnTransformer`` preprocessor.
        cols_for_imputation: Columns passed to the MICE imputer.
        features_highly_skewed: PowerTransformer column list.
        features_with_outliers: RobustScaler column list.
        features_standard: StandardScaler column list.
        seed: Random seed for the MICE imputer.

    Returns:
        Array of shape ``(n_samples, n_targets)`` with OOF predictions.
    """
    from sklearn.base import clone

    oof_preds = np.zeros((len(X_base), y_base.shape[1]))
    for train_idx, val_idx in kf.split(X_base):
        X_tr, X_va = X_base.iloc[train_idx].copy(), X_base.iloc[val_idx].copy()
        y_tr       = y_base.iloc[train_idx]

        for df_fold in [X_tr, X_va]:
            df_fold['atterberg_is_missing'] = df_fold['atterberg_liquid_limit_pct'].isnull().astype(int)
            df_fold['kf_is_missing']        = df_fold['hyd_cond_kf_m_s'].isnull().astype(int)
            df_fold['loi_is_missing']       = df_fold['loss_on_ignition_pct'].isnull().astype(int)

        imputer = get_default_mice_imputer(seed=seed)
        X_tr[cols_for_imputation] = imputer.fit_transform(X_tr[cols_for_imputation])
        X_va[cols_for_imputation] = imputer.transform(X_va[cols_for_imputation])

        for df_fold in [X_tr, X_va]:
            apply_fold_feature_engineering(df_fold)

        if best_is_scaled:
            _valid         = set(X_tr.columns)
            _feat_skewed   = [c for c in features_highly_skewed  if c in _valid]
            _feat_outliers = [c for c in features_with_outliers   if c in _valid]
            _feat_standard = [c for c in features_standard        if c in _valid]
            preprocessor   = get_column_preprocessor(_feat_skewed, _feat_outliers, _feat_standard)
            X_tr_proc = preprocessor.fit_transform(X_tr)
            X_va_proc = preprocessor.transform(X_va)
        else:
            X_tr_proc, X_va_proc = X_tr.values, X_va.values

        model = clone(best_model_template)
        model.fit(X_tr_proc, y_tr)
        oof_preds[val_idx] = model.predict(X_va_proc)

    return oof_preds


def predict_korfiatis_mdd(row: pd.Series) -> float:
    """Estimate Proctor MDD with the Korfiatis & Manikopoulos (1982) formula.

    Valid only for granular soils where the saturation parameter
    ``s = 1 / sqrt(C_U) > 0.2``.  Requires ``s_parameter`` to be pre-computed
    (``1 / sqrt(psd_C_U)``) before calling ``df.apply``.

    Args:
        row: A row from an empirical comparison DataFrame containing
            ``s_parameter``, ``psd_passing_at_0_063mm_pct``, and optionally
            ``grain_density_g_cm3`` (defaults to 2.65 if missing).

    Returns:
        Estimated MDD in g/cm³, or ``np.nan`` outside validity range.
    """
    s   = row.get("s_parameter")
    FF  = row.get("psd_passing_at_0_063mm_pct")
    G_s = row.get("grain_density_g_cm3")
    if pd.isna(s) or pd.isna(FF) or s <= 0.2:
        return np.nan
    G_s = 2.65 if pd.isna(G_s) else float(G_s)
    t   = FF / 100.0
    a, b, c, d, q = 0.6682, 0.0, 0.8565, 0.3282, 0.7035
    eta = (c - d * s) if 0.2 < s <= 0.5738 else (a - b * s)
    if eta <= 0:
        return np.nan
    return G_s / (((1.0 - t) / eta) + (t / q))


def plot_feature_engineering_comparison(
    scores_dict: dict,
    cfg_class: CFG,
    figsize: tuple = (14, 5),
    y_mins: tuple | None = None,
) -> None:
    """Bar chart comparison of CV errors across different feature engineering setups.

    Args:
        scores_dict: Mapping of configuration label → ``(mae_mdd, mae_owc)`` tuple.
        cfg_class: Configuration object providing localised axis labels.
        figsize: Figure size as ``(width, height)`` in inches.
        y_mins: Optional lower bounds ``(min_mdd, min_owc)`` for the y-axes.
            When ``None``, y-axes start at 0.
    """
    labels  = list(scores_dict.keys())
    mae_mdd = [v[0] for v in scores_dict.values()]
    mae_owc = [v[1] for v in scores_dict.values()]

    x = np.arange(len(labels))
    sns.set_theme(style="white")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    color_mdd, color_owc = '#2c5d88', '#b84a39'

    # Left panel: MDD MAE
    bars1 = ax1.bar(x, mae_mdd, color=color_mdd, width=0.5, alpha=0.85)
    ax1.set_title(cfg_class.get_label('mae_mdd_title'), fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylabel(cfg_class.get_label('error_mdd_ylabel'), fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
    sns.despine(ax=ax1)

    # Right panel: OWC MAE
    bars2 = ax2.bar(x, mae_owc, color=color_owc, width=0.5, alpha=0.85)
    ax2.set_title(cfg_class.get_label('mae_owc_title'), fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylabel(cfg_class.get_label('error_owc_ylabel'), fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
    sns.despine(ax=ax2)

    if y_mins is not None:
        ax1.set_ylim(y_mins[0], max(mae_mdd) * 1.15)
        ax2.set_ylim(y_mins[1], max(mae_owc) * 1.15)
    else:
        ax1.set_ylim(0, max(mae_mdd) * 1.15)
        ax2.set_ylim(0, max(mae_owc) * 1.15)

    # Value annotations on each bar
    for ax, bars in zip([ax1, ax2], [bars1, bars2]):
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + (ax.get_ylim()[1] * 0.015),
                f'{yval:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
            )

    fig.suptitle(cfg_class.get_label('feat_eng_title'), fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()


def plot_pipeline_comparison(df_folds: list | pd.DataFrame) -> None:
    """Validation dashboard comparing scaled vs. unscaled pipeline configurations.

    Creates a 1×2 figure:
      - Left: Mean NMAE bar chart per model and scaling status.
      - Right: Fold-level NMAE boxplot with dynamic y-axis clipping to avoid
        visual compression caused by extreme outlier folds.

    Args:
        df_folds: List of per-fold result dicts, or a pre-assembled DataFrame.
            Expected columns: ``'Modell'`` (or ``'Modellkonfiguration'``),
            ``'Skaliert'`` (bool), ``'NMAE'`` (or ``'NMAE (Hauptmetrik)'``).
    """
    df_plot = pd.DataFrame(df_folds) if isinstance(df_folds, list) else df_folds.copy()

    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "0.9"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=120)

    palette = {
        'Skaliert (Mit Pipeline)': '#2c5d88',
        'Unskaliert (Rohdaten)':   '#b84a39',
    }

    # Normalise column names for robustness across notebook versions
    if 'NMAE (Hauptmetrik)' in df_plot.columns:
        df_plot = df_plot.rename(columns={'NMAE (Hauptmetrik)': 'NMAE'})
    if 'Modell' not in df_plot.columns and 'Modellkonfiguration' in df_plot.columns:
        df_plot = df_plot.rename(columns={'Modellkonfiguration': 'Modell'})

    df_plot['Status'] = df_plot['Skaliert'].map({True: 'Skaliert (Mit Pipeline)', False: 'Unskaliert (Rohdaten)'})
    df_means = df_plot.groupby(['Modell', 'Status'])['NMAE'].mean().reset_index()

    # Left: mean NMAE bar chart
    sns.barplot(
        data=df_means, x='Modell', y='NMAE', hue='Status',
        palette=palette, alpha=0.85, ax=ax1, errorbar=None,
    )
    for bar in ax1.patches:
        height = bar.get_height()
        if height > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0, height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#444444',
            )

    ax1.set_title("Mean Total Model Error (NMAE)", fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel("Model Architecture", fontweight='bold', labelpad=10)
    ax1.set_ylabel("Macro NMAE (Lower = Better)", fontweight='bold')
    ax1.set_ylim(0, df_means['NMAE'].max() * 1.15)
    ax1.get_legend().remove()
    sns.despine(left=True, bottom=True, ax=ax1)

    # Right: fold-level boxplot with dynamic y-axis clipping
    sns.boxplot(
        data=df_plot, x='Modell', y='NMAE', hue='Status',
        palette=palette, ax=ax2, width=0.5,
        flierprops={"marker": "x", "markerfacecolor": "#d9534f", "markersize": 8, "markeredgecolor": "#b84a39"},
    )

    # Clip y-axis to prevent extreme outlier folds from compressing the view
    normal_values = df_plot[df_plot['NMAE'] < 1.0]['NMAE']
    y_max_clean   = normal_values.max() * 1.15 if not normal_values.empty else 0.6
    y_max_clean   = max(0.55, y_max_clean)
    ax2.set_ylim(0, y_max_clean)

    if (df_plot['NMAE'] > y_max_clean).any():
        ax2.text(
            0.0, y_max_clean * 0.88,
            "⚠️ Extreme value (Fold 07: 1.5368)\noutside focus scale",
            color='#b84a39', fontsize=9, fontweight='bold', ha='center',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.3'),
        )

    ax2.set_title("Error Distribution & Model Stability Across All Folds", fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel("Model Architecture", fontweight='bold', labelpad=10)
    ax2.set_ylabel("NMAE per Fold", fontweight='bold')
    sns.despine(left=True, bottom=True, ax=ax2)
    ax2.legend(title="Pipeline Status", loc='upper right', frameon=True, facecolor='white', edgecolor='none')

    plt.suptitle(
        "Validation Dashboard: Preprocessing Impact & Model Resilience",
        fontsize=14, fontweight='bold', y=1.02, color='#222222',
    )
    plt.tight_layout()
    plt.show()


def plot_ml_vs_empirical(
    emp_df: pd.DataFrame,
    best_model_name: str,
) -> None:
    """Scatter comparison of ML out-of-fold predictions vs. Korfiatis (1982).

    Restricts the comparison to the Korfiatis-valid subset (granular soils),
    prints MAE and R² for both methods, and renders a 1×2 scatter plot with
    the identity line.

    Args:
        emp_df: DataFrame with columns ``proctor_mdd_g_cm3``,
            ``pred_mdd_ml_oof``, and ``pred_mdd_korfiatis``.
        best_model_name: Label used in the ML panel title.
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    mask   = emp_df["pred_mdd_korfiatis"].notna()
    y_true = emp_df.loc[mask, "proctor_mdd_g_cm3"]
    y_ml   = emp_df.loc[mask, "pred_mdd_ml_oof"]
    y_korf = emp_df.loc[mask, "pred_mdd_korfiatis"]

    mae_ml,   r2_ml   = mean_absolute_error(y_true, y_ml),   r2_score(y_true, y_ml)
    mae_korf, r2_korf = mean_absolute_error(y_true, y_korf), r2_score(y_true, y_korf)
    print(f"ML OOF    → MAE: {mae_ml:.4f} g/cm³  |  R²: {r2_ml:.4f}")
    print(f"Korfiatis → MAE: {mae_korf:.4f} g/cm³  |  R²: {r2_korf:.4f}")

    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "0.9"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=120)

    mn = min(y_true.min(), y_ml.min(), y_korf.min()) * 0.98
    mx = max(y_true.max(), y_ml.max(), y_korf.max()) * 1.02

    for ax, y_pred, color, label, mae_v, r2_v in [
        (ax1, y_ml,   '#2c5d88', f"{best_model_name} (OOF)", mae_ml,   r2_ml),
        (ax2, y_korf, '#c0392b', "Korfiatis (1982)",          mae_korf, r2_korf),
    ]:
        ax.scatter(y_true, y_pred, alpha=0.6, s=40, color=color, edgecolors='white', linewidth=0.4)
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1.2, label='Ideallinie')
        ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
        ax.set_xlabel("Gemessen MDD [g/cm³]", fontweight='bold')
        ax.set_ylabel("Vorhergesagt MDD [g/cm³]", fontweight='bold')
        ax.set_title(f"{label}\nMAE = {mae_v:.4f} g/cm³  |  R² = {r2_v:.3f}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)

    plt.suptitle("Vergleich: ML-Modell vs. Korfiatis-Empirie (nur Schnittmenge)",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def predict_with_pipeline_registry(
    df_new: pd.DataFrame,
    config_label: str,
    registry: dict,
    cols_for_imputation: list,
) -> np.ndarray:
    """Generate leak-free ensemble predictions from a multi-fold model registry.

    Each fold's fitted imputer, scaler, and model are applied independently to
    the raw test data, replicating the exact transformation pipeline used during
    training. Predictions are averaged across all folds.

    Args:
        df_new: Raw test DataFrame in the same feature space as the training data.
        config_label: Key used to look up the fold artifacts in *registry*.
        registry: Dictionary mapping config labels to lists of fold artifact dicts.
            Each dict must contain ``'fitted_imputer'``, ``'fitted_scaler'``
            (or ``None``), and ``'fitted_model'``.
        cols_for_imputation: Columns passed to the fold-specific imputer.

    Returns:
        Array of shape ``(n_samples, n_targets)`` with ensemble-averaged predictions.

    Raises:
        KeyError: If *config_label* is not found in *registry*.
    """
    if config_label not in registry:
        raise KeyError(f"Configuration '{config_label}' not found in registry.")

    fold_predictions    = []
    fold_artifacts_list = registry[config_label]

    for artifacts in fold_artifacts_list:
        X_tmp = df_new.copy()

        # Replicate missing-indicator feature construction from training
        X_tmp['atterberg_is_missing'] = X_tmp['atterberg_liquid_limit_pct'].isnull().astype(int)
        X_tmp['kf_is_missing']        = X_tmp['hyd_cond_kf_m_s'].isnull().astype(int)
        X_tmp['loi_is_missing']       = X_tmp['loss_on_ignition_pct'].isnull().astype(int)

        # Apply fold-specific MICE transformation
        X_tmp[cols_for_imputation] = artifacts['fitted_imputer'].transform(X_tmp[cols_for_imputation])

        # Replicate post-imputation feature engineering (identical to CV-loop)
        apply_fold_feature_engineering(X_tmp)

        # Apply fold-specific scaling if available
        if artifacts['fitted_scaler'] is not None:
            X_proc = artifacts['fitted_scaler'].transform(X_tmp)
        else:
            X_proc = X_tmp.values

        fold_predictions.append(artifacts['fitted_model'].predict(X_proc))

    # Average predictions across all folds for a robust ensemble estimate
    return np.mean(fold_predictions, axis=0)



# ===========================================================================
# 8. CUSTOM HELPER FUNCTIONS
# ===========================================================================

def impute_missing_values2(
    input_df: pd.DataFrame,
    impute_cols: list,
    predictors: list | None = None,
    log_transform_cols: list | None = None,
    applicable_mask: pd.Series | np.ndarray | None = None,
    estimator=None,
    max_iter: int = 20,
    random_state: int = 42,
) -> tuple:
    """
    Impute missing values using MICE (notebook implementation).

    Supports separate target and predictor columns, optional log10
    transformation of selected variables, and subgroup-specific imputation
    via a row mask (e.g. fine- and coarse-grained soils). Rather than
    overwriting the originals, appends ``{col}_completed`` and
    ``{col}_was_missing`` columns (plus ``log10_{col}_completed`` for
    log-transformed variables), allowing multiple subgroup imputations to be
    combined safely on the same DataFrame.

    Returns the updated DataFrame and the fitted ``IterativeImputer``.
    """
    output_df = input_df.copy()

    if predictors is None:
        predictors = []

    if log_transform_cols is None:
        log_transform_cols = []

    if estimator is None:
        estimator = BayesianRidge()

    # Remove duplicates while preserving order
    mice_columns = list(dict.fromkeys(impute_cols + predictors))

    missing_columns = [col for col in mice_columns if col not in output_df.columns]
    if missing_columns:
        raise KeyError(f"These columns are not in the dataframe: {missing_columns}")

    if applicable_mask is None:
        applicable_mask = pd.Series(True, index=output_df.index)
    else:
        applicable_mask = pd.Series(applicable_mask, index=output_df.index).astype(bool)

    # Work only on rows where these variables are applicable
    mice_data = output_df.loc[applicable_mask, mice_columns].copy()

    # Ensure numeric input
    mice_data = mice_data.apply(pd.to_numeric, errors="coerce")

    # Apply log10 transformation where requested
    transformed_column_names = {}

    for col in log_transform_cols:
        if col not in mice_columns:
            raise ValueError(
                f"{col!r} is in log_transform_cols but not in "
                "impute_cols or predictors."
            )

        transformed_name = f"__log10_{col}"
        transformed_column_names[col] = transformed_name

        values = mice_data[col].astype(float)

        # Non-positive values are invalid for log10
        valid_positive = values > 0

        mice_data[transformed_name] = np.nan
        mice_data.loc[valid_positive, transformed_name] = np.log10(values.loc[valid_positive])

        mice_data = mice_data.drop(columns=col)

    # Names actually passed into IterativeImputer
    model_columns = mice_data.columns.tolist()

    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        tol=1e-3,
        initial_strategy="median",
        imputation_order="ascending",
        skip_complete=True,
        sample_posterior=False,
        random_state=random_state,
    )

    imputed_array = imputer.fit_transform(mice_data)
    imputed_data  = pd.DataFrame(imputed_array, index=mice_data.index, columns=model_columns)

    # Convert log-transformed variables back to their original scale
    for original_col, transformed_col in transformed_column_names.items():
        imputed_data[original_col] = 10 ** imputed_data[transformed_col]

    # Create or update completed columns and missing indicators
    for col in impute_cols:
        completed_col      = f"{col}_completed"
        missing_indicator  = f"{col}_was_missing"

        # Create these only on the first call.
        # If they already exist, preserve previous imputations.
        if completed_col not in output_df.columns:
            output_df[completed_col] = output_df[col].copy()

        if missing_indicator not in output_df.columns:
            output_df[missing_indicator] = output_df[col].isna().astype(int)

        # Fill only originally missing values inside the current subset
        fill_mask = applicable_mask & output_df[col].isna()

        output_df.loc[fill_mask, completed_col] = imputed_data.loc[fill_mask, col]

        # Recalculate the log-completed column using all currently completed values
        if col in log_transform_cols:
            log_completed_col = f"log10_{completed_col}"
            output_df[log_completed_col] = np.nan

            positive_mask = (
                output_df[completed_col].notna()
                & np.isfinite(output_df[completed_col])
                & (output_df[completed_col] > 0)
            )

            output_df.loc[positive_mask, log_completed_col] = np.log10(
                output_df.loc[positive_mask, completed_col]
            )

    return output_df, imputer