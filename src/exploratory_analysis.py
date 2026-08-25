"""
exploratory_analysis.py
-----------------------
Reusable Exploratory Data Analysis (EDA) module for the Cleveland Heart Disease dataset.
Generates publication-quality clinical visualizations and computes comprehensive
descriptive, demographic, and feature-relationship statistics.

Figures saved to reports/figures/:
  1. target_distribution.png
  2. age_distribution.png
  3. age_by_target.png
  4. heart_disease_by_sex.png
  5. chest_pain_by_target.png
  6. blood_pressure_by_target.png
  7. cholesterol_by_target.png
  8. max_heart_rate_by_target.png
  9. exercise_angina_by_target.png
  10. st_depression_by_target.png
  11. vessels_ca_by_target.png
  12. thalassemia_by_target.png
  13. correlation_heatmap.png
  14. feature_target_summary.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    get_processed_data_path,
    get_project_root,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Aesthetic styling configuration
PALETTE_TARGET = {0: "#2b5c8f", 1: "#d9534f"}  # Steel Blue (No Disease) vs Terracotta Red (Disease)
PALETTE_MAP = {"No Heart Disease": "#2b5c8f", "Heart Disease Present": "#d9534f"}
TARGET_LABELS = {0: "No Heart Disease", 1: "Heart Disease Present"}

# Category display maps for human-readable charts
CHEST_PAIN_MAP = {
    0: "Typical Angina (0)",
    1: "Atypical Angina (1)",
    2: "Non-Anginal (2)",
    3: "Asymptomatic (3)",
}
SEX_MAP = {0: "Female (0)", 1: "Male (1)"}
EXANG_MAP = {0: "No Angina (0)", 1: "Angina Present (1)"}
THAL_MAP = {1: "Normal (1)", 2: "Fixed Defect (2)", 3: "Reversible Defect (3)"}
SLOPE_MAP = {0: "Upsloping (0)", 1: "Flat (1)", 2: "Downsloping (2)"}
RESTECG_MAP = {0: "Normal (0)", 1: "ST-T Abnormality (1)", 2: "LV Hypertrophy (2)"}


def get_figures_dir() -> Path:
    """Return the path to the reports/figures directory, creating it if needed."""
    root = get_project_root()
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def set_plot_theme() -> None:
    """Configure matplotlib and seaborn aesthetic defaults."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.autolayout": True,
        "savefig.dpi": 300,
    })


def compute_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute full summary statistics for the dataset, stratified by target class.

    Args:
        df: Processed DataFrame.

    Returns:
        Dictionary of numerical and categorical summary metrics.
    """
    total = len(df)
    target_counts = df[TARGET_COLUMN].value_counts().to_dict()

    # Numerical statistics
    num_stats = {}
    for col in NUMERICAL_FEATURES:
        num_stats[col] = {
            "overall_mean": float(df[col].mean()),
            "overall_std": float(df[col].std()),
            "overall_median": float(df[col].median()),
            "overall_iqr": float(df[col].quantile(0.75) - df[col].quantile(0.25)),
            "overall_min": float(df[col].min()),
            "overall_max": float(df[col].max()),
            "by_target": {
                int(k): {
                    "mean": float(v["mean"]),
                    "std": float(v["std"]),
                    "median": float(v["median"]),
                    "min": float(v["min"]),
                    "max": float(v["max"]),
                }
                for k, v in df.groupby(TARGET_COLUMN)[col].agg(["mean", "std", "median", "min", "max"]).to_dict("index").items()
            }
        }

    # Categorical distributions
    cat_stats = {}
    for col in CATEGORICAL_FEATURES:
        ct = pd.crosstab(df[col], df[TARGET_COLUMN], normalize="index") * 100
        counts = pd.crosstab(df[col], df[TARGET_COLUMN])
        cat_stats[col] = {
            "counts": counts.to_dict(),
            "percentages": ct.round(2).to_dict(),
        }

    # Outlier detection via IQR
    outlier_summary = {}
    for col in NUMERICAL_FEATURES:
        q25 = float(df[col].quantile(0.25))
        q75 = float(df[col].quantile(0.75))
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)][col].tolist()
        outlier_summary[col] = {
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": len(outliers),
            "outlier_percentage": round((len(outliers) / total) * 100, 2),
            "outlier_values": [float(v) for v in outliers],
        }

    return {
        "record_count": total,
        "target_distribution": {
            "counts": {int(k): int(v) for k, v in target_counts.items()},
            "percentages": {int(k): round((v / total) * 100, 2) for k, v in target_counts.items()},
        },
        "numerical_statistics": num_stats,
        "categorical_statistics": cat_stats,
        "outlier_analysis": outlier_summary,
        "correlations_with_target": df.corr()[TARGET_COLUMN].round(4).to_dict(),
    }


def plot_target_distribution(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """Plot and save the target variable distribution bar chart."""
    set_plot_theme()
    out_dir = output_dir or get_figures_dir()
    out_path = out_dir / "target_distribution.png"

    counts = df[TARGET_COLUMN].value_counts().sort_index()
    total = len(df)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [TARGET_LABELS[0], TARGET_LABELS[1]],
        [counts[0], counts[1]],
        color=[PALETTE_TARGET[0], PALETTE_TARGET[1]],
        width=0.5,
        edgecolor="black",
        linewidth=1.2,
    )

    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(
            f"{int(height)} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_title("Distribution of Heart Disease Diagnosis in Cleveland Dataset", pad=15, fontweight="bold")
    ax.set_ylabel("Number of Patients")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_age_analysis(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Plot and save age distribution histogram and age by target boxplot."""
    set_plot_theme()
    out_dir = output_dir or get_figures_dir()

    # 1. Overall Age Distribution
    p1 = out_dir / "age_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        df["age"],
        bins=15,
        kde=True,
        color="#2b5c8f",
        edgecolor="black",
        ax=ax,
    )
    mean_age = df["age"].mean()
    median_age = df["age"].median()
    ax.axvline(mean_age, color="red", linestyle="--", linewidth=1.8, label=f"Mean ({mean_age:.1f} yrs)")
    ax.axvline(median_age, color="orange", linestyle=":", linewidth=2, label=f"Median ({median_age:.1f} yrs)")
    ax.set_title("Patient Age Distribution (Cleveland Heart Disease Cohort)", pad=15, fontweight="bold")
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Patient Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close()

    # 2. Age by Target Class
    p2 = out_dir / "age_by_target.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    df_plot = df.copy()
    df_plot["target_label"] = df_plot[TARGET_COLUMN].map(TARGET_LABELS)
    sns.boxplot(
        data=df_plot,
        x="target_label",
        y="age",
        hue="target_label",
        palette=PALETTE_MAP,
        legend=False,
        width=0.45,
        ax=ax,
    )
    sns.stripplot(
        data=df_plot,
        x="target_label",
        y="age",
        color="black",
        alpha=0.35,
        jitter=0.2,
        ax=ax,
    )
    ax.set_title("Age Distribution by Heart Disease Diagnosis", pad=15, fontweight="bold")
    ax.set_xlabel("Diagnostic Class")
    ax.set_ylabel("Age (Years)")
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close()

    logger.info(f"Saved: {p1}, {p2}")
    return p1, p2


def plot_sex_analysis(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """Plot and save heart disease prevalence broken down by biological sex."""
    set_plot_theme()
    out_dir = output_dir or get_figures_dir()
    out_path = out_dir / "heart_disease_by_sex.png"

    df_plot = df.copy()
    df_plot["sex_label"] = df_plot["sex"].map(SEX_MAP)
    df_plot["target_label"] = df_plot[TARGET_COLUMN].map(TARGET_LABELS)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df_plot,
        x="sex_label",
        hue="target_label",
        palette=PALETTE_MAP,
        edgecolor="black",
        ax=ax,
    )

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    ax.set_title("Heart Disease Prevalence by Biological Sex", pad=15, fontweight="bold")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Patient Count")
    ax.legend(title="Diagnosis")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_chest_pain_analysis(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """Plot and save chest pain type vs heart disease diagnosis."""
    set_plot_theme()
    out_dir = output_dir or get_figures_dir()
    out_path = out_dir / "chest_pain_by_target.png"

    df_plot = df.copy()
    df_plot["cp_label"] = df_plot["cp"].map(CHEST_PAIN_MAP)
    df_plot["target_label"] = df_plot[TARGET_COLUMN].map(TARGET_LABELS)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.countplot(
        data=df_plot,
        x="cp_label",
        hue="target_label",
        palette=PALETTE_MAP,
        edgecolor="black",
        order=[CHEST_PAIN_MAP[0], CHEST_PAIN_MAP[1], CHEST_PAIN_MAP[2], CHEST_PAIN_MAP[3]],
        ax=ax,
    )

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    ax.set_title("Chest Pain Type vs Heart Disease Diagnosis", pad=15, fontweight="bold")
    ax.set_xlabel("Chest Pain Category")
    ax.set_ylabel("Patient Count")
    ax.legend(title="Diagnosis")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_vitals_analysis(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """Plot and save distributions and boxplots for blood pressure, cholesterol, and maximum heart rate."""
    set_plot_theme()
    out_dir = output_dir or get_figures_dir()
    df_plot = df.copy()
    df_plot["target_label"] = df_plot[TARGET_COLUMN].map(TARGET_LABELS)

    # 1. Blood pressure
    p1 = out_dir / "blood_pressure_by_target.png"
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.boxplot(
        data=df_plot,
        x="target_label",
        y="trestbps",
        hue="target_label",
        palette=PALETTE_MAP,
        legend=False,
        width=0.45,
        ax=ax,
    )
    sns.stripplot(
        data=df_plot,
        x="target_label",
        y="trestbps",
        color="black",
        alpha=0.3,
        jitter=0.2,
        ax=ax,
    )
    ax.set_title("Resting Blood Pressure by Heart Disease Diagnosis", pad=15, fontweight="bold")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Resting BP (mm Hg)")
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close()

    # 2. Cholesterol
    p2 = out_dir / "cholesterol_by_target.png"
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.boxplot(
        data=df_plot,
        x="target_label",
        y="chol",
        hue="target_label",
        palette=PALETTE_MAP,
        legend=False,
        width=0.45,
        ax=ax,
    )
    sns.stripplot(
        data=df_plot,
        x="target_label",
        y="chol",
        color="black",
        alpha=0.3,
        jitter=0.2,
        ax=ax,
    )
    ax.set_title("Serum Cholesterol Levels by Heart Disease Diagnosis", pad=15, fontweight="bold")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Serum Cholesterol (mg/dl)")
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close()

    # 3. Maximum Heart Rate (thalach)
    p3 = out_dir / "max_heart_rate_by_target.png"
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.boxplot(
        data=df_plot,
        x="target_label",
        y="thalach",
        hue="target_label",
        palette=PALETTE_MAP,
        legend=False,
        width=0.45,
        ax=ax,
    )
    sns.stripplot(
        data=df_plot,
        x="target_label",
        y="thalach",
        color="black",
        alpha=0.3,
        jitter=0.2,
        ax=ax,
    )
    ax.set_title("Maximum Heart Rate Achieved (thalach) by Diagnosis", pad=15, fontweight="bold")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Maximum Heart Rate (bpm)")
    plt.tight_layout()
    plt.savefig(p3, dpi=300)
    plt.close()

    logger.info(f"Saved: {p1}, {p2}, {p3}")
    return p1, p2, p3


def plot_clinical_stress_features(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Tuple[Path, Path, Path, Path]:
    """Plot and save exercise angina, ST depression, major vessels, and thalassemia by target."""
    set_plot_theme()
    out_dir = output_dir or get_figures_dir()
    df_plot = df.copy()
    df_plot["target_label"] = df_plot[TARGET_COLUMN].map(TARGET_LABELS)

    # 1. Exercise Angina
    p1 = out_dir / "exercise_angina_by_target.png"
    df_plot["exang_label"] = df_plot["exang"].map(EXANG_MAP)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.countplot(
        data=df_plot,
        x="exang_label",
        hue="target_label",
        palette=PALETTE_MAP,
        edgecolor="black",
        ax=ax,
    )
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", xy=(p.get_x() + p.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Exercise-Induced Angina vs Heart Disease", pad=15, fontweight="bold")
    ax.set_xlabel("Exercise Angina")
    ax.set_ylabel("Patient Count")
    ax.legend(title="Diagnosis")
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close()

    # 2. ST Depression (oldpeak)
    p2 = out_dir / "st_depression_by_target.png"
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.boxplot(
        data=df_plot,
        x="target_label",
        y="oldpeak",
        hue="target_label",
        palette=PALETTE_MAP,
        legend=False,
        width=0.45,
        ax=ax,
    )
    sns.stripplot(
        data=df_plot,
        x="target_label",
        y="oldpeak",
        color="black",
        alpha=0.3,
        jitter=0.2,
        ax=ax,
    )
    ax.set_title("ST Depression (oldpeak) by Heart Disease Diagnosis", pad=15, fontweight="bold")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("ST Depression (mm)")
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close()

    # 3. Major Vessels (ca)
    p3 = out_dir / "vessels_ca_by_target.png"
    df_plot["ca_str"] = df_plot["ca"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df_plot,
        x="ca_str",
        hue="target_label",
        palette=PALETTE_MAP,
        edgecolor="black",
        order=["0", "1", "2", "3"],
        ax=ax,
    )
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", xy=(p.get_x() + p.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Major Vessels Colored by Fluoroscopy (ca) vs Heart Disease", pad=15, fontweight="bold")
    ax.set_xlabel("Number of Major Vessels (0 - 3)")
    ax.set_ylabel("Patient Count")
    ax.legend(title="Diagnosis")
    plt.tight_layout()
    plt.savefig(p3, dpi=300)
    plt.close()

    # 4. Thalassemia (thal)
    p4 = out_dir / "thalassemia_by_target.png"
    df_plot["thal_label"] = df_plot["thal"].map(THAL_MAP)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.countplot(
        data=df_plot,
        x="thal_label",
        hue="target_label",
        palette=PALETTE_MAP,
        edgecolor="black",
        order=[THAL_MAP[1], THAL_MAP[2], THAL_MAP[3]],
        ax=ax,
    )
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", xy=(p.get_x() + p.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Thalassemia Defect Status vs Heart Disease", pad=15, fontweight="bold")
    ax.set_xlabel("Thalassemia Status")
    ax.set_ylabel("Patient Count")
    ax.legend(title="Diagnosis")
    plt.tight_layout()
    plt.savefig(p4, dpi=300)
    plt.close()

    logger.info(f"Saved: {p1}, {p2}, {p3}, {p4}")
    return p1, p2, p3, p4


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """Plot and save full correlation matrix heatmap."""
    set_plot_theme()
    out_dir = output_dir or get_figures_dir()
    out_path = out_dir / "correlation_heatmap.png"

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 15, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.6,
        vmin=-0.6,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Clinical Attributes & Heart Disease Target", pad=18, fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_feature_target_summary(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """Plot and save ranked absolute correlation of all features with the heart disease target."""
    set_plot_theme()
    out_dir = output_dir or get_figures_dir()
    out_path = out_dir / "feature_target_summary.png"

    target_corr = df.corr()[TARGET_COLUMN].drop(index=[TARGET_COLUMN]).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2b5c8f" if c < 0 else "#d9534f" for c in target_corr.values]
    bars = ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor="black")

    for bar in bars:
        width = bar.get_width()
        offset = 0.02 if width >= 0 else -0.02
        ha = "left" if width >= 0 else "right"
        ax.annotate(
            f"{width:+.2f}",
            xy=(width + offset, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha=ha,
            fontweight="bold",
            fontsize=10,
        )

    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Linear Correlation of Clinical Features with Heart Disease (Target)", pad=15, fontweight="bold")
    ax.set_xlabel("Pearson Correlation with Target Class (0 = Absence, 1 = Presence)")
    ax.set_xlim(min(target_corr.values) - 0.1, max(target_corr.values) + 0.1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


def generate_all_eda_figures(df: Optional[pd.DataFrame] = None, output_dir: Optional[Path] = None) -> List[Path]:
    """
    Execute all plotting routines and save publication figures to reports/figures/.

    Args:
        df: Optional processed DataFrame. If None, loaded from processed data path.
        output_dir: Target directory for figure outputs.

    Returns:
        List of generated figure filepaths.
    """
    if df is None:
        processed_path = get_processed_data_path()
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {processed_path}. Run Phase 2 preprocessing first.")
        df = pd.read_csv(processed_path)

    out_dir = output_dir or get_figures_dir()
    logger.info(f"Generating all EDA figures into: {out_dir}")

    generated_files: List[Path] = []
    generated_files.append(plot_target_distribution(df, out_dir))
    a1, a2 = plot_age_analysis(df, out_dir)
    generated_files.extend([a1, a2])
    generated_files.append(plot_sex_analysis(df, out_dir))
    generated_files.append(plot_chest_pain_analysis(df, out_dir))
    v1, v2, v3 = plot_vitals_analysis(df, out_dir)
    generated_files.extend([v1, v2, v3])
    s1, s2, s3, s4 = plot_clinical_stress_features(df, out_dir)
    generated_files.extend([s1, s2, s3, s4])
    generated_files.append(plot_correlation_heatmap(df, out_dir))
    generated_files.append(plot_feature_target_summary(df, out_dir))

    logger.info(f"All {len(generated_files)} EDA figures generated successfully.")
    return generated_files


if __name__ == "__main__":
    df_data = pd.read_csv(get_processed_data_path())
    stats = compute_summary_statistics(df_data)
    figures = generate_all_eda_figures(df_data)
    print(f"\nEDA Execution Summary:")
    print(f"  Records Analyzed: {stats['record_count']}")
    print(f"  Target 0 (No Disease): {stats['target_distribution']['counts'][0]} ({stats['target_distribution']['percentages'][0]}%)")
    print(f"  Target 1 (Disease Present): {stats['target_distribution']['counts'][1]} ({stats['target_distribution']['percentages'][1]}%)")
    print(f"  Figures Saved ({len(figures)}): {[f.name for f in figures]}")
