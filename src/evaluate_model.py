"""
evaluate_model.py
-----------------
Reusable evaluation routines and visualization utilities for healthcare classification models.
Responsible for:
  - Metric computation (Accuracy, Precision, Recall, F1, ROC-AUC)
  - Confusion matrix plotting and export
  - Model comparison visualization
  - Feature importance extraction and plotting (with proper OHE column alignment)
  - Decision Tree visualization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import plot_tree

from src.data_preprocessing import get_project_root
from src.exploratory_analysis import set_plot_theme

logger = logging.getLogger(__name__)

CLASS_NAMES = ["No Disease (0)", "Disease Present (1)"]


def compute_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_prob: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Calculate core evaluation metrics for binary classification.

    Args:
        y_true: True target labels.
        y_pred: Predicted class labels.
        y_prob: Predicted probabilities for positive class (Class 1).

    Returns:
        Dictionary of rounded metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else np.nan,
    }
    return metrics


def plot_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    model_name: str,
    output_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Optional[Path]]:
    """
    Generate and save a publication-quality confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name: Name of the model for display and logging.
        output_path: File path to save the chart image.

    Returns:
        Tuple of (confusion matrix array, saved output path).
    """
    set_plot_theme()
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )

    ax.set_title(f"Confusion Matrix: {model_name}", pad=14, fontweight="bold")
    ax.set_xlabel("Predicted Diagnosis", labelpad=10, fontweight="bold")
    ax.set_ylabel("Actual Diagnosis", labelpad=10, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        logger.info(f"Saved confusion matrix for {model_name} to: {output_path}")
    plt.close()

    return cm, output_path


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Generate a grouped bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC across models.

    Args:
        comparison_df: DataFrame with columns ['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'].
        output_path: Destination path for figure.

    Returns:
        Saved output path.
    """
    set_plot_theme()
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

    df_melted = comparison_df.melt(
        id_vars=["model"],
        value_vars=available_metrics,
        var_name="metric",
        value_name="score",
    )

    metric_name_map = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall (Sensitivity)",
        "f1_score": "F1-Score",
        "roc_auc": "ROC-AUC",
    }
    df_melted["metric_label"] = df_melted["metric"].map(metric_name_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ["#2b5c8f", "#5cb85c", "#f0ad4e", "#d9534f", "#6f42c1"]

    sns.barplot(
        data=df_melted,
        x="model",
        y="score",
        hue="metric_label",
        palette=palette[: len(available_metrics)],
        edgecolor="black",
        ax=ax,
    )

    ax.set_title("Comparative Healthcare Model Performance (Untouched Test Set)", pad=16, fontweight="bold")
    ax.set_xlabel("Classification Model", labelpad=10, fontweight="bold")
    ax.set_ylabel("Metric Score", labelpad=10, fontweight="bold")
    ax.set_ylim(0.0, 1.08)
    ax.legend(title="Evaluation Metric", bbox_to_anchor=(1.02, 1), loc="upper left")

    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(
                f"{h:.2f}",
                xy=(p.get_x() + p.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved model comparison plot to: {output_path}")
    plt.close()

    return output_path


def extract_feature_names_from_pipeline(pipeline_or_transformer) -> List[str]:
    """
    Extract transformed feature names from a scikit-learn Pipeline or ColumnTransformer.

    Args:
        pipeline_or_transformer: Fitted scikit-learn Pipeline or ColumnTransformer.

    Returns:
        List of transformed feature names.
    """
    transformer = pipeline_or_transformer
    if hasattr(pipeline_or_transformer, "named_steps") and "preprocessor" in pipeline_or_transformer.named_steps:
        transformer = pipeline_or_transformer.named_steps["preprocessor"]

    if hasattr(transformer, "get_feature_names_out"):
        names = list(transformer.get_feature_names_out())
        # Clean prefix names (e.g., 'cat__', 'num__')
        clean_names = [n.split("__")[-1] for n in names]
        return clean_names
    return []


def plot_decision_tree_structure(
    dt_model,
    feature_names: List[str],
    output_path: Optional[Path] = None,
    max_depth: int = 3,
) -> Optional[Path]:
    """
    Visualize the decision tree rules with clear medical feature and class labels.

    Args:
        dt_model: Trained DecisionTreeClassifier or pipeline ending with one.
        feature_names: Names of features after preprocessing.
        output_path: Destination path for plot.
        max_depth: Depth of tree visualization for readable display.

    Returns:
        Saved output path.
    """
    set_plot_theme()
    tree = dt_model.named_steps["classifier"] if hasattr(dt_model, "named_steps") else dt_model

    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=["No Disease", "Disease Present"],
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=10,
        ax=ax,
    )
    ax.set_title(
        f"Decision Tree Architecture (Top {max_depth} Levels) - Clinical Decision Flow",
        pad=18,
        fontweight="bold",
        fontsize=14,
    )
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved Decision Tree visualization to: {output_path}")
    plt.close()

    return output_path


def compute_and_plot_feature_importance(
    tree_or_rf_model,
    feature_names: List[str],
    model_name: str = "Decision Tree",
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """
    Extract Gini feature importances, format into a sorted DataFrame, and plot bar chart.

    Args:
        tree_or_rf_model: Trained tree/forest model or pipeline.
        feature_names: Preprocessed feature names.
        model_name: Label for plot title.
        output_path: Destination image path.

    Returns:
        Tuple of (feature importance DataFrame, saved figure path).
    """
    set_plot_theme()
    model = tree_or_rf_model.named_steps["classifier"] if hasattr(tree_or_rf_model, "named_steps") else tree_or_rf_model
    importances = model.feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    # Plot top features
    fig, ax = plt.subplots(figsize=(9, 6))
    top_fi = fi_df[fi_df["importance"] > 0.001].copy()
    if top_fi.empty:
        top_fi = fi_df.head(10)

    sns.barplot(
        data=top_fi,
        y="feature",
        x="importance",
        color="#2b5c8f",
        edgecolor="black",
        ax=ax,
    )

    for p in ax.patches:
        w = p.get_width()
        ax.annotate(
            f"{w:.3f}",
            xy=(w, p.get_y() + p.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_title(f"{model_name}: Feature Importance (Gini Impurity Reduction)", pad=15, fontweight="bold")
    ax.set_xlabel("Relative Importance Score", labelpad=10, fontweight="bold")
    ax.set_ylabel("Clinical Attribute", labelpad=10, fontweight="bold")
    ax.set_xlim(0, max(fi_df["importance"]) * 1.18)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved feature importance plot to: {output_path}")
    plt.close()

    return fi_df, output_path
