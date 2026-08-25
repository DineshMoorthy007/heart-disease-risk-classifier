"""
train_model.py
--------------
Model training, cross-validation, and comparative evaluation pipeline for Heart Disease Risk Prediction.
Trains and compares 4 classification algorithms:
  1. Logistic Regression (Linear baseline)
  2. Decision Tree Classifier (Primary explainable model)
  3. Random Forest Classifier (Ensemble comparison)
  4. K-Nearest Neighbors (Distance-based comparison)

Ensures zero data leakage:
  - 80/20 Stratified train/test split
  - 5-Fold Stratified Cross-Validation strictly on training split
  - Preprocessing transformers fitted strictly on training data
  - Evaluation conducted on untouched test split
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    get_processed_data_path,
    get_project_root,
    load_raw_data,
    separate_features_target,
)
from src.evaluate_model import (
    compute_and_plot_feature_importance,
    compute_classification_metrics,
    extract_feature_names_from_pipeline,
    plot_confusion_matrix,
    plot_decision_tree_structure,
    plot_model_comparison,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def get_models_dir() -> Path:
    """Return the models/ directory path."""
    root = get_project_root()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_results_dir() -> Path:
    """Return the reports/results/ directory path."""
    root = get_project_root()
    res_dir = root / "reports" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    return res_dir


def get_figures_dir() -> Path:
    """Return the reports/figures/ directory path."""
    root = get_project_root()
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def build_preprocessor(scale_numerical: bool = False) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numerical and categorical features.

    Args:
        scale_numerical: Whether to apply StandardScaler to numerical features.

    Returns:
        ColumnTransformer.
    """
    num_transformer = StandardScaler() if scale_numerical else "passthrough"
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL_FEATURES),
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def create_model_pipelines() -> Dict[str, Pipeline]:
    """
    Construct preprocessing and model pipelines for all 4 classifiers.

    Returns:
        Dictionary mapping model names to scikit-learn Pipelines.
    """
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numerical=True)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numerical=False)),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=4,
                        min_samples_leaf=3,
                        min_samples_split=6,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numerical=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=6,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numerical=True)),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }
    return models


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training (80%) and testing (20%) sets with target stratification.

    Args:
        df: Processed DataFrame.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X, y = separate_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    logger.info(
        f"Data split completed: Train size = {len(X_train)} ({len(X_train)/len(df):.0%}), "
        f"Test size = {len(X_test)} ({len(X_test)/len(df):.0%})"
    )
    return X_train, X_test, y_train, y_test


def perform_cross_validation(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """
    Execute 5-Fold Stratified Cross-Validation on the training data.

    Args:
        models: Dictionary of model pipelines.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        DataFrame summarizing CV metrics for each model.
    """
    logger.info(f"Starting {CV_FOLDS}-Fold Stratified Cross-Validation on training data...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results_list = []

    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        cv_entry = {
            "model": name,
            "cv_accuracy_mean": round(float(np.mean(scores["test_accuracy"])), 4),
            "cv_accuracy_std": round(float(np.std(scores["test_accuracy"])), 4),
            "cv_precision_mean": round(float(np.mean(scores["test_precision"])), 4),
            "cv_recall_mean": round(float(np.mean(scores["test_recall"])), 4),
            "cv_f1_mean": round(float(np.mean(scores["test_f1"])), 4),
            "cv_roc_auc_mean": round(float(np.mean(scores["test_roc_auc"])), 4),
        }
        cv_results_list.append(cv_entry)
        logger.info(
            f"CV [{name}]: Accuracy = {cv_entry['cv_accuracy_mean']:.4f} (+/- {cv_entry['cv_accuracy_std']:.4f}), "
            f"Recall = {cv_entry['cv_recall_mean']:.4f}, F1 = {cv_entry['cv_f1_mean']:.4f}"
        )

    cv_df = pd.DataFrame(cv_results_list)
    return cv_df


def train_and_evaluate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, Any]]:
    """
    Fit each model pipeline on X_train and evaluate on the untouched X_test set.

    Returns:
        Tuple of (comparison DataFrame, trained pipelines dict, metadata dict).
    """
    logger.info("Fitting models on training split and evaluating on untouched test split...")
    fig_dir = get_figures_dir()
    comparison_records = []
    trained_pipelines: Dict[str, Pipeline] = {}
    detailed_metrics: Dict[str, Any] = {}

    for name, pipeline in models.items():
        # Fit on training split only
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predictions on test set
        y_pred = pipeline.predict(X_test)
        y_prob = (
            pipeline.predict_proba(X_test)[:, 1]
            if hasattr(pipeline, "predict_proba")
            else None
        )

        metrics = compute_classification_metrics(y_test, y_pred, y_prob)

        # Retrieve matching CV mean accuracy
        cv_row = cv_df[cv_df["model"] == name].iloc[0]
        cv_acc_str = f"{cv_row['cv_accuracy_mean']:.4f} ± {cv_row['cv_accuracy_std']:.4f}"

        record = {
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "cv_accuracy": cv_acc_str,
            "cv_f1_mean": cv_row["cv_f1_mean"],
            "cv_recall_mean": cv_row["cv_recall_mean"],
        }
        comparison_records.append(record)
        detailed_metrics[name] = metrics

        # Generate and save confusion matrix
        safe_name = name.lower().replace(" ", "_")
        cm_path = fig_dir / f"confusion_matrix_{safe_name}.png"
        plot_confusion_matrix(y_test, y_pred, model_name=name, output_path=cm_path)

        logger.info(
            f"Test [{name}]: Accuracy = {metrics['accuracy']:.4f}, Recall = {metrics['recall']:.4f}, "
            f"Precision = {metrics['precision']:.4f}, F1 = {metrics['f1_score']:.4f}, ROC-AUC = {metrics['roc_auc']}"
        )

    comp_df = pd.DataFrame(comparison_records)

    # Generate model comparison plot
    comp_plot_path = fig_dir / "model_performance_comparison.png"
    plot_model_comparison(comp_df, output_path=comp_plot_path)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "train_records": len(X_train),
        "test_records": len(X_test),
        "class_distribution_train": y_train.value_counts().to_dict(),
        "class_distribution_test": y_test.value_counts().to_dict(),
        "test_metrics": detailed_metrics,
    }

    return comp_df, trained_pipelines, metadata


def save_trained_models_and_artifacts(
    trained_pipelines: Dict[str, Pipeline],
    comp_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> None:
    """
    Persist trained model pipelines and benchmark result tables to disk.
    """
    models_dir = get_models_dir()
    results_dir = get_results_dir()
    fig_dir = get_figures_dir()

    # 1. Save CSV results
    comp_csv_path = results_dir / "model_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    logger.info(f"Saved model comparison table to: {comp_csv_path}")

    cv_csv_path = results_dir / "cross_validation_results.csv"
    cv_df.to_csv(cv_csv_path, index=False)
    logger.info(f"Saved cross-validation results to: {cv_csv_path}")

    # 2. Save individual baseline models
    for name, pipeline in trained_pipelines.items():
        safe_name = name.lower().replace(" ", "_")
        model_path = models_dir / f"{safe_name}_model.joblib"
        joblib.dump(pipeline, model_path)
        logger.info(f"Saved trained pipeline [{name}] to: {model_path}")

    # 3. Decision Tree Specific Visualizations & Feature Importance
    dt_pipeline = trained_pipelines["Decision Tree"]
    feature_names = extract_feature_names_from_pipeline(dt_pipeline)

    # Save DT feature importance table & plot
    fi_df, fi_plot_path = compute_and_plot_feature_importance(
        dt_pipeline,
        feature_names=feature_names,
        model_name="Decision Tree",
        output_path=fig_dir / "decision_tree_feature_importance.png",
    )
    fi_csv_path = results_dir / "decision_tree_feature_importance.csv"
    fi_df.to_csv(fi_csv_path, index=False)
    logger.info(f"Saved Decision Tree feature importances to: {fi_csv_path}")

    # Decision tree visualization
    dt_tree_path = fig_dir / "decision_tree_visualization.png"
    plot_decision_tree_structure(
        dt_pipeline,
        feature_names=feature_names,
        output_path=dt_tree_path,
        max_depth=3,
    )

    # 4. Save metadata JSON
    metadata["feature_names_transformed"] = feature_names
    dt_classifier = dt_pipeline.named_steps["classifier"]
    metadata["decision_tree_properties"] = {
        "max_depth": int(dt_classifier.get_depth()),
        "n_leaves": int(dt_classifier.get_n_leaves()),
        "criterion": dt_classifier.criterion,
        "top_features": fi_df.head(5).to_dict(orient="records"),
    }

    meta_path = models_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata JSON to: {meta_path}")


def run_training_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Pipeline]]:
    """
    Execute the end-to-end Phase 4 machine learning training and comparison pipeline.

    Returns:
        Tuple of (model comparison DataFrame, CV DataFrame, trained pipelines dict).
    """
    logger.info("=== Starting Machine Learning Model Development & Comparison Pipeline ===")
    processed_path = get_processed_data_path()
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {processed_path}. Run Phase 2 first.")

    df = pd.read_csv(processed_path)

    # 1. Stratified Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 2. Build Pipelines
    models = create_model_pipelines()

    # 3. Cross-Validation on Training Split
    cv_df = perform_cross_validation(models, X_train, y_train)

    # 4. Fit & Test Set Evaluation
    comp_df, trained_pipelines, metadata = train_and_evaluate_models(
        models, X_train, X_test, y_train, y_test, cv_df
    )

    # 5. Persist Models and Artifacts
    save_trained_models_and_artifacts(trained_pipelines, comp_df, cv_df, metadata)

    logger.info("=== Phase 4 Model Development & Comparison Completed Successfully ===")
    return comp_df, cv_df, trained_pipelines


if __name__ == "__main__":
    comparison_table, cv_table, _ = run_training_pipeline()
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS (5-Fold Stratified CV on Training Set):")
    print("=" * 80)
    print(cv_table.to_string(index=False))

    print("\n" + "=" * 80)
    print("TEST SET BENCHMARK COMPARISON (Untouched 20% Test Set):")
    print("=" * 80)
    print(comparison_table.to_string(index=False))
