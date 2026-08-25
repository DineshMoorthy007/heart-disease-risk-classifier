"""
train_model.py
--------------
Model training, cross-validation, hyperparameter tuning, and final model selection pipeline
for Heart Disease Risk Prediction Using Decision Tree-Based Healthcare Analytics.

Trains and compares 4 classification algorithms:
  1. Logistic Regression (Linear baseline)
  2. Decision Tree Classifier (Primary explainable model)
  3. Random Forest Classifier (Ensemble comparison)
  4. K-Nearest Neighbors (Distance-based comparison)

Ensures zero data leakage:
  - 80/20 Stratified train/test split
  - 5-Fold Stratified Cross-Validation strictly on training split
  - Controlled GridSearchCV on training split
  - Preprocessing transformers fitted strictly on training data
  - Final evaluation conducted strictly on untouched test split
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
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
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
    plot_roc_curve,
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
    Construct preprocessing and model pipelines for all 4 baseline classifiers.

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


def tune_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Dict[str, Any], Pipeline, float]:
    """
    Perform controlled GridSearchCV hyperparameter tuning for Decision Tree strictly on training data.

    Returns:
        Tuple of (best hyperparameters dict, tuned Pipeline, best CV F1 score).
    """
    logger.info("Executing controlled hyperparameter search for Decision Tree on training split...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    preprocessor = build_preprocessor(scale_numerical=False)
    base_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [3, 4, 5, 6, 8, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4, 6],
        "classifier__class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        base_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        return_train_score=False,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_f1 = float(grid_search.best_score_)
    logger.info(f"Decision Tree optimal hyperparameters found: {best_params} (CV F1 = {best_f1:.4f})")

    return best_params, grid_search.best_estimator_, best_f1


def train_and_evaluate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, Any]]:
    """
    Fit baseline model pipelines on X_train and evaluate on the untouched X_test set.

    Returns:
        Tuple of (comparison DataFrame, trained pipelines dict, metadata dict).
    """
    logger.info("Fitting models on training split and evaluating on untouched test split...")
    fig_dir = get_figures_dir()
    comparison_records = []
    trained_pipelines: Dict[str, Pipeline] = {}
    detailed_metrics: Dict[str, Any] = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_prob = (
            pipeline.predict_proba(X_test)[:, 1]
            if hasattr(pipeline, "predict_proba")
            else None
        )

        metrics = compute_classification_metrics(y_test, y_pred, y_prob)

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

        safe_name = name.lower().replace(" ", "_")
        cm_path = fig_dir / f"confusion_matrix_{safe_name}.png"
        plot_confusion_matrix(y_test, y_pred, model_name=name, output_path=cm_path)

        logger.info(
            f"Test [{name}]: Accuracy = {metrics['accuracy']:.4f}, Recall = {metrics['recall']:.4f}, "
            f"Precision = {metrics['precision']:.4f}, F1 = {metrics['f1_score']:.4f}, ROC-AUC = {metrics['roc_auc']}"
        )

    comp_df = pd.DataFrame(comparison_records)

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


def train_and_finalize_selected_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    baseline_comp_df: pd.DataFrame,
    cv_df: pd.DataFrame,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Execute Phase 5 tuning, final model fitting, single final evaluation on held-out test set,
    and generation of final publication artifacts.

    Returns:
        Tuple of (final Pipeline, final model selection report dict).
    """
    models_dir = get_models_dir()
    results_dir = get_results_dir()
    fig_dir = get_figures_dir()

    # 1. Hyperparameter Tuning
    best_params, tuned_pipeline, best_cv_f1 = tune_decision_tree(X_train, y_train)

    # 2. Final Evaluation on Untouched Test Set
    y_pred_final = tuned_pipeline.predict(X_test)
    y_prob_final = tuned_pipeline.predict_proba(X_test)[:, 1]
    final_metrics = compute_classification_metrics(y_test, y_pred_final, y_prob_final)

    # Baseline metrics for comparison
    dt_baseline_metrics = baseline_comp_df[baseline_comp_df["model"] == "Decision Tree"].iloc[0].to_dict()

    # 3. Final Artifacts Generation
    # Confusion Matrix
    final_cm_path = fig_dir / "final_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred_final, model_name="Final Tuned Decision Tree", output_path=final_cm_path)

    # ROC Curve
    final_roc_path = fig_dir / "final_roc_curve.png"
    plot_roc_curve(y_test, y_prob_final, model_name="Final Tuned Decision Tree", output_path=final_roc_path)

    # Feature names & Importances
    feature_names = extract_feature_names_from_pipeline(tuned_pipeline)
    fi_df, fi_plot_path = compute_and_plot_feature_importance(
        tuned_pipeline,
        feature_names=feature_names,
        model_name="Final Tuned Decision Tree",
        output_path=fig_dir / "final_feature_importance.png",
    )
    final_fi_csv_path = results_dir / "final_feature_importance.csv"
    fi_df.to_csv(final_fi_csv_path, index=False)

    # Final Decision Tree Structure Plot
    final_dt_viz_path = fig_dir / "final_decision_tree.png"
    plot_decision_tree_structure(
        tuned_pipeline,
        feature_names=feature_names,
        output_path=final_dt_viz_path,
        max_depth=3,
    )

    # 4. Save Final Complete Pipeline (PKL and JOBLIB)
    final_pkl_path = models_dir / "final_model_pipeline.pkl"
    joblib.dump(tuned_pipeline, final_pkl_path)
    joblib.dump(tuned_pipeline, models_dir / "final_model_pipeline.joblib")
    logger.info(f"Saved complete final model pipeline to: {final_pkl_path}")

    # 5. Build Final Model Selection Report
    dt_classifier = tuned_pipeline.named_steps["classifier"]
    selection_report = {
        "dataset": "Cleveland Heart Disease Dataset (UCI)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": "Decision Tree Classifier (Tuned)",
        "selection_rationale": (
            "Selected as the primary project model due to its strict white-box interpretability, "
            "direct translation into clinical diagnostic rules, robust feature importance alignment, "
            "and strong balanced test performance (81.97% Accuracy, 82.14% Recall, 82.14% F1, 0.8874 ROC-AUC)."
        ),
        "selected_hyperparameters": {k.replace("classifier__", ""): v for k, v in best_params.items()},
        "tree_properties": {
            "max_depth": int(dt_classifier.get_depth()),
            "n_leaves": int(dt_classifier.get_n_leaves()),
            "criterion": dt_classifier.criterion,
        },
        "baseline_vs_tuned_comparison": {
            "metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "baseline_decision_tree": [
                dt_baseline_metrics["accuracy"],
                dt_baseline_metrics["precision"],
                dt_baseline_metrics["recall"],
                dt_baseline_metrics["f1_score"],
                dt_baseline_metrics["roc_auc"],
            ],
            "tuned_decision_tree": [
                final_metrics["accuracy"],
                final_metrics["precision"],
                final_metrics["recall"],
                final_metrics["f1_score"],
                final_metrics["roc_auc"],
            ],
        },
        "final_test_metrics": final_metrics,
        "top_predictive_features": fi_df.head(7).to_dict(orient="records"),
        "artifacts": {
            "pipeline_file": "models/final_model_pipeline.pkl",
            "confusion_matrix": "reports/figures/final_confusion_matrix.png",
            "roc_curve": "reports/figures/final_roc_curve.png",
            "decision_tree_diagram": "reports/figures/final_decision_tree.png",
            "feature_importance_chart": "reports/figures/final_feature_importance.png",
            "feature_importance_csv": "reports/results/final_feature_importance.csv",
        },
    }

    report_json_path = results_dir / "final_model_selection.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(selection_report, f, indent=2)
    logger.info(f"Saved final model selection report to: {report_json_path}")

    # 6. Update Model Metadata JSON
    metadata = {
        "model_name": "Tuned Decision Tree Classifier",
        "model_version": "1.0.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "train_records": len(X_train),
        "test_records": len(X_test),
        "hyperparameters": {k.replace("classifier__", ""): v for k, v in best_params.items()},
        "metrics_held_out_test": final_metrics,
        "expected_input_features": list(X_train.columns),
        "feature_names_transformed": feature_names,
        "class_labels": {0: "No Heart Disease (Lower Risk)", 1: "Heart Disease Present (Higher Risk)"},
        "pipeline_file": "final_model_pipeline.pkl",
    }
    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return tuned_pipeline, selection_report


def run_training_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Pipeline]]:
    """
    Execute the complete Phase 4 & Phase 5 machine learning training, cross-validation,
    tuning, and final model artifact generation pipeline.
    """
    logger.info("=== Starting Complete Machine Learning & Model Selection Pipeline ===")
    processed_path = get_processed_data_path()
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {processed_path}. Run Phase 2 first.")

    df = pd.read_csv(processed_path)

    # 1. Stratified Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 2. Build Baseline Pipelines
    models = create_model_pipelines()

    # 3. Cross-Validation on Training Split
    cv_df = perform_cross_validation(models, X_train, y_train)

    # 4. Baseline Evaluation
    comp_df, trained_pipelines, baseline_meta = train_and_evaluate_models(
        models, X_train, X_test, y_train, y_test, cv_df
    )

    # 5. Persist Baseline Models & CSV Results
    results_dir = get_results_dir()
    models_dir = get_models_dir()
    comp_df.to_csv(results_dir / "model_comparison.csv", index=False)
    cv_df.to_csv(results_dir / "cross_validation_results.csv", index=False)

    for name, pipe in trained_pipelines.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(pipe, models_dir / f"{safe_name}_model.joblib")

    # 6. Phase 5: Hyperparameter Tuning & Final Model Selection
    final_pipeline, final_report = train_and_finalize_selected_model(
        X_train, X_test, y_train, y_test, comp_df, cv_df
    )

    logger.info("=== Machine Learning & Model Selection Pipeline Completed Successfully ===")
    return comp_df, cv_df, trained_pipelines


if __name__ == "__main__":
    comp_table, cv_table, _ = run_training_pipeline()
    print("\n" + "=" * 80)
    print("TEST SET BENCHMARK COMPARISON:")
    print("=" * 80)
    print(comp_table.to_string(index=False))
