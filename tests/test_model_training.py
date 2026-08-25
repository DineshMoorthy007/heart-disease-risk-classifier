"""
test_model_training.py
----------------------
Unit tests for model pipeline construction, cross-validation, train/test splitting,
and model serialization.
"""

from pathlib import Path
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data_preprocessing import get_processed_data_path
from src.train_model import (
    build_preprocessor,
    create_model_pipelines,
    get_models_dir,
    get_results_dir,
    perform_cross_validation,
    run_training_pipeline,
    split_data,
)


@pytest.fixture
def processed_df():
    """Load processed dataset."""
    path = get_processed_data_path()
    assert path.exists()
    return pd.read_csv(path)


def test_split_data(processed_df):
    """Test 80/20 train/test split and target stratification."""
    X_train, X_test, y_train, y_test = split_data(processed_df)

    assert len(X_train) == 242
    assert len(X_test) == 61
    assert len(y_train) == 242
    assert len(y_test) == 61
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns

    # Check stratification proportions
    train_pct = y_train.mean()
    test_pct = y_test.mean()
    assert abs(train_pct - test_pct) < 0.05


def test_create_model_pipelines():
    """Test that all 4 model pipelines are correctly constructed."""
    models = create_model_pipelines()
    expected_models = ["Logistic Regression", "Decision Tree", "Random Forest", "KNN"]

    for name in expected_models:
        assert name in models
        assert isinstance(models[name], Pipeline)
        assert "preprocessor" in models[name].named_steps
        assert "classifier" in models[name].named_steps


def test_perform_cross_validation(processed_df):
    """Test 5-Fold Stratified Cross-Validation execution on training subset."""
    X_train, _, y_train, _ = split_data(processed_df)
    models = create_model_pipelines()

    # Test with subset of models to keep test fast
    test_models = {"Decision Tree": models["Decision Tree"]}
    cv_df = perform_cross_validation(test_models, X_train, y_train)

    assert isinstance(cv_df, pd.DataFrame)
    assert len(cv_df) == 1
    assert "cv_accuracy_mean" in cv_df.columns
    assert 0.60 <= cv_df.loc[0, "cv_accuracy_mean"] <= 1.0


def test_run_training_pipeline_e2e():
    """Test the complete Phase 4 training pipeline execution and output artifact generation."""
    comp_df, cv_df, trained_models = run_training_pipeline()

    assert isinstance(comp_df, pd.DataFrame)
    assert len(comp_df) == 4
    assert set(comp_df["model"]) == {"Logistic Regression", "Decision Tree", "Random Forest", "KNN"}

    assert isinstance(cv_df, pd.DataFrame)
    assert len(cv_df) == 4

    # Verify model files exist
    models_dir = get_models_dir()
    for name in ["decision_tree_model.joblib", "logistic_regression_model.joblib", "random_forest_model.joblib", "knn_model.joblib"]:
        assert (models_dir / name).exists()

    assert (models_dir / "model_metadata.json").exists()

    # Verify result tables exist
    results_dir = get_results_dir()
    assert (results_dir / "model_comparison.csv").exists()
    assert (results_dir / "cross_validation_results.csv").exists()
    assert (results_dir / "decision_tree_feature_importance.csv").exists()
