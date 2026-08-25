"""
test_model_evaluation.py
------------------------
Unit tests for metric computation, confusion matrix plotting,
and feature importance extraction.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.evaluate_model import (
    compute_and_plot_feature_importance,
    compute_classification_metrics,
    extract_feature_names_from_pipeline,
    plot_confusion_matrix,
    plot_model_comparison,
)
from src.train_model import create_model_pipelines, split_data
from src.data_preprocessing import get_processed_data_path


@pytest.fixture
def processed_df():
    path = get_processed_data_path()
    return pd.read_csv(path)


def test_compute_classification_metrics():
    """Test metric computation with synthetic perfect and mixed predictions."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.95])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_plot_confusion_matrix():
    """Test confusion matrix calculation and figure export."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "test_cm.png"
        cm, saved = plot_confusion_matrix(y_true, y_pred, "TestModel", output_path=out_path)

        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 1  # TP
        assert saved.exists()


def test_feature_importance_extraction(processed_df):
    """Test extracting Gini feature importances from a trained Decision Tree pipeline."""
    X_train, _, y_train, _ = split_data(processed_df)
    pipelines = create_model_pipelines()
    dt_pipeline = pipelines["Decision Tree"]
    dt_pipeline.fit(X_train, y_train)

    feature_names = extract_feature_names_from_pipeline(dt_pipeline)
    assert len(feature_names) > 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "fi_test.png"
        fi_df, saved = compute_and_plot_feature_importance(
            dt_pipeline,
            feature_names=feature_names,
            model_name="Decision Tree",
            output_path=out_path,
        )

        assert isinstance(fi_df, pd.DataFrame)
        assert "feature" in fi_df.columns
        assert "importance" in fi_df.columns
        assert abs(fi_df["importance"].sum() - 1.0) < 0.01
        assert saved.exists()
