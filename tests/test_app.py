"""
test_app.py
-----------
Unit tests for Streamlit application loading, cached data accessors,
and figure resolution.
"""

from pathlib import Path
import pandas as pd
import pytest

from app.app import (
    get_cached_cv_results,
    get_cached_metadata,
    get_cached_model_comparison,
    get_cached_model_pipeline,
    get_cached_processed_data,
    get_cached_selection_report,
    get_figure_path,
)


def test_get_cached_processed_data():
    """Verify processed dataset loader in app."""
    df = get_cached_processed_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 303
    assert "target" in df.columns


def test_get_cached_metadata_and_selection():
    """Verify JSON metadata and selection report loaders in app."""
    meta = get_cached_metadata()
    assert isinstance(meta, dict)
    assert "metrics_held_out_test" in meta

    sel = get_cached_selection_report()
    assert isinstance(sel, dict)
    assert "selected_model" in sel


def test_get_cached_comparison_and_cv_results():
    """Verify model comparison and CV results CSV loaders."""
    comp_df = get_cached_model_comparison()
    assert isinstance(comp_df, pd.DataFrame)
    assert len(comp_df) == 4

    cv_df = get_cached_cv_results()
    assert isinstance(cv_df, pd.DataFrame)
    assert len(cv_df) == 4


def test_get_cached_model_pipeline():
    """Verify final model pipeline loading in app."""
    pipeline = get_cached_model_pipeline()
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")


def test_get_figure_paths():
    """Verify figure path resolutions for critical dashboard visualizations."""
    expected_figures = [
        "final_confusion_matrix.png",
        "final_roc_curve.png",
        "final_decision_tree.png",
        "final_feature_importance.png",
        "correlation_heatmap.png",
        "target_distribution.png",
    ]
    for fig_name in expected_figures:
        fig_path = get_figure_path(fig_name)
        assert fig_path is not None, f"Figure {fig_name} could not be resolved"
        assert fig_path.exists()
