"""
test_exploratory_analysis.py
----------------------------
Unit tests for the exploratory analysis module and figure generation.
"""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.data_preprocessing import get_processed_data_path
from src.exploratory_analysis import (
    compute_summary_statistics,
    generate_all_eda_figures,
    plot_target_distribution,
    plot_correlation_heatmap,
    get_figures_dir,
)


@pytest.fixture
def processed_df():
    """Load the processed dataset for testing."""
    path = get_processed_data_path()
    assert path.exists(), f"Processed data file does not exist at {path}"
    return pd.read_csv(path)


def test_compute_summary_statistics(processed_df):
    """Test that summary statistics dictionary contains all required fields and correct values."""
    stats = compute_summary_statistics(processed_df)

    assert stats["record_count"] == 303
    assert stats["target_distribution"]["counts"][0] == 164
    assert stats["target_distribution"]["counts"][1] == 139
    assert stats["target_distribution"]["percentages"][0] == 54.13
    assert stats["target_distribution"]["percentages"][1] == 45.87

    # Check numerical statistics
    for num_col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        assert num_col in stats["numerical_statistics"]
        assert "overall_mean" in stats["numerical_statistics"][num_col]
        assert "by_target" in stats["numerical_statistics"][num_col]

    # Check outlier analysis
    for num_col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        assert num_col in stats["outlier_analysis"]
        assert "iqr" in stats["outlier_analysis"][num_col]
        assert "outlier_count" in stats["outlier_analysis"][num_col]


def test_plot_target_distribution(processed_df):
    """Test individual target distribution plot generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        out_fig = plot_target_distribution(processed_df, output_dir=tmp_path)
        assert out_fig.exists()
        assert out_fig.stat().st_size > 1000


def test_plot_correlation_heatmap(processed_df):
    """Test correlation heatmap figure generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        out_fig = plot_correlation_heatmap(processed_df, output_dir=tmp_path)
        assert out_fig.exists()
        assert out_fig.stat().st_size > 1000


def test_generate_all_eda_figures(processed_df):
    """Test batch generation of all 14 publication figures."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        figures = generate_all_eda_figures(processed_df, output_dir=tmp_path)
        assert len(figures) == 14
        for fig in figures:
            assert fig.exists()
            assert fig.stat().st_size > 1000
