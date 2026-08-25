"""
test_data_preprocessing.py
--------------------------
Unit tests for data preprocessing, schema validation, quality checks,
target transformation, and zero-leakage feature separation.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    EXPECTED_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    clean_data,
    generate_data_quality_report,
    get_feature_types,
    get_processed_data_path,
    get_raw_data_path,
    load_raw_data,
    run_preprocessing_pipeline,
    save_processed_data,
    separate_features_target,
    transform_target,
    validate_columns,
)


@pytest.fixture
def raw_df():
    """Load the actual raw dataset for testing."""
    return load_raw_data()


@pytest.fixture
def sample_raw_df():
    """Create a synthetic mini-DataFrame replicating Cleveland structure."""
    data = {
        "age": [63, 67, 67, 37, 41],
        "sex": [1, 1, 1, 1, 0],
        "cp": [0, 3, 3, 2, 1],
        "trestbps": [145, 160, 120, 130, 130],
        "chol": [233, 286, 229, 250, 204],
        "fbs": [1, 0, 0, 0, 0],
        "restecg": [2, 2, 2, 0, 2],
        "thalach": [150, 108, 129, 187, 172],
        "exang": [0, 1, 1, 0, 0],
        "oldpeak": [2.3, 1.5, 2.6, 3.5, 1.4],
        "slope": [2, 1, 1, 2, 0],
        "ca": [0, 3, 2, 0, 0],
        "thal": [2, 1, 3, 1, 1],
        "target": [0, 1, 2, 0, 0],  # Includes multi-class value '2' for testing mapping
    }
    return pd.DataFrame(data)


def test_raw_data_file_exists():
    """Test that the raw dataset file exists at the expected path."""
    path = get_raw_data_path()
    assert path.exists(), f"Raw data file does not exist at {path}"


def test_load_raw_data(raw_df):
    """Test that raw data loads cleanly and has expected dimensions."""
    assert isinstance(raw_df, pd.DataFrame)
    assert len(raw_df) == 303, f"Expected 303 rows, got {len(raw_df)}"
    assert len(raw_df.columns) == 14, f"Expected 14 columns, got {len(raw_df.columns)}"
    for col in EXPECTED_COLUMNS:
        assert col in raw_df.columns, f"Expected column '{col}' not found in raw data"


def test_validate_columns(sample_raw_df):
    """Test that column validation succeeds on valid data and raises on missing columns."""
    # Should pass without error
    validate_columns(sample_raw_df)

    # Missing column should raise ValueError
    invalid_df = sample_raw_df.drop(columns=["age"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_columns(invalid_df)


def test_clean_data_duplicates():
    """Test duplicate detection and removal."""
    data = {col: [1, 1, 2] for col in EXPECTED_COLUMNS}
    df_with_dups = pd.DataFrame(data)
    cleaned_df, meta = clean_data(df_with_dups, drop_duplicates=True)
    assert meta["duplicates_detected"] == 1
    assert meta["duplicates_removed"] == 1
    assert len(cleaned_df) == 2


def test_clean_data_missing_values():
    """Test imputation of missing values."""
    data = {
        "age": [60.0, np.nan, 40.0],
        "sex": [1, 0, 1],
        "cp": [0, 1, 2],
        "trestbps": [120, 130, 140],
        "chol": [200, 220, 240],
        "fbs": [0, 1, 0],
        "restecg": [0, 1, 2],
        "thalach": [150, 160, 170],
        "exang": [0, 1, 0],
        "oldpeak": [1.0, 2.0, 3.0],
        "slope": [0, 1, 2],
        "ca": [0, 1, 0],
        "thal": [1, 2, 3],
        "target": [0, 1, 0],
    }
    df_missing = pd.DataFrame(data)
    cleaned_df, meta = clean_data(df_missing)
    assert meta["missing_values_detected"] == 1
    assert cleaned_df["age"].isnull().sum() == 0
    # Median of [60, 40] is 50.0
    assert cleaned_df.loc[1, "age"] == 50.0


def test_transform_target(sample_raw_df):
    """Test standardisation of target variable to binary (0 and 1)."""
    transformed_df, stats = transform_target(sample_raw_df)
    unique_targets = set(transformed_df["target"].unique())
    assert unique_targets.issubset({0, 1})
    # Value 2 in sample_raw_df row index 2 should have been converted to 1
    assert transformed_df.loc[2, "target"] == 1
    assert stats["total_records"] == 5
    assert stats["counts"]["no_disease_0"] == 3
    assert stats["counts"]["disease_1"] == 2


def test_separate_features_target(sample_raw_df):
    """Test feature and target separation and verify zero target leakage."""
    X, y = separate_features_target(sample_raw_df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(sample_raw_df)
    assert len(y) == len(sample_raw_df)
    assert TARGET_COLUMN not in X.columns
    assert len(X.columns) == 13


def test_feature_types_taxonomy():
    """Test that feature types dictionary covers all features with no overlap."""
    taxonomy = get_feature_types()
    all_feats = set(taxonomy["all_features"])
    num_feats = set(taxonomy["numerical_features"])
    cat_feats = set(taxonomy["categorical_features"])

    assert len(all_feats) == 13
    assert num_feats.isdisjoint(cat_feats), "Numerical and categorical features must not overlap"
    assert num_feats.union(cat_feats) == all_feats, "Union of numerical and categorical features must equal all features"


def test_save_and_reload_processed_data(sample_raw_df):
    """Test saving and re-loading processed CSV dataset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test_processed.csv"
        saved_path = save_processed_data(sample_raw_df, output_path=tmp_path)
        assert saved_path.exists()
        reloaded = pd.read_csv(saved_path)
        assert len(reloaded) == len(sample_raw_df)
        assert list(reloaded.columns) == list(sample_raw_df.columns)


def test_run_preprocessing_pipeline_e2e():
    """Test the full end-to-end preprocessing pipeline on real data."""
    processed_df, report = run_preprocessing_pipeline()
    assert isinstance(processed_df, pd.DataFrame)
    assert len(processed_df) == 303
    assert len(processed_df.columns) == 14
    assert processed_df["target"].isin([0, 1]).all()
    assert report["summary"]["processed_row_count"] == 303
    assert report["data_leakage_prevented"] is True

    processed_file = get_processed_data_path()
    assert processed_file.exists()
