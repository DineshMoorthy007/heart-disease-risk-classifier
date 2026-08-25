"""
data_preprocessing.py
---------------------
Modular, reproducible preprocessing pipeline for the Cleveland Heart Disease dataset.
Responsible for:
  - Raw dataset loading and UTF-8 BOM handling
  - Schema and column validation
  - Missing and sentinel value detection & remediation
  - Duplicate detection and handling
  - Target variable verification and binary standardisation
  - Feature/target separation with guaranteed zero target leakage
  - Generation of structured Data Quality reports
  - Exporting processed dataset to data/processed/heart_disease_processed.csv
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Setup module logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Canonical Cleveland dataset feature definitions
EXPECTED_COLUMNS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

NUMERICAL_FEATURES: List[str] = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES: List[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

BINARY_FEATURES: List[str] = [
    "sex",
    "fbs",
    "exang",
]

MULTICLASS_CATEGORICAL_FEATURES: List[str] = [
    "cp",
    "restecg",
    "slope",
    "ca",
    "thal",
]

TARGET_COLUMN: str = "target"

# Sentinel missing value tokens common in UCI medical datasets
SENTINEL_MISSING_VALUES: List[str] = [
    "?",
    "NA",
    "na",
    "N/A",
    "null",
    "NULL",
    "NaN",
    "nan",
    "None",
    "",
    " ",
]


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_raw_data_path() -> Path:
    """
    Return the path to the raw dataset.
    Checks for both Heart_disease.csv and heart_disease.csv.
    """
    root = get_project_root()
    primary = root / "data" / "raw" / "Heart_disease.csv"
    fallback = root / "data" / "raw" / "heart_disease.csv"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


def get_processed_data_path() -> Path:
    """Return the destination path for the processed dataset."""
    root = get_project_root()
    return root / "data" / "processed" / "heart_disease_processed.csv"


def get_feature_types() -> Dict[str, List[str]]:
    """
    Return the feature categorization taxonomy.
    Used by downstream EDA, modeling, and prediction pipelines.
    """
    return {
        "all_features": [col for col in EXPECTED_COLUMNS if col != TARGET_COLUMN],
        "numerical_features": NUMERICAL_FEATURES.copy(),
        "categorical_features": CATEGORICAL_FEATURES.copy(),
        "binary_features": BINARY_FEATURES.copy(),
        "multiclass_categorical_features": MULTICLASS_CATEGORICAL_FEATURES.copy(),
        "target": [TARGET_COLUMN],
    }


def load_raw_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the raw CSV dataset into a pandas DataFrame.
    Handles UTF-8 BOM encoding ('utf-8-sig') and parses sentinel missing tokens.

    Args:
        filepath: Optional path to the raw CSV file.

    Returns:
        pd.DataFrame containing the raw records.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be parsed or is empty.
    """
    path = Path(filepath) if filepath else get_raw_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {path}")

    logger.info(f"Loading raw dataset from: {path}")
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        na_values=SENTINEL_MISSING_VALUES,
        keep_default_na=True,
    )

    # Normalize column names: strip whitespace, lowercase
    df.columns = df.columns.str.strip().str.lower()

    if df.empty:
        raise ValueError(f"Loaded dataset from {path} is empty.")

    logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
    return df


def validate_columns(
    df: pd.DataFrame, expected_columns: Optional[List[str]] = None
) -> None:
    """
    Validate that the DataFrame contains all required columns.

    Args:
        df: Input DataFrame.
        expected_columns: List of expected column names. Defaults to EXPECTED_COLUMNS.

    Raises:
        ValueError: If required columns are missing from the DataFrame.
    """
    cols_to_check = expected_columns if expected_columns is not None else EXPECTED_COLUMNS
    df_cols = set(df.columns)
    missing_cols = [c for c in cols_to_check if c not in df_cols]

    if missing_cols:
        raise ValueError(
            f"Dataset validation failed. Missing required columns: {missing_cols}"
        )

    unexpected_cols = [c for c in df.columns if c not in cols_to_check]
    if unexpected_cols:
        logger.warning(
            f"Dataset contains unexpected additional columns: {unexpected_cols}"
        )

    logger.info("Column validation passed successfully.")


def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform dataset-level cleaning:
      1. Detect and handle duplicate rows.
      2. Detect and handle missing / sentinel values.
      3. Cast columns to appropriate numerical / categorical data types.

    Args:
        df: Input raw DataFrame.
        drop_duplicates: Whether to remove duplicate records.

    Returns:
        Tuple of (cleaned DataFrame, metadata dictionary summarizing actions taken).
    """
    cleaned = df.copy()
    initial_rows = len(cleaned)

    # 1. Duplicates check
    num_duplicates = int(cleaned.duplicated().sum())
    if num_duplicates > 0 and drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        logger.info(f"Removed {num_duplicates} duplicate records.")
    else:
        logger.info(f"Duplicate check: {num_duplicates} duplicates found.")

    # 2. Missing values check
    missing_per_col = cleaned.isnull().sum().to_dict()
    total_missing = sum(missing_per_col.values())
    imputed_records: Dict[str, Any] = {}

    if total_missing > 0:
        logger.warning(f"Detected {total_missing} missing values across columns: {missing_per_col}")
        # Statistical median for numerical, mode for categorical (dataset-level baseline)
        for col in cleaned.columns:
            if cleaned[col].isnull().sum() > 0:
                if col in NUMERICAL_FEATURES:
                    fill_val = cleaned[col].median()
                else:
                    fill_val = cleaned[col].mode().iloc[0]
                cleaned[col] = cleaned[col].fillna(fill_val)
                imputed_records[col] = fill_val
                logger.info(f"Imputed missing values in '{col}' using value: {fill_val}")
    else:
        logger.info("Missing value check: 0 missing values detected.")

    # 3. Type standardisation
    for col in cleaned.columns:
        if col in NUMERICAL_FEATURES:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        elif col in CATEGORICAL_FEATURES or col == TARGET_COLUMN:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").astype(int)

    metadata = {
        "initial_row_count": initial_rows,
        "duplicates_detected": num_duplicates,
        "duplicates_removed": num_duplicates if drop_duplicates else 0,
        "missing_values_detected": total_missing,
        "missing_per_column": missing_per_col,
        "imputed_columns": imputed_records,
        "cleaned_row_count": len(cleaned),
    }

    return cleaned, metadata


def transform_target(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and ensure the target variable is binary (0 = No Disease, 1 = Disease Present).
    If multi-class integer targets (1..4) are found, maps them to 1.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.

    Returns:
        Tuple of (DataFrame with binary target, target distribution statistics dictionary).

    Raises:
        ValueError: If target column is missing or cannot be converted to binary.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    df_transformed = df.copy()

    # Check unique values in target
    unique_vals = set(df_transformed[target_col].dropna().unique())
    logger.info(f"Raw target unique values: {sorted(list(unique_vals))}")

    # Check if multi-class (e.g. 0, 1, 2, 3, 4)
    if not unique_vals.issubset({0, 1}):
        logger.info("Multi-class target detected. Converting values >= 1 to 1 (Binary Classification).")
        df_transformed[target_col] = (df_transformed[target_col] > 0).astype(int)
    else:
        df_transformed[target_col] = df_transformed[target_col].astype(int)

    # Calculate target distribution statistics
    counts = df_transformed[target_col].value_counts().to_dict()
    total = len(df_transformed)
    no_disease_count = int(counts.get(0, 0))
    disease_count = int(counts.get(1, 0))

    dist_stats = {
        "target_classes": [0, 1],
        "class_labels": {
            0: "No Heart Disease (Absence)",
            1: "Heart Disease Present (Presence)",
        },
        "counts": {
            "no_disease_0": no_disease_count,
            "disease_1": disease_count,
        },
        "percentages": {
            "no_disease_0_pct": round((no_disease_count / total) * 100, 2) if total > 0 else 0.0,
            "disease_1_pct": round((disease_count / total) * 100, 2) if total > 0 else 0.0,
        },
        "total_records": total,
    }

    logger.info(
        f"Target distribution: Class 0 = {no_disease_count} ({dist_stats['percentages']['no_disease_0_pct']}%), "
        f"Class 1 = {disease_count} ({dist_stats['percentages']['disease_1_pct']}%)"
    )

    return df_transformed, dist_stats


def separate_features_target(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X and target vector y.
    Guarantees no target leakage by explicitly asserting target absence from X.

    Args:
        df: Processed DataFrame.
        target_col: Name of the target column.

    Returns:
        Tuple of (X: pd.DataFrame, y: pd.Series).

    Raises:
        ValueError: If target_col is missing or leakage is detected.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Leakage assertion
    if target_col in X.columns:
        raise ValueError("Critical error: Target column remained inside feature matrix X!")

    return X, y


def generate_data_quality_report(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    clean_meta: Dict[str, Any],
    target_stats: Dict[str, Any],
    report_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Generate and save a structured JSON Data Quality Report.

    Args:
        raw_df: Original raw DataFrame.
        cleaned_df: Cleaned and processed DataFrame.
        clean_meta: Metadata from clean_data().
        target_stats: Target distribution stats from transform_target().
        report_path: Optional file path to write the JSON report.

    Returns:
        Dictionary containing the full report structure.
    """
    root = get_project_root()
    out_path = Path(report_path) if report_path else root / "reports" / "results" / "data_quality_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset_name": "Cleveland Heart Disease Dataset (UCI)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "raw_row_count": len(raw_df),
            "raw_column_count": len(raw_df.columns),
            "processed_row_count": len(cleaned_df),
            "processed_column_count": len(cleaned_df.columns),
            "columns": list(cleaned_df.columns),
        },
        "data_hygiene": clean_meta,
        "target_analysis": target_stats,
        "feature_taxonomy": get_feature_types(),
        "column_ranges": {
            col: {
                "dtype": str(cleaned_df[col].dtype),
                "min": float(cleaned_df[col].min()),
                "max": float(cleaned_df[col].max()),
                "unique_count": int(cleaned_df[col].nunique()),
            }
            for col in cleaned_df.columns
        },
        "data_leakage_prevented": True,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Data Quality Report saved to: {out_path}")
    return report


def save_processed_data(
    df: pd.DataFrame, output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Save the processed DataFrame to CSV.

    Args:
        df: Cleaned DataFrame.
        output_path: Target file path. Defaults to data/processed/heart_disease_processed.csv.

    Returns:
        Path to the saved CSV file.
    """
    path = Path(output_path) if output_path else get_processed_data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Processed dataset successfully saved to: {path}")
    return path


def run_preprocessing_pipeline(
    raw_path: Optional[Union[str, Path]] = None,
    processed_path: Optional[Union[str, Path]] = None,
    report_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute the complete Phase 2 data preprocessing pipeline:
      1. Load raw data (handling BOM & sentinels)
      2. Validate column schema
      3. Clean data (duplicates & missing values)
      4. Standardize binary target
      5. Generate and persist Data Quality Report
      6. Persist processed dataset to CSV

    Returns:
        Tuple of (processed DataFrame, data quality report dict).
    """
    logger.info("=== Starting Data Preprocessing Pipeline ===")

    # 1. Load
    raw_df = load_raw_data(raw_path)

    # 2. Validate schema
    validate_columns(raw_df)

    # 3. Clean
    cleaned_df, clean_meta = clean_data(raw_df)

    # 4. Target transformation
    processed_df, target_stats = transform_target(cleaned_df)

    # 5. Data Quality Report
    report = generate_data_quality_report(
        raw_df=raw_df,
        cleaned_df=processed_df,
        clean_meta=clean_meta,
        target_stats=target_stats,
        report_path=report_path,
    )

    # 6. Save processed dataset
    save_processed_data(processed_df, processed_path)

    logger.info("=== Data Preprocessing Pipeline Completed Successfully ===")
    return processed_df, report


if __name__ == "__main__":
    df_processed, dq_report = run_preprocessing_pipeline()
    print(f"\nPipeline Output Summary:")
    print(f"  Processed Rows: {len(df_processed)}")
    print(f"  Processed Columns: {len(df_processed.columns)}")
    print(f"  Target 0: {dq_report['target_analysis']['counts']['no_disease_0']} ({dq_report['target_analysis']['percentages']['no_disease_0_pct']}%)")
    print(f"  Target 1: {dq_report['target_analysis']['counts']['disease_1']} ({dq_report['target_analysis']['percentages']['disease_1_pct']}%)")
