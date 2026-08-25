"""
app.py
------
Heart Disease Risk Prediction Dashboard & Healthcare Analytics Application.
Interactive Streamlit application implementing:
  1. Home: Overview, dynamic key metrics, ML pipeline methodology
  2. Dataset Overview: Cleveland dataset provenance, schema taxonomy, interactive table
  3. Healthcare Analytics: Demographic, physiological, stress, and correlation analytics
  4. Model Performance: 4-model comparison, CV stability, tuned Decision Tree evaluation, ROC & Tree diagrams
  5. Risk Prediction: Real-time patient risk assessment with clinical input validation & explanation
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    EXPECTED_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    get_processed_data_path,
    get_project_root,
)
from src.prediction import (
    FEATURE_CONSTRAINTS,
    FEATURE_ORDER,
    MEDICAL_DISCLAIMER,
    get_default_model_path,
    load_model,
    predict_risk,
    validate_patient_input,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioRisk Analytics | Heart Disease Risk Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS for Professional Academic Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #495057;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #6c757d;
        margin-top: 4px;
    }

    /* Pipeline Step Box */
    .pipeline-step {
        background-color: #ffffff;
        border-left: 4px solid #1e3a8a;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 0 6px 6px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }

    /* Risk Badges */
    .risk-badge-low {
        background-color: #e8f5e9;
        color: #1b5e20;
        border: 1px solid #81c784;
        padding: 14px 20px;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
    }
    .risk-badge-high {
        background-color: #ffebee;
        color: #b71c1c;
        border: 1px solid #e57373;
        padding: 14px 20px;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
    }

    /* Educational Disclaimer Banner */
    .disclaimer-box {
        background-color: #fff9db;
        border-left: 4px solid #d97706;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #495057;
        margin-top: 14px;
    }

    .badge-tag {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #e2e8f0;
        color: #334155;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Data & Model Loaders
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_processed_data() -> pd.DataFrame:
    """Load and cache the processed Cleveland dataset."""
    path = get_processed_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset missing at {path}. Run Phase 2 first.")
    return pd.read_csv(path)


@st.cache_data
def get_cached_metadata() -> Dict[str, Any]:
    """Load model metadata JSON."""
    root = get_project_root()
    path = root / "models" / "model_metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_cached_selection_report() -> Dict[str, Any]:
    """Load final model selection decision report."""
    root = get_project_root()
    path = root / "reports" / "results" / "final_model_selection.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_cached_model_comparison() -> pd.DataFrame:
    """Load Phase 4 baseline model comparison table."""
    root = get_project_root()
    path = root / "reports" / "results" / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def get_cached_cv_results() -> pd.DataFrame:
    """Load Phase 4 cross-validation results table."""
    root = get_project_root()
    path = root / "reports" / "results" / "cross_validation_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def get_cached_feature_importance() -> pd.DataFrame:
    """Load final Decision Tree feature importance ranking."""
    root = get_project_root()
    path = root / "reports" / "results" / "final_feature_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def get_cached_model_pipeline():
    """Load and cache the trained final model pipeline."""
    return load_model()


def get_figure_path(filename: str) -> Optional[Path]:
    """Resolve figure path in reports/figures/."""
    root = get_project_root()
    fig_path = root / "reports" / "figures" / filename
    return fig_path if fig_path.exists() else None


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
def render_sidebar():
    """Render the application sidebar with clean, professional navigation."""
    with st.sidebar:
        st.markdown("### CardioRisk Analytics")
        st.caption("Heart Disease Risk Prediction System")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "Home",
                "Dataset Overview",
                "Healthcare Analytics",
                "Model Performance",
                "Risk Prediction",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("#### Technical Specifications")
        st.markdown(
            """
            * **Core Runtime:** Python 3.14
            * **ML Library:** Scikit-Learn
            * **Data Processing:** Pandas, NumPy
            * **Analytical Charts:** Matplotlib, Seaborn
            * **Framework:** Streamlit
            """
        )

        st.markdown("#### Selected Primary Model")
        st.info(
            "**Tuned Decision Tree Classifier**\n\n"
            "- Depth: 3 (8 Leaf Nodes)\n"
            "- Criterion: Gini Impurity (Balanced)\n"
            "- Architecture: 100% White-Box Explainability"
        )

        st.markdown("---")
        st.markdown(
            f"""
            <div class="disclaimer-box">
            <strong>Educational Notice:</strong> {MEDICAL_DISCLAIMER}
            </div>
            """,
            unsafe_allow_html=True,
        )

        return page


# -----------------------------------------------------------------------------
# 1. Home Page
# -----------------------------------------------------------------------------
def render_home_page(df: pd.DataFrame, meta: Dict[str, Any], sel_report: Dict[str, Any]):
    """Render the Home overview page with dynamic metrics and methodology."""
    st.title("Heart Disease Risk Prediction Using Decision Tree-Based Healthcare Analytics")
    st.markdown(
        "##### An academic healthcare analytics and explainable machine-learning system for cardiovascular risk classification."
    )
    st.markdown("---")

    # Dynamic Key Metrics Cards
    total_records = len(df) if not df.empty else 303
    total_features = len([c for c in df.columns if c != TARGET_COLUMN]) if not df.empty else 13
    final_metrics = sel_report.get("final_test_metrics", {})
    test_acc = final_metrics.get("accuracy", 0.8689)
    test_recall = final_metrics.get("recall", 0.8571)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Cohort Size</div>
                <div class="metric-value">{total_records}</div>
                <div class="metric-sub">Cleveland Patients</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Clinical Attributes</div>
                <div class="metric-value">{total_features}</div>
                <div class="metric-sub">Predictive Features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Primary Model</div>
                <div class="metric-value" style="font-size: 1.15rem;">Decision Tree</div>
                <div class="metric-sub">Depth 3 (Tuned)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Test Accuracy</div>
                <div class="metric-value">{test_acc:.1%}</div>
                <div class="metric-sub">Held-Out Test Set</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Test Recall</div>
                <div class="metric-value">{test_recall:.1%}</div>
                <div class="metric-sub">Sensitivity (24/28 TP)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Project Methodology and Overview
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("Project Overview & Clinical Motivation")
        st.markdown(
            """
            This application provides an interactive decision-support interface developed for healthcare analytics.
            Unlike black-box algorithms whose internal logic is opaque, this project prioritizes **clinical explainability**
            via a tuned **Decision Tree Classifier**, enabling clinicians and students to inspect exact decision pathways.

            **Clinical Dimensions Evaluated:**
            * **Demographics:** Patient chronological age and biological sex distributions.
            * **Symptom Presentations:** Chest pain classification (typical angina, atypical angina, non-anginal pain, asymptomatic).
            * **Physiological Stress Markers:** Resting blood pressure, serum cholesterol, peak heart rate achieved, exercise-induced ST depression.
            * **Diagnostic Imaging:** Fluoroscopic coronary vessel calcification (`ca`) and thallium scintigraphy defect status (`thal`).
            """
        )

        st.subheader("End-to-End Methodology Pipeline")
        steps = [
            ("1. Data Preprocessing & Validation", "Handling UTF-8 BOM encoding, schema validation, zero data leakage separation."),
            ("2. Exploratory Healthcare Analytics", "Cohort statistical profiling, correlation heatmaps, diagnostic marker comparisons."),
            ("3. Model Benchmarking & Comparison", "Evaluating Logistic Regression, Random Forest, KNN, and Decision Tree via 5-Fold Stratified CV."),
            ("4. Hyperparameter Tuning & Model Selection", "Grid search optimization preserving tree interpretability (depth 3, Gini criterion)."),
            ("5. Interactive Prediction Engine", "Clinical input validation, risk stratification, and factor attribution."),
        ]
        for title, desc in steps:
            st.markdown(
                f"""
                <div class="pipeline-step">
                    <strong>{title}</strong><br>
                    <span style="font-size: 0.85rem; color: #495057;">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.subheader("Primary Model Architecture")
        dt_fig = get_figure_path("final_decision_tree.png")
        if dt_fig:
            st.image(str(dt_fig), caption="Tuned Decision Tree Structure (Max Depth = 3)", use_container_width=True)

        st.info(
            "**White-Box Explainability:** Every node in the decision tree represents an interpretable clinical threshold, "
            "allowing full verification of why a patient is categorized as higher or lower risk."
        )


# -----------------------------------------------------------------------------
# 2. Dataset Overview Page
# -----------------------------------------------------------------------------
def render_dataset_page(df: pd.DataFrame):
    """Render the Dataset Overview page with provenance, schema, and data table."""
    st.title("Dataset Overview & Clinical Taxonomy")
    st.markdown("Comprehensive inspection of the Cleveland Heart Disease database.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Dataset Provenance & Cohort Attributes")
        st.markdown(
            """
            * **Source Repository:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
            * **Clinical Origin:** Cleveland Clinic Foundation (Dr. Robert Detrano et al.)
            * **Total Records:** 303 patient entries
            * **Predictive Features:** 13 clinical variables (5 numerical, 8 categorical)
            * **Target Variable:** Binary classification (`0`: No Heart Disease, `1`: Heart Disease Present)
            * **Data Hygiene:** 0 missing values, 0 duplicate records after validation.
            """
        )

    with col2:
        st.subheader("Diagnostic Target Distribution")
        target_counts = df[TARGET_COLUMN].value_counts()
        c0 = target_counts.get(0, 0)
        c1 = target_counts.get(1, 0)
        p0 = (c0 / len(df)) * 100
        p1 = (c1 / len(df)) * 100

        col_a, col_b = st.columns(2)
        col_a.metric("No Heart Disease (0)", f"{c0} patients", f"{p0:.1f}%")
        col_b.metric("Heart Disease Present (1)", f"{c1} patients", f"{p1:.1f}%")

        target_fig = get_figure_path("target_distribution.png")
        if target_fig:
            st.image(str(target_fig), caption="Target Class Distribution in Cleveland Cohort", use_container_width=True)

    st.markdown("---")
    st.subheader("Clinical Features Taxonomy & Supported Bounds")

    taxonomy_data = [
        {"Feature": "age", "Type": "Numerical", "Clinical Meaning": "Patient age", "Unit": "Years", "Dataset Range / Valid Values": "29 – 77"},
        {"Feature": "sex", "Type": "Categorical", "Clinical Meaning": "Biological sex", "Unit": "Binary", "Dataset Range / Valid Values": "0 = Female, 1 = Male"},
        {"Feature": "cp", "Type": "Categorical", "Clinical Meaning": "Chest pain presentation", "Unit": "Ordinal (4 levels)", "Dataset Range / Valid Values": "0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic"},
        {"Feature": "trestbps", "Type": "Numerical", "Clinical Meaning": "Resting blood pressure on hospital admission", "Unit": "mm Hg", "Dataset Range / Valid Values": "94 – 200"},
        {"Feature": "chol", "Type": "Numerical", "Clinical Meaning": "Serum cholesterol level", "Unit": "mg/dl", "Dataset Range / Valid Values": "126 – 564"},
        {"Feature": "fbs", "Type": "Categorical", "Clinical Meaning": "Fasting blood sugar > 120 mg/dl", "Unit": "Binary", "Dataset Range / Valid Values": "0 = False (≤120), 1 = True (>120)"},
        {"Feature": "restecg", "Type": "Categorical", "Clinical Meaning": "Resting electrocardiographic results", "Unit": "Nominal (3 levels)", "Dataset Range / Valid Values": "0: Normal, 1: ST-T wave abnormality, 2: LV hypertrophy"},
        {"Feature": "thalach", "Type": "Numerical", "Clinical Meaning": "Maximum peak heart rate achieved", "Unit": "bpm", "Dataset Range / Valid Values": "71 – 202"},
        {"Feature": "exang", "Type": "Categorical", "Clinical Meaning": "Exercise-induced angina", "Unit": "Binary", "Dataset Range / Valid Values": "0 = No, 1 = Yes"},
        {"Feature": "oldpeak", "Type": "Numerical", "Clinical Meaning": "ST depression induced by exercise relative to rest", "Unit": "mm", "Dataset Range / Valid Values": "0.0 – 6.2"},
        {"Feature": "slope", "Type": "Categorical", "Clinical Meaning": "Slope of the peak exercise ST segment", "Unit": "Ordinal (3 levels)", "Dataset Range / Valid Values": "0: Upsloping, 1: Flat, 2: Downsloping"},
        {"Feature": "ca", "Type": "Categorical", "Clinical Meaning": "Number of major coronary vessels colored by fluoroscopy", "Unit": "Count (0-3)", "Dataset Range / Valid Values": "0, 1, 2, 3 vessels"},
        {"Feature": "thal", "Type": "Categorical", "Clinical Meaning": "Thallium scintigraphy defect status", "Unit": "Nominal (3 levels)", "Dataset Range / Valid Values": "1: Normal, 2: Fixed defect, 3: Reversible defect"},
    ]
    st.dataframe(pd.DataFrame(taxonomy_data), use_container_width=True, hide_index=True)

    with st.expander("Inspect Processed Dataset (All 303 Records)", expanded=False):
        st.dataframe(df, use_container_width=True)
        st.caption("Displaying cleaned, standardized dataset from `data/processed/heart_disease_processed.csv`.")


# -----------------------------------------------------------------------------
# 3. Healthcare Analytics Page
# -----------------------------------------------------------------------------
def render_analytics_page(df: pd.DataFrame):
    """Render the Healthcare Analytics page with interactive filters and EDA charts."""
    st.title("Healthcare Analytics & Exploratory Insights")
    st.markdown("Analytical visualizations exploring clinical risk correlations across demographic and physiological dimensions.")
    st.markdown("---")

    # Interactive Cohort Filter Bar
    with st.container():
        st.markdown("#### Cohort Filter Controls")
        f_col1, f_col2, f_col3 = st.columns(3)

        min_age, max_age = int(df["age"].min()), int(df["age"].max())
        selected_age = f_col1.slider("Filter Age Range (Years)", min_age, max_age, (min_age, max_age))

        sex_options = {"All Sexes": None, "Male Only (sex=1)": 1, "Female Only (sex=0)": 0}
        selected_sex_label = f_col2.selectbox("Filter Biological Sex", list(sex_options.keys()))
        selected_sex = sex_options[selected_sex_label]

        target_options = {"All Diagnoses": None, "No Heart Disease (0)": 0, "Heart Disease Present (1)": 1}
        selected_target_label = f_col3.selectbox("Filter Diagnostic Outcome", list(target_options.keys()))
        selected_target = target_options[selected_target_label]

        # Apply filtering
        filtered_df = df.copy()
        filtered_df = filtered_df[(filtered_df["age"] >= selected_age[0]) & (filtered_df["age"] <= selected_age[1])]
        if selected_sex is not None:
            filtered_df = filtered_df[filtered_df["sex"] == selected_sex]
        if selected_target is not None:
            filtered_df = filtered_df[filtered_df["target"] == selected_target]

        st.caption(f"Showing **{len(filtered_df)}** of **{len(df)}** patients matching active filter criteria.")

    st.markdown("---")

    # Analytics Sub-Sections using Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Demographics & Chest Pain",
        "Physiological Biomarkers",
        "Exercise Stress & Imaging",
        "Correlation Matrix & Ranking",
    ])

    with tab1:
        st.subheader("Demographic & Symptomatic Patterns")
        c1, c2 = st.columns(2)
        with c1:
            age_fig = get_figure_path("age_by_target.png")
            if age_fig:
                st.image(str(age_fig), caption="Patient Age Distribution by Diagnostic Class", use_container_width=True)
            st.markdown(
                r"**Key Insight:** Patients diagnosed with heart disease present at an older average age "
                r"(mean $56.63 \pm 7.94$ vs $52.59 \pm 9.51$ years)."
            )

        with c2:
            cp_fig = get_figure_path("chest_pain_by_target.png")
            if cp_fig:
                st.image(str(cp_fig), caption="Chest Pain Category vs Diagnosis", use_container_width=True)
            st.markdown(
                "**Key Insight:** Asymptomatic chest pain presentation (`cp=3`) exhibits the highest cardiac risk rate "
                "at **72.92%** (105/144 patients)."
            )

        c3, c4 = st.columns(2)
        with c3:
            sex_fig = get_figure_path("heart_disease_by_sex.png")
            if sex_fig:
                st.image(str(sex_fig), caption="Heart Disease Prevalence by Biological Sex", use_container_width=True)

        with c4:
            age_dist = get_figure_path("age_distribution.png")
            if age_dist:
                st.image(str(age_dist), caption="Overall Cohort Age Distribution Histogram & KDE", use_container_width=True)

    with tab2:
        st.subheader("Physiological Biomarkers: Blood Pressure & Cholesterol")
        c1, c2 = st.columns(2)
        with c1:
            bp_fig = get_figure_path("blood_pressure_by_target.png")
            if bp_fig:
                st.image(str(bp_fig), caption="Resting Blood Pressure by Diagnosis", use_container_width=True)
            st.markdown(
                "**Key Insight:** Resting BP demonstrates modest positive elevation in the heart disease cohort "
                "(mean $134.57$ vs $129.28$ mm Hg)."
            )

        with c2:
            chol_fig = get_figure_path("cholesterol_by_target.png")
            if chol_fig:
                st.image(str(chol_fig), caption="Serum Cholesterol by Diagnosis", use_container_width=True)
            st.markdown(
                "**Key Insight:** High cholesterol levels above 240 mg/dl are frequent across both cohorts, reflecting the clinical referral nature of the dataset."
            )

    with tab3:
        st.subheader("Exercise Stress Testing & Fluoroscopic Imaging")
        c1, c2 = st.columns(2)
        with c1:
            thalach_fig = get_figure_path("max_heart_rate_by_target.png")
            if thalach_fig:
                st.image(str(thalach_fig), caption="Peak Heart Rate Achieved (thalach) by Target", use_container_width=True)
            st.markdown(
                "**Key Insight:** Peak heart rate is strongly inversely correlated ($r = -0.417$). Disease patients achieved lower maximum rate ($139.26$ vs $158.38$ bpm)."
            )

        with c2:
            oldpeak_fig = get_figure_path("st_depression_by_target.png")
            if oldpeak_fig:
                st.image(str(oldpeak_fig), caption="Exercise ST Depression (oldpeak) by Target", use_container_width=True)
            st.markdown(
                "**Key Insight:** Exercise-induced ST depression reflects myocardial ischemia (mean $1.57$ mm in disease vs $0.59$ mm in non-disease)."
            )

        c3, c4 = st.columns(2)
        with c3:
            ca_fig = get_figure_path("vessels_ca_by_target.png")
            if ca_fig:
                st.image(str(ca_fig), caption="Major Coronary Vessels Colored by Fluoroscopy", use_container_width=True)
            st.markdown("**Key Insight:** Disease prevalence escalates sharply from 26.1% ($ca=0$) to 85.0% ($ca=3$).")

        with c4:
            thal_fig = get_figure_path("thalassemia_by_target.png")
            if thal_fig:
                st.image(str(thal_fig), caption="Thallium Scintigraphy Defect Status vs Diagnosis", use_container_width=True)
            st.markdown("**Key Insight:** Reversible defect status (`thal=3`) is associated with a 76.07% disease rate.")

    with tab4:
        st.subheader("Correlation Heatmap & Feature Ranking")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            corr_fig = get_figure_path("correlation_heatmap.png")
            if corr_fig:
                st.image(str(corr_fig), caption="Pearson Correlation Matrix Heatmap", use_container_width=True)
        with c2:
            rank_fig = get_figure_path("feature_target_summary.png")
            if rank_fig:
                st.image(str(rank_fig), caption="Linear Correlation with Target Outcome", use_container_width=True)


# -----------------------------------------------------------------------------
# 4. Model Performance Page
# -----------------------------------------------------------------------------
def render_performance_page(comp_df: pd.DataFrame, cv_df: pd.DataFrame, sel_report: Dict[str, Any]):
    """Render the Model Performance & Evaluation page."""
    st.title("Machine Learning Model Performance & Evaluation")
    st.markdown("Comparative evaluation across 4 classification algorithms and deep-dive analysis of the final Decision Tree model.")
    st.markdown("---")

    st.subheader("1. Baseline Algorithm Benchmarking (Untouched 20% Test Split)")
    if not comp_df.empty:
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        comp_plot = get_figure_path("model_performance_comparison.png")
        if comp_plot:
            st.image(str(comp_plot), caption="Test Benchmark Comparison Across 4 Classifiers", use_container_width=True)

    with c2:
        st.markdown("#### 5-Fold Stratified Cross-Validation Stability")
        if not cv_df.empty:
            st.dataframe(cv_df, use_container_width=True, hide_index=True)
        st.info(
            "Cross-validation was evaluated strictly on the 80% training split ($N=242$) to prevent data leakage."
        )

    st.markdown("---")
    st.subheader("2. Final Model Deep-Dive: Tuned Decision Tree")

    st.markdown(
        """
        **Selection Rationale:** The Decision Tree Classifier was selected as the central architecture for this healthcare analytics
        project due to its strict **white-box interpretability**. While ensemble models achieve high accuracy, the Decision Tree provides
        transparent Boolean decision rules that allow full verification during clinical and academic review.
        """
    )

    # Baseline vs Tuned Comparison Table
    st.markdown("#### Baseline vs. Tuned Decision Tree Comparison")
    b_vs_t = sel_report.get("baseline_vs_tuned_comparison", {})
    if b_vs_t:
        comp_table = pd.DataFrame({
            "Evaluation Metric": b_vs_t.get("metric", []),
            "Baseline Decision Tree (Depth 4)": b_vs_t.get("baseline_decision_tree", []),
            "Tuned Decision Tree (Depth 3, Balanced)": b_vs_t.get("tuned_decision_tree", []),
        })
        st.dataframe(comp_table, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        final_cm = get_figure_path("final_confusion_matrix.png")
        if final_cm:
            st.image(str(final_cm), caption="Final Confusion Matrix (Untouched Test Set)", use_container_width=True)

    with col_b:
        final_roc = get_figure_path("final_roc_curve.png")
        if final_roc:
            st.image(str(final_roc), caption="Final Receiver Operating Characteristic (ROC) Curve", use_container_width=True)

    st.markdown("---")
    st.subheader("3. Decision Tree Visualization & Feature Importance")

    col_tree, col_fi = st.columns([1.3, 1])
    with col_tree:
        final_tree = get_figure_path("final_decision_tree.png")
        if final_tree:
            st.image(str(final_tree), caption="Final Interpretable Decision Tree Diagram (Depth = 3)", use_container_width=True)

    with col_fi:
        final_fi = get_figure_path("final_feature_importance.png")
        if final_fi:
            st.image(str(final_fi), caption="Gini Feature Importance Reduction Ranking", use_container_width=True)
        st.caption(
            "**Methodological Note:** Feature importance measures how influential an attribute was in reducing impurity "
            "within this statistical model; it does not establish medical causality."
        )


# -----------------------------------------------------------------------------
# 5. Risk Prediction Page
# -----------------------------------------------------------------------------
def render_prediction_page():
    """Render the interactive Patient Risk Prediction interface."""
    st.title("Interactive Patient Heart Disease Risk Prediction")
    st.markdown("Enter patient clinical attributes below to assess estimated cardiovascular risk using the trained Decision Tree pipeline.")
    st.markdown("---")

    # Preset Patient Profiles for Fast Demonstration
    st.markdown("#### Demonstration Presets")
    col_pre1, col_pre2, col_pre3 = st.columns(3)

    if "profile" not in st.session_state:
        st.session_state.profile = {
            "age": 55,
            "sex": 1,
            "cp": 0,
            "trestbps": 130,
            "chol": 240,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        }

    if col_pre1.button("Load Low-Risk Profile Preset", use_container_width=True):
        st.session_state.profile = {
            "age": 42,
            "sex": 0,
            "cp": 1,
            "trestbps": 115,
            "chol": 190,
            "fbs": 0,
            "restecg": 0,
            "thalach": 175,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        }
        st.rerun()

    if col_pre2.button("Load High-Risk Profile Preset", use_container_width=True):
        st.session_state.profile = {
            "age": 62,
            "sex": 1,
            "cp": 3,
            "trestbps": 155,
            "chol": 290,
            "fbs": 1,
            "restecg": 2,
            "thalach": 115,
            "exang": 1,
            "oldpeak": 3.0,
            "slope": 1,
            "ca": 2,
            "thal": 3,
        }
        st.rerun()

    if col_pre3.button("Reset to Default Values", use_container_width=True):
        st.session_state.profile = {
            "age": 55,
            "sex": 1,
            "cp": 0,
            "trestbps": 130,
            "chol": 240,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        }
        st.rerun()

    p = st.session_state.profile

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form
    with st.form("risk_prediction_form"):
        st.markdown("### Clinical Parameter Entry Form")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("##### 1. Demographics & Vitals")
            age = st.number_input("Age (Years)", min_value=18, max_value=100, value=int(p["age"]), step=1)
            sex_label = st.selectbox(
                "Biological Sex",
                options=["Female (0)", "Male (1)"],
                index=int(p["sex"]),
            )
            sex = 1 if "Male" in sex_label else 0

            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=80,
                max_value=220,
                value=int(p["trestbps"]),
                step=1,
                help="Resting blood pressure on hospital admission.",
            )

            chol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=100,
                max_value=600,
                value=int(p["chol"]),
                step=1,
                help="Serum cholesterol measurement.",
            )

        with c2:
            st.markdown("##### 2. Exercise & ECG Diagnostics")
            thalach = st.number_input(
                "Max Heart Rate Achieved (bpm)",
                min_value=60,
                max_value=220,
                value=int(p["thalach"]),
                step=1,
                help="Maximum heart rate reached during treadmill exercise stress testing.",
            )

            oldpeak = st.number_input(
                "ST Depression (oldpeak mm)",
                min_value=0.0,
                max_value=8.0,
                value=float(p["oldpeak"]),
                step=0.1,
                format="%.1f",
                help="ST depression induced by exercise relative to rest.",
            )

            exang_label = st.selectbox(
                "Exercise-Induced Angina",
                options=["No (0)", "Yes (1)"],
                index=int(p["exang"]),
                help="Did the patient experience angina during exercise?",
            )
            exang = 1 if "Yes" in exang_label else 0

            slope_options = ["0: Upsloping", "1: Flat", "2: Downsloping"]
            slope_idx = min(int(p["slope"]), 2)
            slope_label = st.selectbox(
                "Peak Exercise ST Slope",
                options=slope_options,
                index=slope_idx,
            )
            slope = int(slope_label.split(":")[0])

        with c3:
            st.markdown("##### 3. Symptoms & Scintigraphy")
            cp_options = [
                "0: Typical Angina",
                "1: Atypical Angina",
                "2: Non-Anginal Pain",
                "3: Asymptomatic",
            ]
            cp_idx = min(int(p["cp"]), 3)
            cp_label = st.selectbox("Chest Pain Type", options=cp_options, index=cp_idx)
            cp = int(cp_label.split(":")[0])

            fbs_label = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                options=["False (0: ≤ 120 mg/dl)", "True (1: > 120 mg/dl)"],
                index=int(p["fbs"]),
            )
            fbs = 1 if "True" in fbs_label else 0

            restecg_options = [
                "0: Normal",
                "1: ST-T Wave Abnormality",
                "2: Left Ventricular Hypertrophy",
            ]
            restecg_idx = min(int(p["restecg"]), 2)
            restecg_label = st.selectbox("Resting ECG Results", options=restecg_options, index=restecg_idx)
            restecg = int(restecg_label.split(":")[0])

            ca_options = ["0 vessels", "1 vessel", "2 vessels", "3 vessels"]
            ca_idx = min(int(p["ca"]), 3)
            ca_label = st.selectbox("Major Vessels Colored by Fluoroscopy (ca)", options=ca_options, index=ca_idx)
            ca = int(ca_label.split(" ")[0])

            thal_options = [
                "1: Normal",
                "2: Fixed Defect",
                "3: Reversible Defect",
            ]
            thal_val = int(p["thal"])
            thal_idx = 0 if thal_val == 1 else (1 if thal_val == 2 else 2)
            thal_label = st.selectbox("Thalassemia Scintigraphy Status", options=thal_options, index=thal_idx)
            thal = int(thal_label.split(":")[0])

        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("Run Heart Disease Risk Assessment", type="primary", use_container_width=True)

    # Process Prediction upon Submit
    if submit_button:
        patient_dict = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }

        try:
            pipeline = get_cached_model_pipeline()
            result = predict_risk(patient_dict, model=pipeline)

            st.markdown("---")
            st.subheader("Model Risk Assessment Results")

            r_col1, r_col2 = st.columns([1.2, 1])

            with r_col1:
                is_high_risk = result["prediction"] == 1
                badge_class = "risk-badge-high" if is_high_risk else "risk-badge-low"

                st.markdown(
                    f"""
                    <div class="{badge_class}">
                        Model Assessment: {result['predicted_category']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Estimated Risk Tier:** `{result['risk_level']}`")
                st.markdown(f"**Model-Estimated Disease Probability:** `{result['heart_disease_percentage']}%`")
                st.progress(float(result["heart_disease_probability"]))

                st.markdown(f"**Classification Confidence Score:** `{result['confidence_score']:.1%}`")

            with r_col2:
                st.markdown("##### Identified Contributing Clinical Factors")
                factors = result.get("contributing_clinical_factors", [])
                if factors:
                    for f in factors:
                        st.markdown(f"* **{f}**")
                else:
                    st.markdown("No severe clinical risk markers triggered in this patient profile.")

            with st.expander("Review Submitted Clinical Feature Values", expanded=False):
                st.json(result["input_features"])

            st.markdown(
                f"""
                <div class="disclaimer-box">
                <strong>Important Notice:</strong> {result['medical_disclaimer']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            logger.error(f"Prediction failure: {e}", exc_info=True)


# -----------------------------------------------------------------------------
# Main Application Controller
# -----------------------------------------------------------------------------
def main():
    """Main application entry point."""
    try:
        df = get_cached_processed_data()
        meta = get_cached_metadata()
        sel_report = get_cached_selection_report()
        comp_df = get_cached_model_comparison()
        cv_df = get_cached_cv_results()
    except Exception as e:
        st.error(f"Failed to load project artifacts: {e}. Please ensure Phases 1-5 have completed.")
        return

    # Render Sidebar & Get Selected Page
    selected_page = render_sidebar()

    # Route to Selected Page
    if selected_page == "Home":
        render_home_page(df, meta, sel_report)
    elif selected_page == "Dataset Overview":
        render_dataset_page(df)
    elif selected_page == "Healthcare Analytics":
        render_analytics_page(df)
    elif selected_page == "Model Performance":
        render_performance_page(comp_df, cv_df, sel_report)
    elif selected_page == "Risk Prediction":
        render_prediction_page()


if __name__ == "__main__":
    main()
