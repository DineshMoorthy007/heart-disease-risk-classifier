# Heart Disease Risk Prediction Using Decision Tree-Based Healthcare Analytics

An academic machine-learning and healthcare analytics system to predict heart disease risk based on clinical features from the Cleveland Heart Disease dataset.

---

## 1. Project Overview

This project implements an end-to-end healthcare analytics and machine-learning pipeline:
* **Exploratory Data Analysis (EDA)** on demographic, symptomatic, physiological, and fluoroscopic markers.
* **Explainable Machine Learning** using Decision Tree classification as the central, white-box model.
* **Comparative Model Benchmarking** across Logistic Regression, Random Forest, and K-Nearest Neighbors.
* **Controlled Hyperparameter Tuning** strictly within training cross-validation folds.
* **Reusable Risk Prediction Pipeline** with clinical sanity validation and stratified risk scoring.
* **Interactive Risk Prediction Dashboard** built with Streamlit (Phase 6).

---

## 2. Dataset Attribution

* **Source:** [UCI Machine Learning Repository — Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
* **Subset:** Cleveland Heart Disease Database (303 records, 13 predictive clinical features, 1 target variable).

---

## 3. Project Structure

```text
heart-disease-risk-classifier/
│
├── data/
│   ├── raw/
│   │   └── Heart_disease.csv             # Immutable raw Cleveland dataset (303 rows)
│   └── processed/
│       └── heart_disease_processed.csv   # Cleaned, standardized dataset (303 rows)
│
├── notebooks/
│   └── exploratory_analysis.ipynb        # Phase 3: Executed exploratory analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py             # Phase 2: Modular preprocessing pipeline
│   ├── exploratory_analysis.py           # Phase 3: Analytical visualizations
│   ├── train_model.py                    # Phase 4 & 5: Training, tuning, & final persistence
│   ├── evaluate_model.py                 # Phase 4 & 5: Metrics, curves, trees, & feature importances
│   └── prediction.py                     # Phase 5: Reusable inference & validation pipeline
│
├── models/
│   ├── final_model_pipeline.pkl          # Serialized complete final model pipeline
│   ├── final_model_pipeline.joblib
│   ├── decision_tree_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   ├── knn_model.joblib
│   └── model_metadata.json               # Schema, hyperparameters, & test metrics
│
├── reports/
│   ├── figures/                          # 25 publication-quality analytical and ML figures
│   │   ├── final_confusion_matrix.png
│   │   ├── final_roc_curve.png
│   │   ├── final_decision_tree.png
│   │   ├── final_feature_importance.png
│   │   └── ...
│   └── results/
│       ├── data_quality_report.json
│       ├── model_comparison.csv
│       ├── cross_validation_results.csv
│       ├── final_model_selection.json
│       └── final_feature_importance.csv
│
├── tests/
│   ├── test_data_preprocessing.py        # 10 unit tests for data preprocessing
│   ├── test_exploratory_analysis.py      # 4 unit tests for EDA
│   ├── test_model_training.py            # 4 unit tests for training & splitting
│   ├── test_model_evaluation.py          # 3 unit tests for evaluation & trees
│   └── test_prediction.py                # 8 unit tests for inference & validation
│
├── requirements.txt
├── pytest.ini
├── conftest.py
├── README.md
└── .gitignore
```

---

## 4. Data Preprocessing & Quality Pipeline (Phase 2)

The data preprocessing module (`src/data_preprocessing.py`) handles:
1. **Raw Data Preservation:** The raw dataset in `data/raw/` is treated as immutable and never modified.
2. **Encoding & BOM Handling:** Handles UTF-8 BOM (`utf-8-sig`) without column corruption.
3. **Schema Validation:** Strict verification of all 13 clinical features and the target variable.
4. **Data Hygiene:** Explicit checks for duplicate rows and sentinel missing-value tokens (`?`, `NA`, `null`).
5. **Target Standardisation:** Validates binary classification representation ($0 = \text{No Heart Disease}$, $1 = \text{Heart Disease Present}$).
6. **Leakage Prevention:** Strict feature/target separation ($X$ and $y$) with assertions ensuring zero target leakage. Downstream scalers and encoders are fitted solely on training splits.
7. **Automated Data Quality Reporting:** Generates structured JSON reports saved to `reports/results/data_quality_report.json`.

---

## 5. Exploratory Data Analysis & Healthcare Analytics (Phase 3)

The exploratory analysis module (`src/exploratory_analysis.py`) and notebook (`notebooks/exploratory_analysis.ipynb`) provide publication-quality visualizations and statistical evaluations across all 303 patient records:

### Key Empirical Findings:
* **Cohort & Balance:** 164 patients (54.13%) without heart disease; 139 patients (45.87%) with heart disease.
* **Demographics:** Heart disease patients are slightly older (mean $56.63 \pm 7.94$ vs $52.59 \pm 9.51$ years). Males exhibited a 55.34% disease rate (114/206) compared to 25.77% in females (25/97) in this clinical referral cohort.
* **Chest Pain Types (`cp`):** Asymptomatic presentations (`cp=3`) had the highest disease rate at 72.92% (105/144), whereas non-anginal (`cp=2`, 20.93%) and atypical angina (`cp=1`, 18.00%) had significantly lower rates.
* **Exercise Stress Markers:**
  - **Peak Heart Rate (`thalach`):** Inversely correlated ($r = -0.417$); disease cohort achieved lower mean peak heart rate ($139.26$ bpm vs $158.38$ bpm).
  - **Exercise Angina (`exang`):** Positively correlated ($r = 0.432$); 76.77% disease prevalence when exercise angina is present.
  - **ST Depression (`oldpeak`):** Positively correlated ($r = 0.425$); mean depression of $1.57$ mm in disease vs $0.59$ mm in non-disease.
* **Fluoroscopy & Thalassemia:**
  - **Major Vessels (`ca`):** Escalating disease rate from 26.11% ($ca=0$) to 85.00% ($ca=3$) ($r = 0.460$).
  - **Thalassemia (`thal`):** Reversible defect status (`thal=3`) had a 76.07% heart disease rate ($r = 0.516$).

---

## 6. Machine Learning Baseline Benchmarking (Phase 4)

Four classification architectures were evaluated on an untouched 20% test split ($N=61$) after 5-Fold Stratified Cross-Validation on the 80% training split ($N=242$):

| Model | Test Accuracy | Test Precision | Test Recall (Sensitivity) | Test F1-Score | Test ROC-AUC | 5-Fold CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8852** | 0.8387 | **0.9286** | **0.8814** | **0.9665** | $0.8471 \pm 0.0100$ |
| **KNN ($k=5$)** | **0.8852** | **0.8621** | 0.8929 | 0.8772 | 0.9529 | $0.8222 \pm 0.0288$ |
| **Random Forest** | 0.8525 | 0.8065 | 0.8929 | 0.8475 | 0.9567 | $0.8096 \pm 0.0321$ |
| **Decision Tree (Baseline)** | 0.7869 | 0.7586 | 0.7857 | 0.7719 | 0.8755 | $0.7357 \pm 0.0342$ |

---

## 7. Model Selection & Hyperparameter Tuning (Phase 5)

### Hyperparameter Tuning Methodology:
To optimize the primary Decision Tree for clinical screening sensitivity while preserving white-box interpretability, a controlled `GridSearchCV` was executed strictly on the 242-sample training set using 5-Fold Stratified Cross-Validation across:
* `criterion`: `['gini', 'entropy']`
* `max_depth`: `[3, 4, 5, 6, 8, None]`
* `min_samples_split`: `[2, 5, 10]`
* `min_samples_leaf`: `[1, 2, 4, 6, 8]`
* `class_weight`: `[None, 'balanced']`

### Optimal Selected Hyperparameters:
* `criterion`: `'gini'`
* `max_depth`: `3`
* `min_samples_leaf`: `1`
* `min_samples_split`: `2`
* `class_weight`: `'balanced'`

### Baseline Decision Tree vs. Tuned Decision Tree (Held-Out Test Split):

| Evaluation Metric | Baseline Decision Tree (`depth=4`) | Tuned Decision Tree (`depth=3, balanced`) | Clinical Impact |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 0.7869 | **0.8689** | $+8.20\%$ overall diagnostic classification accuracy |
| **Precision** | 0.7586 | **0.8571** | $+9.85\%$ precision (fewer false alarms) |
| **Recall (Sensitivity)** | 0.7857 | **0.8571** | $+7.14\%$ disease detection ($24/28$ cardiac patients detected) |
| **F1-Score** | 0.7719 | **0.8571** | $+8.52\%$ balanced F1 score |
| **ROC-AUC** | 0.8755 | **0.8712** | Stable discriminative power |
| **Tree Complexity** | 16 leaves, depth 4 | **8 leaves, depth 3** | Simpler, more robust, highly interpretable decision rules |

### Top Predictive Features (Tuned Decision Tree):
1. `thal_1` (Normal Thalassemia Status): **54.01%**
2. `cp_3` (Asymptomatic Chest Pain): **18.18%**
3. `ca_0` (Zero Fluoroscopy Major Vessels): **18.10%**
4. `oldpeak` (ST Depression): **5.73%**
5. `age`: **3.97%**

---

## 8. Prediction Pipeline & Input Validation (Phase 5)

The inference pipeline in [`src/prediction.py`](file:///d:/Sem_Project/heart-disease-risk-classifier/src/prediction.py) provides:
* **Strict Clinical Input Validation:** Checks types, non-null values, categorical values, and sanity bounds.
* **Key Order Invariance:** Automatically reorders inputs to match training schema.
* **Risk Stratification:** Assigns categories ("Lower Predicted Risk" / "Higher Predicted Risk") and risk tiers (Low $<40\%$, Moderate $40\%-70\%$, High $\ge 70\%$).
* **Contributing Factor Extraction:** Pinpoints key physiological and symptomatic risk indicators.
* **Mandatory Medical Disclaimer:** Enforces standard non-diagnostic academic disclaimer.

---

## 9. Project Limitations & Academic Healthcare Scope

For academic viva evaluation and healthcare context, the following limitations are documented:
1. **Sample Size:** The dataset contains 303 records from a single medical center (Cleveland Clinic Foundation).
2. **Referral Cohort Demographics:** The cohort contains a higher proportion of males (68%) and referred patients undergoing angiography, which may not reflect general population distributions.
3. **Historical Data:** The dataset was collected in the late 1980s; diagnostic thresholds and imaging techniques have evolved.
4. **Clinical Validation:** The model serves as an educational and decision-support risk screening tool; it has not undergone formal clinical trial or FDA/regulatory validation.

---

## 10. Installation & Execution

1. **Activate Virtual Environment:**
   ```bash
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Preprocessing Pipeline:**
   ```bash
   python -m src.data_preprocessing
   ```

4. **Generate Exploratory Visualizations:**
   ```bash
   python -m src.exploratory_analysis
   ```

5. **Train, Tune, & Evaluate Models:**
   ```bash
   python -m src.train_model
   ```

6. **Run Single Sample Prediction:**
   ```bash
   python -m src.prediction
   ```

7. **Run Pytest Test Suite:**
   ```bash
   pytest tests/ -v
   ```

---

## 11. Medical Disclaimer

> **IMPORTANT:** This prediction system is generated by a machine-learning model for **educational and analytical purposes only**. It is **not a medical diagnosis** and must not be used as a substitute for professional medical advice, clinical evaluation, or treatment decisions.
