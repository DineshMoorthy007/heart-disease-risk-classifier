# Heart Disease Risk Prediction Using Decision Tree-Based Healthcare Analytics

An academic machine-learning and healthcare analytics system to predict heart disease risk based on clinical features from the Cleveland Heart Disease dataset.

---

## 1. Project Overview

This project implements an end-to-end healthcare analytics and machine-learning pipeline:
* **Exploratory Data Analysis (EDA)** on key demographic and physiological features.
* **Explainable Machine Learning** using Decision Tree classification as the central model.
* **Baseline Model Comparison** across Logistic Regression, Random Forest, and K-Nearest Neighbors.
* **Interactive Risk Prediction Dashboard** built with Streamlit.

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
│   │   └── Heart_disease.csv             # Immutable raw Cleveland dataset
│   └── processed/
│       └── heart_disease_processed.csv   # Cleaned, standardized dataset
│
├── notebooks/
│   └── exploratory_analysis.ipynb        # Phase 3: Executed exploratory analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py             # Phase 2: Modular preprocessing pipeline
│   ├── exploratory_analysis.py           # Phase 3: Analytical visualizations
│   ├── train_model.py                    # Phase 4: Model training routines
│   ├── evaluate_model.py                 # Phase 4 & 5: Metrics & explainability
│   └── prediction.py                     # Phase 6: Prediction inference pipeline
│
├── models/
│   ├── decision_tree_model.joblib        # Phase 4: Trained baseline pipelines
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   ├── knn_model.joblib
│   └── model_metadata.json
│
├── app/
│   └── app.py                            # Phase 7: Streamlit dashboard
│
├── reports/
│   ├── figures/                          # EDA and ML evaluation charts (21 figures)
│   └── results/
│       ├── data_quality_report.json
│       ├── model_comparison.csv
│       ├── cross_validation_results.csv
│       └── decision_tree_feature_importance.csv
│
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_exploratory_analysis.py
│   ├── test_model_training.py
│   └── test_model_evaluation.py
│
├── requirements.txt
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
6. **Leakage Prevention:** Strict feature/target separation ($X$ and $y$) with assertions ensuring zero target leakage. Whole-dataset scaling is avoided to ensure downstream scalers are fitted solely on training splits.
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
* **Outlier Assessment:** IQR analysis detected clinically valid extreme physiological values (e.g. resting BP up to 200 mm Hg, cholesterol up to 564 mg/dl, oldpeak up to 6.2 mm), all representing severe cardiac cases rather than data entry anomalies.

---

## 6. Machine Learning Model Development & Benchmarking (Phase 4)

The modeling pipeline (`src/train_model.py`) evaluates 4 classification architectures using an untouched 20% test split ($N=61$) and 5-Fold Stratified Cross-Validation ($N=242$):

### Evaluation Strategy:
* **Stratified Split:** 80% Train ($N=242$), 20% Test ($N=61$, 33 class 0, 28 class 1).
* **Leakage-Free Pipelines:** One-hot encoding on categorical features; standard scaling for distance/linear models (Logistic Regression, KNN); untouched numerical features for tree models.
* **Primary Algorithm:** **Decision Tree Classifier** (`max_depth=4`, `min_samples_leaf=3`, `min_samples_split=6`), prioritized for clinical interpretability and transparent decision logic.
* **Comparative Baselines:** Logistic Regression (linear baseline), Random Forest (ensemble baseline), and K-Nearest Neighbors (distance-based baseline).

### Benchmark Comparison (Untouched Test Set):

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | 5-Fold CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8852** | 0.8387 | **0.9286** | **0.8814** | **0.9665** | $0.8471 \pm 0.0100$ |
| **KNN ($k=5$)** | **0.8852** | **0.8621** | 0.8929 | 0.8772 | 0.9529 | $0.8222 \pm 0.0288$ |
| **Random Forest** | 0.8525 | 0.8065 | 0.8929 | 0.8475 | 0.9567 | $0.8096 \pm 0.0321$ |
| **Decision Tree (Primary)** | 0.7869 | 0.7586 | 0.7857 | 0.7719 | 0.8755 | $0.7357 \pm 0.0342$ |

### Decision Tree Feature Importance:
1. `thal_1` (Normal Thalassemia Status): **45.0%**
2. `cp_3` (Asymptomatic Chest Pain): **16.3%**
3. `ca_0` (Zero Fluoroscopy Vessels): **14.1%**
4. `chol` (Serum Cholesterol): **7.2%**
5. `oldpeak` (ST Depression): **5.6%**
6. `sex_1` (Male): **3.2%**
7. `thalach` (Maximum Heart Rate): **3.0%**

---

## 7. Installation & Setup

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

4. **Generate EDA Visualizations:**
   ```bash
   python -m src.exploratory_analysis
   ```

5. **Train & Evaluate Machine Learning Models:**
   ```bash
   python -m src.train_model
   ```

6. **Run Test Suite:**
   ```bash
   pytest tests/
   ```

---

## 8. Medical Disclaimer

> **IMPORTANT:** This prediction system is generated by a machine-learning model for **educational and analytical purposes only**. It is **not a medical diagnosis** and must not be used as a substitute for professional medical advice, clinical evaluation, or treatment decisions.
