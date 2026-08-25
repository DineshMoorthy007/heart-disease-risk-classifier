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
│   ├── evaluate_model.py                 # Phase 5: Metrics & explainability
│   └── prediction.py                     # Phase 6: Prediction inference pipeline
│
├── models/
│   ├── decision_tree_model.pkl
│   └── model_metadata.json
│
├── app/
│   └── app.py                            # Phase 7: Streamlit dashboard
│
├── reports/
│   ├── figures/                          # Phase 3: 14 publication-quality EDA charts
│   └── results/
│       └── data_quality_report.json      # Preprocessing data hygiene report
│
├── tests/
│   ├── test_data_preprocessing.py        # Unit tests for preprocessing
│   └── test_exploratory_analysis.py      # Unit tests for exploratory analysis
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

### Generated Visualizations (`reports/figures/`):
1. `target_distribution.png` — Target class distribution
2. `age_distribution.png` — Overall patient age histogram & KDE
3. `age_by_target.png` — Age distribution by diagnostic class
4. `heart_disease_by_sex.png` — Prevalence by biological sex
5. `chest_pain_by_target.png` — Chest pain presentation vs diagnosis
6. `blood_pressure_by_target.png` — Resting blood pressure boxplot
7. `cholesterol_by_target.png` — Serum cholesterol boxplot
8. `max_heart_rate_by_target.png` — Peak heart rate achieved boxplot
9. `exercise_angina_by_target.png` — Exercise-induced angina count plot
10. `st_depression_by_target.png` — ST depression (oldpeak) boxplot
11. `vessels_ca_by_target.png` — Fluoroscopy major vessels vs diagnosis
12. `thalassemia_by_target.png` — Thalassemia status vs diagnosis
13. `correlation_heatmap.png` — Pearson correlation matrix heatmap
14. `feature_target_summary.png` — Ranked linear correlation with target

---

## 6. Installation & Setup

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

5. **Run Tests:**
   ```bash
   pytest tests/
   ```

---

## 7. Medical Disclaimer

> **IMPORTANT:** This prediction system is generated by a machine-learning model for **educational and analytical purposes only**. It is **not a medical diagnosis** and must not be used as a substitute for professional medical advice, clinical evaluation, or treatment decisions.
