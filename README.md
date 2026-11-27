# Cardio-Metabolic Risk Prediction from Wearable Accelerometer Data (NHANES 2011–2014)

An end-to-end, leakage-free Machine Learning pipeline for predicting **Metabolic Syndrome (MetS)** risk using objective wrist-worn triaxial accelerometer patterns (`PAXMIN` / `PAXDAY`) and clinical survey data from the **NHANES 2011–2014 (Cycles G & H)** cohorts.

---

## 1. Completed Data Engineering & Preprocessing Pipeline (Member 1 – Anushka K.R.)

All data ingestion, cohort validation, minute-level accelerometer stream filtering, clinical target construction, quality control, and exploratory data analysis tasks (`M1-01` through `M1-08`) have been implemented and verified (`100% Complete`).

### Task Completion Matrix

| Task ID | Task Description | Script / Notebook | Verified Outcome & Deliverable |
| :--- | :--- | :--- | :--- |
| **M1-01** | Minute-level accelerometer stream ingestion & cohort filtering (`PAXMIN_G/H.xpt`) | `notebooks/01_data_preprocessing.ipynb` & `src/data/wearable_processor.py` | Filtered `78.13M` Cycle G rows $\rightarrow$ **`26,256,025`** rows (`minute_streams_cycle_g.parquet`) & `88.22M` Cycle H rows $\rightarrow$ **`26,242,498`** rows (`minute_streams_cycle_h.parquet`) (**`52,498,523`** total minute-level records) |
| **M1-02** | NHANES 2011–2014 (Cycles G & H) clinical, lab & exam `.XPT` ingestion (18 files) | `src/data/build_cohort.py` | Automated ingestion of `DEMO`, `BMX`, `BPX`, `GLU`, `TRIGLY`, `HDL`, `RXQ_RX`, `SMQ`, and `PAXDAY` |
| **M1-03** | Multi-table relational merge across survey cycles | `src/data/build_cohort.py` | Merged `n = 19,931` total participants via unique identifier `SEQN` |
| **M1-04** | Adult cohort eligibility & complete fasting lab filtering | `src/data/build_cohort.py` | Filtered `n = 11,329` adults (`RIDAGEYR >= 20`) $\rightarrow$ `n = 5,295` with complete fasting biomarkers |
| **M1-05** | Accelerometer wear-time compliance validation | `notebooks/01_data_preprocessing.ipynb` & `src/data/wearable_processor.py` | Validated `>= 3` valid days ($\ge 600$ awake wear mins/day = `36,000` seconds) $\rightarrow$ Final cohort **`n = 4,603`** |
| **M1-06** | Missing value audit & covariate outlier inspection | `notebooks/02_eda.ipynb` | Confirmed **`0.0%` missing values** across all base columns; generated boxplots (`m1_06_covariate_boxplots.png`) |
| **M1-07** | Harmonized binary Metabolic Syndrome target label construction | `src/data/build_cohort.py` | Created `metabolic_syndrome_target` ($\ge 3$ of 5 criteria) & quarantined biomarkers to prevent target leakage |
| **M1-08** | Exploratory Data Analysis (EDA) & Participant Attrition Flow Diagram | `notebooks/02_eda.ipynb` | Generated class balance, age/sex risk distributions, and publication-ready attrition flowchart |

---

## 2. Verified Cohort Attrition & Numerical Outcomes

### Step-by-Step Participant Selection Funnel
1. **Total NHANES 2011–2014 Participants (Cycles G & H):** `n = 19,931`
2. **Adult Cohort Filter (`RIDAGEYR >= 20` years):** `n = 11,329` *(Excluded: 8,602 participants under age 20)*
3. **Complete Fasting Clinical & Laboratory Biomarkers:** `n = 5,295` *(Excluded: 6,034 non-fasting / incomplete lab records)*
4. **Valid Accelerometer Wear Compliance (`>= 3` valid days at $\ge 600$ awake mins/day):** **`n = 4,603`** *(Excluded: 692 insufficient wear compliance)*

### Final Analytical Cohort Summary Statistics (`n = 4,603`)
- **Missing Data Rate:** `0.0%` (`0` missing values across all 10 columns in `clean_cohort_base.parquet`)
- **Target Class Distribution (`metabolic_syndrome_target`):**
  - **Class `0` (Low Cardio-Metabolic Risk):** `2,618` participants (**`56.9%`**)
  - **Class `1` (High Risk / Metabolic Syndrome):** `1,985` participants (**`43.1%`**)
- **Wearable Compliance & Demographic Distributions:**
  - **Mean Daily Awake Wear Time (`mean_daily_wear_mins`):** Mean = **`765.21` mins/day** (`12.75` hours/day)
  - **Valid Accelerometer Days (`valid_days_count`):** Mean = `8.90` days (SD = `0.62`), Median = `9.0` days (Range: `3` – `9` days)
  - **Age (`RIDAGEYR`):** Mean = `49.37` years (SD = `17.55`), Median = `49.0` years (IQR: `34.0` – `63.0`, Range: `20` – `80`)
  - **Biological Sex (`RIAGENDR`):** Balanced representation across Male (`1.0`) and Female (`2.0`), with MetS prevalence rising sharply after age 50.
- **Processed Minute-Level Accelerometer Streams:**
  - **`minute_streams_cycle_g.parquet` (2011–2012):** `26,256,025` filtered minute-level rows across 8 core accelerometer variables (`SEQN`, `PAXDAYM`, `PAXDAYWM`, `PAXSSNMP`, `PAXMTSM`, `PAXTRANM`, `PAXPREDM`, `PAXFLGSM`)
  - **`minute_streams_cycle_h.parquet` (2013–2014):** `26,242,498` filtered minute-level rows across 8 core accelerometer variables

---

## 3. Clinical Target Definition & Anti-Leakage Architecture

### Harmonized NCEP ATP III / IDF Metabolic Syndrome Criteria (`M1-07`)
A participant is labeled positive (`metabolic_syndrome_target = 1`) if **3 or more** of the following 5 clinical criteria are met:
1. **Elevated Waist Circumference (`BMXWAIST`):** $\ge 102\text{ cm}$ (Men) or $\ge 88\text{ cm}$ (Women)
2. **Elevated Triglycerides (`LBXTR`):** $\ge 150\text{ mg/dL}$ or lipid-lowering medication
3. **Reduced HDL Cholesterol (`LBDHDD`):** $< 40\text{ mg/dL}$ (Men) or $< 50\text{ mg/dL}$ (Women) or medication
4. **Elevated Blood Pressure (`BPXSY1–3`, `BPXDI1–3`):** Systolic $\ge 130\text{ mmHg}$ or Diastolic $\ge 85\text{ mmHg}$ or antihypertensive therapy
5. **Elevated Fasting Glucose (`LBXGLU`):** $\ge 100\text{ mg/dL}$ or antidiabetic medication

### Strict Quarantine Protocol (Zero Target Leakage)
To guarantee that downstream models predict risk strictly from **wearable behavioral patterns and safe demographics**, all clinical lab values, anthropometric measurements, blood pressure readings, and medications used to compute the target are isolated into `data/interim/quarantined_clinical_biomarkers.parquet`.

#### Handover Schema (`data/processed/clean_cohort_base.parquet`)

| Column Name | Data Type | Role | Description |
| :--- | :--- | :--- | :--- |
| `SEQN` | `float64` | Primary Key | Unique NHANES participant identifier for joining wearable streams |
| `survey_cycle` | `str` | Metadata | NHANES survey wave (`2011-2012` or `2013-2014`) |
| `RIDAGEYR` | `float64` | Safe Covariate | Age in years at screening (`20` – `80`) |
| `RIAGENDR` | `float64` | Safe Covariate | Biological sex (`1.0` = Male, `2.0` = Female) |
| `RIDRETH3` | `float64` | Safe Covariate | Race / Hispanic origin category |
| `smoking_status` | `str` | Safe Covariate | Derived smoking status (`Never`, `Former`, `Current`, `Unknown`) |
| `valid_days_count` | `int64` | Compliance Metric | Number of valid accelerometer wear days (`3` – `9`) |
| `mean_daily_wear_mins` | `float64` | Compliance Metric | Mean daily awake accelerometer wear duration in minutes (`765.21` mean) |
| `WTSAF2YR` | `float64` | Survey Weight | Fasting subsample 2-year MEC weight *(use for weighted evaluation only)* |
| `metabolic_syndrome_target` | `int64` | **Target (`y`)** | Binary cardio-metabolic risk label (`0` = Low Risk, `1` = High Risk) |

---

## 4. Visual Reports & Diagnostics (`reports/figures/`)

All diagnostic figures generated in `notebooks/02_eda.ipynb` are saved at `300 DPI` in `reports/figures/`:
- **`m1_06_covariate_boxplots.png`:** Outlier and quartile distribution analysis for participant age and accelerometer wear compliance.
- **`m1_08_eda_distributions.png`:** Three-panel exploratory summary showing target class balance (`2,618` vs `1,985`), age-stratified MetS prevalence with KDE curves, and sex-stratified risk counts.
- **`m1_08_attrition_flow_diagram.png`:** Complete cohort attrition diagram from `n = 19,931` raw records to the final `n = 4,603` leakage-free analytical dataset.

Model evaluation figures from `notebooks/06_evaluation_insights.ipynb` (Member 4) are listed in [Section 6](#6-model-evaluation-explainability--sensitivity-member-4--kasuntha-smm).

---

## 5. Repository Structure & Pipeline Roadmap

```text
mechine-learning-/
├── data/                         # Excluded from Git (.gitignore) — shared via Google Drive
│   ├── raw/
│   │   ├── nhanes/               # Raw NHANES 2011–2014 .XPT clinical/lab files (18 files)
│   │   └── physionet/            # Raw PAXMIN_G.xpt & PAXMIN_H.xpt accelerometer streams
│   ├── interim/
│   │   ├── nhanes_2011_2014_clinical_labels.parquet
│   │   └── quarantined_clinical_biomarkers.parquet   # HELD OUT (Leakage Prevention)
│   └── processed/                # Handover files for downstream modeling
│       ├── clean_cohort_base.parquet               # 4,603 rows x 10 columns
│       ├── minute_streams_cycle_g.parquet          # 26,256,025 rows x 8 columns
│       └── minute_streams_cycle_h.parquet          # 26,242,498 rows x 8 columns
├── notebooks/
│   ├── 01_data_preprocessing.ipynb      # [Completed] Ingestion, filtering, target creation
│   ├── 02_eda.ipynb                     # [Completed] QC, outlier inspection, attrition diagram
│   ├── 03_feature_engineering.ipynb     # [Completed] Behavioral, circadian & intensity features, PCA, K-means
│   ├── 04_feature_selection.ipynb       # [In progress] Feature selection notebook (logic in src/features/selectors.py)
│   ├── 05_model_training.ipynb          # [Completed - progress scope] LR, RF, GB x Sets 0/A/B, 5-fold CV
│   └── 06_evaluation_insights.ipynb     # [Completed] Test metrics, bootstrap CIs, DeLong, importance, E4-E6
├── models/
│   └── tuned_pipelines.joblib           # Git-ignored; rebuilt and verified by notebook 06 if missing
├── reports/
│   ├── figures/                         # Saved high-resolution diagnostic plots (300 DPI)
│   └── tables/                          # Evaluation tables (CSV), committed as deliverables
├── src/
│   ├── data/                            # Member 1: cohort, wear-time filters, labels
│   ├── features/                        # Member 2: feature sets, PCA / K-means, selectors
│   ├── models/                          # Member 3: classifiers + leakage-safe pipeline
│   └── evaluation/                      # Member 4: metrics, bootstrap, explainability, sensitivity, plots
├── tests/                               # pytest suite (incl. tests/test_evaluation.py)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 6. Model Evaluation, Explainability & Sensitivity (Member 4 – Kasuntha S.M.M.)

All evaluation tasks (`M4-01` – `M4-07`) are implemented in `src/evaluation/` and run end-to-end in `notebooks/06_evaluation_insights.ipynb`.
The held-out test set (`n = 918`, prevalence `0.43`) is only **scored** here; every threshold and hyperparameter was fixed on training data.

### Task Completion Matrix

| Task ID | Task Description | Code | Deliverable |
| :--- | :--- | :--- | :--- |
| **M4-01** | Accuracy, precision, recall, F1, balanced accuracy | `src/evaluation/metrics.py` | `reports/tables/model_performance_metrics.csv` |
| **M4-02** | ROC-AUC, PR-AUC, Brier score + DeLong test for correlated AUCs | `src/evaluation/metrics.py` | same table, `reports/tables/statistical_tests.csv` |
| **M4-03** | ROC / PR curves, confusion matrices (τ = 0.5 and F1-optimal), calibration | `src/evaluation/plots.py` | `reports/figures/m4_03_*.png` |
| **M4-04** | 95% percentile CIs from 1,000 bootstrap resamples; paired Set 0 → A → B differences | `src/evaluation/bootstrap.py` | `model_performance_summary.csv`, `m4_04_*_forest_plot.png` |
| **M4-05** | LR odds ratios, RF Gini, permutation importance (R = 20), selection stability across folds | `src/evaluation/explainability.py` | `feature_importance_*.csv`, `feature_selection_stability.csv`, `m4_05_*.png` |
| **M4-06** | E4: refit with fasting survey weights `WTSAF2YR / 2` | `src/evaluation/sensitivity.py` | `reports/tables/sensitivity_analysis_results.csv` |
| **M4-07** | E5: cross-cycle transfer (2011–12 ⇄ 2013–14); E6: feature-group ablation | `src/evaluation/sensitivity.py` | same table, `m4_07_sensitivity_analyses.png` |
| **M4-08** | Integration: notebook 06, README, findings | `notebooks/06_evaluation_insights.ipynb` | Section 9 of notebook 06 |

### Headline Results (test set, Feature Set B, F1-optimal threshold, 95% bootstrap CI)

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Brier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.761** [0.730, 0.793] | 0.645 | **0.843** | 0.582 | **0.689** | 0.201 |
| Gradient Boosting | 0.754 [0.725, 0.787] | **0.665** | 0.810 | **0.585** | 0.679 | **0.199** |
| Random Forest | 0.744 [0.715, 0.780] | 0.650 | 0.825 | 0.565 | 0.671 | 0.206 |
| Majority-class baseline | 0.500 | 0.429 | 0.000 | – | 0.000 | – |

**Value of wearable data (RQ2), test ROC-AUC:**

| Model | Set 0 (demographics) | Set A (+ basic activity) | Set B (+ engineered) | Set 0 → A (DeLong p) |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.732 | 0.751 | 0.761 | +0.019 (p < 0.001) |
| Random Forest | 0.726 | 0.743 | 0.744 | +0.017 (p = 0.045) |
| Gradient Boosting | 0.733 | 0.751 | 0.754 | +0.018 (p = 0.008) |

### Key Findings
1. **Moderate, genuine discrimination** (ROC-AUC 0.74–0.76) from demographics and wrist activity alone, without waist, blood pressure or lab values.
2. **Wearable activity data adds significant value** over demographics for every model (+0.017 to +0.019 ROC-AUC).
3. **Engineered features help the linear model only** (Set A → B: +0.010, p = 0.013 for LR; not significant for RF / GB).
4. **Logistic Regression is the best model**: significantly better than Random Forest (DeLong p = 0.012), level with Gradient Boosting (p = 0.23).
5. **Most robust activity predictors:** mean daily MIMS (OR 0.33 per SD), minutes of high-intensity activity, upper MIMS percentiles, day-to-day variability and 24-h profile PCA scores. Age is the strongest predictor overall.
6. **Robustness:** survey weighting (E4) changes ROC-AUC by < 0.01; models transfer across NHANES cycles (E5, ROC-AUC 0.736–0.768); dropping any single engineered feature group changes CV ROC-AUC by ≤ 0.004 (E6), so activity information is redundant across groups.

### Methodological Notes
- **Fitted pipelines.** `models/tuned_pipelines.joblib` is git-ignored. Notebook 06 loads it if present; otherwise it rebuilds all 9 pipelines from `cv_results_summary.parquet`. In both cases it **asserts** that they reproduce Member 3's test predictions exactly before continuing.
- **E4 survey weights.** 418 participants (9.1%) have `WTSAF2YR = 0` (outside the official fasting subsample). Following NHANES guidelines, they get weight 0 in the weighted analysis only.
- **E6 adaptation.** The proposal planned "single step algorithm vs. all five". The cohort uses NHANES `PAXMIN` (MIMS only), so no step-count columns exist; E6 was run as a feature-group ablation on training-set CV instead.
- **Scope at progress evaluation.** 3 of the 8 planned classifiers (LR, RF, GB) are trained. The evaluation code handles any number of models, so adding the remaining classifiers to `test_predictions.parquet` and re-running notebook 06 updates every table and figure.

### Evaluation Figures (`reports/figures/`, 300 DPI)
| Figure | Shows |
| :--- | :--- |
| `m4_03_roc_pr_curves_by_feature_set.png` | ROC and PR curves for every model, Set 0 vs A vs B |
| `m4_03_roc_pr_curves_set_b.png` | ROC and PR curves of all models on Set B |
| `m4_03_confusion_matrices_set_b.png` | Confusion matrices at τ = 0.5 and the F1-optimal τ |
| `m4_03_calibration_set_b.png` | Reliability curves with Brier scores |
| `m4_04_roc_auc_forest_plot.png` / `m4_04_pr_auc_forest_plot.png` | Test ROC-AUC / PR-AUC with 95% bootstrap CIs |
| `m4_05_feature_importance.png` | LR odds ratios, RF Gini importance, permutation importance |
| `m4_05_selection_stability.png` | Selection frequency across 5 folds (filter vs L1) |
| `m4_07_sensitivity_analyses.png` | E4 survey weights, E5 cross-cycle transfer, E6 ablation |

---

## 7. How to Reproduce

```bash
# 1. Create and activate a virtual environment (Python 3.11+)
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the test suite
python -m pytest tests -q

# 4. Run the evaluation notebook end to end (about 15 minutes)
jupyter nbconvert --to notebook --execute --inplace notebooks/06_evaluation_insights.ipynb
```

Notebooks `01`–`05` regenerate the upstream data files in `data/processed/`; notebook `06` only needs the processed files already in the repository.
