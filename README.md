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
│   ├── 03_feature_engineering.ipynb     # [Next] Behavioral, circadian & intensity features
│   ├── 04_feature_selection.ipynb       # [Planned] Feature selection & dimensionality reduction
│   ├── 05_model_training.ipynb          # [Planned] Model training & cross-validation
│   └── 06_evaluation_insights.ipynb     # [Planned] ROC-AUC, SHAP explainability & clinical insights
├── reports/
│   └── figures/                         # Saved high-resolution diagnostic plots (300 DPI)
├── src/
│   └── data/
│       ├── build_cohort.py              # NHANES clinical ingestion & MetS target logic
│       ├── wearable_processor.py        # Wear-time validation & minute-stream chunked filtering
│       └── prepare_handover.py          # Leakage-free handover verification script
├── .gitignore
├── requirements.txt
└── README.md