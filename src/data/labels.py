import pandas as pd
import numpy as np


def compute_metabolic_syndrome_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign binary cardio-metabolic risk label (Metabolic Syndrome) using 
    harmonized clinical criteria (>= 3 of 5 abnormal components).
    """
    data = df.copy()

    # 1. Waist Circumference criterion
    male_waist = (data["RIAGENDR"] == 1) & (data["BMXWAIST"] > 102.0)
    female_waist = (data["RIAGENDR"] == 2) & (data["BMXWAIST"] >= 88.0)
    c_waist = (male_waist | female_waist).astype(int)

    # 2. Elevated Triglycerides criterion (>= 150 mg/dL or medication)
    med_lipid = data["BPQ100D"] == 1
    c_trig = ((data["LBXTR"] >= 150.0) | med_lipid).astype(int)

    # 3. Reduced HDL Cholesterol criterion (< 40 mg/dL male, < 50 mg/dL female or medication)
    male_hdl = (data["RIAGENDR"] == 1) & (data["LBDHDD"] < 40.0)
    female_hdl = (data["RIAGENDR"] == 2) & (data["LBDHDD"] < 50.0)
    c_hdl = ((male_hdl | female_hdl) | med_lipid).astype(int)

    # 4. Elevated Blood Pressure criterion (Systolic >= 130 or Diastolic >= 85 or anti-hypertensives)
    bp_sys = data[["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]].mean(axis=1)
    bp_dia = data[["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]].mean(axis=1)
    med_bp = data["BPQ050A"] == 1
    c_bp = ((bp_sys >= 130.0) | (bp_dia >= 85.0) | med_bp).astype(int)

    # 5. Elevated Fasting Glucose criterion (>= 100 mg/dL or diabetes medication)
    med_glucose = (data["DIQ050"] == 1) | (data["DIQ070"] == 1)
    c_glu = ((data["LBXGLU"] >= 100.0) | med_glucose).astype(int)

    # Aggregate criteria
    total_criteria = c_waist + c_trig + c_hdl + c_bp + c_glu
    data["metabolic_syndrome_target"] = (total_criteria >= 3).astype(int)
    
    # Store individual flags for downstream validation (quarantined from ML inputs)
    data["c_waist"] = c_waist
    data["c_trig"] = c_trig
    data["c_hdl"] = c_hdl
    data["c_bp"] = c_bp
    data["c_glu"] = c_glu
    data["criteria_sum"] = total_criteria

    return data