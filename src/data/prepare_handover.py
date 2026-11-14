import os
import pandas as pd


# Columns used to define the target (MUST be quarantined from ML feature pipeline)
LEAKY_BIOMARKER_COLS = [
    "BMXWAIST", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
    "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4",
    "BPQ050A", "BPQ100D", "LBXGLU", "LBXTR", "LBDHDD",
    "DIQ050", "DIQ070", "c_waist", "c_trig", "c_hdl",
    "c_bp", "c_glu", "criteria_sum"
]

# Safe demographic and behavioral baseline columns allowed for Member 2
SAFE_COHORT_COLS = [
    "SEQN", "survey_cycle", "RIDAGEYR", "RIAGENDR", "RIDRETH3",
    "SMQ020", "SMQ040", "WTSAF2YR", "valid_days_count",
    "mean_daily_wear_mins", "metabolic_syndrome_target"
]


def create_clean_member2_handover(
    clinical_df: pd.DataFrame,
    valid_wear_df: pd.DataFrame,
    processed_dir: str,
    interim_dir: str
) -> pd.DataFrame:
    """
    Merge clinical cohort with valid accelerometer compliance table,
    quarantine target-defining biomarkers to prevent leakage, and export
    the clean cohort contract for Member 2.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(interim_dir, exist_ok=True)

    # Inner join retains only participants with both complete clinical labs AND valid wear days
    merged = clinical_df.merge(valid_wear_df, on="SEQN", how="inner")
    merged = merged.drop_duplicates(subset=["SEQN"]).reset_index(drop=True)

    # 1. Save quarantined clinical biomarkers separately for audit/reporting only
    quarantine_cols = ["SEQN"] + [c for c in LEAKY_BIOMARKER_COLS if c in merged.columns]
    quarantine_path = os.path.join(interim_dir, "quarantined_clinical_biomarkers.parquet")
    merged[quarantine_cols].to_parquet(quarantine_path, index=False)

    # 2. Export strictly leakage-free cohort table for Member 2
    safe_cols = [c for c in SAFE_COHORT_COLS if c in merged.columns]
    clean_cohort = merged[safe_cols].copy()

    # Encode smoking status cleanly into a categorical baseline feature
    if "SMQ020" in clean_cohort.columns and "SMQ040" in clean_cohort.columns:
        clean_cohort["smoking_status"] = "never"
        clean_cohort.loc[(clean_cohort["SMQ020"] == 1) & (clean_cohort["SMQ040"] == 3), "smoking_status"] = "former"
        clean_cohort.loc[(clean_cohort["SMQ020"] == 1) & (clean_cohort["SMQ040"].isin([1, 2])), "smoking_status"] = "current"
        clean_cohort = clean_cohort.drop(columns=["SMQ020", "SMQ040"])

    clean_path = os.path.join(processed_dir, "clean_cohort_base.parquet")
    clean_cohort.to_parquet(clean_path, index=False)

    return clean_cohort