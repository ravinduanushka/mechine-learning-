import os
import pandas as pd
from src.data.load_data import NHANES_FILES, download_nhanes_file, read_nhanes_xpt
from src.data.filters import filter_adult_cohort
from src.data.labels import compute_metabolic_syndrome_target


def process_single_nhanes_cycle(cycle_label: str, nhanes_dir: str) -> pd.DataFrame:
    """
    Download, read, filter, merge, and label a single NHANES survey cycle
    (e.g., '2011-2012' or '2013-2014').
    """
    file_map = NHANES_FILES[cycle_label]

    # Ensure all required XPT files exist locally
    for key, fname in file_map.items():
        download_nhanes_file(cycle_label, fname, output_dir=nhanes_dir)

    # Read required variables
    demo_df = read_nhanes_xpt(os.path.join(nhanes_dir, file_map["DEMO"]), ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"])
    adults_df = filter_adult_cohort(demo_df, min_age=20)

    bmx_df = read_nhanes_xpt(os.path.join(nhanes_dir, file_map["BMX"]), ["SEQN", "BMXWAIST"])
    bpx_df = read_nhanes_xpt(os.path.join(nhanes_dir, file_map["BPX"]), [
        "SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
        "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"
    ])
    bpq_df = read_nhanes_xpt(os.path.join(nhanes_dir, file_map["BPQ"]), ["SEQN", "BPQ050A", "BPQ100D"])
    glu_df = read_nhanes_xpt(os.path.join(nhanes_dir, file_map["GLU"]), ["SEQN", "LBXGLU", "WTSAF2YR"])
    trig_df = read_nhanes_xpt(os.path.join(nhanes_dir, file_map["TRIGLY"]), ["SEQN", "LBXTR"])
    hdl_df = read_nhanes_xpt(os.path.join(nhanes_dir, file_map["HDL"]), ["SEQN", "LBDHDD"])
    diq_df = read_nhanes_xpt(os.path.join(nhanes_dir, file_map["DIQ"]), ["SEQN", "DIQ050", "DIQ070"])
    smq_df = read_nhanes_xpt(os.path.join(nhanes_dir, file_map["SMQ"]), ["SEQN", "SMQ020", "SMQ040"])

    # Merge clinical records by SEQN
    cohort = adults_df.merge(bmx_df, on="SEQN", how="inner")
    cohort = cohort.merge(bpx_df, on="SEQN", how="inner")
    cohort = cohort.merge(bpq_df, on="SEQN", how="left")
    cohort = cohort.merge(glu_df, on="SEQN", how="inner")
    cohort = cohort.merge(trig_df, on="SEQN", how="inner")
    cohort = cohort.merge(hdl_df, on="SEQN", how="inner")
    cohort = cohort.merge(diq_df, on="SEQN", how="left")
    cohort = cohort.merge(smq_df, on="SEQN", how="left")

    # Compute Metabolic Syndrome binary target
    labeled_cohort = compute_metabolic_syndrome_target(cohort)
    labeled_cohort["survey_cycle"] = cycle_label

    return labeled_cohort


def build_combined_nhanes_clinical_cohort(nhanes_dir: str) -> pd.DataFrame:
    """Combine 2011-2012 (Cycle G) and 2013-2014 (Cycle H) clinical cohorts."""
    df_g = process_single_nhanes_cycle("2011-2012", nhanes_dir)
    df_h = process_single_nhanes_cycle("2013-2014", nhanes_dir)

    combined = pd.concat([df_g, df_h], axis=0, ignore_index=True)
    combined = combined.drop_duplicates(subset=["SEQN"]).reset_index(drop=True)
    return combined