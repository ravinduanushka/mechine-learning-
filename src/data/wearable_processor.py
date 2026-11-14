import os
import pandas as pd
import numpy as np
import pyreadstat
from typing import List, Set
from src.data.load_data import download_nhanes_file


def get_valid_wear_participants_from_paxday(
    nhanes_dir: str,
    min_wake_minutes: int = 600,
    min_valid_days: int = 3
) -> pd.DataFrame:
    """
    Download and parse NHANES PAXDAY_G and PAXDAY_H files to identify participants
    meeting protocol compliance (>= 3 days with >= 600 valid awake-wear minutes).
    """
    day_files = [
        ("2011-2012", "PAXDAY_G.XPT"),
        ("2013-2014", "PAXDAY_H.XPT"),
    ]

    valid_summaries = []

    for cycle, fname in day_files:
        fpath = download_nhanes_file(cycle, fname, output_dir=nhanes_dir)
        df, _ = pyreadstat.read_xport(fpath, encoding="latin1")

        # Keep required day-level wear columns
        cols = [c for c in ["SEQN", "PAXDAYD", "PAXAISMD", "PAXVMIND"] if c in df.columns]
        sub = df[cols].copy()

        # Use valid awake wear minutes (PAXAISMD) or total valid wear minutes (PAXVMIND)
        wear_col = "PAXAISMD" if "PAXAISMD" in sub.columns else "PAXVMIND"
        sub = sub[sub[wear_col] >= min_wake_minutes]

        # Count compliant days per SEQN
        counts = (
            sub.groupby("SEQN")
            .agg(
                valid_days_count=(wear_col, "count"),
                mean_daily_wear_mins=(wear_col, "mean")
            )
            .reset_index()
        )
        qualified = counts[counts["valid_days_count"] >= min_valid_days]
        valid_summaries.append(qualified)

    combined_valid = pd.concat(valid_summaries, axis=0, ignore_index=True)
    combined_valid = combined_valid.drop_duplicates(subset=["SEQN"]).reset_index(drop=True)
    return combined_valid


def extract_filtered_minute_streams(
    minute_file_path: str,
    target_seqns: Set[float],
    output_parquet_path: str,
    chunk_size: int = 100000
) -> int:
    """
    Stream large minute-level accelerometer files (CSV.GZ or XPT chunks)
    and save only rows belonging to the final analytical cohort SEQNs.
    """
    filtered_chunks: List[pd.DataFrame] = []
    keep_cols = ["SEQN", "PAXDAYM", "PAXDAYWM", "PAXSSNMP", "PAXMTSM", "PAXPREDM", "PAXFLGSM"]

    if minute_file_path.endswith(".csv") or minute_file_path.endswith(".csv.gz"):
        reader = pd.read_csv(minute_file_path, chunksize=chunk_size, compression="infer")
        for chunk in reader:
            available = [c for c in keep_cols if c in chunk.columns]
            sub = chunk[chunk["SEQN"].isin(target_seqns)][available]
            if not sub.empty:
                filtered_chunks.append(sub)
    else:
        # Read via pyreadstat in chunks for XPT files
        offset = 0
        while True:
            df_chunk, _ = pyreadstat.read_xport(
                minute_file_path,
                encoding="latin1",
                row_limit=chunk_size,
                row_offset=offset
            )
            if df_chunk.empty:
                break
            available = [c for c in keep_cols if c in df_chunk.columns]
            sub = df_chunk[df_chunk["SEQN"].isin(target_seqns)][available]
            if not sub.empty:
                filtered_chunks.append(sub)
            offset += chunk_size

    if not filtered_chunks:
        raise ValueError("No matching SEQNs found in the minute-level stream file.")

    final_minute_df = pd.concat(filtered_chunks, axis=0, ignore_index=True)
    final_minute_df.to_parquet(output_parquet_path, index=False)
    return len(final_minute_df)