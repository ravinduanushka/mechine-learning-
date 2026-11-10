import os
import requests
import pandas as pd
import pyreadstat
from typing import Generator, List

NHANES_CDN_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

NHANES_FILES = {
    "2011-2012": {
        "DEMO": "DEMO_G.XPT",
        "BMX": "BMX_G.XPT",
        "BPX": "BPX_G.XPT",
        "BPQ": "BPQ_G.XPT",
        "GLU": "GLU_G.XPT",
        "TRIGLY": "TRIGLY_G.XPT",
        "HDL": "HDL_G.XPT",
        "DIQ": "DIQ_G.XPT",
        "SMQ": "SMQ_G.XPT",
    },
    "2013-2014": {
        "DEMO": "DEMO_H.XPT",
        "BMX": "BMX_H.XPT",
        "BPX": "BPX_H.XPT",
        "BPQ": "BPQ_H.XPT",
        "GLU": "GLU_H.XPT",
        "TRIGLY": "TRIGLY_H.XPT",
        "HDL": "HDL_H.XPT",
        "DIQ": "DIQ_H.XPT",
        "SMQ": "SMQ_H.XPT",
    }
}


def download_nhanes_file(cycle_label: str, filename: str, output_dir: str = "data/raw/nhanes") -> str:
    """Download verified SAS XPT file from CDC portal or use local cached file."""
    os.makedirs(output_dir, exist_ok=True)
    destination_path = os.path.join(output_dir, filename)

    # Use local file if it already exists and is non-empty
    if os.path.exists(destination_path) and os.path.getsize(destination_path) > 1024:
        return destination_path

    year_cycle = cycle_label.split("-")[0]
    candidate_urls = [
        f"{NHANES_CDN_URL}/{year_cycle}/DataFiles/{filename}",
        f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle_label}/{filename}",
    ]

    headers = {"User-Agent": "Mozilla/5.0"}
    download_success = False

    for url in candidate_urls:
        try:
            response = requests.get(url, stream=True, timeout=60, headers=headers)
            if response.status_code == 200:
                with open(destination_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                download_success = True
                break
        except requests.RequestException:
            continue

    if not download_success:
        raise RuntimeError(f"Could not download {filename}. Please download manually to {destination_path}")

    return destination_path


def read_nhanes_xpt(file_path: str, selected_columns: List[str] = None) -> pd.DataFrame:
    """Read a SAS XPT file using pyreadstat C-engine."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df, _ = pyreadstat.read_xport(file_path, encoding="latin1")

    if selected_columns:
        valid_cols = [c for c in selected_columns if c in df.columns]
        df = df[valid_cols]

    return df


def load_physionet_minute_data_chunks(
    file_path: str, 
    chunk_size: int = 5000
) -> Generator[pd.DataFrame, None, None]:
    """Stream massive PhysioNet CSV files in memory-safe chunks."""
    return pd.read_csv(file_path, chunksize=chunk_size, compression="infer")