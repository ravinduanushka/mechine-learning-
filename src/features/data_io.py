"""Project paths, data discovery, loading and schema validation (Member 2).

Nothing here fabricates data: when a required input is missing, the functions
report exactly which file and columns are needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- schemas ----
TARGET = "metabolic_syndrome_target"

COHORT_REQUIRED = [
    "SEQN", "survey_cycle", "RIDAGEYR", "RIAGENDR", "RIDRETH3",
    "smoking_status", TARGET,
]
# Present in Member 1's file but never used as prediction features:
#   WTSAF2YR (survey weight), valid_days_count / mean_daily_wear_mins (wear QC).
COHORT_NOT_FEATURES = ["WTSAF2YR", "valid_days_count", "mean_daily_wear_mins"]

MINUTE_REQUIRED = ["SEQN", "PAXDAYM", "PAXMTSM", "PAXPREDM"]
MINUTE_FILE_PATTERNS = ["minute_streams*", "minute_level*"]
COHORT_FILE_PATTERNS = ["clean_cohort_base*"]

SUPPORTED_SUFFIXES = (".parquet", ".csv", ".feather", ".xpt")

# Target-defining clinical variables (NCEP-style MetS criteria). They must never
# be prediction inputs; Member 1 quarantines them in data/interim/.
LEAKY_PREFIXES = ("BMX", "BPX", "LBX", "LBD", "RXD", "RXQ", "DIQ", "BPQ")
LEAKY_NAMES = {
    TARGET, "waist", "bmi", "glucose", "triglycerides", "hdl", "systolic",
    "diastolic", "blood_pressure_meds", "diabetes_meds",
}


def assert_no_target_leakage(feature_columns) -> None:
  """Raise if any feature name looks like the target or a target-defining lab."""
  bad = [
      c for c in feature_columns
      if c in LEAKY_NAMES or str(c).upper().startswith(LEAKY_PREFIXES)
  ]
  if bad:
    raise ValueError(f"Target-defining / leaky columns used as features: {bad}")


# ------------------------------------------------------------------ paths ----
def find_project_root(start: Path | None = None) -> Path:
  """Nearest folder (from `start` upwards) that contains src/features/."""
  start = Path(start or Path.cwd()).resolve()
  for folder in [start, *start.parents]:
    if (folder / "src" / "features").is_dir():
      return folder
  # fall back to this file's location: <root>/src/features/data_io.py
  return Path(__file__).resolve().parents[2]


PROJECT_ROOT = find_project_root(Path(__file__).parent)


def candidate_data_dirs(root: Path = PROJECT_ROOT) -> list[Path]:
  """Existing data/processed folders: env override, root, then child repos."""
  cands: list[Path] = []
  if os.environ.get("NHANES_DATA_DIR"):
    cands.append(Path(os.environ["NHANES_DATA_DIR"]))
  cands.append(root / "data" / "processed")
  cands += sorted(p / "data" / "processed" for p in root.iterdir() if p.is_dir())
  seen, out = set(), []
  for c in cands:
    if c.is_dir() and c.resolve() not in seen:
      seen.add(c.resolve())
      out.append(c)
  return out


def find_files(data_dir: Path, patterns: list[str]) -> list[Path]:
  """Files in data_dir (parquet/csv/feather/xpt) whose stem matches a glob pattern."""
  found: list[Path] = []
  for pat in patterns:
    for ext in SUPPORTED_SUFFIXES:
      found += sorted(Path(data_dir).glob(f"{pat}{ext}"))
  return found


def locate_data_dir(patterns: list[str] = COHORT_FILE_PATTERNS,
                    root: Path = PROJECT_ROOT) -> Path | None:
  """First candidate data dir that contains a file matching `patterns`."""
  for d in candidate_data_dirs(root):
    if find_files(d, patterns):
      return d
  return None


def inventory_data_files(root: Path = PROJECT_ROOT) -> pd.DataFrame:
  """All supported data files under `root` (skips .git, caches, virtualenvs)."""
  skip = {".git", "__pycache__", ".ipynb_checkpoints", "venv", ".venv", "site-packages"}
  rows = []
  for dirpath, dirnames, files in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in skip]
    for f in files:
      p = Path(dirpath) / f
      if p.suffix.lower() in SUPPORTED_SUFFIXES:
        rows.append({"path": str(p.relative_to(root)), "type": p.suffix.lower(),
                     "size_kb": round(p.stat().st_size / 1024, 1)})
  return pd.DataFrame(rows, columns=["path", "type", "size_kb"])


# ---------------------------------------------------------------- reading ----
def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
  path = Path(path)
  suffix = path.suffix.lower()
  if suffix == ".parquet":
    return pd.read_parquet(path, columns=columns)
  if suffix == ".csv":
    return pd.read_csv(path, usecols=columns)
  if suffix == ".feather":
    return pd.read_feather(path, columns=columns)
  if suffix == ".xpt":
    try:
      import pyreadstat
    except ImportError as e:  # optional dependency, listed in requirements.txt
      raise ImportError("pyreadstat is required to read .xpt files") from e
    df, _ = pyreadstat.read_xport(str(path), encoding="latin1")
    return df if columns is None else df[columns]
  raise ValueError(f"Unsupported file type: {path}")


def file_columns(path: Path) -> list[str]:
  """Column names without loading the data (parquet/csv/feather)."""
  path = Path(path)
  if path.suffix.lower() == ".parquet":
    import pyarrow.parquet as pq
    return list(pq.ParquetFile(path).schema_arrow.names)
  if path.suffix.lower() == ".csv":
    return list(pd.read_csv(path, nrows=0).columns)
  return list(read_table(path).columns)


# ------------------------------------------------------------ availability ----
@dataclass
class InputStatus:
  name: str
  patterns: list[str]
  required_columns: list[str]
  files: list[Path]
  missing_columns: list[str]

  @property
  def available(self) -> bool:
    return bool(self.files) and not self.missing_columns

  def describe(self) -> str:
    if not self.files:
      return (f"MISSING: {self.name}: expected a parquet/csv file named "
              f"{' or '.join(self.patterns)} with columns {self.required_columns}")
    if self.missing_columns:
      return f"INVALID: {self.name}: {[f.name for f in self.files]} lacks columns {self.missing_columns}"
    return f"OK: {self.name}: {[f.name for f in self.files]}"


def check_input(name: str, data_dir: Path | None, patterns: list[str],
                required: list[str]) -> InputStatus:
  files = find_files(data_dir, patterns) if data_dir else []
  missing: list[str] = []
  if files:
    have = set(file_columns(files[0]))
    missing = [c for c in required if c not in have]
  return InputStatus(name, patterns, required, files, missing)


def check_data_availability(root: Path = PROJECT_ROOT):
  """Status of the cohort file and the minute-level streams, searched everywhere."""
  cohort_dir = locate_data_dir(COHORT_FILE_PATTERNS, root)
  minute_dir = locate_data_dir(MINUTE_FILE_PATTERNS, root)
  return {
      "cohort": check_input("participant cohort", cohort_dir,
                            COHORT_FILE_PATTERNS, COHORT_REQUIRED),
      "minute": check_input("minute-level accelerometer streams", minute_dir,
                            MINUTE_FILE_PATTERNS, MINUTE_REQUIRED),
  }


# ------------------------------------------------------------- validation ----
def validate_cohort(cohort: pd.DataFrame) -> dict:
  """Hard checks raise ValueError; soft findings are returned in a report."""
  missing = [c for c in COHORT_REQUIRED if c not in cohort.columns]
  if missing:
    raise ValueError(f"Cohort is missing required columns: {missing}")
  if cohort["SEQN"].isna().any():
    raise ValueError("Cohort has missing SEQN values")
  dup = int(cohort["SEQN"].duplicated().sum())
  if dup:
    raise ValueError(f"Cohort has {dup} duplicate SEQN values")
  if cohort[TARGET].isna().any():
    raise ValueError(f"Target '{TARGET}' has missing values")
  if not set(cohort[TARGET].unique()) <= {0, 1}:
    raise ValueError(f"Target '{TARGET}' must be binary 0/1")
  if cohort[TARGET].nunique() < 2:
    raise ValueError(f"Target '{TARGET}' has only one class")
  na = cohort[COHORT_REQUIRED].isna().sum()
  return {
      "n_participants": len(cohort),
      "target_prevalence": float(cohort[TARGET].mean()),
      "missing_by_column": na[na > 0].to_dict(),
  }


def load_cohort(data_dir: Path | None = None) -> pd.DataFrame:
  """Load + validate Member 1's participant table, indexed by integer SEQN."""
  data_dir = data_dir or locate_data_dir(COHORT_FILE_PATTERNS)
  files = find_files(data_dir, COHORT_FILE_PATTERNS) if data_dir else []
  if not files:
    raise FileNotFoundError(
        f"clean_cohort_base.(parquet|csv) not found. Required columns: {COHORT_REQUIRED}")
  cohort = read_table(files[0])
  validate_cohort(cohort)
  cohort = cohort.copy()
  cohort["SEQN"] = cohort["SEQN"].astype("int64")
  return cohort.set_index("SEQN")
