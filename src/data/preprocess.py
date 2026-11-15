"""Leakage-safe preprocessing pipeline for cardiometabolic risk modelling.

Steps (in this order):
  1. stratified train/test split   <- done FIRST so no statistic sees the test set
  2. missing-value imputation      median / KNN (numeric), most-frequent (categorical)
  3. one-hot encoding              low-cardinality / text columns
  4. scaling                       StandardScaler or MinMaxScaler (numeric only)

Imputers, encoder and scaler are fitted on the TRAINING rows only and then applied
unchanged to the test rows (and to any future data).

Quick use:
    from src.data.preprocess import preprocess
    res = preprocess("data/processed/clean_cohort_base.parquet")
    res.X_train, res.X_test, res.y_train, res.y_test      # DataFrames / Series
    res.pipeline                                          # fitted, reusable

Command line:
    python -m src.data.preprocess --file data/processed/clean_cohort_base.parquet --out data/processed
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

DEFAULT_TARGET = "metabolic_syndrome_target"
# Never used as predictors: row identifier and survey sampling weight.
DEFAULT_DROP = ("SEQN", "WTSAF2YR")
MAX_CAT_UNIQUE = 10  # numeric columns with <= this many distinct values count as coded categories


@dataclass
class PreprocessResult:
  """Everything produced by :func:`preprocess`."""
  X_train: pd.DataFrame
  X_test: pd.DataFrame
  y_train: pd.Series
  y_test: pd.Series
  pipeline: ColumnTransformer
  numeric_cols: list[str]
  categorical_cols: list[str]
  dropped_cols: list[str] = field(default_factory=list)

  @property
  def feature_names(self) -> list[str]:
    return list(self.X_train.columns)


# ------------------------------------------------------------------ loading ----
def load_data(path: str | Path) -> pd.DataFrame:
  """Read a parquet, csv or feather file."""
  path = Path(path)
  readers = {".parquet": pd.read_parquet, ".csv": pd.read_csv, ".feather": pd.read_feather}
  if path.suffix.lower() not in readers:
    raise ValueError(f"Unsupported file type '{path.suffix}' (use parquet, csv or feather)")
  if not path.exists():
    raise FileNotFoundError(path)
  return readers[path.suffix.lower()](path)


# ------------------------------------------------------------- column types ----
def infer_column_types(
    X: pd.DataFrame, max_cat_unique: int = MAX_CAT_UNIQUE
) -> tuple[list[str], list[str]]:
  """Split columns into (numeric, categorical).

  Categorical = text / category / bool columns, plus numeric columns with at most
  ``max_cat_unique`` distinct values (e.g. sex or race stored as codes).
  """
  numeric, categorical = [], []
  for col in X.columns:
    s = X[col]
    if (not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
        or s.nunique(dropna=True) <= max_cat_unique):
      categorical.append(col)
    else:
      numeric.append(col)
  return numeric, categorical


# ------------------------------------------------------------- the pipeline ----
def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    imputer: str = "median",
    scaler: str = "standard",
    knn_neighbors: int = 5,
) -> ColumnTransformer:
  """Unfitted ColumnTransformer: impute -> (scale | one-hot) per column group.

  Args:
    imputer: ``"median"`` (robust, default) or ``"knn"`` for numeric columns.
      Categorical columns always use the most frequent value.
    scaler: ``"standard"`` (zero mean, unit variance) or ``"minmax"`` (0-1).
  """
  if imputer not in {"median", "knn"}:
    raise ValueError(f"imputer must be 'median' or 'knn', got {imputer!r}")
  if scaler not in {"standard", "minmax"}:
    raise ValueError(f"scaler must be 'standard' or 'minmax', got {scaler!r}")

  num_imputer = (KNNImputer(n_neighbors=knn_neighbors) if imputer == "knn"
                 else SimpleImputer(strategy="median"))
  num_scaler = StandardScaler() if scaler == "standard" else MinMaxScaler()

  transformers = []
  if numeric_cols:
    transformers.append(("num", Pipeline([("impute", num_imputer), ("scale", num_scaler)]),
                         numeric_cols))
  if categorical_cols:
    transformers.append(("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # 'if_binary' keeps one column for 2-level variables; unseen levels -> all zeros
        ("encode", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ]), categorical_cols))
  if not transformers:
    raise ValueError("No feature columns to preprocess")

  pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
  return pre.set_output(transform="pandas")


# ---------------------------------------------------------------- splitting ----
def split_data(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
  """Reproducible train/test split, stratified on ``target`` by default.

  Rows with a missing target are dropped (with a warning). The split keeps the
  original index (e.g. SEQN), so rows can be traced back.
  """
  if target not in df.columns:
    raise KeyError(f"Target '{target}' not found. Columns: {list(df.columns)}")
  if not 0 < test_size < 1:
    raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
  n_missing = int(df[target].isna().sum())
  if n_missing:
    warnings.warn(f"Dropping {n_missing} rows with a missing target")
    df = df.dropna(subset=[target])
  y = df[target]
  if y.nunique() < 2:
    raise ValueError("Target has fewer than two classes")
  if stratify and y.value_counts().min() < 2:
    raise ValueError("Cannot stratify: a class has fewer than 2 rows")
  X = df.drop(columns=[target])
  return train_test_split(X, y, test_size=test_size, random_state=random_state,
                          stratify=y if stratify else None)


# --------------------------------------------------------------- main entry ----
def preprocess(
    data: pd.DataFrame | str | Path,
    target: str = DEFAULT_TARGET,
    drop_cols: tuple[str, ...] | list[str] = DEFAULT_DROP,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    imputer: str = "median",
    scaler: str = "standard",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    max_cat_unique: int = MAX_CAT_UNIQUE,
) -> PreprocessResult:
  """Split, then impute / encode / scale with statistics learned on train only.

  Args:
    data: DataFrame or path to a parquet/csv/feather file.
    target: name of the binary label column.
    drop_cols: columns that are never features (IDs, survey weights, ...).
    numeric_cols / categorical_cols: override automatic column typing.
    imputer, scaler: see :func:`build_preprocessor`.

  Returns:
    :class:`PreprocessResult` with transformed ``X_train`` / ``X_test`` (named
    DataFrames sharing the original index), the labels, and the fitted pipeline.
  """
  df = load_data(data) if isinstance(data, (str, Path)) else data.copy()
  X_train, X_test, y_train, y_test = split_data(df, target, test_size, random_state, stratify)

  dropped = [c for c in drop_cols if c in X_train.columns]
  X_train, X_test = X_train.drop(columns=dropped), X_test.drop(columns=dropped)

  if numeric_cols is None and categorical_cols is None:
    numeric_cols, categorical_cols = infer_column_types(X_train, max_cat_unique)
  else:
    given = set(numeric_cols or []) | set(categorical_cols or [])
    rest = [c for c in X_train.columns if c not in given]
    auto_num, auto_cat = infer_column_types(X_train[rest], max_cat_unique) if rest else ([], [])
    numeric_cols = list(numeric_cols or []) + auto_num
    categorical_cols = list(categorical_cols or []) + auto_cat
  missing = [c for c in numeric_cols + categorical_cols if c not in X_train.columns]
  if missing:
    raise KeyError(f"Columns not in data: {missing}")

  pipe = build_preprocessor(numeric_cols, categorical_cols, imputer, scaler)
  Xtr = pipe.fit_transform(X_train)   # statistics learned from TRAIN only
  Xte = pipe.transform(X_test)        # applied unchanged to TEST
  return PreprocessResult(Xtr, Xte, y_train, y_test, pipe, numeric_cols, categorical_cols, dropped)


# ------------------------------------------------------------- persistence ----
def save_pipeline(pipeline: ColumnTransformer, path: str | Path) -> Path:
  """Save the fitted preprocessor so new data can be transformed identically."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  joblib.dump(pipeline, path)
  return path


def load_pipeline(path: str | Path) -> ColumnTransformer:
  return joblib.load(path)


def save_splits(res: PreprocessResult, out_dir: str | Path) -> list[Path]:
  """Write X_train/X_test/y_train/y_test as parquet files into ``out_dir``."""
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  files = {
      "X_train": res.X_train, "X_test": res.X_test,
      "y_train": res.y_train.to_frame(), "y_test": res.y_test.to_frame(),
  }
  paths = []
  for name, frame in files.items():
    p = out / f"{name}.parquet"
    frame.to_parquet(p)
    paths.append(p)
  return paths


def summarize(res: PreprocessResult) -> None:
  """Print a short report of what the pipeline did."""
  print(f"train: {res.X_train.shape} | test: {res.X_test.shape}")
  print(f"dropped (not features): {res.dropped_cols}")
  print(f"numeric ({len(res.numeric_cols)}): {res.numeric_cols}")
  print(f"categorical ({len(res.categorical_cols)}): {res.categorical_cols}")
  print(f"model features after encoding: {len(res.feature_names)}")
  print(f"target rate  train: {res.y_train.mean():.3f}  test: {res.y_test.mean():.3f}")
  print(f"NaNs left: train {int(res.X_train.isna().sum().sum())}, test {int(res.X_test.isna().sum().sum())}")


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description="Split, impute, encode and scale a dataset")
  ap.add_argument("--file", required=True, help="parquet/csv/feather file")
  ap.add_argument("--target", default=DEFAULT_TARGET)
  ap.add_argument("--imputer", choices=["median", "knn"], default="median")
  ap.add_argument("--scaler", choices=["standard", "minmax"], default="standard")
  ap.add_argument("--test-size", type=float, default=0.2)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--out", help="folder to save X/y parquet files and preprocessor.joblib")
  a = ap.parse_args()
  result = preprocess(a.file, a.target, imputer=a.imputer, scaler=a.scaler,
                      test_size=a.test_size, random_state=a.seed)
  summarize(result)
  if a.out:
    paths = save_splits(result, a.out) + [save_pipeline(result.pipeline, Path(a.out) / "preprocessor.joblib")]
    print("saved:", [p.name for p in paths])
