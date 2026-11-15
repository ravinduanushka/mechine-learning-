"""Initial Exploratory Data Analysis (EDA) for cardiometabolic risk prediction.

Run as a script:
    python eda_initial.py                         # auto-detects a file under data/
    python eda_initial.py --file data/processed/clean_cohort_base.parquet
    python eda_initial.py --target metabolic_syndrome_target --save --no-show

Or in a Jupyter notebook:
    from eda_initial import run_eda
    df, report = run_eda()                        # or run_eda("path/to/file.parquet")

Every step is a small function, so you can also call them one by one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SUPPORTED = (".parquet", ".csv", ".feather")
TARGET_CANDIDATES = ["metabolic_syndrome_target", "cardiometabolic_risk", "target", "label"]
MAX_CAT_UNIQUE = 10        # numeric columns with <= this many distinct values are treated as coded categories
sns.set_theme(style="whitegrid")


# --------------------------------------------------------------- 1. loading ----
def find_data_files(data_dir: Path) -> list[Path]:
  """All parquet/csv/feather files under data_dir (searches sub-folders too)."""
  return sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in SUPPORTED)


def locate_data_dir(start: Path | None = None) -> Path:
  """Nearest 'data' folder: here, in a parent, or inside a child repo folder."""
  start = Path(start or Path.cwd()).resolve()
  for folder in [start, *start.parents]:
    for cand in [folder / "data", *sorted(folder.glob("*/data"))]:
      if cand.is_dir() and find_data_files(cand):
        return cand
  raise FileNotFoundError("No data/ folder with parquet/csv/feather files found near " + str(start))


def load_dataset(path: Path) -> pd.DataFrame:
  """Read a parquet, csv or feather file."""
  path = Path(path)
  readers = {".parquet": pd.read_parquet, ".csv": pd.read_csv, ".feather": pd.read_feather}
  if path.suffix.lower() not in readers:
    raise ValueError(f"Unsupported file type: {path.suffix} (use {SUPPORTED})")
  return readers[path.suffix.lower()](path)


def pick_file(data_dir: Path) -> Path:
  """Prefer the cleaned cohort table; otherwise the first file found."""
  files = find_data_files(data_dir)
  preferred = [f for f in files if "clean_cohort" in f.name]
  return (preferred or files)[0]


# --------------------------------------------------------------- 2. overview ----
def overview(df: pd.DataFrame, n: int = 5) -> None:
  """Shape, first rows, column names and data types."""
  print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
  print(f"First {n} rows:")
  print(df.head(n).to_string(), "\n")
  print("Columns and data types:")
  print(df.dtypes.rename("dtype").to_frame().assign(n_unique=df.nunique()).to_string(), "\n")


# ----------------------------------------------------------- 3. data quality ----
def find_id_columns(df: pd.DataFrame) -> list[str]:
  """Columns that identify a row (e.g. SEQN); excluded from statistics."""
  return [c for c in df.columns if c.upper() in {"SEQN", "ID"} or c.upper().endswith("_ID")]


def quality_report(df: pd.DataFrame) -> dict:
  """Missing values, duplicate rows and duplicate IDs."""
  miss = df.isna().sum()
  miss_df = pd.DataFrame({"missing": miss, "missing_%": (miss / len(df) * 100).round(2)})
  miss_df = miss_df[miss_df["missing"] > 0].sort_values("missing", ascending=False)
  report = {
      "missing": miss_df,
      "duplicate_rows": int(df.duplicated().sum()),
      "duplicate_ids": {c: int(df[c].duplicated().sum()) for c in find_id_columns(df)},
  }
  print("Missing values:", "none" if miss_df.empty else "\n" + miss_df.to_string())
  print("Duplicate rows:", report["duplicate_rows"])
  print("Duplicate IDs :", report["duplicate_ids"] or "no ID column found", "\n")
  return report


# ------------------------------------------------------ 4. column type split ----
def split_columns(df: pd.DataFrame, target: str | None = None, max_cat_unique: int = MAX_CAT_UNIQUE):
  """Return (numerical, categorical, id_columns), excluding the target.

  Categorical = text/category/bool columns, plus numeric columns with at most
  `max_cat_unique` distinct values (e.g. sex or race stored as codes).
  """
  ids = find_id_columns(df)
  numerical, categorical = [], []
  for c in df.columns:
    if c in ids or c == target:
      continue
    s = df[c]
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
      categorical.append(c)
    elif s.nunique(dropna=True) <= max_cat_unique:
      categorical.append(c)
    else:
      numerical.append(c)
  print(f"ID columns ({len(ids)}): {ids}")
  print(f"Numerical ({len(numerical)}): {numerical}")
  print(f"Categorical ({len(categorical)}): {categorical}")
  print(f"(numeric columns with <= {max_cat_unique} distinct values are treated as coded categories)\n")
  return numerical, categorical, ids


def describe_numeric(df: pd.DataFrame, numerical: list[str]) -> pd.DataFrame:
  """Mean, median, standard deviation, min, max and skewness."""
  if not numerical:
    return pd.DataFrame()
  d = df[numerical].agg(["mean", "median", "std", "min", "max", "skew"]).T.round(3)
  print("Descriptive statistics (numerical columns):")
  print(d.to_string(), "\n")
  return d


def outlier_report(df: pd.DataFrame, numerical: list[str], iqr_k: float = 1.5,
                   z_thresh: float = 3.0, flag_pct: float = 5.0) -> pd.DataFrame:
  """Outliers per numerical column by the IQR rule and by z-score.

  - IQR outlier : value outside [Q1 - k*IQR, Q3 + k*IQR]  (k = ``iqr_k``, default 1.5)
  - far outlier : same rule with 2*k (default 3.0) - very extreme values
  - z outlier   : |z| > ``z_thresh``
  A column is flagged when more than ``flag_pct`` % of its values are IQR outliers.
  Outliers are only REPORTED here; nothing is removed or changed.
  """
  if iqr_k <= 0 or z_thresh <= 0:
    raise ValueError("iqr_k and z_thresh must be positive")
  rows = []
  for c in numerical:
    s = df[c].dropna()
    if s.empty:
      continue
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    far_lo, far_hi = q1 - 2 * iqr_k * iqr, q3 + 2 * iqr_k * iqr
    std = s.std()
    n_iqr = int(((s < lo) | (s > hi)).sum())
    rows.append({
        "column": c,
        "n_iqr_outliers": n_iqr,
        "iqr_outlier_%": round(n_iqr / len(s) * 100, 2),
        "n_far_outliers": int(((s < far_lo) | (s > far_hi)).sum()),
        "n_z_outliers": int((((s - s.mean()) / std).abs() > z_thresh).sum()) if std > 0 else 0,
        "lower_fence": round(lo, 3),
        "upper_fence": round(hi, 3),
        "min": round(s.min(), 3),
        "max": round(s.max(), 3),
    })
  out = pd.DataFrame(rows)
  if out.empty:
    print("Outliers: no numerical columns to check\n")
    return out
  out["flag"] = np.where(out["iqr_outlier_%"] > flag_pct, "CHECK", "")
  out = out.set_index("column")
  print(f"Outlier report (IQR k={iqr_k}, z>{z_thresh}); reported only, nothing removed:")
  print(out.to_string())
  flagged = out.index[out["flag"] == "CHECK"].tolist()
  print("Columns with more than", f"{flag_pct}% IQR outliers:", flagged or "none",
        "\n  (heavy right skew often explains these; consider a log transform or robust scaling.)\n")
  return out


# ---------------------------------------------------------------- 5. target ----
def detect_target(df: pd.DataFrame, target: str | None = None) -> str | None:
  """Use the given target, else the first well-known target name or *_target column."""
  if target:
    if target not in df.columns:
      raise KeyError(f"Target '{target}' not in columns: {list(df.columns)}")
    return target
  for c in TARGET_CANDIDATES:
    if c in df.columns:
      return c
  hits = [c for c in df.columns if c.lower().endswith(("_target", "_label"))]
  return hits[0] if hits else None


def target_summary(df: pd.DataFrame, target: str) -> dict:
  """Class counts/percentages, imbalance ratio and a plain-language verdict."""
  counts = df[target].value_counts(dropna=False).sort_index()
  pct = (counts / counts.sum() * 100).round(2)
  minority = pct.drop(index=[i for i in pct.index if pd.isna(i)], errors="ignore").min()
  ratio = counts.max() / counts.min()
  if minority >= 40:
    verdict = "roughly BALANCED - plain stratified splits and accuracy-style metrics are fine"
  elif minority >= 20:
    verdict = "MODERATELY imbalanced - use stratified splits; report ROC-AUC and PR-AUC, not only accuracy"
  else:
    verdict = "STRONGLY imbalanced - use stratification, class weights or resampling, and PR-AUC/recall"
  print(f"Target '{target}':")
  print(pd.DataFrame({"count": counts, "percent": pct}).to_string())
  print(f"Imbalance ratio (majority/minority): {ratio:.2f}  -> {verdict}\n")
  return {"counts": counts, "percent": pct, "ratio": float(ratio), "verdict": verdict}


def visualization_plan(target: str | None, numerical: list[str], categorical: list[str]) -> None:
  """Print what the plots below show and what to check next."""
  print("VISUALISATION PLAN")
  if target:
    print(f"  1. Target balance bar chart ({target})")
    print(f"  2. Target rate by each categorical column ({len(categorical)} columns)")
    print(f"  3. Numeric distributions split by target ({len(numerical)} columns)")
  print("  4. Boxplots of numerical columns (outliers, split by target)")
  print("  5. Correlation heatmap of numerical columns (look for |r| > 0.9 pairs)")
  print("  6. Missing-value bar chart (only if anything is missing)")
  print("  Next steps: check outliers, decide imputation, confirm no target-defining columns are features\n")


# ------------------------------------------------------------------ 6. plots ----
def _finish(fig, name: str, save_dir: Path | None, show: bool) -> None:
  fig.tight_layout()
  if save_dir:
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"eda_{name}.png", dpi=150, bbox_inches="tight")
  plt.show() if show else plt.close(fig)


def plot_target(df, target, save_dir=None, show=True):
  fig, ax = plt.subplots(figsize=(4.5, 3.5))
  order = sorted(df[target].dropna().unique())
  sns.countplot(data=df, x=target, order=order, ax=ax, color="#4C78A8")
  for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=9)
  ax.set_title(f"Target distribution: {target}")
  _finish(fig, "target", save_dir, show)


def plot_numeric(df, numerical, target=None, save_dir=None, show=True, max_cols=12):
  cols = numerical[:max_cols]
  if not cols:
    return
  ncols = 3
  nrows = int(np.ceil(len(cols) / ncols))
  fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows), squeeze=False)
  for ax, c in zip(axes.ravel(), cols):
    if target:
      sns.histplot(data=df, x=c, hue=target, bins=30, stat="density", common_norm=False,
                   element="step", ax=ax, palette="deep")
    else:
      sns.histplot(data=df, x=c, bins=30, ax=ax, color="#4C78A8")
    ax.set_title(c, fontsize=10)
  for ax in axes.ravel()[len(cols):]:
    ax.axis("off")
  _finish(fig, "numeric_distributions", save_dir, show)


def plot_boxplots(df, numerical, target=None, save_dir=None, show=True, max_cols=12):
  """One boxplot per numerical column (split by target when given) to see outliers."""
  cols = numerical[:max_cols]
  if not cols:
    return
  ncols = 3
  nrows = int(np.ceil(len(cols) / ncols))
  fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows), squeeze=False)
  for ax, c in zip(axes.ravel(), cols):
    if target:
      sns.boxplot(data=df, x=target, y=c, ax=ax, color="#4C78A8", fliersize=2)
    else:
      sns.boxplot(data=df, y=c, ax=ax, color="#4C78A8", fliersize=2)
    ax.set_title(c, fontsize=10)
  for ax in axes.ravel()[len(cols):]:
    ax.axis("off")
  _finish(fig, "boxplots", save_dir, show)


def plot_categorical(df, categorical, target=None, save_dir=None, show=True, max_cols=9):
  cols = categorical[:max_cols]
  if not cols:
    return
  ncols = 3
  nrows = int(np.ceil(len(cols) / ncols))
  fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows), squeeze=False)
  for ax, c in zip(axes.ravel(), cols):
    if target and pd.api.types.is_numeric_dtype(df[target]):
      rate = df.groupby(c, observed=True)[target].mean()      # target rate per category
      rate.plot.bar(ax=ax, color="#F58518")
      ax.set_ylabel(f"mean({target})")
    else:
      df[c].value_counts().plot.bar(ax=ax, color="#4C78A8")
    ax.set_title(c, fontsize=10)
  for ax in axes.ravel()[len(cols):]:
    ax.axis("off")
  _finish(fig, "categorical", save_dir, show)


def plot_correlation(df, numerical, save_dir=None, show=True):
  if len(numerical) < 2:
    return
  corr = df[numerical].corr()
  fig, ax = plt.subplots(figsize=(0.7 * len(numerical) + 3, 0.6 * len(numerical) + 2.5))
  sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
  ax.set_title("Correlation of numerical columns")
  _finish(fig, "correlation", save_dir, show)
  pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
  high = pairs[pairs.abs() > 0.9]
  print("Highly correlated pairs (|r| > 0.9):", "none" if high.empty else "\n" + high.round(3).to_string(), "\n")


def plot_missing(report, save_dir=None, show=True):
  miss = report["missing"]
  if miss.empty:
    return
  fig, ax = plt.subplots(figsize=(6, 0.4 * len(miss) + 1.5))
  miss["missing_%"].sort_values().plot.barh(ax=ax, color="#E45756")
  ax.set_xlabel("% missing")
  _finish(fig, "missing", save_dir, show)


# --------------------------------------------------------------- 7. run all ----
def run_eda(path: str | Path | None = None, target: str | None = None,
            save: bool = False, show: bool = True, iqr_k: float = 1.5):
  """Run the whole EDA. Returns (DataFrame, dict with the computed summaries).

  ``iqr_k`` sets the IQR multiplier for outlier detection (1.5 standard, 3.0 strict).
  """
  path = Path(path) if path else pick_file(locate_data_dir())
  print(f"Loading: {path}\n")
  df = load_dataset(path)
  save_dir = None
  if save:  # <root>/data/.../file -> <root>/reports/figures
    save_dir = next((p / "reports" / "figures" for p in Path(path).resolve().parents
                     if (p / "data").is_dir()), Path("reports/figures"))

  overview(df)
  report = quality_report(df)
  target = detect_target(df, target)
  numerical, categorical, ids = split_columns(df, target)
  stats = describe_numeric(df, numerical)
  outliers = outlier_report(df, numerical, iqr_k=iqr_k)

  summary = None
  if target:
    summary = target_summary(df, target)
  else:
    print("No target column detected - pass target='your_column' to analyse class balance.\n")
  visualization_plan(target, numerical, categorical)

  if target:
    plot_target(df, target, save_dir, show)
  plot_numeric(df, numerical, target, save_dir, show)
  plot_boxplots(df, numerical, target, save_dir, show)
  plot_categorical(df, categorical, target, save_dir, show)
  plot_correlation(df, numerical, save_dir, show)
  plot_missing(report, save_dir, show)

  return df, {"quality": report, "numerical": numerical, "categorical": categorical,
              "ids": ids, "stats": stats, "outliers": outliers, "target": target,
              "target_summary": summary}


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description="Initial EDA for cardiometabolic risk data")
  ap.add_argument("--file", help="parquet/csv/feather file (default: auto-detect under data/)")
  ap.add_argument("--target", help="target column (default: auto-detect)")
  ap.add_argument("--iqr-k", type=float, default=1.5,
                  help="IQR multiplier for outliers (1.5 standard, 3.0 strict)")
  ap.add_argument("--save", action="store_true", help="save figures to reports/figures/")
  ap.add_argument("--no-show", action="store_true", help="do not open plot windows")
  a = ap.parse_args()
  run_eda(a.file, a.target, save=a.save, show=not a.no_show, iqr_k=a.iqr_k)
