"""Load Member 1's long-format minute streams and reshape them for build_features.

Input columns (see src/data/wearable_processor.py): SEQN, PAXDAYM, PAXDAYWM,
PAXSSNMP, PAXMTSM (MIMS), PAXPREDM (wear code), PAXFLGSM. There is no
minute-of-day column, so the minute index is the row order within each
(SEQN, PAXDAYM) day. Only days with exactly 1440 rows are used, so that minute k
means the same clock minute for every day. Rows must be in chronological order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.data_io import (  # noqa: F401  (re-exported for older notebooks)
    MINUTE_FILE_PATTERNS,
    MINUTE_REQUIRED as NEEDED,
    find_files,
    locate_data_dir,
    read_table,
)

MIN_PER_DAY = 1440


def validate_minute_stream(long_df: pd.DataFrame) -> None:
  """Raise if the long minute table lacks required columns or has bad values."""
  missing = [c for c in NEEDED if c not in long_df.columns]
  if missing:
    raise ValueError(f"Minute stream is missing required columns: {missing}")
  if long_df["SEQN"].isna().any():
    raise ValueError("Minute stream has missing SEQN values")


def minute_stream_to_profiles(long_df: pd.DataFrame, min_days: int = 3):
  """Long minute stream -> (wide typical-day profiles, complete days per participant).

  Wide columns are ``(signal, minute)`` with signals mims, wake_wear, sleep_wear
  and one ``dailymims_<k>`` column per valid day. MIMS is averaged over days
  using wear minutes only (PAXPREDM in 1, 2); negative sentinel values (-0.01)
  and non-wear minutes are ignored, and any remaining gaps are interpolated.
  """
  validate_minute_stream(long_df)
  df = long_df[NEEDED].copy()
  df["SEQN"] = df["SEQN"].astype("int64")
  df = df.sort_values(["SEQN", "PAXDAYM"], kind="stable")
  df["_m"] = df.groupby(["SEQN", "PAXDAYM"], sort=False).cumcount()
  size = df.groupby(["SEQN", "PAXDAYM"], sort=False)["_m"].transform("size")
  full = df[size == MIN_PER_DAY]
  days_per_id = full.groupby("SEQN")["PAXDAYM"].nunique()
  keep_ids = days_per_id[days_per_id >= min_days].index
  full = full[full["SEQN"].isin(keep_ids)]
  if full.empty:
    raise ValueError("No participant has >= %d complete 1440-minute days." % min_days)

  ids = full["SEQN"].to_numpy().reshape(-1, MIN_PER_DAY)[:, 0]
  mims = full["PAXMTSM"].to_numpy(dtype="float32").reshape(-1, MIN_PER_DAY)
  pred = full["PAXPREDM"].to_numpy(dtype="float32").reshape(-1, MIN_PER_DAY)
  wake, sleep = pred == 1, pred == 2
  mims = np.where((wake | sleep) & (mims >= 0), mims, np.nan)

  # one row per participant-day -> aggregate to one row per participant
  day_mean = pd.DataFrame(np.nanmean(mims, axis=1), index=ids)
  day_mean["_k"] = day_mean.groupby(level=0).cumcount()
  daily = day_mean.reset_index().pivot(index="index", columns="_k", values=0)
  daily.columns = pd.MultiIndex.from_product(
      [[f"dailymims_{k}" for k in daily.columns], [0]]
  )

  def profile(arr: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(arr, index=ids).groupby(level=0).mean()

  m = profile(mims)  # NaN-skipping mean over days
  m = m.T.interpolate(limit_direction="both").T.fillna(0.0)
  w = profile(wake.astype("float32"))
  s = profile(sleep.astype("float32"))

  cols = pd.RangeIndex(MIN_PER_DAY)
  wide = pd.concat(
      {
          "mims": m.set_axis(cols, axis=1),
          "wake_wear": w.set_axis(cols, axis=1),
          "sleep_wear": s.set_axis(cols, axis=1),
      },
      axis=1,
  ).join(daily)
  wide.index.name = "SEQN"
  return wide, days_per_id.rename("n_full_days")


def load_minute_profiles(data_dir: Path | None = None, min_days: int = 3):
  """Find minute_streams*/minute_level* files, concatenate cycles, reshape."""
  data_dir = data_dir or locate_data_dir(MINUTE_FILE_PATTERNS)
  files = find_files(data_dir, MINUTE_FILE_PATTERNS) if data_dir else []
  if not files:
    raise FileNotFoundError(
        f"No minute-stream file found in {data_dir}. Expected a parquet or CSV "
        "file named minute_streams* or minute_level* containing "
        f"{', '.join(NEEDED)}."
    )
  frames = [read_table(f, NEEDED) for f in files]
  print("minute files:", [f.name for f in files])
  return minute_stream_to_profiles(pd.concat(frames, ignore_index=True), min_days)
