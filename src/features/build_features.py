"""Feature Engineering pipeline for Cardiometabolic Risk Prediction (Member 2).

Expected input (``df_minutes``): one row per participant (index = SEQN) and
MultiIndex columns ``(signal, minute_of_day)`` holding a typical-day 1440-minute
profile per signal. Recognised signals:

  steps_<algorithm>   e.g. steps_oak, steps_sc, steps_ve, steps_vs, steps_ad
  mims                MIMS per minute
  activity_counts     activity counts per minute
  wear_prediction     1 = wake wear, 2 = sleep wear (anything else = other)
  wake_wear/sleep_wear  fraction of valid days a minute was wake/sleep wear
  dailysteps_<k>      (optional) total steps of valid day k, for day-to-day CV
  dailymims_<k>       (optional) mean MIMS of valid day k, for day-to-day CV

Signals that are absent are skipped, so the pipeline also works on MIMS-only data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

EPS = 1e-5
# MIMS/min below this counts as sedentary. ASSUMPTION: confirm against the proposal.
SEDENTARY_MIMS = 10.5


def _signals(df: pd.DataFrame, prefix: str) -> list[str]:
  names = df.columns.get_level_values(0).unique()
  return [s for s in names if s == prefix or s.startswith(prefix + "_")]


def _steps_by_algorithm(df: pd.DataFrame) -> dict[str, np.ndarray]:
  return {s: df[s].to_numpy(dtype=float) for s in _signals(df, "steps")}


def _mean_steps_profile(df: pd.DataFrame) -> np.ndarray | None:
  algos = _steps_by_algorithm(df)
  return np.mean(list(algos.values()), axis=0) if algos else None


def _rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
  """Rolling mean along axis 1 (valid windows only)."""
  c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), axis=1)
  return (c[:, window:] - c[:, :-window]) / window


def _run_lengths(row: np.ndarray) -> np.ndarray:
  """Lengths of consecutive True runs in a boolean vector."""
  padded = np.concatenate(([0], row.astype(np.int8), [0]))
  d = np.diff(padded)
  return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def calculate_basic_summaries(df_minutes: pd.DataFrame) -> pd.DataFrame:
  """Set A: mean daily steps per algorithm, MIMS, AC, wake/sleep wear time."""
  out = pd.DataFrame(index=df_minutes.index)
  algos = _steps_by_algorithm(df_minutes)
  for name, arr in algos.items():
    out[f"mean_daily_{name}"] = arr.sum(axis=1)
  if algos:
    out["mean_daily_steps"] = np.mean(
        [a.sum(axis=1) for a in algos.values()], axis=0
    )
  if "mims" in df_minutes.columns.get_level_values(0):
    out["mean_daily_mims"] = df_minutes["mims"].mean(axis=1)
  if "activity_counts" in df_minutes.columns.get_level_values(0):
    out["mean_daily_ac"] = df_minutes["activity_counts"].mean(axis=1)
  names = df_minutes.columns.get_level_values(0)
  if "wake_wear" in names:  # fraction-of-days profiles -> mean daily minutes
    out["wake_wear_min"] = df_minutes["wake_wear"].sum(axis=1)
  if "sleep_wear" in names:
    out["sleep_wear_min"] = df_minutes["sleep_wear"].sum(axis=1)
  if "wear_prediction" in names:
    wear = df_minutes["wear_prediction"]
    out["wake_wear_min"] = (wear == 1).sum(axis=1)
    out["sleep_wear_min"] = (wear == 2).sum(axis=1)
  return out


def _intensity_gradient(mims: np.ndarray) -> np.ndarray:
  """Slope of log10(minutes in band) vs log10(band midpoint) (Rowlands-style)."""
  edges = np.array([0, 1, 2, 4, 8, 16, 32, 64, np.inf])
  mids = np.array([0.5, 1.5, 3, 6, 12, 24, 48, 96])
  x = np.log10(mids)
  slopes = np.full(mims.shape[0], np.nan)
  for i, row in enumerate(mims):
    counts = np.histogram(row, bins=edges)[0]
    keep = counts > 0
    if keep.sum() >= 3:
      slopes[i] = np.polyfit(x[keep], np.log10(counts[keep]), 1)[0]
  return slopes


def engineer_intensity_features(df_minutes: pd.DataFrame) -> pd.DataFrame:
  """Peak cadences, minutes above 100/130 steps/min, intensity gradient."""
  out = pd.DataFrame(index=df_minutes.index)
  steps = _mean_steps_profile(df_minutes)
  if steps is not None:
    out["peak_1min_cadence"] = steps.max(axis=1)
    out["peak_30min_cadence"] = _rolling_mean(steps, 30).max(axis=1)
    out["min_above_100"] = (steps >= 100).sum(axis=1)
    out["min_above_130"] = (steps >= 130).sum(axis=1)
  if "mims" in df_minutes.columns.get_level_values(0):
    out["intensity_gradient"] = _intensity_gradient(
        df_minutes["mims"].to_numpy(dtype=float)
    )
  return out


def engineer_sedentary_features(df_minutes: pd.DataFrame) -> pd.DataFrame:
  """Sedentary fraction, bouts >= 30 min, breaks per sedentary hour."""
  out = pd.DataFrame(index=df_minutes.index)
  if "mims" not in df_minutes.columns.get_level_values(0):
    return out
  sed = df_minutes["mims"].to_numpy(dtype=float) < SEDENTARY_MIMS
  out["sedentary_fraction"] = sed.mean(axis=1)
  out["sedentary_bouts_30min"] = [(_run_lengths(r) >= 30).sum() for r in sed]
  breaks = (sed[:, :-1] & ~sed[:, 1:]).sum(axis=1)  # sedentary -> active
  out["sedentary_breaks_per_hour"] = breaks / (sed.sum(axis=1) / 60 + EPS)
  return out


def engineer_activity_patterns(df_minutes: pd.DataFrame) -> pd.DataFrame:
  """Relative amplitude (M10/L5) and day-to-day CV of steps / MIMS."""
  out = pd.DataFrame(index=df_minutes.index)
  if "mims" in df_minutes.columns.get_level_values(0):
    mims = df_minutes["mims"].to_numpy(dtype=float)
    m10 = _rolling_mean(mims, 600).max(axis=1)
    l5 = _rolling_mean(mims, 300).min(axis=1)
    out["relative_amplitude"] = (m10 - l5) / (m10 + l5 + EPS)

  for prefix, col in (("dailysteps", "day_to_day_cv_steps"),
                      ("dailymims", "day_to_day_cv_mims")):
    daily = _signals(df_minutes, prefix)
    if daily:
      d = df_minutes[daily].to_numpy(dtype=float)
      out[col] = np.nanstd(d, axis=1) / (np.nanmean(d, axis=1) + EPS)
  return out


def engineer_unsupervised_features(
    df_minutes: pd.DataFrame,
    fit_index: pd.Index | None = None,
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
  """Algorithm-disagreement CV, PCA(1-3) of log10-MIMS, K-Means phenotype.

  PCA and K-Means are fitted only on ``fit_index`` (the training participants)
  and then applied to everyone, so no test information leaks into the features.
  """
  out = pd.DataFrame(index=df_minutes.index)
  algos = _steps_by_algorithm(df_minutes)
  if algos:
    totals = np.column_stack([a.sum(axis=1) for a in algos.values()])
    out["algorithm_disagreement_cv"] = totals.std(axis=1) / (totals.mean(axis=1) + EPS)

  if "mims" in df_minutes.columns.get_level_values(0):
    log_mims = np.log10(df_minutes["mims"].fillna(0) + 1)
    if fit_index is None:
      warnings.warn("fit_index not given: PCA/K-Means fitted on ALL rows (leakage).")
      fit_rows = log_mims
    else:
      fit_rows = log_mims.loc[fit_index]
    pca = PCA(n_components=3, random_state=random_state).fit(fit_rows)
    scores = pca.transform(log_mims)
    for k in range(3):
      out[f"pca_score_{k + 1}"] = scores[:, k]
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(fit_rows)
    out["activity_phenotype_cluster"] = km.predict(log_mims)
  return out


# ------------------------------------------------------------------------
# Member 2 additions: Set 0, pending-feature reporting, feature-set assembly
# ------------------------------------------------------------------------
# Features that need step-level data. The NHANES handover has MIMS + wear codes
# only, so these are reported as PENDING rather than approximated from MIMS.
PENDING_STEP_FEATURES = {
    "mean_daily_steps_<algorithm> (x5), mean_daily_steps": "steps_<algorithm> minute profiles",
    "peak_1min_cadence": "steps_<algorithm> minute profiles",
    "peak_30min_cadence": "steps_<algorithm> minute profiles",
    "min_above_100, min_above_130": "steps_<algorithm> minute profiles",
    "algorithm_disagreement_cv": "steps from >= 2 algorithms",
    "day_to_day_cv_steps": "dailysteps_<k> (per-day step totals)",
    "mean_daily_ac": "activity_counts minute profile",
}


def pending_step_features(df_minutes: pd.DataFrame | None = None) -> dict[str, str]:
  """Feature groups that cannot be computed from the data at hand -> dependency."""
  if df_minutes is None:
    return dict(PENDING_STEP_FEATURES)
  names = set(df_minutes.columns.get_level_values(0))
  has_steps = any(n.startswith("steps") for n in names)
  has_daily_steps = any(n.startswith("dailysteps") for n in names)
  pending = {}
  for key, dep in PENDING_STEP_FEATURES.items():
    if key.startswith("mean_daily_ac"):
      ok = "activity_counts" in names
    elif key.startswith("day_to_day"):
      ok = has_daily_steps
    elif key.startswith("algorithm_disagreement"):
      ok = sum(n.startswith("steps_") for n in names) >= 2
    else:
      ok = has_steps
    if not ok:
      pending[key] = dep
  return pending


def build_demographic_features(cohort: pd.DataFrame) -> pd.DataFrame:
  """Set 0: age, sex, race/ethnicity, smoking, survey cycle (M2-01).

  Uses only demographic/lifestyle columns; the target, survey weight and wear-QC
  columns are never included. Nominal variables are one-hot encoded.
  """
  out = pd.DataFrame(index=cohort.index)
  out["age"] = cohort["RIDAGEYR"].astype(float)
  out["female"] = (cohort["RIAGENDR"] == 2).astype(int)  # NHANES: 1 male, 2 female
  out["cycle_2013_2014"] = (cohort["survey_cycle"] == "2013-2014").astype(int)
  race = pd.get_dummies(cohort["RIDRETH3"].astype("Int64"), prefix="race", dtype=int)
  smoke = pd.get_dummies(cohort["smoking_status"], prefix="smoking", dtype=int)
  return out.join(race).join(smoke)


def build_static_features(cohort: pd.DataFrame, minutes: pd.DataFrame) -> dict:
  """Deterministic (non-learned) feature blocks, aligned on SEQN.

  Returns dict with: set0, summaries (Set A extras), intensity, sedentary,
  patterns. Nothing here is fitted, so it is safe to compute on all rows.
  """
  idx = cohort.index.intersection(minutes.index)
  cohort, minutes = cohort.loc[idx], minutes.loc[idx]
  return {
      "set0": build_demographic_features(cohort),
      "summaries": calculate_basic_summaries(minutes),
      "intensity": engineer_intensity_features(minutes),
      "sedentary": engineer_sedentary_features(minutes),
      "patterns": engineer_activity_patterns(minutes),
  }


def assemble_feature_sets(static: dict, unsup: pd.DataFrame) -> dict:
  """Nested sets: Set 0 ⊂ Set A (+summaries) ⊂ Set B (+engineered, PCA, K-Means)."""
  set0 = static["set0"]
  set_a = set0.join(static["summaries"])
  unsup = pd.get_dummies(unsup, columns=["activity_phenotype_cluster"],
                         prefix="phenotype", drop_first=True, dtype=int)
  engineered = pd.concat(
      [static["intensity"], static["sedentary"], static["patterns"], unsup], axis=1
  )
  return {"set0": set0, "setA": set_a, "setB": set_a.join(engineered)}


def log_mims_matrix(minutes: pd.DataFrame) -> pd.DataFrame:
  """log10(MIMS + 1) profile matrix used by PCA and K-Means."""
  return np.log10(minutes["mims"].fillna(0) + 1)


def pca_variance_table(minutes: pd.DataFrame, fit_index, n_components: int = 10):
  """Fit PCA on training participants; return (fitted PCA, variance table)."""
  X = log_mims_matrix(minutes)
  k = min(n_components, len(fit_index), X.shape[1])
  pca = PCA(n_components=k, random_state=42).fit(X.loc[fit_index])
  table = pd.DataFrame({
      "component": np.arange(1, k + 1),
      "explained_variance_ratio": pca.explained_variance_ratio_,
      "cumulative": np.cumsum(pca.explained_variance_ratio_),
  })
  return pca, table


def kmeans_k_scan(minutes: pd.DataFrame, fit_index, k_range=range(2, 9),
                  random_state: int = 42) -> pd.DataFrame:
  """Inertia and silhouette for several k, fitted on training participants only."""
  from sklearn.metrics import silhouette_score
  X = log_mims_matrix(minutes).loc[fit_index]
  rows = []
  for k in k_range:
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
    rows.append({"k": k, "inertia": km.inertia_,
                 "silhouette": silhouette_score(X, km.labels_)})
  return pd.DataFrame(rows)
