"""SYNTHETIC TEST FIXTURES - for unit tests only, never used by the analysis notebook."""
import numpy as np
import pandas as pd

N_MIN = 1440


def make_test_cohort(n=200, seed=0) -> pd.DataFrame:
  """Synthetic participant table with the same columns as clean_cohort_base."""
  rng = np.random.default_rng(seed)
  age = rng.integers(20, 80, n).astype(float)
  y = (rng.random(n) < 1 / (1 + np.exp(-(age - 50) / 12))).astype(int)
  return pd.DataFrame({
      "SEQN": np.arange(1000, 1000 + n, dtype=float),
      "survey_cycle": rng.choice(["2011-2012", "2013-2014"], n),
      "RIDAGEYR": age,
      "RIAGENDR": rng.integers(1, 3, n).astype(float),
      "RIDRETH3": rng.choice([1.0, 3.0, 4.0, 6.0], n),
      "WTSAF2YR": rng.uniform(1e4, 3e5, n),
      "valid_days_count": 7,
      "mean_daily_wear_mins": 900.0,
      "metabolic_syndrome_target": y,
      "smoking_status": rng.choice(["never", "former", "current"], n),
  })


def make_test_minutes(ids, seed=0, with_steps=False) -> pd.DataFrame:
  """Synthetic wide minute profiles (signal, minute) indexed by SEQN."""
  rng = np.random.default_rng(seed)
  n, t = len(ids), np.arange(N_MIN)
  circ = np.clip(np.sin((t - 360) / N_MIN * 2 * np.pi), 0, None)
  level = rng.lognormal(0, 0.4, n)[:, None]
  mims = level * 20 * circ * rng.gamma(3, 0.35, (n, N_MIN))
  wake = np.tile(((t >= 360) & (t <= 1320)).astype(float), (n, 1))
  blocks = {"mims": mims, "wake_wear": wake, "sleep_wear": 1 - wake}
  if with_steps:
    for a in ("oak", "sc"):
      blocks[f"steps_{a}"] = mims * 5 * rng.uniform(0.9, 1.1, (n, 1))
  frames = [pd.DataFrame(v, index=ids, columns=pd.MultiIndex.from_product([[k], t]))
            for k, v in blocks.items()]
  for d in range(4):
    frames.append(pd.DataFrame(mims.mean(1) * rng.uniform(0.8, 1.2, n), index=ids,
                               columns=pd.MultiIndex.from_product([[f"dailymims_{d}"], [0]])))
  out = pd.concat(frames, axis=1)
  out.index.name = "SEQN"
  return out


def make_test_long_minutes(ids, days=4, partial_last_day=True, seed=0) -> pd.DataFrame:
  """Synthetic long-format minute stream in Member 1's column layout."""
  rng = np.random.default_rng(seed)
  rows = []
  for sid in ids:
    for d in range(1, days + 1 + int(partial_last_day)):
      n = N_MIN if d <= days else 500
      rows.append(pd.DataFrame({
          "SEQN": float(sid), "PAXDAYM": d, "PAXDAYWM": d, "PAXSSNMP": 80.0,
          "PAXMTSM": rng.gamma(2, 3, n), "PAXPREDM": np.ones(n), "PAXFLGSM": 0.0}))
  return pd.concat(rows, ignore_index=True)
