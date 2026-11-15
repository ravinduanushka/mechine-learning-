import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "scripts", ROOT):          # repo layout: scripts/eda_initial.py
  if (d / "eda_initial.py").exists():
    sys.path.insert(0, str(d))
    break

import eda_initial as eda  # noqa: E402


@pytest.fixture()
def df():
  rng = np.random.default_rng(0)
  base = rng.normal(50, 5, 500)
  d = pd.DataFrame({"x": base, "flat": 1.0, "skewed": rng.lognormal(0, 1.5, 500)})
  d.loc[0, "x"] = 500.0                      # one extreme value
  d["target"] = rng.integers(0, 2, 500)
  return d


def test_outlier_report_finds_known_outlier(df):
  r = eda.outlier_report(df, ["x"])
  assert r.loc["x", "n_iqr_outliers"] >= 1
  assert r.loc["x", "n_far_outliers"] >= 1 and r.loc["x", "n_z_outliers"] >= 1
  assert r.loc["x", "max"] == 500.0


def test_outlier_report_constant_column_and_flagging(df):
  r = eda.outlier_report(df, ["flat", "skewed"])
  assert r.loc["flat", "n_iqr_outliers"] == 0 and r.loc["flat", "n_z_outliers"] == 0
  assert r.loc["skewed", "iqr_outlier_%"] > 0        # skewed data has tail outliers


def test_outlier_report_does_not_modify_data_and_validates(df):
  before = df.copy()
  eda.outlier_report(df, ["x", "skewed"])
  pd.testing.assert_frame_equal(df, before)           # report only
  with pytest.raises(ValueError):
    eda.outlier_report(df, ["x"], iqr_k=0)
  assert eda.outlier_report(df, []).empty


def test_stricter_k_gives_fewer_outliers(df):
  loose = eda.outlier_report(df, ["skewed"], iqr_k=1.5).loc["skewed", "n_iqr_outliers"]
  strict = eda.outlier_report(df, ["skewed"], iqr_k=3.0).loc["skewed", "n_iqr_outliers"]
  assert strict < loose


def test_run_eda_includes_outliers_and_boxplots(df, tmp_path):
  import matplotlib
  matplotlib.use("Agg")
  f = tmp_path / "data" / "d.csv"
  f.parent.mkdir()
  df.to_csv(f, index=False)
  _, rep = eda.run_eda(f, target="target", save=True, show=False)
  assert "x" in rep["outliers"].index
  assert (tmp_path / "reports" / "figures" / "eda_boxplots.png").exists()
