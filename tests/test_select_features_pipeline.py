import numpy as np
import pandas as pd
import pytest

from src.features.select_features import (
    SelectionConfig, drop_non_features, load_selected_features, near_constant_filter,
    ranked_correlation_filter, run_selection, save_selection, build_selection_record)


def _data(n=400, seed=0):
  rng = np.random.default_rng(seed)
  signal = rng.normal(size=n)
  y = pd.Series((signal + rng.normal(scale=0.7, size=n) > 0).astype(int))
  X = pd.DataFrame({"signal": signal, "signal_copy": signal * 2 + 1e-3 * rng.normal(size=n),
                    "noise1": rng.normal(size=n), "noise2": rng.normal(size=n),
                    "const": 1.0, "rare": (rng.random(n) > 0.995).astype(int),
                    "SEQN": np.arange(n), "WTSAF2YR": rng.random(n), "split": "train"})
  return X, y


def test_non_features_and_constants_dropped():
  X, y = _data()
  clean, dropped = drop_non_features(X)
  assert {"SEQN", "WTSAF2YR", "split"} <= set(dropped)
  assert set(near_constant_filter(clean)) >= {"const", "rare"}


def test_ranked_corr_keeps_more_relevant_twin():
  X, y = _data()
  kept, dropped = ranked_correlation_filter(X[["signal", "signal_copy", "noise1"]], y)
  assert len(dropped) == 1 and set(dropped) | set(kept) == {"signal", "signal_copy", "noise1"}


def test_run_selection_finds_signal_and_excludes_metadata(tmp_path):
  X, y = _data()
  cfg = SelectionConfig(n_estimators_rf=50, n_estimators_gb=30, n_jobs=1)
  res = run_selection(X, y, cfg)
  for subset in res.subsets.values():
    assert not {"SEQN", "WTSAF2YR", "split", "const"} & set(subset)
  assert any(f in res.subsets["consensus"] for f in ("signal", "signal_copy"))
  p = save_selection({"S": build_selection_record("consensus", res, cfg, X.columns, len(X))},
                     tmp_path / "sel.json")
  assert load_selected_features("S", p, X.columns) == res.subsets["consensus"]
  with pytest.raises(ValueError):
    load_selected_features("S", p, ["signal"])
