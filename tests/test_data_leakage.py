import warnings

import numpy as np
import pandas as pd
import pytest

from src.features import build_features as bf
from src.features import evaluate_sets as ev
from src.features import select_features as sf
from tests.fixtures import make_test_cohort, make_test_minutes


def _cohort(n=200):
  return make_test_cohort(n).assign(SEQN=lambda d: d.SEQN.astype("int64")).set_index("SEQN")


def test_split_is_disjoint_stratified_reproducible():
  y = _cohort()["metabolic_syndrome_target"]
  tr, te = ev.stratified_split(y, 0.2, seed=42)
  tr2, te2 = ev.stratified_split(y, 0.2, seed=42)
  assert set(tr).isdisjoint(te) and len(tr) + len(te) == len(y)
  assert len(te) == round(0.2 * len(y))
  assert abs(y.loc[tr].mean() - y.loc[te].mean()) < 0.03
  assert tr.equals(tr2) and te.equals(te2)


def test_pca_kmeans_ignore_test_participants():
  minutes = make_test_minutes(_cohort(150).index)
  train, test = minutes.index[:110], minutes.index[110:]
  base = bf.engineer_unsupervised_features(minutes, fit_index=train)
  changed = minutes.copy()
  mcols = [c for c in changed.columns if c[0] == "mims"]
  changed.loc[test, mcols] = changed.loc[test, mcols].to_numpy() * 7 + 3
  after = bf.engineer_unsupervised_features(changed, fit_index=train)
  pd.testing.assert_frame_equal(base.loc[train], after.loc[train])   # train rows unchanged
  assert not np.allclose(base.loc[test, "pca_score_1"], after.loc[test, "pca_score_1"])


def test_unsupervised_without_fit_index_warns():
  minutes = make_test_minutes(_cohort(60).index)
  with pytest.warns(UserWarning, match="fit_index not given"):
    bf.engineer_unsupervised_features(minutes)


def test_cv_fits_learned_steps_on_training_fold_only(monkeypatch):
  c = _cohort(160)
  y = c["metabolic_syndrome_target"]
  train, test = ev.stratified_split(y)
  s0 = bf.build_demographic_features(c)
  fit_calls, selector_rows = [], []

  def builder(fit_ids):
    fit_calls.append(set(fit_ids))
    return s0

  orig = sf.apply_feature_selection

  def spy(X, *a, **k):
    selector_rows.append(set(X.index))
    return orig(X, *a, **k)

  monkeypatch.setattr(sf, "apply_feature_selection", spy)
  folds, _ = ev.cross_validate_set(builder, y, train, n_splits=4)
  assert len(fit_calls) == len(selector_rows) == 4
  for fit_ids, rows in zip(fit_calls, selector_rows):
    assert fit_ids <= set(train) and fit_ids.isdisjoint(test)
    assert rows == fit_ids                      # selector saw exactly the training fold
    assert abs(len(fit_ids) - 0.75 * len(train)) <= 1
  assert folds["roc_auc"].between(0, 1).all()


def test_compare_sets_use_identical_folds_and_participants():
  c = _cohort(160)
  y = c["metabolic_syndrome_target"]
  train, _ = ev.stratified_split(y)
  s0 = bf.build_demographic_features(c)
  seen = {"a": [], "b": []}

  def make(key, frame):
    def build(ids):
      seen[key].append(frozenset(ids))
      return frame
    return build

  summary, _ = ev.compare_feature_sets(
      {"a": make("a", s0), "b": make("b", s0.assign(extra=1.0))}, y, train, n_splits=3)
  assert seen["a"] == seen["b"]                     # matched folds
  assert (summary["n_participants"] == len(train)).all()


def test_target_never_in_feature_sets():
  c = _cohort(100)
  assert "metabolic_syndrome_target" not in bf.build_demographic_features(c).columns
