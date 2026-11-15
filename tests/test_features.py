import numpy as np
import pandas as pd
import pytest

from src.features import build_features as bf
from src.features.data_io import assert_no_target_leakage
from src.features.select_features import (
    LeakageSafeSelector, apply_feature_selection, correlation_filter)
from tests.fixtures import make_test_cohort, make_test_minutes


@pytest.fixture(scope="module")
def cohort():
  return make_test_cohort(120).assign(SEQN=lambda d: d.SEQN.astype("int64")).set_index("SEQN")


@pytest.fixture(scope="module")
def minutes(cohort):
  return make_test_minutes(cohort.index)


def test_set0_columns_no_target_or_weights(cohort):
  s0 = bf.build_demographic_features(cohort)
  assert len(s0) == len(cohort) and s0.index.equals(cohort.index)
  assert {"age", "female", "cycle_2013_2014"} <= set(s0.columns)
  assert not {"metabolic_syndrome_target", "WTSAF2YR", "valid_days_count"} & set(s0.columns)
  assert s0.drop(columns="age").isin([0, 1]).all().all()
  assert_no_target_leakage(s0.columns)


def test_leakage_guard_blocks_target_and_clinical_labs():
  for bad in ("metabolic_syndrome_target", "BMXWAIST", "LBXTR", "BPXSY1"):
    with pytest.raises(ValueError):
      assert_no_target_leakage(["age", bad])


def test_basic_summaries_known_values():
  idx = pd.Index([1, 2], name="SEQN")
  cols = pd.MultiIndex.from_product([["mims"], range(1440)])
  m = pd.DataFrame(np.vstack([np.full(1440, 2.0), np.full(1440, 4.0)]), index=idx, columns=cols)
  wake = pd.DataFrame(np.ones((2, 1440)), index=idx,
                      columns=pd.MultiIndex.from_product([["wake_wear"], range(1440)]))
  out = bf.calculate_basic_summaries(m.join(wake))
  assert out["mean_daily_mims"].tolist() == [2.0, 4.0]
  assert out["wake_wear_min"].tolist() == [1440, 1440]


def test_sedentary_known_runs():
  # 100 sedentary, 10 active, 40 sedentary, rest active -> bouts>=30: 2, fraction 140/1440
  v = np.full(1440, 50.0)
  v[:100] = 0.0
  v[110:150] = 0.0
  m = pd.DataFrame([v], index=[1], columns=pd.MultiIndex.from_product([["mims"], range(1440)]))
  out = bf.engineer_sedentary_features(m)
  assert out["sedentary_bouts_30min"].iloc[0] == 2
  assert out["sedentary_fraction"].iloc[0] == pytest.approx(140 / 1440)


def test_activity_pattern_and_intensity_ranges(minutes):
  pat = bf.engineer_activity_patterns(minutes)
  assert pat["relative_amplitude"].between(0, 1).all()
  assert "day_to_day_cv_mims" in pat and (pat["day_to_day_cv_mims"] >= 0).all()
  inten = bf.engineer_intensity_features(minutes)
  assert inten["intensity_gradient"].notna().all()


def test_step_features_pending_not_invented(minutes):
  assert "peak_1min_cadence" not in bf.engineer_intensity_features(minutes).columns
  assert "mean_daily_steps" not in bf.calculate_basic_summaries(minutes).columns
  pending = bf.pending_step_features(minutes)
  assert any(k.startswith("peak_1min_cadence") for k in pending)
  with_steps = make_test_minutes(minutes.index, with_steps=True)
  assert "peak_1min_cadence" in bf.engineer_intensity_features(with_steps).columns
  assert not any(k.startswith("peak_1min") for k in bf.pending_step_features(with_steps))


def _mims_frame(rows):
  return pd.DataFrame(rows, index=range(len(rows)),
                      columns=pd.MultiIndex.from_product([["mims"], range(1440)]))


def test_extended_features_known_values():
  flat = np.full(1440, 5.0)                       # even, always sedentary
  v = np.zeros(1440)
  v[100:160] = 30.0                               # one 60-min high-intensity bout
  out = bf.engineer_extended_mims_features(_mims_frame([flat, v]))
  assert out.loc[0, "activity_entropy"] == pytest.approx(1.0, abs=1e-3)
  assert out.loc[1, "activity_entropy"] < 0.7     # concentrated in one hour
  assert out.loc[1, "min_high_intensity"] == 60
  assert out.loc[1, "longest_active_bout"] == 60
  assert out.loc[1, "longest_sedentary_bout"] == 1440 - 160
  assert out.loc[0, "longest_active_bout"] == 0 and out.loc[0, "longest_sedentary_bout"] == 1440
  assert out.loc[1, "mims_q1"] == pytest.approx(30.0 * 60 / 360)
  assert out.loc[1, ["mims_q2", "mims_q3", "mims_q4"]].eq(0).all()
  assert out.loc[0, "mims_median"] == 5.0 and out.loc[0, "mims_p90"] == 5.0


def test_extended_features_in_set_b_and_skipped_without_mims(cohort, minutes):
  static = bf.build_static_features(cohort, minutes)
  assert {"mims_q1", "activity_entropy", "longest_active_bout"} <= set(static["extended"].columns)
  unsup = bf.engineer_unsupervised_features(minutes, fit_index=minutes.index[:90])
  assert set(static["extended"].columns) <= set(bf.assemble_feature_sets(static, unsup)["setB"].columns)
  no_mims = minutes.drop(columns="mims", level=0)
  assert bf.engineer_extended_mims_features(no_mims).shape[1] == 0


def test_pca_and_kmeans_shapes(minutes):
  fit_ids = minutes.index[:90]
  u = bf.engineer_unsupervised_features(minutes, fit_index=fit_ids)
  assert len(u) == len(minutes)
  assert [c for c in u.columns if c.startswith("pca_")] == ["pca_score_1", "pca_score_2", "pca_score_3"]
  assert set(u["activity_phenotype_cluster"]) <= {0, 1, 2, 3}
  pca, table = bf.pca_variance_table(minutes, fit_ids, n_components=5)
  assert pca.components_.shape == (5, 1440) and table["cumulative"].iloc[-1] <= 1.0
  scan = bf.kmeans_k_scan(minutes, fit_ids, k_range=range(2, 5))
  assert list(scan["k"]) == [2, 3, 4]


def test_feature_set_nesting(cohort, minutes):
  static = bf.build_static_features(cohort, minutes)
  unsup = bf.engineer_unsupervised_features(minutes, fit_index=minutes.index[:90])
  sets = bf.assemble_feature_sets(static, unsup)
  assert set(sets["set0"].columns) < set(sets["setA"].columns) < set(sets["setB"].columns)
  assert sets["setB"].index.is_unique and len(sets["setB"]) == len(cohort)


def _toy(n=300, seed=1):
  rng = np.random.default_rng(seed)
  sig = rng.normal(size=n)
  y = pd.Series((sig + rng.normal(0, 0.5, n) > 0).astype(int))
  X = pd.DataFrame({"signal": sig, "signal_copy": sig * 1.001 + 1e-4,
                    "noise1": rng.normal(size=n), "noise2": rng.normal(size=n)})
  return X, y


def test_correlation_filter_drops_one_of_pair():
  X, _ = _toy()
  kept = correlation_filter(X, 0.9)
  assert ("signal" in kept) != ("signal_copy" in kept) and "noise1" in kept


def test_apply_feature_selection_stages_and_signal():
  X, y = _toy()
  sel, rep = apply_feature_selection(X, y, return_report=True, verbose=False)
  assert set(sel) <= set(rep["after_rfe"]) <= set(rep["after_mi"]) <= set(rep["after_correlation"])
  assert "signal" in sel or "signal_copy" in sel
  assert len(sel) == len(set(sel)) >= 1


def test_selector_works_with_missing_values():
  X, y = _toy()
  X.loc[::7, "noise1"] = np.nan
  assert apply_feature_selection(X, y, verbose=False)


def test_leakage_safe_selector_transform_columns():
  X, y = _toy()
  s = LeakageSafeSelector().fit(X, y)
  assert list(s.transform(X).columns) == s.selected_
