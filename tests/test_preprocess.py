import numpy as np
import pandas as pd
import pytest

from src.data import preprocess as pp
from tests.fixtures import make_test_cohort


@pytest.fixture()
def df():
  d = make_test_cohort(300)
  d.loc[d.index[:20], "RIDAGEYR"] = np.nan          # numeric NaNs
  d.loc[d.index[20:35], "smoking_status"] = np.nan  # categorical NaNs
  return d.set_index("SEQN")


def test_split_is_stratified_reproducible_and_disjoint(df):
  a = pp.split_data(df)
  b = pp.split_data(df)
  X_tr, X_te, y_tr, y_te = a
  assert set(X_tr.index).isdisjoint(X_te.index) and len(X_tr) + len(X_te) == len(df)
  assert abs(y_tr.mean() - y_te.mean()) < 0.03
  assert X_tr.index.equals(b[0].index)                 # same seed -> same split
  assert pp.DEFAULT_TARGET not in X_tr.columns


def test_no_missing_values_after_preprocessing(df):
  res = pp.preprocess(df)
  assert res.X_train.isna().sum().sum() == 0 and res.X_test.isna().sum().sum() == 0
  assert res.X_train.index.equals(res.y_train.index)   # SEQN index preserved


def test_scaler_fitted_on_train_only(df):
  res = pp.preprocess(df)
  assert res.X_train["RIDAGEYR"].mean() == pytest.approx(0, abs=1e-9)
  assert res.X_train["RIDAGEYR"].std(ddof=0) == pytest.approx(1, abs=1e-9)
  assert abs(res.X_test["RIDAGEYR"].mean()) > 1e-6     # test is NOT re-centred on itself


def test_imputer_uses_train_median_not_test(df):
  X_tr, X_te, y_tr, y_te = pp.split_data(df)
  train_median = X_tr["RIDAGEYR"].median()
  res = pp.preprocess(df)
  imp = res.pipeline.named_transformers_["num"].named_steps["impute"]
  i = res.numeric_cols.index("RIDAGEYR")
  assert imp.statistics_[i] == pytest.approx(train_median)


def test_id_weight_and_target_never_features(df):
  res = pp.preprocess(df)
  assert set(res.dropped_cols) == {"WTSAF2YR"}         # SEQN is the index here
  assert not any(c.startswith(("WTSAF2YR", "metabolic")) for c in res.feature_names)


def test_one_hot_encoding_and_unseen_category(df):
  res = pp.preprocess(df)
  assert any(c.startswith("smoking_status") for c in res.feature_names)
  assert res.X_train.select_dtypes(exclude="number").empty
  new = df.drop(columns=[pp.DEFAULT_TARGET, "WTSAF2YR"]).head(5).copy()
  new["smoking_status"] = "never_seen_before"
  out = res.pipeline.transform(new)                    # must not raise
  smoke = [c for c in out.columns if c.startswith("smoking_status")]
  assert (out[smoke].sum(axis=1) == 0).all()


def test_knn_and_minmax_options(df):
  res = pp.preprocess(df, imputer="knn", scaler="minmax")
  assert res.X_train.isna().sum().sum() == 0
  assert res.X_train["RIDAGEYR"].between(0, 1).all()


def test_invalid_options_and_inputs(df):
  with pytest.raises(ValueError):
    pp.build_preprocessor(["a"], [], imputer="mean")
  with pytest.raises(ValueError):
    pp.build_preprocessor(["a"], [], scaler="robust")
  with pytest.raises(KeyError):
    pp.split_data(df, target="nope")
  with pytest.raises(ValueError):
    pp.split_data(df, test_size=1.5)
  with pytest.raises(ValueError):
    pp.split_data(df.assign(metabolic_syndrome_target=0))
  with pytest.raises(FileNotFoundError):
    pp.load_data("does_not_exist.parquet")


def test_missing_target_rows_dropped_with_warning(df):
  df.loc[df.index[:5], pp.DEFAULT_TARGET] = np.nan
  with pytest.warns(UserWarning, match="missing target"):
    res = pp.preprocess(df)
  assert len(res.X_train) + len(res.X_test) == len(df) - 5


def test_pipeline_roundtrip_and_saved_splits(df, tmp_path):
  res = pp.preprocess(df)
  path = pp.save_pipeline(res.pipeline, tmp_path / "pre.joblib")
  again = pp.load_pipeline(path)
  raw_test = df.loc[res.X_test.index].drop(columns=[pp.DEFAULT_TARGET, "WTSAF2YR"])
  pd.testing.assert_frame_equal(again.transform(raw_test), res.X_test)
  assert len(pp.save_splits(res, tmp_path)) == 4


def test_manual_column_override(df):
  res = pp.preprocess(df, numeric_cols=["RIDAGEYR"], categorical_cols=["RIAGENDR"])
  assert "RIDAGEYR" in res.numeric_cols and "RIAGENDR" in res.categorical_cols
