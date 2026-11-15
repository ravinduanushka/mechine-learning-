import pytest

from src.features import data_io as io
from src.features.minute_loader import (
    load_minute_profiles, minute_stream_to_profiles, validate_minute_stream)
from tests.fixtures import make_test_cohort, make_test_long_minutes


def test_find_files_and_read_table_all_formats(tmp_path):
  df = make_test_cohort(10)
  df.to_parquet(tmp_path / "clean_cohort_base.parquet")
  assert io.find_files(tmp_path, ["clean_cohort_base*"])[0].suffix == ".parquet"
  df.to_csv(tmp_path / "other.csv", index=False)
  df.to_feather(tmp_path / "other2.feather")
  for name in ("other.csv", "other2.feather", "clean_cohort_base.parquet"):
    assert io.read_table(tmp_path / name).shape == df.shape
  assert io.file_columns(tmp_path / "clean_cohort_base.parquet") == list(df.columns)


def test_load_cohort_indexes_by_int_seqn(tmp_path):
  make_test_cohort(20).to_parquet(tmp_path / "clean_cohort_base.parquet")
  c = io.load_cohort(tmp_path)
  assert c.index.name == "SEQN" and c.index.dtype == "int64" and len(c) == 20


@pytest.mark.parametrize("drop", ["metabolic_syndrome_target", "RIDAGEYR", "SEQN"])
def test_validate_cohort_requires_columns(drop):
  with pytest.raises(ValueError, match="missing required columns"):
    io.validate_cohort(make_test_cohort(20).drop(columns=drop))


def test_validate_cohort_rejects_duplicate_ids():
  c = make_test_cohort(20)
  c.loc[1, "SEQN"] = c.loc[0, "SEQN"]
  with pytest.raises(ValueError, match="duplicate SEQN"):
    io.validate_cohort(c)


def test_validate_cohort_target_checks():
  c = make_test_cohort(20)
  c.loc[0, io.TARGET] = 2
  with pytest.raises(ValueError, match="binary"):
    io.validate_cohort(c)
  c = make_test_cohort(20)
  c[io.TARGET] = c[io.TARGET].astype(float)
  c.loc[0, io.TARGET] = float("nan")
  with pytest.raises(ValueError, match="missing"):
    io.validate_cohort(c)


def test_validate_cohort_reports_missing_values():
  c = make_test_cohort(50)
  c.loc[:4, "RIDAGEYR"] = float("nan")
  assert io.validate_cohort(c)["missing_by_column"] == {"RIDAGEYR": 5}


def test_missing_minute_data_is_reported_not_faked(tmp_path):
  with pytest.raises(FileNotFoundError, match="PAXMTSM"):
    load_minute_profiles(tmp_path)
  make_test_cohort(10).to_parquet(tmp_path / "clean_cohort_base.parquet")
  st = io.check_input("minute", tmp_path, io.MINUTE_FILE_PATTERNS, io.MINUTE_REQUIRED)
  assert not st.available and st.files == [] and st.describe().startswith("MISSING")


def test_minute_schema_validation():
  df = make_test_long_minutes([1, 2]).drop(columns="PAXMTSM")
  with pytest.raises(ValueError, match="PAXMTSM"):
    validate_minute_stream(df)


def test_minute_profiles_drop_partial_days_and_shape():
  long_df = make_test_long_minutes([1, 2, 3], days=4)
  wide, days = minute_stream_to_profiles(long_df, min_days=3)
  assert list(days) == [4, 4, 4]               # the 500-row partial day is ignored
  assert wide["mims"].shape == (3, 1440) and wide.index.is_unique
  assert wide["wake_wear"].sum(axis=1).eq(1440).all()
  assert not wide["mims"].isna().any().any()


def test_minute_profiles_require_enough_days():
  with pytest.raises(ValueError, match="complete"):
    minute_stream_to_profiles(make_test_long_minutes([1], days=2), min_days=3)


@pytest.mark.skipif(io.locate_data_dir() is None, reason="real cohort file not present")
def test_real_cohort_schema_and_quality():
  c = io.load_cohort()                          # validates columns, duplicates, target
  assert c.index.is_unique and set(c[io.TARGET].unique()) == {0, 1}
  assert c[io.COHORT_REQUIRED[1:]].notna().all().all()
