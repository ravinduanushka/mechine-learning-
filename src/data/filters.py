import pandas as pd
import numpy as np


def filter_adult_cohort(demo_df: pd.DataFrame, min_age: int = 20) -> pd.DataFrame:
    """Filter demographic cohort to include only participants aged >= min_age."""
    return demo_df[demo_df["RIDAGEYR"] >= min_age].copy()


def calculate_daily_wake_wear_minutes(
    wear_pred_series: pd.Series, 
    quality_flag_series: pd.Series
) -> int:
    """
    Compute total valid awake-wear minutes for a single day:
    W_ij = sum(1[wear_prediction == 1] * 1[quality_indicator == 0])
    """
    valid_mask = (wear_pred_series == 1) & (quality_flag_series == 0)
    return int(np.sum(valid_mask))


def filter_valid_accelerometer_participants(
    daily_wear_summary: pd.DataFrame,
    min_minutes_per_day: int = 600,
    min_valid_days: int = 3
) -> pd.DataFrame:
    """
    Filter participants meeting minimum protocol compliance:
    Retain participants with at least 3 days having >= 600 valid awake minutes.
    """
    valid_day_mask = daily_wear_summary["wake_wear_minutes"] >= min_minutes_per_day
    compliant_days = daily_wear_summary[valid_day_mask]

    day_counts = (
        compliant_days.groupby("SEQN")["day"]
        .count()
        .reset_index(name="valid_day_count")
    )

    qualified_seqns = day_counts[day_counts["valid_day_count"] >= min_valid_days]
    return qualified_seqns