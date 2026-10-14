"""Time-based train/validation/test split on whole Pacific-time days."""
import pandas as pd

from src.features import LOCAL_TZ

VAL_START = "2015-02-10"    # first local day of validation
TEST_START = "2015-02-14"   # first local day of test


def local_date(created_at):
    """Calendar day in the stores' time zone (created_at is naive UTC)."""
    return (created_at.dt.tz_localize("UTC")
            .dt.tz_convert(LOCAL_TZ)
            .dt.tz_localize(None)
            .dt.normalize())


def time_split(df, val_start=VAL_START, test_start=TEST_START):
    day = local_date(df["created_at"])
    train = df[day < pd.Timestamp(val_start)]
    val = df[(day >= pd.Timestamp(val_start)) & (day < pd.Timestamp(test_start))]
    test = df[day >= pd.Timestamp(test_start)]
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))