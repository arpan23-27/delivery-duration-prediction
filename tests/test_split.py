import pandas as pd

from src.split import time_split


def make_df(stamps):
    return pd.DataFrame({"created_at": pd.to_datetime(stamps), "x": range(len(stamps))})


def test_split_boundaries_use_pacific_days():
    df = make_df(["2015-02-10 07:59:00",   # Feb 9 23:59 Pacific  -> train
                  "2015-02-10 08:00:00",   # Feb 10 00:00 Pacific -> val
                  "2015-02-14 07:59:00",   # Feb 13 23:59 Pacific -> val
                  "2015-02-14 08:00:00"])  # Feb 14 00:00 Pacific -> test
    train, val, test = time_split(df)
    assert list(train["x"]) == [0]
    assert list(val["x"]) == [1, 2]
    assert list(test["x"]) == [3]


def test_split_partitions_and_orders_rows():
    df = make_df(pd.date_range("2015-01-21", "2015-02-18", freq="3h"))
    train, val, test = time_split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert train["created_at"].max() < val["created_at"].min()
    assert val["created_at"].max() < test["created_at"].min()