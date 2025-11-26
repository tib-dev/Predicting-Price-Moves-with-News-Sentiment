import pandas as pd
from src.fns_project.utils import datetime_utils as du


def test_normalize_and_naive():
    df = pd.DataFrame({"date": ["2023-09-18 08:00:00", "2023-09-19 09:00:00"]})
    df2 = du.normalize_dates(df, "date")
    assert df2["date"].dt.normalize().equals(df2["date"])
