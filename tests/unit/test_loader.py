import pandas as pd
import io
from src.fns_project.data import loader


def test_load_news_csv_stringio():
    s = io.StringIO(
        "date,headline,stock\n2023-09-18 08:00:00,Hello AAPL,AAPL\n")
    df = loader.load_news(s, date_col="date",
                          source_tz="Etc/GMT+4", target_tz="UTC")
    assert not df.empty
    assert "headline" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_prices_stringio():
    s = io.StringIO(
        "Date,Open,High,Low,Close,Volume\n2023-09-18,100,101,99,100,1000\n")
    df = loader.load_prices(
        s, date_col="Date", source_tz="UTC", target_tz="UTC")
    assert "date" in df.columns
    assert "Close" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
