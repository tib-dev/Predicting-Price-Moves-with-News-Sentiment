# fns_project/analysis/correlation.py
from __future__ import annotations
from typing import Optional
import logging

import pandas as pd
import numpy as np

# local imports to reuse your align/aggregate helpers
from fns_project.data.loader import load_price_csv
from fns_project.align_dates import align_news_to_trading_days, aggregate_headlines, merge_with_prices

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def compute_daily_returns(price_df: pd.DataFrame, close_col: str = "Close", date_col: str = "date") -> pd.DataFrame:
    """
    Produce a DataFrame indexed by normalized date with column 'daily_return' (pct change).
    """
    df = price_df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df[date_col] = df[date_col].dt.normalize()
        df = df.set_index(date_col)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = df.index.normalize()
    else:
        raise ValueError(
            "price_df must contain a date column or a DatetimeIndex")

    df = df.sort_index()
    # coerce close to float
    if close_col not in df.columns:
        raise ValueError(
            f"close_col '{close_col}' not found in price dataframe")
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df["daily_return"] = df[close_col].pct_change().fillna(0.0)
    return df[["daily_return"]]


def aggregate_daily_sentiment_from_news(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    sentiment_col: str = "sentiment_ensemble",
    news_date_col: str = "date",
    price_date_col: str = "date",
) -> pd.DataFrame:
    """
    Align news to trading days using your aligner, and produce aggregated sentiment by trading_date.
    Returns a DataFrame indexed by trading_date with columns: news_count, sentiment_mean, sentiment_median, sentiment_std
    """
    # Align news items to trading days
    news_aligned = align_news_to_trading_days(
        news_df, price_df, news_date_col=news_date_col, price_date_col=price_date_col)
    # Aggregate headlines (we keep trading_date grouping)
    agg_head = aggregate_headlines(news_aligned, group_by_cols=[
                                   "trading_date"], headline_col="headline")
    # Merge back sentiment column(s) by trading_date: we need to collect sentiment per article first
    # Ensure sentiment column present in news_aligned
    if sentiment_col not in news_aligned.columns:
        raise ValueError(
            f"Sentiment column '{sentiment_col}' missing from news dataframe. Run sentiment pipeline first.")

    # group by trading_date directly on news_aligned to aggregate numeric sentiment
    agg = news_aligned.groupby("trading_date").agg(
        news_count=(sentiment_col, "size"),
        sentiment_mean=(sentiment_col, "mean"),
        sentiment_median=(sentiment_col, "median"),
        sentiment_std=(sentiment_col, "std"),
    )
    agg = agg.fillna(
        {"sentiment_mean": 0.0, "sentiment_median": 0.0, "sentiment_std": 0.0})
    # join textual aggregates for convenience
    agg = agg.join(agg_head.set_index("trading_date")[
                   ["headline_count", "combined_headlines"]], how="left")
    # normalize index and sorts
    agg.index = pd.to_datetime(agg.index).normalize()
    agg = agg.sort_index()
    return agg


def correlation_with_returns(
    daily_sentiment: pd.DataFrame,
    daily_returns: pd.DataFrame,
    max_lag: int = 5,
    sentiment_col: str = "sentiment_mean",
    returns_col: str = "daily_return",
) -> pd.DataFrame:
    """
    Compute correlation for lags 0..max_lag. Positive lag = sentiment leads returns.
    Returns a DataFrame indexed by lag with columns: correlation, n
    """
    # align indices
    s = daily_sentiment.copy()
    r = daily_returns.copy()

    # ensure indices normalized date-only
    s.index = pd.to_datetime(s.index).normalize()
    if isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index).normalize()
    else:
        raise ValueError("daily_returns must be indexed by datetime index")

    merged = s.join(r[[returns_col]], how="inner")
    results = []
    for lag in range(0, max_lag + 1):
        dfc = merged.copy()
        dfc["shifted_return"] = dfc[returns_col].shift(-lag)
        valid = dfc.dropna(subset=[sentiment_col, "shifted_return"])
        corr = valid[sentiment_col].corr(
            valid["shifted_return"]) if len(valid) > 1 else np.nan
        results.append({"lag": lag, "correlation": corr, "n": len(valid)})

    return pd.DataFrame(results).set_index("lag")


# Example pipeline (quick-run)
if __name__ == "__main__":
    # The following is a minimal example. Replace file paths with your data.
    try:
        # load price file (ensure path exists in your environment)
        price_df = load_price_csv(
            "data/raw/price_sample.csv", date_col="Date", tz="Etc/GMT+4")
        # sample news
        news_df = pd.DataFrame({
            "date": ["2023-09-18 08:10:00", "2023-09-18 14:00:00", "2023-09-19 09:00:00"],
            "headline": ["Company X raises guidance", "Company Y under investigation", "Company X announces buyback"]
        })
        # Normally run sentiment pipeline before calling aggregation
        from fns_project.nlp.sentiment import add_sentiment_columns
        news_scored = add_sentiment_columns(news_df)

        daily_sent = aggregate_daily_sentiment_from_news(news_scored, price_df)
        daily_returns = compute_daily_returns(
            price_df, close_col="Close", date_col="date")
        corrs = correlation_with_returns(daily_sent, daily_returns, max_lag=3)
        print(corrs)
    except Exception as exc:
        logger.exception("Pipeline example failed: %s", exc)
