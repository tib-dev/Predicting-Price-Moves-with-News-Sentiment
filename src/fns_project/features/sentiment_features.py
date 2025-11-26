"""Compute daily sentiment features for modeling."""

from __future__ import annotations
from typing import Optional, List
import logging

import pandas as pd

from fns_project.analysis.sentiment import add_sentiment_columns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    date_col: str = "date",
    sentiment_col: str = "sentiment_ensemble"
) -> pd.DataFrame:
    """
    Aggregate sentiment by date.
    Returns a DataFrame indexed by date with columns:
    - news_count
    - sentiment_mean
    - sentiment_median
    - sentiment_std
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}'")
    if sentiment_col not in df.columns:
        raise ValueError(f"Missing sentiment column '{sentiment_col}'")

    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    agg = df.groupby(date_col).agg(
        news_count=(sentiment_col, "count"),
        sentiment_mean=(sentiment_col, "mean"),
        sentiment_median=(sentiment_col, "median"),
        sentiment_std=(sentiment_col, "std")
    ).fillna(0.0)
    return agg


def add_rolling_sentiment(
    daily_sentiment: pd.DataFrame,
    window_sizes: Optional[List[int]] = None,
    sentiment_col: str = "sentiment_mean"
) -> pd.DataFrame:
    """
    Add rolling mean and std for specified window sizes.
    Returns a new DataFrame copy.
    """
    window_sizes = window_sizes or [3, 5, 7]
    df = daily_sentiment.copy()
    for w in window_sizes:
        df[f"{sentiment_col}_roll_mean_{w}"] = df[sentiment_col].rolling(
            w, min_periods=1).mean()
        df[f"{sentiment_col}_roll_std_{w}"] = df[sentiment_col].rolling(
            w, min_periods=1).std().fillna(0.0)
    return df


def add_lagged_sentiment(
    daily_sentiment: pd.DataFrame,
    lags: Optional[List[int]] = None,
    sentiment_col: str = "sentiment_mean"
) -> pd.DataFrame:
    """
    Add lagged features for sentiment.
    Positive lag = previous day's sentiment
    """
    lags = lags or [1, 2, 3]
    df = daily_sentiment.copy()
    for lag in lags:
        df[f"{sentiment_col}_lag_{lag}"] = df[sentiment_col].shift(
            lag).fillna(0.0)
    return df


def create_sentiment_features(
    news_df: pd.DataFrame,
    date_col: str = "date",
    headline_col: str = "headline",
    preprocess_args: Optional[dict] = None,
    rolling_windows: Optional[List[int]] = None,
    lag_days: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Full pipeline:
    1. Compute sentiment columns
    2. Aggregate by day
    3. Add rolling features
    4. Add lagged features
    """
    df_scored = add_sentiment_columns(
        news_df, headline_col=headline_col, preprocess=True, preprocess_args=preprocess_args)
    daily_sent = aggregate_daily_sentiment(
        df_scored, date_col=date_col, sentiment_col="sentiment_ensemble")
    daily_sent = add_rolling_sentiment(
        daily_sent, window_sizes=rolling_windows)
    daily_sent = add_lagged_sentiment(daily_sent, lags=lag_days)
    return daily_sent


def aggregate_daily_sentiment_from_news(
    news_df: pd.DataFrame,
    date_col: str = "date",
    headline_col: str = "headline",
    preprocess_args: Optional[dict] = None,
    rolling_windows: Optional[List[int]] = None,
    lag_days: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Wrapper for create_sentiment_features to produce daily sentiment features from raw news.

    Returns a DataFrame indexed by date with:
    - news_count
    - sentiment_mean / median / std
    - rolling and lagged sentiment features
    """
    return create_sentiment_features(
        news_df=news_df,
        date_col=date_col,
        headline_col=headline_col,
        preprocess_args=preprocess_args,
        rolling_windows=rolling_windows,
        lag_days=lag_days
    )
