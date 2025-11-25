# src/fns_project/analysis/correlation.py
"""Correlation analysis between daily sentiment and stock returns."""

from __future__ import annotations
from typing import Optional
import logging

import pandas as pd
import numpy as np

from fns_project.data.loader import load_price_csv
from fns_project.features.sentiment_features import aggregate_daily_sentiment_from_news
from fns_project.analysis.returns import compute_daily_returns
from fns_project.logging_config import get_logger

logger = get_logger(__name__)


def correlation_with_returns(
    daily_sentiment: pd.DataFrame,
    daily_returns: pd.DataFrame,
    max_lag: int = 5,
    sentiment_col: str = "sentiment_mean",
    returns_col: str = "daily_return",
) -> pd.DataFrame:
    """
    Compute correlation between sentiment and stock returns for lags 0..max_lag.
    Positive lag means sentiment leads returns.

    Parameters
    ----------
    daily_sentiment : pd.DataFrame
        Aggregated daily sentiment indexed by date.
    daily_returns : pd.DataFrame
        Daily returns indexed by date.
    max_lag : int
        Maximum lag to compute correlation.
    sentiment_col : str
        Column name in daily_sentiment to use.
    returns_col : str
        Column name in daily_returns to use.

    Returns
    -------
    pd.DataFrame
        Indexed by lag, with columns: correlation, n (number of overlapping days)
    """
    # Normalize index
    s = daily_sentiment.copy()
    r = daily_returns.copy()
    s.index = pd.to_datetime(s.index).normalize()
    if isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index).normalize()
    else:
        raise ValueError("daily_returns must be indexed by a DatetimeIndex")

    merged = s.join(r[[returns_col]], how="inner")
    results = []

    for lag in range(0, max_lag + 1):
        dfc = merged.copy()
        dfc["shifted_return"] = dfc[returns_col].shift(-lag)
        valid = dfc.dropna(subset=[sentiment_col, "shifted_return"])
        corr = valid[sentiment_col].corr(valid["shifted_return"]) if len(valid) > 1 else np.nan
        results.append({"lag": lag, "correlation": corr, "n": len(valid)})

    return pd.DataFrame(results).set_index("lag")

