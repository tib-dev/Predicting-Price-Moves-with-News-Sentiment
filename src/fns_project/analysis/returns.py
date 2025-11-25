# src/fns_project/analysis/returns.py
"""Compute stock returns."""

from typing import Optional
import pandas as pd
import numpy as np
from fns_project.logging_config import get_logger

logger = get_logger(__name__)


def compute_daily_returns(
    price_df: pd.DataFrame, close_col: str = "Close", date_col: str = "date"
) -> pd.DataFrame:
    """
    Compute daily returns (percentage change) from price data.

    Parameters
    ----------
    price_df : pd.DataFrame
        DataFrame with price data.
    close_col : str
        Column name for closing price.
    date_col : str
        Column name for date.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with 'daily_return' column.
    """
    df = price_df.copy()

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df[date_col] = df[date_col].dt.normalize()
        df = df.set_index(date_col)
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.normalize()
    else:
        raise ValueError("price_df must have a datetime index or date column")

    df = df.sort_index()
    if close_col not in df.columns:
        raise ValueError(f"Column '{close_col}' not found in price_df")
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")

    df["daily_return"] = df[close_col].pct_change().fillna(0.0)
    return df[["daily_return"]]
