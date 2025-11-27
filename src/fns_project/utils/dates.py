"""Helpers to normalize and convert date/time columns for consistent downstream merging."""

from __future__ import annotations
from typing import Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def ensure_tz_aware(df: pd.DataFrame, date_col: str, tz: str = "UTC") -> pd.DataFrame:
    """Return copy where date_col is timezone-aware (localized to tz if naive)."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].dt.tz is None:
        out[date_col] = out[date_col].dt.tz_localize(tz)
    else:
        out[date_col] = out[date_col].dt.tz_convert(tz)
    return out


def ensure_tz_naive(df: pd.DataFrame, date_col: str, tz: Optional[str] = None) -> pd.DataFrame:
    """Return copy where date_col is tz-naive."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].dt.tz is not None:
        if tz is not None:
            out[date_col] = out[date_col].dt.tz_convert(
                tz).dt.tz_localize(None)
        else:
            out[date_col] = out[date_col].dt.tz_localize(None)
    return out


def normalize_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Normalize timestamps to midnight of their day (preserving timezone info if present)."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    try:
        out[date_col] = out[date_col].dt.normalize()
    except Exception:
        out[date_col] = pd.to_datetime(out[date_col].dt.date)
    return out


def datetime_to_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Convert datetime column to date-only (midnight) while preserving tz info."""
    out = df.copy()
    out[date_col] = pd.to_datetime(
        out[date_col], errors="coerce").dt.normalize()
    return out


def align_news_to_prices(news_df: pd.DataFrame, price_df: pd.DataFrame,
                         news_date_col: str = "date", price_date_col: str = "date") -> pd.DataFrame:
    """Return subset of news_df where its normalized dates appear in price_df's normalized dates."""
    n = normalize_dates(news_df, news_date_col)
    p = normalize_dates(price_df, price_date_col)
    price_dates = pd.Series(pd.to_datetime(
        p[price_date_col]).dt.normalize().unique())
    return n[n[news_date_col].isin(price_dates)].reset_index(drop=True)

import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def datetime_to_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Convert datetime column to date-only (midnight) while preserving tz info."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    return out


def normalize_timestamps(
    df: pd.DataFrame, date_col="date", original_tz="Etc/GMT+4",
    target_tz="UTC", normalize_to_date: bool = False, from_index=False
) -> pd.DataFrame:
    """
    Convert a datetime column or index to a timezone-aware column in the desired target timezone.
    Handles mixed tz-naive and tz-aware values.
    Optionally normalize to date-only (midnight).

    Args:
        df: Input DataFrame
        date_col: Name of the datetime column (ignored if from_index=True)
        original_tz: Original timezone of naive timestamps
        target_tz: Desired timezone
        normalize_to_date: If True, strip time component
        from_index: If True, normalize the DataFrame's index instead of a column

    Returns:
        DataFrame with normalized timestamp column or index
    """
    out = df.copy()

    if from_index:
        if not isinstance(out.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex")
        ts = pd.to_datetime(out.index, errors="coerce")
    else:
        if date_col not in out.columns:
            raise ValueError(f"Column '{date_col}' not found in DataFrame.")
        ts = pd.to_datetime(out[date_col], errors="coerce")

    n_dropped = ts.isna().sum()
    if n_dropped > 0:
        logger.warning("Dropped %d rows due to invalid timestamps", n_dropped)
    ts = ts.dropna()

    # Convert naive timestamps to original_tz first
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(original_tz)
    ts = ts.dt.tz_convert(target_tz)

    if normalize_to_date:
        ts = ts.dt.normalize()

    if from_index:
        out = out.loc[ts.index]  # keep only valid rows
        out.index = ts
    else:
        out = out.loc[ts.index]  # keep only valid rows
        out[date_col] = ts

    logger.info("Normalized timestamps to %s (rows=%d)", target_tz, len(out))
    return out
