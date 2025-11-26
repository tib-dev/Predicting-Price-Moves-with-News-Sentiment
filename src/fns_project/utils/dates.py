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
    """Return copy where date_col is tz-naive.

    If tz is supplied, convert to tz first then drop tz info; otherwise drop tz info directly.
    """
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
    # use dt.normalize which preserves tz-aware dtype in newer pandas; safe fallback below if needed
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    try:
        out[date_col] = out[date_col].dt.normalize()
    except Exception:
        # fallback: strip time components manually
        out[date_col] = pd.to_datetime(out[date_col].dt.date)
    return out


def align_news_to_prices(news_df: pd.DataFrame, price_df: pd.DataFrame, news_date_col: str = "date", price_date_col: str = "date") -> pd.DataFrame:
    """Return subset of news_df where its normalized dates appear in price_df's normalized dates.

    This is a convenience wrapper used in quick checks; prefer the robust `align.align_news_to_trading_days`
    for production mapping rules.
    """
    n = news_df.copy()
    p = price_df.copy()
    n = normalize_dates(n, news_date_col)
    p = normalize_dates(p, price_date_col)
    price_dates = pd.Series(pd.to_datetime(
        p[price_date_col]).dt.normalize().unique())
    return n[n[news_date_col].isin(price_dates)].reset_index(drop=True)


def normalize_timestamps(
    df, date_col="date", original_tz="Etc/GMT+4", target_tz="UTC"
) -> pd.DataFrame:
    """
    Convert a datetime column to a timezone-aware column in the desired target timezone.

    Handles both naive and already tz-aware timestamps.

    Args:
        df: Input DataFrame
        date_col: Name of the datetime column
        original_tz: Original timezone of timestamps (used only for naive timestamps)
        target_tz: Desired timezone to convert to (UTC by default)

    Returns:
        DataFrame with normalized timestamp column
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])

    # Check if timestamps are naive or tz-aware
    if out[date_col].dt.tz is None:
        # naive -> localize then convert
        out[date_col] = out[date_col].dt.tz_localize(
            original_tz).dt.tz_convert(target_tz)
    else:
        # already tz-aware -> just convert
        out[date_col] = out[date_col].dt.tz_convert(target_tz)

    logger.info("Normalized '%s' timestamps to %s (rows=%d)",
                date_col, target_tz, len(out))
    return out
