"""Date/time helper functions."""

from datetime import datetime, timedelta
import pandas as pd

def parse_dates(series: pd.Series, fmt: str = None) -> pd.Series:
    """Convert a series to datetime, optionally with a format string."""
    return pd.to_datetime(series, format=fmt, errors='coerce')

def today() -> datetime:
    """Return today's date at midnight."""
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

def date_range(start: str, end: str, freq: str = 'D') -> pd.DatetimeIndex:
    """Generate a date range."""
    return pd.date_range(start=start, end=end, freq=freq)
