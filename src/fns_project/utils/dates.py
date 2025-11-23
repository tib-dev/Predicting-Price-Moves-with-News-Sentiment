# date.py
"""Module for date parsing and extraction."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Optional

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def parse_dates_from_csv(
    path: Path | str,
    date_col: str = "date",
    parse_dates: bool = True,
    tz: Optional[str] = None
) -> pd.DataFrame:
    """
    Load a CSV file and parse date column into datetime.

    Parameters
    ----------
    path : Path | str
        Path to CSV file
    date_col : str
        Column name containing dates
    parse_dates : bool
        Whether to parse as datetime
    tz : Optional[str]
        Timezone, e.g., "UTC", "America/New_York"

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed date column
    """
    path = Path(path)
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"Column {date_col} not found in CSV.")

    if parse_dates:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if tz:
            df[date_col] = df[date_col].dt.tz_localize("UTC").dt.tz_convert(tz)
    logger.info(f"Parsed {len(df)} dates from {path}")
    return df


def extract_date_parts(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add columns for year, month, day, and hour from a datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing date column
    date_col : str
        Name of the datetime column

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns: year, month, day, hour
    """
    df = df.copy()
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["hour"] = df[date_col].dt.hour
    return df


if __name__ == "__main__":
    # Example usage
    import io

    sample_csv = io.StringIO("""date,headline
2025-11-23 08:15:00,Market opens higher
2025-11-23 12:45:00,Stocks dip slightly
2025-11-24 09:30:00,Tech rally continues
""")

    df = parse_dates_from_csv(sample_csv, date_col="date")
    df_parts = extract_date_parts(df)
    print(df_parts)
