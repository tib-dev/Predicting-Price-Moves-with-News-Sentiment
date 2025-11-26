# src/fns_project/analysis/returns.py
import pandas as pd
import numpy as np

def compute_daily_returns(
    price_df: pd.DataFrame,
    close_col: str = "Close",
    date_col: str = "date",
    include_log: bool = True,
    fill_na: bool = False
) -> pd.DataFrame:
    """
    Compute daily and log returns from price data, keeping all original columns.

    Parameters
    ----------
    price_df : pd.DataFrame
        DataFrame containing OHLCV data.
    close_col : str
        Column name for closing price.
    date_col : str
        Column name for date.
    include_log : bool
        Whether to compute log returns.
    fill_na : bool
        Whether to fill first-row NaNs with 0.0.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional 'daily_return' and optionally 'log_return'.
    """
    df = price_df.copy()

    # Ensure datetime index
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df[date_col] = df[date_col].dt.normalize()
        df = df.set_index(date_col)
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.normalize()
    else:
        raise ValueError("price_df must have a datetime index or a date column")

    df = df.sort_index()

    # Ensure numeric close prices
    if close_col not in df.columns:
        raise ValueError(f"Column '{close_col}' not found in price_df")
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")

    # Compute returns
    df["daily_return"] = df[close_col].pct_change()
    if include_log:
        df["log_return"] = np.log(df[close_col]).diff()

    if fill_na:
        df[["daily_return", "log_return"]] = df[["daily_return", "log_return"]].fillna(0.0)

    return df
