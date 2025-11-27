# src/fns_project/analysis/correlation.py
"""Correlation analysis between daily sentiment and stock returns."""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import logging


from fns_project.analysis.returns import compute_daily_returns

logger = logging.getLogger("notebook")  # use global logger


def correlation_with_returns(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame,
    sentiment_col: str = "sentiment_mean",
    returns_col: str = "daily_return",
    max_lag: int = 5,
    return_pvalues: bool = False
) -> pd.DataFrame:
    """
    Compute correlation between aggregated daily sentiment (already prepared)
    and stock price returns across multiple lag days.

    Parameters
    ----------
    sentiment_df : pd.DataFrame
        Must already contain daily sentiment features.
        Index should be daily dates.
    price_df : pd.DataFrame
        Must contain a 'daily_return' column.
    sentiment_col : str
        Which sentiment column to correlate with returns.
    returns_col : str
        Column in price_df containing daily returns.
    max_lag : int
        Maximum lag of returns you want to test.
    return_pvalues : bool
        Whether to include p-values.

    Returns
    -------
    pd.DataFrame
        Indexed by lag, with correlation, sample size, and optional p-values.
    """

    # Ensure sentiment_df has datetime index
    s = sentiment_df[[sentiment_col]].copy()
    s.index = pd.to_datetime(s.index).normalize()

    # Use existing daily returns
    r = price_df[[returns_col]].copy()
    r.index = pd.to_datetime(r.index).normalize()
    r = r.rename(columns={returns_col: "daily_return"})

    # Merge on matching dates
    merged = s.join(r, how="inner")

    if merged.empty:
        logger.warning("No overlapping dates for correlation computation.")
        cols = ["correlation", "n"]
        if return_pvalues:
            cols.append("p_value")
        return pd.DataFrame(columns=cols)

    # Compute correlation for each lag
    rows = []
    for lag in range(max_lag + 1):
        shifted = merged["daily_return"].shift(-lag)
        temp = merged[[sentiment_col]].copy()
        temp["shifted_return"] = shifted
        temp = temp.dropna()

        if len(temp) > 1:
            corr, pval = pearsonr(temp[sentiment_col], temp["shifted_return"])
        else:
            corr, pval = np.nan, np.nan

        row = {
            "lag": lag,
            "correlation": corr,
            "n": len(temp),
        }
        if return_pvalues:
            row["p_value"] = pval

        rows.append(row)

    result = pd.DataFrame(rows).set_index("lag")
    logger.info("Correlation analysis complete (0 to %d lags).", max_lag)
    return result
