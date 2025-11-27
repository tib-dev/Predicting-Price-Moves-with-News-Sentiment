import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from fns_project.analysis.sentiment import _score_vader_series, _score_textblob_series
from fns_project.data.align_dates import align_news_to_trading_days
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# 1️⃣ Score sentiment for cleaned headlines
# -----------------------------


def add_sentiment_columns_cleaned(news_df: pd.DataFrame, headline_col: str = "headline") -> pd.DataFrame:
    """
    Compute sentiment columns for already cleaned headlines.
    Skips preprocessing since data is pre-cleaned.
    """
    out = news_df.copy()
    texts = out[headline_col].astype(str)

    vader_df = _score_vader_series(texts)
    tb_df = _score_textblob_series(texts)

    out = pd.concat([out, vader_df, tb_df], axis=1)
    # ensemble score: weighted combination
    vader = out["vader_compound"].astype(float)
    tb = out["textblob_polarity"].astype(float)
    out["sentiment_ensemble"] = (0.7 * vader + 0.3 * tb).clip(-1, 1)

    return out

# -----------------------------
# 2️⃣ Aggregate daily sentiment
# -----------------------------


def aggregate_daily_sentiment(news_df: pd.DataFrame, price_df: pd.DataFrame, headline_col: str = "headline") -> pd.DataFrame:
    # Align news to trading dates
    news_aligned = align_news_to_trading_days(
        news_df, price_df, news_date_col="date", price_date_col="date")

    # Group by trading day
    grouped = news_aligned.groupby("trading_date")["sentiment_ensemble"].agg(
        sentiment_mean="mean",
        sentiment_std="std",
        count="count"
    ).reset_index()

    return grouped.set_index("trading_date")

# -----------------------------
# 3️⃣ Compute correlation with returns
# -----------------------------


def correlation_with_returns(sentiment_df: pd.DataFrame, price_df: pd.DataFrame, sentiment_col: str = "sentiment_mean",
                             returns_col: str = "daily_return", max_lag: int = 5, return_pvalues: bool = True) -> pd.DataFrame:
    """
    Compute correlation between daily sentiment and stock returns for multiple lags.
    """
    # Ensure datetime index
    s = sentiment_df[[sentiment_col]].copy()
    s.index = pd.to_datetime(s.index).normalize()

    r = price_df[[returns_col]].copy()
    r.index = pd.to_datetime(price_df["date"]).normalize()

    # Merge on dates
    merged = s.join(r, how="inner")
    if merged.empty:
        logger.warning("No overlapping dates for correlation computation.")
        cols = ["correlation", "n"]
        if return_pvalues:
            cols.append("p_value")
        return pd.DataFrame(columns=cols)

    # Compute correlations
    rows = []
    for lag in range(max_lag + 1):
        shifted = merged[returns_col].shift(-lag)
        temp = merged[[sentiment_col]].copy()
        temp["shifted_return"] = shifted
        temp = temp.dropna()

        if len(temp) > 1:
            corr, pval = pearsonr(temp[sentiment_col], temp["shifted_return"])
        else:
            corr, pval = np.nan, np.nan

        row = {"lag": lag, "correlation": corr, "n": len(temp)}
        if return_pvalues:
            row["p_value"] = pval
        rows.append(row)

    return pd.DataFrame(rows).set_index("lag")


