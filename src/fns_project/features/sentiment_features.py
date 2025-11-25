"""Generate daily/rolling sentiment features from news sentiment scores."""
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def aggregate_daily_sentiment(news_df: pd.DataFrame, sentiment_col: str = "sentiment_ensemble") -> pd.DataFrame:
    """Aggregate sentiment per trading day."""
    df = news_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    agg = df.groupby("date").agg(
        sentiment_mean=(sentiment_col, "mean"),
        sentiment_median=(sentiment_col, "median"),
        sentiment_std=(sentiment_col, "std"),
        news_count=(sentiment_col, "size")
    )
    return agg


def rolling_sentiment(df: pd.DataFrame, col: str = "sentiment_mean", window: int = 3) -> pd.DataFrame:
    """Add rolling sentiment features."""
    df = df.copy()
    df[f"{col}_rolling_{window}"] = df[col].rolling(window=window).mean()
    return df
