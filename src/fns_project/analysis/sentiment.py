"""
Module for scoring news sentiment using VADER and TextBlob on already-cleaned headlines.
Also normalizes timestamps and adds an 'hour' column.
"""

from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


# -----------------------------
# Lazy imports
# -----------------------------
def _get_vader():
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def _get_textblob():
    from textblob import TextBlob
    return TextBlob


# -----------------------------
# Sentiment scoring helpers
# -----------------------------
def _score_vader_series(series: pd.Series) -> pd.DataFrame:
    sia = _get_vader()
    rows = [sia.polarity_scores(str(txt)) for txt in series.fillna("")]
    df = pd.DataFrame(rows, index=series.index)
    df.columns = [f"vader_{c}" for c in df.columns]
    return df


def _score_textblob_series(series: pd.Series) -> pd.DataFrame:
    TextBlob = _get_textblob()
    polarity, subjectivity = [], []
    for txt in series.fillna("").astype(str):
        tb = TextBlob(txt)
        polarity.append(tb.sentiment.polarity)
        subjectivity.append(tb.sentiment.subjectivity)
    return pd.DataFrame(
        {"textblob_polarity": polarity, "textblob_subjectivity": subjectivity},
        index=series.index
    )


def ensemble_score(df: pd.DataFrame, weight_vader: float = 0.7, weight_tb: float = 0.3) -> pd.Series:
    """Weighted ensemble of VADER and TextBlob polarity scores."""
    # Ensure we have the correct Series
    vader = df.get("vader_compound", pd.Series(0.0, index=df.index))
    tb = df.get("textblob_polarity", pd.Series(0.0, index=df.index))

    # Convert to float Series explicitly
    vader = pd.Series(vader, index=df.index, dtype=float)
    tb = pd.Series(tb, index=df.index, dtype=float)

    score = weight_vader * vader + weight_tb * tb
    # Ensure result is a Series with name
    return pd.Series(score, index=df.index, name="sentiment_ensemble").clip(-1.0, 1.0)


# -----------------------------
# Main function
# -----------------------------
def add_sentiment_columns(
    df: pd.DataFrame,
    headline_col: str = "headline",
    date_col: str = "date",
    original_tz: str = None,
    target_tz: str = "UTC"
) -> pd.DataFrame:
    """
    Compute VADER/TextBlob sentiment scores on cleaned headlines and normalize timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        Input news DataFrame with cleaned headlines.
    headline_col : str
        Column containing headlines.
    date_col : str
        Column with datetime for timestamp normalization.
    original_tz : str
        Original timezone of the date column.
    target_tz : str
        Target timezone (default: UTC).

    Returns
    -------
    pd.DataFrame
        DataFrame with VADER, TextBlob, ensemble sentiment scores, normalized date, and hour.
    """
    out = df.copy()

    if headline_col not in out.columns:
        raise ValueError(f"DataFrame missing headline column '{headline_col}'")

    # Sentiment scoring
    texts = out[headline_col].astype(str)
    vader_df = _score_vader_series(texts)
    tb_df = _score_textblob_series(texts)
    out = pd.concat([out, vader_df, tb_df], axis=1)
    out["sentiment_ensemble"] = ensemble_score(out)

    # Timestamp normalization
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        if original_tz:
            out[date_col] = out[date_col].dt.tz_localize(original_tz, ambiguous='NaT', nonexistent='NaT')
        if target_tz:
            out[date_col] = out[date_col].dt.tz_convert(target_tz)
        out['hour'] = out[date_col].dt.hour
        # Optional: keep date only
        out[date_col] = out[date_col].dt.date

    return out


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    sample = pd.DataFrame({
        "headline": [
            "Apple reports blowout earnings, stock jumps",
            "Regulator sues small fintech startup over compliance failures",
            "Mixed signals from Fed minutes; markets uncertain"
        ],
        "date": ["2025-11-26 14:30:00", "2025-11-26 09:15:00", "2025-11-25 16:45:00"]
    })
    scored = add_sentiment_columns(sample, date_col="date", original_tz="Etc/GMT+3")
    print(scored[[
        "headline", "vader_compound", "textblob_polarity", "sentiment_ensemble", "date", "hour"
    ]])
