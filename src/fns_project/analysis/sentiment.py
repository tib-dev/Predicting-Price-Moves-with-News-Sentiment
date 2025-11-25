"""Module for scoring news sentiment using VADER and TextBlob."""

from __future__ import annotations
from typing import Optional
import logging

import pandas as pd

from fns_project.preprocess import preprocess_headlines, add_headline_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


# -----------------------------
# Lazy imports for heavy libraries
# -----------------------------
def _get_vader():
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except Exception as e:
        raise ImportError(
            "VADER not available. Install nltk and download 'vader_lexicon'."
        ) from e
    return SentimentIntensityAnalyzer()


def _get_textblob():
    try:
        from textblob import TextBlob
    except Exception as e:
        raise ImportError(
            "TextBlob not available. Install: pip install textblob") from e
    return TextBlob


def _score_vader_series(series: pd.Series) -> pd.DataFrame:
    sia = _get_vader()
    rows = []
    for txt in series.fillna("").astype(str):
        rows.append(sia.polarity_scores(txt))
    df = pd.DataFrame(rows, index=series.index)
    df.columns = [f"vader_{c}" for c in df.columns]
    return df


def _score_textblob_series(series: pd.Series) -> pd.DataFrame:
    TextBlob = _get_textblob()
    polarity = []
    subjectivity = []
    for txt in series.fillna("").astype(str):
        tb = TextBlob(txt)
        polarity.append(tb.sentiment.polarity)
        subjectivity.append(tb.sentiment.subjectivity)
    return pd.DataFrame({"textblob_polarity": polarity, "textblob_subjectivity": subjectivity}, index=series.index)


def ensemble_score(df: pd.DataFrame, weight_vader: float = 0.7, weight_tb: float = 0.3) -> pd.Series:
    vader = df.get("vader_compound", pd.Series(
        0.0, index=df.index)).astype(float)
    tb = df.get("textblob_polarity", pd.Series(
        0.0, index=df.index)).astype(float)
    score = weight_vader * vader + weight_tb * tb
    return score.clip(-1.0, 1.0).rename("sentiment_ensemble")


def add_sentiment_columns(
    df: pd.DataFrame,
    headline_col: str = "headline",
    preprocess: bool = True,
    preprocess_args: Optional[dict] = None,
) -> pd.DataFrame:
    preprocess_args = preprocess_args or {}
    out = df.copy()

    if headline_col not in out.columns:
        raise ValueError(f"DataFrame missing headline column '{headline_col}'")

    # optional preprocessing
    if preprocess:
        out["_headline_original"] = out[headline_col].astype(str)
        cleaned = preprocess_headlines(
            out,
            text_col=headline_col,
            remove_stopwords=preprocess_args.get("remove_stopwords", True),
            lemmatize=preprocess_args.get("lemmatize", False)
        )
        cleaned_map = cleaned[headline_col].to_dict()
        out["_headline_for_sentiment"] = out.index.map(
            lambda i: cleaned_map.get(i, out.at[i, headline_col]))
    else:
        out["_headline_for_sentiment"] = out[headline_col].astype(str)

    texts = out["_headline_for_sentiment"]

    vader_df = _score_vader_series(texts)
    tb_df = _score_textblob_series(texts)

    out = pd.concat([out, vader_df, tb_df], axis=1)
    out["sentiment_ensemble"] = ensemble_score(out)

    # optional headline metrics
    try:
        out = add_headline_metrics(out, text_col=headline_col)
    except Exception:
        logger.debug("Could not add headline metrics (skipping)")

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
        ]
    })
    scored = add_sentiment_columns(sample)
    print(scored[[
        "headline", "vader_compound", "textblob_polarity", "sentiment_ensemble"
    ]])
