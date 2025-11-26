"""Module description."""

# fns_project/nlp/sentiment.py
from __future__ import annotations
from typing import Optional, Iterable
import logging

import pandas as pd

# local imports from your modules
from fns_project.data.preprocess import clean_text, preprocess_headlines, add_headline_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


# Lazy imports for heavy libs
def _get_vader():
    """Function description."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except Exception as e:
        raise ImportError(
            "VADER not available. Install: pip install nltk && "
            "python -c \"import nltk; nltk.download('vader_lexicon')\""
        ) from e
    return SentimentIntensityAnalyzer()


def _get_textblob():
    """Function description."""
    try:
        from textblob import TextBlob
    except Exception as e:
        raise ImportError(
            "TextBlob not available. Install: pip install textblob") from e
    return TextBlob


def _score_vader_series(series: pd.Series) -> pd.DataFrame:
    """Function description."""
    sia = _get_vader()
    rows = []
    for txt in series.fillna("").astype(str):
        scores = sia.polarity_scores(txt)
        rows.append(scores)
    df = pd.DataFrame(rows, index=series.index)
    df.columns = [f"vader_{c}" for c in df.columns]
    return df


def _score_textblob_series(series: pd.Series) -> pd.DataFrame:
    """Function description."""
    TextBlob = _get_textblob()
    polarity = []
    subjectivity = []
    for txt in series.fillna("").astype(str):
        tb = TextBlob(txt)
        polarity.append(tb.sentiment.polarity)
        subjectivity.append(tb.sentiment.subjectivity)
    return pd.DataFrame({"textblob_polarity": polarity, "textblob_subjectivity": subjectivity}, index=series.index)


def ensemble_score(df: pd.DataFrame, weight_vader: float = 0.7, weight_tb: float = 0.3) -> pd.Series:
    """
    Combine vader_compound and textblob_polarity into a single ensemble score in [-1, 1].
    If a field is missing, it's treated as zero.
    """
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
    """
    Compute sentiment columns for news DataFrame.

    - Optionally run your preprocess_headlines (cleaning + optional lemmatization).
    - Adds: vader_compound, vader_pos, vader_neu, vader_neg, textblob_polarity, textblob_subjectivity, sentiment_ensemble
    - Preserves original DataFrame and returns a new copy.
    """
    preprocess_args = preprocess_args or {}
    out = df.copy()

    if headline_col not in out.columns:
        raise ValueError(f"DataFrame missing headline column '{headline_col}'")

    # Optionally preprocess (clean) the headline text for scoring
    if preprocess:
        # keep original in case user wants it later
        out["_headline_original_for_sentiment"] = out[headline_col].astype(str)
        cleaned = preprocess_headlines(out, text_col=headline_col, remove_stopwords=preprocess_args.get("remove_stopwords", True),
                                       lemmatize=preprocess_args.get("lemmatize", False))
        # preprocess_headlines returns a filtered DF; reindex to original to keep alignment
        # create mapping index->cleaned headline
        cleaned_map = cleaned[headline_col].to_dict()
        out["_headline_for_sentiment"] = out.index.map(lambda i: cleaned_map.get(
            i, out.at[i, headline_col] if i in out.index else ""))
    else:
        out["_headline_for_sentiment"] = out[headline_col].astype(str)

    # Score in batches to avoid memory issues for huge datasets
    texts = out["_headline_for_sentiment"]

    try:
        vader_df = _score_vader_series(texts)
        tb_df = _score_textblob_series(texts)
    except Exception as e:
        logger.exception("Sentiment scoring failed: %s", e)
        raise

    out = pd.concat([out, vader_df, tb_df], axis=1)
    out["sentiment_ensemble"] = ensemble_score(out)
    # optional additional metrics
    try:
        out = add_headline_metrics(out, text_col=headline_col)
    except Exception:
        # not fatal — metrics are convenience
        logger.debug("Could not add headline metrics (skipping)")

    # drop helper columns if you don't want them; keep them for traceability
    return out


# Example quick-run
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
