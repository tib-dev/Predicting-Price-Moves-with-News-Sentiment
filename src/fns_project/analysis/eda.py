# src/fns_project/analysis/eda.py
"""Exploratory Data Analysis for financial news headlines."""

from typing import Dict
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# =========================================================
# 1. SAFE DATE HANDLING
# =========================================================


def ensure_datetime(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Ensure the date column is datetime and drop NaTs."""
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}'.")

    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    return df


# =========================================================
# 2. DESCRIPTIVE STATS
# =========================================================
def headline_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive stats for headline length and word count."""
    needed = ["headline_len_chars", "headline_word_count"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns for stats: {missing}")
        return pd.DataFrame()

    stats = df[needed].describe()
    logger.info("Computed headline length statistics.")
    return stats


def count_articles_per_publisher(df: pd.DataFrame, col: str = "publisher") -> pd.DataFrame:
    """Count number of articles per publisher."""
    if col not in df.columns:
        logger.warning("Publisher column missing.")
        return pd.DataFrame()

    counts = df[col].fillna("Unknown").value_counts().reset_index()
    counts.columns = ["publisher", "article_count"]
    return counts


def publication_trend_by_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Return daily article counts."""
    df = ensure_datetime(df, date_col)
    trend = df.groupby(df[date_col].dt.date).size(
    ).reset_index(name="article_count")
    return trend


def publication_trend_by_time(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Return article counts by hour of day."""
    df = ensure_datetime(df, date_col)
    df["hour"] = df[date_col].dt.hour
    trend = df["hour"].value_counts().sort_index().reset_index()
    trend.columns = ["hour", "article_count"]
    return trend


# =========================================================
# 3. KEYWORD EXTRACTION
# =========================================================
def extract_top_keywords(
    df: pd.DataFrame,
    text_col: str = "headline",
    top_n: int = 20,
    max_features: int = 4000,
    min_df: int = 3
) -> pd.DataFrame:
    """Return top N keywords from headlines."""
    if text_col not in df.columns:
        logger.warning(f"Column '{text_col}' missing for keyword extraction.")
        return pd.DataFrame()

    vectorizer = CountVectorizer(
        stop_words="english", max_features=max_features, min_df=min_df
    )
    X = vectorizer.fit_transform(df[text_col])
    words = vectorizer.get_feature_names_out()
    counts = np.asarray(X.sum(axis=0)).flatten()
    top_idx = counts.argsort()[::-1][:top_n]

    return pd.DataFrame({"keyword": words[top_idx], "count": counts[top_idx]})


# =========================================================
# 4. TOPIC MODELING
# =========================================================
def topic_modeling(
    df: pd.DataFrame,
    text_col: str = "headline",
    n_topics: int = 5,
    n_words: int = 10,
    max_features: int = 4000,
    min_df: int = 3
) -> Dict[str, list]:
    """Return top words per topic using LDA."""
    if text_col not in df.columns:
        logger.warning(f"Column '{text_col}' missing for topic modeling.")
        return {}

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=min_df,
        dtype=np.float32
    )
    X = tfidf.fit_transform(df[text_col])

    lda = LatentDirichletAllocation(
        n_components=n_topics, learning_method="online", random_state=42
    )
    lda.fit(X)

    feature_names = tfidf.get_feature_names_out()
    topics = {}
    for i, comp in enumerate(lda.components_):
        top_ids = comp.argsort()[:-n_words - 1:-1]
        topics[f"Topic {i+1}"] = [feature_names[j] for j in top_ids]

    return topics


# =========================================================
# 5. DOMAIN EXTRACTION
# =========================================================
def extract_publisher_domains(df: pd.DataFrame, col: str = "publisher") -> pd.DataFrame:
    """Extract domains from publisher emails or identifiers."""
    if col not in df.columns:
        logger.warning("Publisher column missing.")
        return pd.DataFrame()

    df = df.copy()
    df["publisher_domain"] = df[col].apply(
        lambda x: x.split("@")[-1].lower() if isinstance(x,
                                                         str) and "@" in x else None
    )
    domain_counts = df["publisher_domain"].dropna(
    ).value_counts().reset_index()
    domain_counts.columns = ["domain", "count"]
    return domain_counts


# =========================================================
# 6. MAIN WRAPPER
# =========================================================
def run_full_eda(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Run all EDA steps and return a result dictionary."""
    df = df.copy()

    results = {
        "headline_stats": headline_length_stats(df),
        "publisher_counts": count_articles_per_publisher(df),
        "keywords": extract_top_keywords(df),
        "topics": topic_modeling(df),
        "trend_by_date": publication_trend_by_date(df),
        "trend_by_hour": publication_trend_by_time(df),
        "publisher_domains": extract_publisher_domains(df)
    }

    logger.info("Full EDA completed successfully.")
    return results
