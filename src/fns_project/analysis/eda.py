import pandas as pd
import logging
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# =========================================================
# 1. DESCRIPTIVE STATISTICS
# =========================================================


def headline_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive stats for headline length and word count."""
    cols = ["headline_len_chars", "headline_word_count"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns for stats: {missing}")
        return pd.DataFrame()
    stats = df[cols].describe()
    logger.info("Computed headline length statistics.")
    return stats


def count_articles_per_publisher(df: pd.DataFrame, publisher_col: str = "publisher") -> pd.DataFrame:
    """Count number of articles published by each publisher."""
    if publisher_col not in df.columns:
        logger.warning(
            "Publisher column missing; skipping publisher analysis.")
        return pd.DataFrame()
    counts = df[publisher_col].value_counts().reset_index()
    counts.columns = ["publisher", "article_count"]
    return counts


def publication_trend_by_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Count articles per date to reveal trends."""
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}'.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    trend = df.groupby(df[date_col].dt.date).size(
    ).reset_index(name="article_count")
    return trend


def publication_trend_by_time(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Analyze publication frequency by hour of day."""
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}'.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["hour"] = df[date_col].dt.hour
    trend = df["hour"].value_counts().sort_index().reset_index()
    trend.columns = ["hour", "article_count"]
    return trend


# =========================================================
# 2. KEYWORD EXTRACTION (memory-friendly)
# =========================================================
def extract_top_keywords(df: pd.DataFrame, text_col: str = "headline",
                         top_n: int = 20, max_features: int = 5000, min_df: int = 5) -> pd.DataFrame:
    """Extract most frequent keywords using CountVectorizer."""
    if text_col not in df.columns:
        logger.warning(
            f"Text column '{text_col}' missing; skipping keyword extraction.")
        return pd.DataFrame()
    vectorizer = CountVectorizer(
        stop_words="english", max_features=max_features, min_df=min_df)
    X = vectorizer.fit_transform(df[text_col])
    word_counts = np.array(X.sum(axis=0)).flatten()
    words = vectorizer.get_feature_names_out()
    top_idx = np.argsort(word_counts)[::-1][:top_n]
    return pd.DataFrame({"keyword": words[top_idx], "count": word_counts[top_idx]})


# =========================================================
# 3. TOPIC MODELING (memory-friendly LDA)
# =========================================================
def topic_modeling(df: pd.DataFrame, text_col: str = "headline",
                   n_topics: int = 5, n_words: int = 10,
                   max_features: int = 5000, min_df: int = 5):
    """Perform LDA topic modeling on cleaned headlines."""
    if text_col not in df.columns:
        logger.warning(
            f"Text column '{text_col}' missing; skipping topic modeling.")
        return {}
    tfidf = TfidfVectorizer(
        stop_words="english", max_features=max_features, min_df=min_df, dtype=np.float32)
    X = tfidf.fit_transform(df[text_col])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    feature_names = tfidf.get_feature_names_out()
    topics = {}
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words-1:-1]]
        topics[f"Topic {idx+1}"] = top_words
    return topics


# =========================================================
# 4. PUBLISHER DOMAIN ANALYSIS
# =========================================================
def extract_publisher_domains(df: pd.DataFrame, publisher_col: str = "publisher") -> pd.DataFrame:
    """Extract email domains if publisher contains email addresses."""
    if publisher_col not in df.columns:
        logger.warning("Publisher column missing; domain extraction skipped.")
        return pd.DataFrame()
    df["publisher_domain"] = df[publisher_col].apply(lambda p: p.split(
        "@")[-1].lower().strip() if isinstance(p, str) and "@" in p else None)
    domain_counts = df["publisher_domain"].dropna(
    ).value_counts().reset_index()
    domain_counts.columns = ["domain", "count"]
    return domain_counts


# =========================================================
# 5. PIPELINE WRAPPER
# =========================================================
def run_full_eda(df: pd.DataFrame):
    """Run all EDA steps and return structured results."""
    results = {
        "headline_stats": headline_length_stats(df),
        "publisher_counts": count_articles_per_publisher(df),
        "keywords": extract_top_keywords(df),
        "topics": topic_modeling(df),
        "trend_by_date": publication_trend_by_date(df),
        "trend_by_hour": publication_trend_by_time(df),
        "publisher_domains": extract_publisher_domains(df)
    }
    logger.info("Full EDA complete.")
    return results
