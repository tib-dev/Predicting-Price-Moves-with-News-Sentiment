# scripts/run_pipeline.py
"""Full pipeline: EDA → Sentiment → Price Returns → Correlation."""

import logging
from pathlib import Path
import pandas as pd

from fns_project.analysis.eda import run_full_eda
from fns_project.analysis.correlation import (
    compute_daily_returns,
    aggregate_daily_sentiment_from_news,
    correlation_with_returns,
)
from fns_project.nlp.sentiment import add_sentiment_columns
from fns_project.data.loader import load_price_csv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def run_pipeline(
    news_csv: str,
    price_csv: str,
    output_dir: str = "data/interim/pipeline_results",
):
    news_path = Path(news_csv)
    price_path = Path(price_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # 1️⃣ Load data
    # -------------------------------
    if not news_path.exists() or not price_path.exists():
        raise FileNotFoundError("News or price CSV file not found.")

    logger.info("Loading news data from %s", news_path)
    news_df = pd.read_csv(news_path)

    logger.info("Loading price data from %s", price_path)
    price_df = load_price_csv(price_path, date_col="Date", tz="Etc/GMT+4")

    # -------------------------------
    # 2️⃣ Run EDA
    # -------------------------------
    logger.info("Running full EDA on news dataset")
    eda_results = run_full_eda(news_df)
    for key, df in eda_results.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(output_path / f"eda_{key}.csv", index=False)
            logger.info("Saved EDA results: %s", key)

    # -------------------------------
    # 3️⃣ Sentiment analysis
    # -------------------------------
    logger.info("Computing sentiment scores for news")
    news_scored = add_sentiment_columns(news_df)

    # -------------------------------
    # 4️⃣ Compute daily sentiment aggregates
    # -------------------------------
    daily_sentiment = aggregate_daily_sentiment_from_news(
        news_scored, price_df)
    daily_sentiment.to_csv(output_path / "daily_sentiment.csv")
    logger.info("Saved daily sentiment aggregates")

    # -------------------------------
    # 5️⃣ Compute daily returns
    # -------------------------------
    daily_returns = compute_daily_returns(
        price_df, close_col="Close", date_col="date")
    daily_returns.to_csv(output_path / "daily_returns.csv")
    logger.info("Saved daily returns")

    # -------------------------------
    # 6️⃣ Correlation analysis
    # -------------------------------
    corrs = correlation_with_returns(daily_sentiment, daily_returns, max_lag=5)
    corrs.to_csv(output_path / "sentiment_return_correlation.csv")
    logger.info("Saved sentiment-return correlation results")

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full FNS pipeline")
    parser.add_argument("--news_csv", required=True, help="Path to news CSV")
    parser.add_argument("--price_csv", required=True, help="Path to price CSV")
    parser.add_argument(
        "--output_dir",
        default="data/interim/pipeline_results",
        help="Directory to save pipeline outputs",
    )
    args = parser.parse_args()

    run_pipeline(args.news_csv, args.price_csv, args.output_dir)
