#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the full pipeline: 
1) Load stock price & news
2) Compute indicators (SMA, EMA, RSI, MACD)
3) Compute sentiment scores on news
4) Align dates and compute correlation between news sentiment and stock returns
"""

from fns_project.analysis.correlation import compute_correlation
from fns_project.nlp.sentiment import compute_sentiment_scores
from fns_project.features.indicators import compute_all_indicators
from fns_project.data.align import align_dates
from fns_project.data.loader import load_prices, load_news
from fns_project.config import ConfigLoader
import sys
from pathlib import Path

import pandas as pd

# Add src to path if running as script
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def main(experiment_file: str = None):
    # -----------------------------
    # Load configs
    # -----------------------------
    cfg = ConfigLoader(experiment_file=experiment_file)

    price_cfg = cfg.get("data")["prices"]
    news_cfg = cfg.get("data")["news"]

    print("Loading data...")
    df_price = load_prices(price_cfg["file_path"])
    df_news = load_news(news_cfg["file_path"])

    # -----------------------------
    # Compute indicators
    # -----------------------------
    print("Computing indicators...")
    df_price_ind = compute_all_indicators(df_price)

    # -----------------------------
    # Compute sentiment scores
    # -----------------------------
    print("Computing sentiment scores...")
    df_news_sent = compute_sentiment_scores(
        df_news, model_cfg=cfg.get_section("sentiment"))

    # -----------------------------
    # Align dates
    # -----------------------------
    print("Aligning dates...")
    df_price_aligned, df_news_aligned = align_dates(df_price_ind, df_news_sent)

    # -----------------------------
    # Compute correlation
    # -----------------------------
    print("Computing correlation...")
    corr_df = compute_correlation(df_price_aligned, df_news_aligned)

    print("\nCorrelation results:")
    print(corr_df.head())

    # Optional: save results
    output_dir = Path(price_cfg.get("output_dir", "../data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    df_price_aligned.to_csv(
        output_dir / "price_with_indicators.csv", index=False)
    df_news_aligned.to_csv(output_dir / "news_with_sentiment.csv", index=False)
    corr_df.to_csv(output_dir / "correlation_results.csv", index=False)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run full stock-news pipeline")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Optional experiment YAML to override defaults")
    args = parser.parse_args()
    main(experiment_file=args.experiment)
