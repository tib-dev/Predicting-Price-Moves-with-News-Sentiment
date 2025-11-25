#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
from fns_project.analysis.correlation import compute_correlation
from fns_project.utils.io_utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run sentiment ↔ price correlation analysis.")
    parser.add_argument("--news", type=str,
                        default="../data/processed/news_sentiment.csv")
    parser.add_argument("--prices", type=str,
                        default="../data/processed/prices_indicators.csv")
    parser.add_argument("--output", type=str,
                        default="../data/processed/correlation_result.csv")
    args = parser.parse_args()

    news_df = pd.read_csv(args.news)
    price_df = pd.read_csv(args.prices)

    ensure_dir(Path(args.output).parent)

    result = compute_correlation(news_df, price_df)
    result.to_csv(args.output, index=False)

    print(f"[✓] Correlation result saved to: {args.output}")


if __name__ == "__main__":
    main()
