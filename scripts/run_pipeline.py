import logging
import pandas as pd
from src.fns_project.data.loader import load_news_csv, load_price_csv
from src.fns_project.data.preprocessor import preprocess_headlines, add_headline_metrics
from src.fns_project.data.align import align_news_to_trading_days, aggregate_headlines
from src.fns_project.analysis.eda_analysis import run_full_eda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample run


def main():
    # Load small sample data
    news_df = pd.DataFrame({
        "date": ["2025-11-20 08:00", "2025-11-20 12:30"],
        "headline": ["Stock jumps after earnings", "Analyst downgrades $TSLA"],
        "publisher": ["news@example.com", "finance@example.org"]
    })

    price_df = pd.DataFrame({
        "date": ["2025-11-20"],
        "Open": [100], "High": [105], "Low": [99], "Close": [104], "Volume": [1000]
    })

    # Pipeline
    news_df = preprocess_headlines(news_df)
    news_df = add_headline_metrics(news_df)
    aligned = align_news_to_trading_days(news_df, price_df)
    agg = aggregate_headlines(aligned)
    eda_results = run_full_eda(agg)

    # Log results
    for key, value in eda_results.items():
        logger.info("%s:\n%s\n", key, value)


if __name__ == "__main__":
    main()
