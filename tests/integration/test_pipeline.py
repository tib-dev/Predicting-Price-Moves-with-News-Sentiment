import unittest
import pandas as pd
from pathlib import Path

from src.fns_project.data.loader import load_news_csv, load_price_csv
from src.fns_project.data.preprocessor import preprocess_headlines, add_headline_metrics
from src.fns_project.data.align import align_news_to_trading_days, aggregate_headlines
from src.fns_project.analysis.eda_analysis import run_full_eda


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load sample data once for all tests."""
        cls.news_path = Path("data/raw/sample_news.csv")
        cls.price_path = Path("data/raw/sample_price.csv")

        # Create tiny sample if not present
        if not cls.news_path.exists():
            sample_news = pd.DataFrame({
                "date": ["2025-11-20 08:00", "2025-11-20 12:30", "2025-11-21 09:15"],
                "headline": [
                    "Apple stock surges after earnings report",
                    "Tesla announces new price target",
                    "FDA approves new drug for cancer treatment"
                ],
                "publisher": ["news@apple.com", "market@tesla.com", "health@fda.gov"]
            })
            cls.news_path.parent.mkdir(parents=True, exist_ok=True)
            sample_news.to_csv(cls.news_path, index=False)

        if not cls.price_path.exists():
            sample_prices = pd.DataFrame({
                "Date": ["2025-11-20", "2025-11-21"],
                "Open": [150, 152],
                "High": [155, 153],
                "Low": [149, 151],
                "Close": [154, 152],
                "Volume": [10000, 12000]
            })
            cls.price_path.parent.mkdir(parents=True, exist_ok=True)
            sample_prices.to_csv(cls.price_path, index=False)

    def test_loader(self):
        news_df = load_news_csv(self.news_path)
        price_df = load_price_csv(self.price_path)
        self.assertGreater(len(news_df), 0)
        self.assertGreater(len(price_df), 0)
        self.assertIn("headline", news_df.columns)
        self.assertIn("date", price_df.columns)

    def test_preprocessor(self):
        news_df = load_news_csv(self.news_path)
        df_clean = preprocess_headlines(news_df)
        df_metrics = add_headline_metrics(df_clean)
        self.assertIn("headline_len_chars", df_metrics.columns)
        self.assertIn("headline_word_count", df_metrics.columns)

    def test_align_and_aggregate(self):
        news_df = load_news_csv(self.news_path)
        price_df = load_price_csv(self.price_path)
        news_clean = preprocess_headlines(news_df)
        news_metrics = add_headline_metrics(news_clean)
        aligned = align_news_to_trading_days(news_metrics, price_df)
        agg = aggregate_headlines(aligned)
        self.assertIn("combined_headlines", agg.columns)
        self.assertIn("headline_count", agg.columns)
        self.assertGreaterEqual(agg["headline_count"].sum(), 1)

    def test_eda_analysis(self):
        news_df = load_news_csv(self.news_path)
        news_clean = preprocess_headlines(news_df)
        news_metrics = add_headline_metrics(news_clean)
        results = run_full_eda(news_metrics)
        self.assertIn("headline_stats", results)
        self.assertIn("publisher_counts", results)
        self.assertIn("keywords", results)
        self.assertIn("topics", results)
        self.assertIn("trend_by_date", results)
        self.assertIn("trend_by_hour", results)
        self.assertIn("publisher_domains", results)


if __name__ == "__main__":
    unittest.main()
