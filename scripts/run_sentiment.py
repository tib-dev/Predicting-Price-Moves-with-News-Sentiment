#!/usr/bin/env python
"""
Compute sentiment scores for news dataset
"""
from src.fns_project.data.loader import load_news
from src.fns_project.nlp.sentiment import compute_sentiment
from src.fns_project.features.sentiment_features import aggregate_daily_sentiment

if __name__ == "__main__":
    news_df = load_news()
    news_df = compute_sentiment(news_df)
    daily_sentiment = aggregate_daily_sentiment(news_df)
    print(daily_sentiment.tail())
