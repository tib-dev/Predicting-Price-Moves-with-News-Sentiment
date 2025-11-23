# fns_project/viz/plots.py
from __future__ import annotations
import logging
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

sns.set_style("whitegrid")


def plot_daily_sentiment(
    daily_sent: pd.DataFrame,
    sentiment_col: str = "sentiment_mean",
    count_col: str = "news_count",
    figsize: tuple[int, int] = (12, 5),
) -> None:
    """Plot daily sentiment and article counts together."""
    df = daily_sent.copy()
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(df.index, df[sentiment_col], color="blue",
             marker="o", label="Sentiment")
    ax1.set_ylabel("Sentiment", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.bar(df.index, df[count_col], alpha=0.3,
            color="gray", label="News Count")
    ax2.set_ylabel("News Count", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    fig.autofmt_xdate()
    fig.suptitle("Daily Sentiment and News Count")
    fig.tight_layout()
    plt.show()


def plot_price_with_indicators(
    df: pd.DataFrame,
    close_col: str = "Close",
    sma_col: Optional[str] = "SMA_20",
    ema_col: Optional[str] = "EMA_20",
    figsize: tuple[int, int] = (12, 5)
) -> None:
    """Plot closing price with SMA/EMA lines."""
    df = df.copy()
    plt.figure(figsize=figsize)
    plt.plot(df.index, df[close_col], label="Close", color="black")
    if sma_col in df.columns:
        plt.plot(df.index, df[sma_col], label=sma_col, linestyle="--")
    if ema_col in df.columns:
        plt.plot(df.index, df[ema_col], label=ema_col, linestyle=":")
    plt.title("Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_lagged_correlation(
    corr_df: pd.DataFrame,
    correlation_col: str = "correlation",
    figsize: tuple[int, int] = (8, 4)
) -> None:
    """Plot correlation vs lag."""
    df = corr_df.copy()
    plt.figure(figsize=figsize)
    sns.barplot(x=df.index.astype(str),
                y=df[correlation_col], palette="coolwarm")
    plt.title("Lagged Correlation (Sentiment -> Returns)")
    plt.xlabel("Lag (days)")
    plt.ylabel("Correlation")
    plt.ylim(-1, 1)
    plt.grid(axis="y")
    plt.show()


def plot_returns_vs_sentiment(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment_mean",
    returns_col: str = "daily_return",
    figsize: tuple[int, int] = (8, 5)
) -> None:
    """Scatter plot of daily returns vs sentiment."""
    df = df.copy()
    plt.figure(figsize=figsize)
    sns.scatterplot(x=sentiment_col, y=returns_col, data=df)
    plt.title("Daily Returns vs Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Daily Return")
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Quick test with dummy data
    import numpy as np
    idx = pd.date_range("2023-09-01", periods=10)
    daily_sent = pd.DataFrame({
        "sentiment_mean": np.random.uniform(-0.5, 0.5, 10),
        "news_count": np.random.randint(1, 10, 10)
    }, index=idx)

    df_price = pd.DataFrame({
        "Close": np.linspace(100, 110, 10),
        "SMA_20": np.linspace(101, 109, 10),
        "EMA_20": np.linspace(100.5, 109.5, 10)
    }, index=idx)

    corr_df = pd.DataFrame({
        "correlation": np.random.uniform(-1, 1, 4)
    }, index=[0, 1, 2, 3])

    plot_daily_sentiment(daily_sent)
    plot_price_with_indicators(df_price)
    plot_lagged_correlation(corr_df)
    plot_returns_vs_sentiment(pd.concat(
        [daily_sent, df_price["Close"].pct_change().fillna(0).rename("daily_return")], axis=1))
