"""
Plotting utilities for price, indicators, sentiment, and correlation analysis.
"""

from __future__ import annotations
import numpy as np
import talib
import logging
from typing import Optional, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import talib as ta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

sns.set_style("whitegrid")

# ============================================================
# NEWS PLOTS
# ============================================================


def plot_headline_lengths(df, char_col="headline_len_chars", word_col="headline_word_count"):
    plt.figure(figsize=(10, 4))
    plt.hist(df[char_col], bins=50, alpha=0.7,
             color="skyblue", edgecolor="black")
    plt.title("Headline Length (Characters)")
    plt.xlabel("Characters")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.hist(df[word_col], bins=50, alpha=0.7,
             color="salmon", edgecolor="black")
    plt.title("Headline Word Count")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.show()


def plot_top_publishers(df, publisher_col="publisher", top_n=10):
    top_pub = df[publisher_col].value_counts().nlargest(top_n)
    plt.figure(figsize=(10, 6))
    top_pub.sort_values().plot(kind="barh", color="#6a5acd", alpha=0.85)

    plt.title(f"Top {top_n} Publishers")
    plt.xlabel("Article Count")
    plt.ylabel("Publisher")

    for i, v in enumerate(top_pub.sort_values()):
        plt.text(v + 0.1, i, str(v), va="center")

    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_publication_times(df, date_col="date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    plt.figure(figsize=(10, 4))
    df[date_col].dt.hour.value_counts().sort_index().plot(
        kind="bar", color="teal")
    plt.title("Articles by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(10, 4))
    order = ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"]
    df[date_col].dt.day_name().value_counts().reindex(
        order).plot(kind="bar", color="orange")
    plt.title("Articles by Day")
    plt.xlabel("Day")
    plt.ylabel("Count")
    plt.show()


# ============================================================
# SENTIMENT PLOTS
# ============================================================

def plot_daily_sentiment(daily_sent, sentiment_col="sentiment_mean", count_col="news_count", figsize=(12, 5)):
    df = daily_sent.copy()
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(df.index, df[sentiment_col], color="blue",
             marker="o", label="Sentiment")
    ax1.set_ylabel("Sentiment", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.bar(df.index, df[count_col], alpha=0.25, color="gray", width=0.8)
    ax2.set_ylabel("News Count", color="gray")

    plt.title("Daily Sentiment vs News Volume")
    fig.tight_layout()
    plt.show()


# ============================================================
# PRICE & TECHNICAL INDICATOR PLOTS
# ============================================================

def plot_price_with_indicators(
    df: pd.DataFrame,
    close_col: str = "Close",
    sma_col: Optional[str] = "SMA_20",
    ema_col: Optional[str] = "EMA_20",
    # ["BB_upper", "BB_middle", "BB_lower"]
    bb_cols: Optional[List[str]] = None,
    show_volume: bool = False,
    figsize: tuple[int, int] = (12, 6),
    title: Optional[str] = None
):
    """
    Clean price plot with SMA, EMA, optional Bollinger Bands & Volume.
    """

    if df.empty:
        print("DataFrame empty. No plot created.")
        return

    df = df.copy()

    if show_volume:
        fig, (ax, ax_vol) = plt.subplots(2, 1, figsize=figsize,
                                         gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Price
    ax.plot(df.index, df[close_col], color="black", label="Close")

    if sma_col in df.columns:
        ax.plot(df.index, df[sma_col], linestyle="--", label=sma_col)

    if ema_col in df.columns:
        ax.plot(df.index, df[ema_col], linestyle=":", label=ema_col)

    # Bollinger Bands
    if bb_cols and len(bb_cols) == 3:
        upper, middle, lower = bb_cols
        if upper in df.columns and lower in df.columns:
            ax.fill_between(
                df.index, df[upper], df[lower], color="gray", alpha=0.15, label="Bollinger Bands")
            ax.plot(df.index, df[middle], color="blue",
                    alpha=0.6, label="BB Middle")

    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Price with Indicators")

    # Volume subplot
    if show_volume and "Volume" in df.columns:
        ax_vol.bar(df.index, df["Volume"], color="steelblue", alpha=0.4)
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_rsi(
    df: pd.DataFrame,
    close_col: str = "Close",
    rsi_col: str = "RSI_14",
    period: int = 14,
    figsize=(12, 4),
    title="RSI",
    ax: plt.Axes = None
):
    df_plot = df.copy()

    if close_col not in df_plot.columns:
        raise ValueError(f"'{close_col}' column not found in DataFrame.")

    # Compute RSI if missing
    if rsi_col not in df_plot.columns:
        df_plot[rsi_col] = talib.RSI(
            df_plot[close_col].values, timeperiod=period)

    # Fill NaNs
    df_plot[rsi_col] = df_plot[rsi_col].fillna(0)

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(df_plot.index, df_plot[rsi_col],
            label=f"RSI ({period})", color="purple", linewidth=1.2)
    ax.axhline(70, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(30, color="green", linestyle="--", linewidth=0.8)

    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if ax is None:
        plt.show()




# Restart kernel before this step if needed

# -----------------------------
# Redefine plot_macd (new improved version)
# -----------------------------


def plot_macd(
    df: pd.DataFrame,
    close_col: str = "Close",
    macd_col: str = "MACD",
    signal_col: str = "MACD_signal",
    hist_col: str = "MACD_hist",
    figsize=(12, 5),
    title="MACD"
):
    df_plot = df.copy()

    if close_col not in df_plot.columns:
        raise ValueError(f"'{close_col}' column not found in DataFrame.")

    # Compute MACD if columns missing
    missing_cols = [c for c in [macd_col, signal_col,
                                hist_col] if c not in df_plot.columns]
    if missing_cols:
        macd, signal, hist = talib.MACD(
            df_plot[close_col], fastperiod=12, slowperiod=26, signalperiod=9)
        df_plot[macd_col] = macd
        df_plot[signal_col] = signal
        df_plot[hist_col] = hist

    plt.figure(figsize=figsize)
    plt.plot(df_plot.index, df_plot[macd_col],
             label="MACD", color="blue", linewidth=1.2)
    plt.plot(df_plot.index, df_plot[signal_col],
             label="Signal", color="red", linewidth=1.2)
    plt.bar(df_plot.index, df_plot[hist_col],
            label="Histogram", color="gray", alpha=0.3, width=1)

    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# CORRELATION / RETURNS PLOTS
# ============================================================

def plot_lagged_correlation(corr_df, correlation_col="correlation", figsize=(8, 4)):
    df = corr_df.copy()
    plt.figure(figsize=figsize)
    sns.barplot(x=df.index.astype(str), y=df[correlation_col])
    plt.title("Lagged Correlation")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.ylim(-1, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def plot_returns_vs_sentiment(df, sentiment_col="sentiment_mean", returns_col="daily_return", figsize=(8, 5)):
    df = df.copy()
    plt.figure(figsize=figsize)
    sns.scatterplot(x=sentiment_col, y=returns_col, data=df)
    plt.title("Returns vs Sentiment")
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.grid(True)
    plt.show()
