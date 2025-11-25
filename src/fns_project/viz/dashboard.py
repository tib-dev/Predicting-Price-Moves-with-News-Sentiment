"""Interactive dashboards for price and sentiment visualization."""

import logging
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

pn.extension("plotly")


def create_price_sentiment_dashboard(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    date_col: str = "date",
    close_col: str = "Close",
    sentiment_col: str = "sentiment_mean",
    sma_cols: list[str] = None,
    rolling_cols: list[str] = None,
    title: str = "Price & Sentiment Dashboard"
):
    """
    Create an interactive dashboard with:
    - top: closing price + SMA lines
    - bottom: daily sentiment + rolling mean
    """
    df_price = price_df.copy()
    df_sent = sentiment_df.copy()

    df_price[date_col] = pd.to_datetime(df_price[date_col])
    df_sent[date_col] = pd.to_datetime(df_sent[date_col])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Price", "Sentiment")
    )

    # Price plot
    fig.add_trace(go.Scatter(
        x=df_price[date_col], y=df_price[close_col],
        mode="lines", name="Close", line=dict(color="blue")
    ), row=1, col=1)

    if sma_cols:
        for col in sma_cols:
            if col in df_price.columns:
                fig.add_trace(go.Scatter(
                    x=df_price[date_col], y=df_price[col],
                    mode="lines", name=col
                ), row=1, col=1)

    # Sentiment plot
    fig.add_trace(go.Scatter(
        x=df_sent[date_col], y=df_sent[sentiment_col],
        mode="lines+markers", name=sentiment_col, line=dict(color="green")
    ), row=2, col=1)

    if rolling_cols:
        for col in rolling_cols:
            if col in df_sent.columns:
                fig.add_trace(go.Scatter(
                    x=df_sent[date_col], y=df_sent[col],
                    mode="lines", name=col, line=dict(dash="dash")
                ), row=2, col=1)

    fig.update_layout(height=600, width=900,
                      title_text=title, template="plotly_white")

    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    dates = pd.date_range("2023-01-01", periods=10)

    # Price sample
    price = pd.Series([100, 101, 102, 101, 103, 105, 104, 106, 107, 108])
    df_price = pd.DataFrame(
        {"date": dates, "Close": price, "SMA_3": price.rolling(3).mean()})

    # Sentiment sample
    sentiment = pd.Series([0.1, 0.2, -0.1, 0.0, 0.3, 0.4, 0.2, -0.2, 0.0, 0.1])
    df_sent = pd.DataFrame({"date": dates, "sentiment_mean": sentiment,
                            "sentiment_mean_roll_3": sentiment.rolling(3).mean()})

    dashboard = create_price_sentiment_dashboard(
        df_price, df_sent,
        sma_cols=["SMA_3"],
        rolling_cols=["sentiment_mean_roll_3"]
    )
    dashboard.show()
