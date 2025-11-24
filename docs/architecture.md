# System Architecture

This project predicts stock price movements based on news sentiment. The architecture consists of the following layers:

## 1. Data Ingestion
- **Stock Prices:** CSV / API (Yahoo Finance, Polygon.io)
- **News Data:** API (NewsAPI, custom feeds)
- **Data Stored:** `data/raw/`

## 2. Data Preprocessing
- Clean and normalize news headlines (`preprocess_text.py`)
- Clean stock prices (`preprocess_prices.py`)
- Align timestamps between datasets (`align.py`)
- Stored in `data/interim/`

## 3. Feature Engineering
- **Technical Indicators:** SMA, EMA, RSI, MACD (`features/indicators.py`)
- **Volatility Measures:** ATR, GARCH (`features/volatility.py`)
- **Sentiment Features:** daily sentiment scores, rolling aggregates (`features/sentiment_features.py`)
- Stored in `data/processed/`

## 4. Analysis
- Compute daily stock returns (`analysis/returns.py`)
- Correlate sentiment and returns (`analysis/correlation.py`)
- Statistical tests & p-values (`analysis/stats_tools.py`)

## 5. Visualization
- Price and indicator plots (`viz/plots.py`)
- Sentiment vs price dashboards (`viz/dashboards.py`)

## 6. Optional Modeling
- ML forecasting: For future expansion (`models/`)

---

### Diagram (ASCII Placeholder)
- News API ---> | Preprocessing | ---> Sentiment Features ---

- Stock API ---> | Preprocessing | ---> Indicators ---------> Analysis ---> Visualization
