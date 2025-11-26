# Pipeline Overview (Task 1–3)

# Pipeline overview — Task 1

This document describes Task 1 (data ingestion, EDA, and preprocessing).

## Files of interest
- `src/fns_project/data/loader.py` — robust CSV/Parquet loaders with timezone handling.
- `src/fns_project/data/preprocess.py` — text cleaning helpers and headline metrics for EDA.
- `src/fns_project/data/align.py` — map news -> trading day and aggregate headlines.
- `src/fns_project/utils/datetime_utils.py` — helper utilities for timezone/normalization.

## Recommended flow (Task 1)
1. Place raw files in `data/raw/`.
2. Run `loader.load_news(...)` -> save to `data/interim/news_clean.parquet`.
3. Run `preprocess.preprocess_headlines(...)` and `add_headline_metrics(...)`.
4. Load prices via `loader.load_prices(...)`.
5. Align news to trading days via `align.align_news_to_trading_days(...)` and aggregate.
6. Save aggregated results to `data/processed/` for Task 2.

## Key design decisions
- All date handling is explicit: pass `source_tz` and `target_tz` to loaders.
- Mapping rule for news->trading date: default is 'next' trading day.
- NLTK resources downloaded lazily on first use.

## Tests
Unit tests are under `tests/unit/`. Run:

## Task 2: Apply Indicators and Metrics
1. Compute technical indicators:
   - SMA, EMA, RSI, MACD
2. Compute volatility metrics (ATR, GARCH, rolling std)
3. Compute financial metrics: daily returns, Sharpe ratio
4. Aggregate sentiment features: daily, rolling windows

## Task 3: Correlation Analysis
1. Calculate daily stock returns.
2. Compute daily sentiment scores.
3. Align sentiment and price data by date.
4. Calculate correlation and statistical significance.
5. Optional: visualize relationships.

### Example Workflow

- Load Data -> Preprocess -> Compute Indicators -> Compute Sentiment -> Align Dates -> Correlation -> Visualization


> Use `run_pipeline.py` to run the end-to-end process.



