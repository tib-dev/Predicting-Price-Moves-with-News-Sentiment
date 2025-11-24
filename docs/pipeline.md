# Pipeline Overview (Task 1–3)

## Task 1: Load and Prepare Data
1. Load stock price and news datasets.
2. Preprocess text and price data.
3. Align dates across news and stock prices.

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



