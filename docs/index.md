# Predicting Price Moves with News Sentiment  
*A clean overview of the system, workflow, and documentation.*

## Overview
This project explores how daily news sentiment relates to stock price movements.  
It combines financial data, news headlines, NLP sentiment scoring, technical indicators, and statistical analysis to understand whether sentiment has predictive value.

The pipeline is modular and production-ready. Each stage has clear responsibilities so you can extend or deploy the system without rewriting core logic.

## Main Goals
- Gather and clean financial and news datasets  
- Generate sentiment scores from news headlines  
- Compute financial indicators and stock returns  
- Align news days with market trading days  
- Measure correlations between sentiment and price movements  
- Produce visual summaries, reports, and dashboards  

## Project Structure (High-Level)



```text

├── src/
│   └── fns_project/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_config.py
│
│       ├── data/                      # ingestion + preprocessing
│       │   ├── __init__.py
│       │   ├── loader.py              # load CSV, Parquet, APIs
│       │   ├── fetch_api.py           # yfinance / polygon / newsapi
│       │   ├── align.py               # timestamp normalization
│       │   ├── preprocess_text.py     # clean headlines
│       │   ├── preprocess_prices.py   # price cleaning filters
│       │   └── pipeline_news.py       # orchestrated news pipeline
│
│       ├── features/                  # feature engineering
│       │   ├── __init__.py
│       │   ├── indicators.py          # TA-Lib/pandas-ta wrappers
│       │   ├── volatility.py          # GARCH, ATR, realized vol
│       │   ├── sentiment_features.py  # daily aggregates, rolling
│       │   └── feature_store.py       # read/write feature datasets
│
│       ├── nlp/                       # NLP + sentiment
│       │   ├── __init__.py
│       │   ├── sentiment.py           # VADER/TextBlob/HF scoring
│       │   ├── vectorizer.py          # TF-IDF, embeddings
│       │   └── topic_model.py         # optional: LDA, NMF
│
│       ├── analysis/                  # statistical analysis
│       │   ├── __init__.py
│       │   ├── returns.py             # returns/log-returns, volatility
│       │   ├── correlation.py         # sentiment ↔ price correlation
│       │   └── stats_tools.py         # p-values, regression, tests
│
│       ├── models/                    # ML/forecasting (future ready)
│       │   ├── __init__.py
│       │   ├── baseline_regressor.py
│       │   └── lstm_predictor.py
│
│       ├── api/                       # optional FastAPI endpoints
│       │   ├── __init__.py
│       │   ├── app.py
│       │   └── routers/
│       │       ├── news.py
│       │       └── indicators.py
│
│       ├── viz/
│       │   ├── __init__.py
│       │   ├── plots.py               # price + sentiment charts
│       │   ├── dashboards.py          # panel, plotly dashboards
│       │   └── report_builder.py      # HTML/PDF summary reports
│
│       └── utils/
│           ├── __init__.py
│           ├── dates.py
│           ├── io_utils.py
│           ├── validators.py          # input validation
│           └── caching.py             # caching for speed

```


## Documentation Index

### 1. Architecture
- How the system is designed  
- Data flow from ingestion to analysis  
- Module roles and responsibilities  
See: **[architecture.md](architecture.md)**

### 2. Pipeline Tasks  
- Clean and align datasets  
- Generate features  
- Run correlation study (Task 1–3 explanation)  
See: **[pipeline.md](pipeline.md)**

### 3. Deployment Guide  
- How to run locally  
- Running full pipeline  
- Optional API service  
- Docker instructions  
See: **[deployment.md](deployment.md)**

## How to Use This Documentation
Each file in this section explains a single part of the system.  
If you're new to the project:
1. Start with **architecture.md**  
2. Then read **pipeline.md**  
3. Use **deployment.md** when you're ready to run or deploy  

## Want to Explore Interactively?
Check the notebooks in the `notebooks/` folder:  
- `eda_news.ipynb` – explore news sentiment  
- `eda_prices.ipynb` – explore stock data  
- `correlation.ipynb` – interactive Task 3 analysis  

## Final Notes
This documentation is designed so you can extend the project:
- Add new sentiment models  
- Add new indicators  
- Run multiple experiments  
- Deploy via API or dashboard  


