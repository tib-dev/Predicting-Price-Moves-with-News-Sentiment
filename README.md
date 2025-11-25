# Predicting Price Moves with News Sentiment — Week 1 Challenge

Analyze the relationship between financial news sentiment and stock price movements to provide actionable insights for Nova Financial Solutions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Objective](#business-objective)
- [Dataset Overview](#dataset-overview)
- [Folder Structure](#folder-structure)
- [Setup & Installation](#setup--installation)
- [Tasks Completed](#tasks-completed)
- [Technologies Used](#technologies-used)
- [Key Insights](#key-insights)

---

## Project Overview

This project covers end-to-end analysis of financial news and stock prices:

- Exploratory Data Analysis (EDA) of news headlines, publishers, and publishing times
- Sentiment analysis of news headlines
- Technical indicators computation using TA-Lib and PyNance
- Correlation analysis between sentiment scores and stock price movements
- Development workflow using Git branches and CI/CD

---

## Business Objective

Nova Financial Solutions aims to strengthen predictive analytics by combining qualitative news sentiment with quantitative stock metrics.

The project focuses on:

- Measuring sentiment in financial news
- Linking sentiment to stock returns
- Highlighting actionable signals for investment strategies

---

## Dataset Overview

**Financial News and Stock Price Integration Dataset (FNSPID)**:

| Column    | Description                     |
| --------- | ------------------------------- |
| headline  | News article title              |
| url       | Link to full article            |
| publisher | Publisher name or domain        |
| date      | Publication date & time (UTC-4) |
| stock     | Stock ticker symbol             |

**Stock Price Dataset**:

- Open, High, Low, Close (OHLC)
- Volume
- Daily returns

---

## Folder Structure

```text
Predicting-Price-Moves-with-News-Sentiment/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── codeql.yml                 # Security scanning
│
├── configs/
│   ├── default.yaml                   # Base pipeline config
│   ├── indicators.yaml                # Indicator selection
│   ├── sentiment.yaml                 # Sentiment model config
│   └── experiment_X.yaml              # Experiment presets
│
├── data/
│   ├── raw/                           # NEVER touched by code
│   ├── interim/                       # Cleaned but not final
│   ├── processed/                     # Ready for modeling
│   └── sample/                        # Small test datasets
│
├── docs/
│   ├── api/
│   ├── architecture.md                # System design diagrams
│   ├── pipeline.md                    # Task1–3 explanations
│   └── deployment.md                  # How to deploy
│
├── notebooks/
│   ├── exploration/                   # EDA, visualization
│   │   ├── eda_news.ipynb
│   │   ├── eda_prices.ipynb
│   │   └── correlation.ipynb
│   └── experiments/                   # Try models/ideas here
│
├── scripts/
│   ├── download_news.py               # CLI script
│   ├── download_prices.py
│   ├── run_pipeline.py                # Runs entire Task 1–3
│   ├── run_sentiment.py
│   └── run_correlation.py
│
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
│       │   ├── dashboard.py          # panel, plotly dashboards
│       │   └── report_builder.py      # HTML/PDF summary reports
│
│       └── utils/
│           ├── __init__.py
│           ├── dates.py
│           └── caching.py             # caching for speed
│
├── tests/
│   ├── unit/
│   │   ├── test_loader.py
│   │   ├── test_sentiment.py
│   │   ├── test_indicators.py
│   │   ├── test_correlation.py
│   │   └── test_date_utils.py
│   └── integration/
│       ├── test_full_news_pipeline.py
│       └── test_full_correlation.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml             # db/cache + API
│
├── requirements.txt
├── pyproject.toml                     # preferred for modern builds
├── README.md
└── .gitignore
```

## Setup & Installation

### Clone the repository:

```bash
git clone https://github.com/<username>/predicting-price-moves-with-news-sentiment.git
cd predicting-price-moves-with-news-sentiment

Create a Python virtual environment and activate it:
```

```bash
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate
```

### Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Launch Jupyter notebooks:

```bash
jupyter notebook
```

## Tasks Completed

### Task 1: Git & Environment Setup

- Repository initialized with branches and CI workflow
- Python virtual environment created
- Exploratory Data Analysis (EDA) of news headlines, publishers, and publishing times

### Task 2: Technical Indicators & Market Analysis

- Stock price data cleaned and prepared
- Technical indicators computed (SMA, EMA, RSI, MACD)
- Visualizations for trends and volume

### Task 3: Sentiment vs Stock Movement Correlation

- Sentiment scores generated from news headlines
- Daily stock returns calculated
- Pearson correlation analysis between sentiment and returns
- Visualizations for patterns and regression

---

## Technologies Used

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- TA-Lib, PyNance
- NLTK, TextBlob
- Jupyter Notebook
- Git & GitHub Actions

---

## Key Insights

- Negative news sentiment often precedes short-term price drops
- Certain publishers have strong positive or negative bias
- Sentiment combined with technical indicators improves predictive power
- Publishing spikes align with major financial events
