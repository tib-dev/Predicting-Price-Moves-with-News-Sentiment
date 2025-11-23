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
predicting-price-moves-with--newssentiment/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI: tests, lint, build
├── docs/                          # Project docs (Sphinx/MD)
│   └── index.md
├── notebooks/                      # Exploratory notebooks (not checked in long-term)
│   ├── eda_news.ipynb
│   ├── eda_prices.ipynb
│   └── correlation.ipynb
├── src/                            # Production Python package (importable)
│   └── fns_project/                # package root (use your project name)
│       ├── __init__.py
│       ├── config.py               # global config, constants, paths
│       ├── logging_config.py       # logger setup
│       ├── data/                   # data ingestion & transforms
│       │   ├── __init__.py
│       │   ├── loader.py           # load news & price files
│       │   ├── align_dates.py      # date normalization & trading-day mapping
│       │   └── preprocess.py       # cleaning, text normalization
│       │   └── news_pipeline.py    # containing the OO wrappers and orchestration layer.
│       ├── features/               # feature engineering & indicators
│       │   ├── __init__.py
│       │   ├── indicators.py       # wrappers for ta/pandas_ta/TA-Lib
│       │   └── sentiment_features.py
│       ├── nlp/                    # NLP utilities & models
│       │   ├── __init__.py
│       │   ├── sentiment.py        # sentiment scoring functions (vader/textblob/hf)
│       │   └── topic_modeling.py   # TF-IDF, LDA helpers
│       ├── analysis/               # statistical analysis, correlation logic
│       │   ├── __init__.py
│       │   ├── correlation.py      # correlation & lag analysis
│       │   └── stats.py            # tests, p-values, regression helpers
│       ├── viz/                    # plotting & dashboards
│       │   ├── __init__.py
│       │   ├── plots.py            # reusable plotting functions
│       │   └── dashboard.py        # streamlit/fastapi endpoints (if any)
│       └── utils/                  # small helpers
│           ├── __init__.py
│           ├── io.py               # read/write helpers (csv/parquet)
│           └── dates.py            # date helpers & timezone utilities
├── scripts/                        # CLI convenience scripts (data download, run)
│   ├── download_data.py
│   ├── run_training.py
│   └── run_analysis.py
├── tests/                          # Unit & integration tests
│   ├── unit/
│   │   ├── test_loader.py
│   │   ├── test_sentiment.py
│   │   └── test_indicators.py
│   └── integration/
│       └── test_pipeline.py
├── data/                           # small sample/test data (gitignored large data)
│   ├── raw_sample/
│   └── processed_sample/
├── configs/                        # YAML/JSON config files for experiments
│   └── default.yaml
├── requirements.txt
├── add_docstrings.py
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
