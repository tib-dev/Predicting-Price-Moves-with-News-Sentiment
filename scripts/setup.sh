#!/bin/bash

# Root directory name (change if you want)
PROJECT_ROOT="project-root"

echo "Creating project structure under $PROJECT_ROOT ..."
mkdir -p $PROJECT_ROOT

# ----------------------------
# GitHub / CI
# ----------------------------
mkdir -p $PROJECT_ROOT/.github/workflows
touch $PROJECT_ROOT/.github/workflows/ci.yml

# ----------------------------
# Docs
# ----------------------------
mkdir -p $PROJECT_ROOT/docs
touch $PROJECT_ROOT/docs/index.md

# ----------------------------
# Notebooks
# ----------------------------
mkdir -p $PROJECT_ROOT/notebooks
touch $PROJECT_ROOT/notebooks/eda_news.ipynb
touch $PROJECT_ROOT/notebooks/eda_prices.ipynb
touch $PROJECT_ROOT/notebooks/correlation.ipynb

# ----------------------------
# Source Package
# ----------------------------
mkdir -p $PROJECT_ROOT/src/fns_project

# Base files
touch $PROJECT_ROOT/src/fns_project/__init__.py
touch $PROJECT_ROOT/src/fns_project/config.py
touch $PROJECT_ROOT/src/fns_project/logging_config.py

# Data module
mkdir -p $PROJECT_ROOT/src/fns_project/data
touch $PROJECT_ROOT/src/fns_project/data/__init__.py
touch $PROJECT_ROOT/src/fns_project/data/loader.py
touch $PROJECT_ROOT/src/fns_project/data/align_dates.py
touch $PROJECT_ROOT/src/fns_project/data/preprocess.py

# Features
mkdir -p $PROJECT_ROOT/src/fns_project/features
touch $PROJECT_ROOT/src/fns_project/features/__init__.py
touch $PROJECT_ROOT/src/fns_project/features/indicators.py
touch $PROJECT_ROOT/src/fns_project/features/sentiment_features.py

# NLP
mkdir -p $PROJECT_ROOT/src/fns_project/nlp
touch $PROJECT_ROOT/src/fns_project/nlp/__init__.py
touch $PROJECT_ROOT/src/fns_project/nlp/sentiment.py
touch $PROJECT_ROOT/src/fns_project/nlp/topic_modeling.py

# Analysis
mkdir -p $PROJECT_ROOT/src/fns_project/analysis
touch $PROJECT_ROOT/src/fns_project/analysis/__init__.py
touch $PROJECT_ROOT/src/fns_project/analysis/correlation.py
touch $PROJECT_ROOT/src/fns_project/analysis/stats.py

# Viz
mkdir -p $PROJECT_ROOT/src/fns_project/viz
touch $PROJECT_ROOT/src/fns_project/viz/__init__.py
touch $PROJECT_ROOT/src/fns_project/viz/plots.py
touch $PROJECT_ROOT/src/fns_project/viz/dashboard.py

# Utils
mkdir -p $PROJECT_ROOT/src/fns_project/utils
touch $PROJECT_ROOT/src/fns_project/utils/__init__.py
touch $PROJECT_ROOT/src/fns_project/utils/io.py
touch $PROJECT_ROOT/src/fns_project/utils/dates.py

# ----------------------------
# Scripts
# ----------------------------
mkdir -p $PROJECT_ROOT/scripts
touch $PROJECT_ROOT/scripts/download_data.py
touch $PROJECT_ROOT/scripts/run_training.py
touch $PROJECT_ROOT/scripts/run_analysis.py

# ----------------------------
# Tests
# ----------------------------
mkdir -p $PROJECT_ROOT/tests/unit
mkdir -p $PROJECT_ROOT/tests/integration

touch $PROJECT_ROOT/tests/unit/test_loader.py
touch $PROJECT_ROOT/tests/unit/test_sentiment.py
touch $PROJECT_ROOT/tests/unit/test_indicators.py
touch $PROJECT_ROOT/tests/integration/test_pipeline.py

# ----------------------------
# Data (gitignored large files)
# ----------------------------
mkdir -p $PROJECT_ROOT/data/raw_sample
mkdir -p $PROJECT_ROOT/data/processed_sample

# ----------------------------
# Configs
# ----------------------------
mkdir -p $PROJECT_ROOT/configs
touch $PROJECT_ROOT/configs/default.yaml

# ----------------------------
# Project metadata
# ----------------------------
touch $PROJECT_ROOT/requirements.txt
touch $PROJECT_ROOT/README.md
touch $PROJECT_ROOT/.gitignore
touch $PROJECT_ROOT/pyproject.toml

echo "Done! Project layout created."
