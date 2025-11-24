#!/usr/bin/env bash
set -e

echo "Initializing project structure..."

# Helper: create directory if not exists
ensure_dir () {
    if [ ! -d "$1" ]; then
        echo "Creating directory: $1"
        mkdir -p "$1"
    else
        echo "Directory exists: $1"
    fi
}

# Helper: create file if not exists
ensure_file () {
    if [ ! -f "$1" ]; then
        echo "Creating file: $1"
        touch "$1"
    else
        echo "File exists: $1"
    fi
}

# -------------------------
# DIRECTORIES
# -------------------------

dirs=(
    ".github/workflows"
    "configs"
    "data/raw"
    "data/interim"
    "data/processed"
    "data/sample"
    "docs/api"
    "notebooks/exploration"
    "notebooks/experiments"
    "scripts"
    "src/fns_project"
    "src/fns_project/data"
    "src/fns_project/features"
    "src/fns_project/nlp"
    "src/fns_project/analysis"
    "src/fns_project/models"
    "src/fns_project/api/routers"
    "src/fns_project/viz"
    "src/fns_project/utils"
    "tests/unit"
    "tests/integration"
    "docker"
)

for d in "${dirs[@]}"; do
    ensure_dir "$d"
done

# -------------------------
# FILES
# -------------------------

files=(
    ".github/workflows/ci.yml"
    ".github/workflows/codeql.yml"
    "configs/default.yaml"
    "configs/indicators.yaml"
    "configs/sentiment.yaml"
    "configs/experiment_X.yaml"
    "docs/architecture.md"
    "docs/pipeline.md"
    "docs/deployment.md"
    "scripts/download_news.py"
    "scripts/download_prices.py"
    "scripts/run_pipeline.py"
    "scripts/run_sentiment.py"
    "scripts/run_correlation.py"
    "src/fns_project/__init__.py"
    "src/fns_project/config.py"
    "src/fns_project/logging_config.py"
    "src/fns_project/data/__init__.py"
    "src/fns_project/data/loader.py"
    "src/fns_project/data/fetch_api.py"
    "src/fns_project/data/align.py"
    "src/fns_project/data/preprocess_text.py"
    "src/fns_project/data/preprocess_prices.py"
    "src/fns_project/data/pipeline_news.py"
    "src/fns_project/features/__init__.py"
    "src/fns_project/features/indicators.py"
    "src/fns_project/features/volatility.py"
    "src/fns_project/features/sentiment_features.py"
    "src/fns_project/features/feature_store.py"
    "src/fns_project/nlp/__init__.py"
    "src/fns_project/nlp/sentiment.py"
    "src/fns_project/nlp/vectorizer.py"
    "src/fns_project/nlp/topic_model.py"
    "src/fns_project/analysis/__init__.py"
    "src/fns_project/analysis/returns.py"
    "src/fns_project/analysis/correlation.py"
    "src/fns_project/analysis/stats_tools.py"
    "src/fns_project/models/__init__.py"
    "src/fns_project/models/baseline_regressor.py"
    "src/fns_project/models/lstm_predictor.py"
    "src/fns_project/api/__init__.py"
    "src/fns_project/api/app.py"
    "src/fns_project/api/routers/news.py"
    "src/fns_project/api/routers/indicators.py"
    "src/fns_project/viz/__init__.py"
    "src/fns_project/viz/plots.py"
    "src/fns_project/viz/dashboards.py"
    "src/fns_project/viz/report_builder.py"
    "src/fns_project/utils/__init__.py"
    "src/fns_project/utils/dates.py"
    "src/fns_project/utils/io_utils.py"
    "src/fns_project/utils/validators.py"
    "src/fns_project/utils/caching.py"
    "tests/unit/test_loader.py"
    "tests/unit/test_sentiment.py"
    "tests/unit/test_indicators.py"
    "tests/unit/test_correlation.py"
    "tests/unit/test_date_utils.py"
    "tests/integration/test_full_news_pipeline.py"
    "tests/integration/test_full_correlation.py"
    "docker/Dockerfile"
    "docker/docker-compose.yml"
    "requirements.txt"
    "pyproject.toml"
    "README.md"
    ".gitignore"
)

for f in "${files[@]}"; do
    ensure_file "$f"
done

echo "Done."
