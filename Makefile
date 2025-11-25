# Makefile for running full FNS pipeline

# Paths
NEWS_CSV = data/raw/news/news_sample.csv
PRICE_CSV = data/raw/price/price_sample.csv
OUTPUT_DIR = data/interim/pipeline_results

# Default target
all: pipeline

# Run full pipeline
pipeline:
	@echo "Running full FNS pipeline..."
	python scripts/run_pipeline.py --news_csv $(NEWS_CSV) --price_csv $(PRICE_CSV) --output_dir $(OUTPUT_DIR)
	@echo "Pipeline completed. Check outputs in $(OUTPUT_DIR)"

# Optional: individual steps
eda:
	@echo "Running EDA only..."
	python -c "from fns_project.analysis.eda import run_full_eda; import pandas as pd; df=pd.read_csv('$(NEWS_CSV)'); result=run_full_eda(df); print(result.keys())"

sentiment:
	@echo "Running sentiment analysis only..."
	python scripts/run_sentiment.py --news_csv $(NEWS_CSV) --output_dir $(OUTPUT_DIR)

returns:
	@echo "Computing daily returns..."
	python scripts/run_correlation.py --price_csv $(PRICE_CSV) --output_dir $(OUTPUT_DIR)

correlation:
	@echo "Running correlation analysis..."
	python scripts/run_correlation.py --news_csv $(NEWS_CSV) --price_csv $(PRICE_CSV) --output_dir $(OUTPUT_DIR)

# Clean outputs
clean:
	@echo "Cleaning pipeline outputs..."
	rm -rf $(OUTPUT_DIR)
	@echo "Done."
