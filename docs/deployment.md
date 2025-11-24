# Deployment Guide

This section explains how to deploy the pipeline and optionally expose APIs.

## 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux / Mac
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```
## 2. Running the Pipeline
```bash
python scripts/run_pipeline.py
# Optional: specify experiment config
python scripts/run_pipeline.py --experiment configs/experiment_X.yaml
```

## 3. Running as a Service (Optional)

### Start FastAPI endpoints for news and indicators:

```bash
uvicorn src.fns_project.api.app:app --reload
```


- Access at: http://127.0.0.1:8000/docs for Swagger UI

## 4. Docker Deployment

- Build Docker image:
```bash
docker build -t fns-project .
```


- Start service with Docker Compose:

``` bash

docker-compose up
```

## 5. Logging & Monitoring

- Logs stored as configured in logging_config.py

- Use dashboards in viz/ to monitor pipeline outputs
