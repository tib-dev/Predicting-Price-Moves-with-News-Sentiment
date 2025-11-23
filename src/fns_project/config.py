from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


DEFAULTS = {
    "market_tz": "America/New_York",  # exchange timezone (ET)
    "news_tz": "Etc/GMT+4",  # dataset stated as UTC-4
}
