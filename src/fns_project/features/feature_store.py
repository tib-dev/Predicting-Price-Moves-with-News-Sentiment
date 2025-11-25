"""Read/write feature datasets to disk or cache."""
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def save_features(df: pd.DataFrame, path: str, file_type: str = "parquet"):
    """Save feature DataFrame to CSV or Parquet."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if file_type == "csv":
        df.to_csv(path, index=True)
    elif file_type == "parquet":
        df.to_parquet(path, index=True)
    else:
        raise ValueError(f"Unsupported file_type '{file_type}'")
    logger.info("Saved features to %s", path)


def load_features(path: str, file_type: str = "parquet") -> pd.DataFrame:
    """Load feature DataFrame from CSV or Parquet."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    if file_type == "csv":
        df = pd.read_csv(path, index_col=0)
    elif file_type == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file_type '{file_type}'")

    return df
