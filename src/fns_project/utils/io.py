"""File I/O helpers."""

from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def read_csv(file_path: str | Path) -> pd.DataFrame:
    """Read a CSV file safely."""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return pd.DataFrame()
    return pd.read_csv(file_path)


def save_csv(df: pd.DataFrame, file_path: str | Path, index: bool = False):
    """Save DataFrame to CSV."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=index)
    logger.info("Saved CSV: %s", file_path)
