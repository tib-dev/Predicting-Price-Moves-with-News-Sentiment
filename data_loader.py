import pandas as pd
from pathlib import Path


class DataLoader:
    """Handles CSV loading and timestamp parsing."""

    def __init__(self, base_path="../data/raw"):
        self.base_path = Path(base_path)

    def load_csv(self, filename: str) -> pd.DataFrame:
        file_path = self.base_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        df = pd.read_csv(file_path)
        df.columns = [c.strip() for c in df.columns]
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        return df
