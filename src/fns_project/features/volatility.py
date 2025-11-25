"""Module for computing volatility measures on financial data."""

import logging
from typing import Optional
import pandas as pd
import numpy as np
import talib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def _ensure_columns(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# -----------------------------
# True Range / Average True Range (ATR)
# -----------------------------
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    _ensure_columns(df, ["High", "Low", "Close"])
    out = df.copy()
    out[f"ATR_{period}"] = talib.ATR(
        out["High"], out["Low"], out["Close"], timeperiod=period)
    return out


# -----------------------------
# Rolling volatility (standard deviation of returns)
# -----------------------------
def compute_rolling_volatility(df: pd.DataFrame, col: str = "Close", period: int = 20) -> pd.DataFrame:
    _ensure_columns(df, [col])
    out = df.copy()
    returns = out[col].pct_change()
    out[f"ROLL_VOL_{period}"] = returns.rolling(
        period).std() * np.sqrt(252)  # annualized
    return out


# -----------------------------
# GARCH placeholder (can integrate ARCH package if needed)
# -----------------------------
def compute_realized_volatility(df: pd.DataFrame, col: str = "Close", period: int = 20) -> pd.DataFrame:
    """
    Realized volatility as rolling std of log returns.
    """
    _ensure_columns(df, [col])
    out = df.copy()
    log_returns = np.log(out[col] / out[col].shift(1))
    out[f"REAL_VOL_{period}"] = log_returns.rolling(
        period).std() * np.sqrt(252)
    return out


# -----------------------------
# All-in-one volatility function
# -----------------------------
def compute_all_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    out = df.copy()
    out = compute_atr(out, period)
    out = compute_rolling_volatility(out, period=period)
    out = compute_realized_volatility(out, period=period)
    return out


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    df = pd.DataFrame({
        "Open": np.arange(100, 110),
        "High": np.arange(101, 111),
        "Low": np.arange(99, 109),
        "Close": np.arange(100, 110)
    })
    df_vol = compute_all_volatility(df, period=5)
    print(df_vol.tail(5))
