"""Module for computing technical indicators on financial data using TA-Lib."""

import logging
from typing import Optional
import pandas as pd
import talib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def _ensure_columns(df: pd.DataFrame, cols: list[str]):
    """Raise error if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# -----------------------------
# Core indicators
# -----------------------------
def compute_sma(df: pd.DataFrame, col: str = "Close", period: int = 20) -> pd.DataFrame:
    _ensure_columns(df, [col])
    out = df.copy()
    out[f"SMA_{period}"] = talib.SMA(out[col], timeperiod=period)
    return out


def compute_ema(df: pd.DataFrame, col: str = "Close", period: int = 20) -> pd.DataFrame:
    _ensure_columns(df, [col])
    out = df.copy()
    out[f"EMA_{period}"] = talib.EMA(out[col], timeperiod=period)
    return out


def compute_rsi(df: pd.DataFrame, col: str = "Close", period: int = 14) -> pd.DataFrame:
    _ensure_columns(df, [col])
    out = df.copy()
    out[f"RSI_{period}"] = talib.RSI(out[col], timeperiod=period)
    return out


def compute_macd(df: pd.DataFrame, col: str = "Close", fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    _ensure_columns(df, [col])
    out = df.copy()
    macd, macdsignal, macdhist = talib.MACD(
        out[col], fastperiod=fast, slowperiod=slow, signalperiod=signal)
    out[f"MACD_{fast}_{slow}_{signal}"] = macd
    out[f"MACD_signal_{fast}_{slow}_{signal}"] = macdsignal
    out[f"MACD_hist_{fast}_{slow}_{signal}"] = macdhist
    return out


# -----------------------------
# Additional Indicators
# -----------------------------
def compute_bollinger_bands(df: pd.DataFrame, col: str = "Close", period: int = 20, nb_std: int = 2) -> pd.DataFrame:
    _ensure_columns(df, [col])
    out = df.copy()
    upper, middle, lower = talib.BBANDS(
        out[col], timeperiod=period, nbdevup=nb_std, nbdevdn=nb_std, matype=0)
    out["BB_upper"] = upper
    out["BB_middle"] = middle
    out["BB_lower"] = lower
    return out


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    _ensure_columns(df, ["High", "Low", "Close"])
    out = df.copy()
    out[f"ATR_{period}"] = talib.ATR(
        out["High"], out["Low"], out["Close"], timeperiod=period)
    return out


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    _ensure_columns(df, ["High", "Low", "Close"])
    out = df.copy()
    slowk, slowd = talib.STOCH(out["High"], out["Low"], out["Close"],
                               fastk_period=k_period, slowk_period=d_period, slowk_matype=0,
                               slowd_period=d_period, slowd_matype=0)
    out["STOCH_K"] = slowk
    out["STOCH_D"] = slowd
    return out


# -----------------------------
# All-in-one function
# -----------------------------
def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = compute_sma(out)
    out = compute_ema(out)
    out = compute_rsi(out)
    out = compute_macd(out)
    out = compute_bollinger_bands(out)
    out = compute_atr(out)
    out = compute_stochastic(out)
    return out


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    import numpy as np
    df = pd.DataFrame({
        "Open": np.arange(100, 110),
        "High": np.arange(101, 111),
        "Low": np.arange(99, 109),
        "Close": np.arange(100, 110),
        "Volume": [1000, 1200, 1300, 1100, 1150, 1400, 1350, 1500, 1450, 1600]
    })
    df_ind = compute_all_indicators(df)
    print(df_ind.tail(5))
