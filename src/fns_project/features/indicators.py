"""Module for computing technical indicators on financial data."""

from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------
# Attempt to import TA-Lib, fallback to pandas_ta
# -----------------------------
try:
    import talib
    _HAS_TALIB = True
    logger.info("TA-Lib loaded successfully.")
except ImportError:
    _HAS_TALIB = False
    try:
        import pandas_ta as pta
        logger.info("pandas_ta loaded as fallback for indicators.")
    except ImportError:
        raise ImportError(
            "Install TA-Lib (`pip install TA-Lib`) or pandas_ta (`pip install pandas_ta`) to compute indicators."
        )

# -----------------------------
# Core indicators
# -----------------------------


def compute_sma(df: pd.DataFrame, col: str = "Close", period: int = 20, out_col: Optional[str] = None) -> pd.DataFrame:
    """Function description."""
    out_col = out_col or f"SMA_{period}"
    df = df.copy()
    if _HAS_TALIB:
        df[out_col] = talib.SMA(df[col], timeperiod=period)
    else:
        df[out_col] = pta.sma(df[col], length=period)
    return df


def compute_ema(df: pd.DataFrame, col: str = "Close", period: int = 20, out_col: Optional[str] = None) -> pd.DataFrame:
    """Function description."""
    out_col = out_col or f"EMA_{period}"
    df = df.copy()
    if _HAS_TALIB:
        df[out_col] = talib.EMA(df[col], timeperiod=period)
    else:
        df[out_col] = pta.ema(df[col], length=period)
    return df


def compute_rsi(df: pd.DataFrame, col: str = "Close", period: int = 14, out_col: Optional[str] = None) -> pd.DataFrame:
    """Function description."""
    out_col = out_col or f"RSI_{period}"
    df = df.copy()
    if _HAS_TALIB:
        df[out_col] = talib.RSI(df[col], timeperiod=period)
    else:
        df[out_col] = pta.rsi(df[col], length=period)
    return df


def compute_macd(df: pd.DataFrame, col: str = "Close", fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Function description."""
    df = df.copy()
    if _HAS_TALIB:
        macd, macdsignal, macdhist = talib.MACD(
            df[col], fastperiod=fast, slowperiod=slow, signalperiod=signal)
    else:
        macd_series = pta.macd(df[col], fast=fast, slow=slow, signal=signal)
        macd = macd_series[f"MACD_{fast}_{slow}_{signal}"]
        macdsignal = macd_series[f"MACDs_{fast}_{slow}_{signal}"]
        macdhist = macd_series[f"MACDh_{fast}_{slow}_{signal}"]
    df["MACD"] = macd
    df["MACD_signal"] = macdsignal
    df["MACD_hist"] = macdhist
    return df

# -----------------------------
# Additional Indicators
# -----------------------------


def compute_bollinger_bands(df: pd.DataFrame, col: str = "Close", period: int = 20, nb_std: int = 2) -> pd.DataFrame:
    """Function description."""
    df = df.copy()
    if _HAS_TALIB:
        upper, middle, lower = talib.BBANDS(
            df[col], timeperiod=period, nbdevup=nb_std, nbdevdn=nb_std, matype=0)
        df["BB_upper"] = upper
        df["BB_middle"] = middle
        df["BB_lower"] = lower
    else:
        bb = pta.bbands(df[col], length=period, std=nb_std)
        df = pd.concat([df, bb], axis=1)
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Function description."""
    df = df.copy()
    if _HAS_TALIB:
        df[f"ATR_{period}"] = talib.ATR(
            df["High"], df["Low"], df["Close"], timeperiod=period)
    else:
        df[f"ATR_{period}"] = pta.atr(
            df["High"], df["Low"], df["Close"], length=period)
    return df


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Function description."""
    df = df.copy()
    if _HAS_TALIB:
        slowk, slowd = talib.STOCH(df["High"], df["Low"], df["Close"], fastk_period=k_period,
                                   slowk_period=d_period, slowk_matype=0, slowd_period=d_period, slowd_matype=0)
        df["STOCH_K"] = slowk
        df["STOCH_D"] = slowd
    else:
        stoch = pta.stoch(df["High"], df["Low"],
                          df["Close"], k=k_period, d=d_period)
        df = pd.concat([df, stoch], axis=1)
    return df

# -----------------------------
# All-in-one indicator function
# -----------------------------


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Function description."""
    df = compute_sma(df)
    df = compute_ema(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)
    df = compute_atr(df)
    df = compute_stochastic(df)
    return df


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    df = pd.DataFrame({
        "Open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "High": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "Low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        "Close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "Volume": [1000, 1200, 1300, 1100, 1150, 1400, 1350, 1500, 1450, 1600]
    })
    df_ind = compute_all_indicators(df)
    print(df_ind.tail(5))
