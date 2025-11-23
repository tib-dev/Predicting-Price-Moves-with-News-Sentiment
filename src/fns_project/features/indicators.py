"""Module description."""

# fns_project/features/indicators.py
from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


# Attempt to import TA-Lib, fallback to pandas_ta
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
            "Install TA-Lib (`pip install TA-Lib`) or pandas_ta (`pip install pandas_ta`) to compute indicators.")


def compute_sma(df: pd.DataFrame, col: str = "Close", period: int = 20, out_col: Optional[str] = None) -> pd.DataFrame:
    """Compute simple moving average."""
    out_col = out_col or f"SMA_{period}"
    df = df.copy()
    if _HAS_TALIB:
        df[out_col] = talib.SMA(df[col], timeperiod=period)
    else:
        df[out_col] = pta.sma(df[col], length=period)
    return df


def compute_ema(df: pd.DataFrame, col: str = "Close", period: int = 20, out_col: Optional[str] = None) -> pd.DataFrame:
    """Compute exponential moving average."""
    out_col = out_col or f"EMA_{period}"
    df = df.copy()
    if _HAS_TALIB:
        df[out_col] = talib.EMA(df[col], timeperiod=period)
    else:
        df[out_col] = pta.ema(df[col], length=period)
    return df


def compute_rsi(df: pd.DataFrame, col: str = "Close", period: int = 14, out_col: Optional[str] = None) -> pd.DataFrame:
    """Compute Relative Strength Index (RSI)."""
    out_col = out_col or f"RSI_{period}"
    df = df.copy()
    if _HAS_TALIB:
        df[out_col] = talib.RSI(df[col], timeperiod=period)
    else:
        df[out_col] = pta.rsi(df[col], length=period)
    return df


def compute_macd(
    df: pd.DataFrame, col: str = "Close",
    fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Compute MACD, returns original df with three new columns:
      MACD, MACD_signal, MACD_hist
    """
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


def compute_all_indicators(
    df: pd.DataFrame, close_col: str = "Close"
) -> pd.DataFrame:
    """
    Compute SMA20, EMA20, RSI14, MACD for a DataFrame.
    """
    df = compute_sma(df, col=close_col, period=20)
    df = compute_ema(df, col=close_col, period=20)
    df = compute_rsi(df, col=close_col, period=14)
    df = compute_macd(df, col=close_col)
    return df


if __name__ == "__main__":
    # quick test
    df = pd.DataFrame({
        "Close": [100, 102, 101, 103, 105, 104, 106, 107, 108, 110, 109, 111, 112, 113, 114]
    })
    df_ind = compute_all_indicators(df)
    print(df_ind.tail(5))
