"""
Module for computing technical indicators using TA-Lib.
Safely preserves index (date), validates columns,
and avoids unintentional data loss.
"""

import logging
import pandas as pd
import talib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _ensure_columns(df: pd.DataFrame, cols: list[str]):
    """Raise error if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _copy(df: pd.DataFrame) -> pd.DataFrame:
    """Guaranteed shallow copy that preserves index."""
    return df.copy()


# ---------------------------------------------------------
# Core Indicators
# ---------------------------------------------------------
def compute_sma(df: pd.DataFrame, col="Close", period=20) -> pd.DataFrame:
    _ensure_columns(df, [col])
    out = _copy(df)
    out[f"SMA_{period}"] = talib.SMA(out[col], timeperiod=period)
    return out


def compute_ema(df: pd.DataFrame, col="Close", period=20) -> pd.DataFrame:
    _ensure_columns(df, [col])
    out = _copy(df)
    out[f"EMA_{period}"] = talib.EMA(out[col], timeperiod=period)
    return out


def compute_rsi(df: pd.DataFrame, col="Close", period=14) -> pd.DataFrame:
    _ensure_columns(df, [col])
    out = _copy(df)
    out[f"RSI_{period}"] = talib.RSI(out[col], timeperiod=period)
    return out


def compute_macd(df: pd.DataFrame, col="Close", fast=12, slow=26, signal=9) -> pd.DataFrame:
    _ensure_columns(df, [col])
    out = _copy(df)
    macd, macdsignal, macdhist = talib.MACD(
        out[col], fastperiod=fast, slowperiod=slow, signalperiod=signal
    )
    out[f"MACD_{fast}_{slow}_{signal}"] = macd
    out[f"MACD_signal_{fast}_{slow}_{signal}"] = macdsignal
    out[f"MACD_hist_{fast}_{slow}_{signal}"] = macdhist
    return out


# ---------------------------------------------------------
# Additional Indicators
# ---------------------------------------------------------
def compute_bollinger_bands(df: pd.DataFrame, col="Close", period=20, nb_std=2) -> pd.DataFrame:
    _ensure_columns(df, [col])
    out = _copy(df)
    upper, middle, lower = talib.BBANDS(
        out[col], timeperiod=period, nbdevup=nb_std, nbdevdn=nb_std, matype=0
    )
    out["BB_upper"] = upper
    out["BB_middle"] = middle
    out["BB_lower"] = lower
    return out


def compute_atr(df: pd.DataFrame, period=14) -> pd.DataFrame:
    _ensure_columns(df, ["High", "Low", "Close"])
    out = _copy(df)
    out[f"ATR_{period}"] = talib.ATR(
        out["High"], out["Low"], out["Close"], timeperiod=period
    )
    return out


def compute_stochastic(df: pd.DataFrame, k_period=14, d_period=3) -> pd.DataFrame:
    _ensure_columns(df, ["High", "Low", "Close"])
    out = _copy(df)
    slowk, slowd = talib.STOCH(
        out["High"], out["Low"], out["Close"],
        fastk_period=k_period,
        slowk_period=d_period, slowk_matype=0,
        slowd_period=d_period, slowd_matype=0
    )
    out["STOCH_K"] = slowk
    out["STOCH_D"] = slowd
    return out


# ---------------------------------------------------------
# All Indicators At Once (Safe)
# ---------------------------------------------------------
def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a full set of TA-Lib indicators.
    Index (usually 'date') is preserved exactly.
    No reset_index() so no column loss.
    """
    out = _copy(df)
    n_rows = len(out)

    # dynamic period adjustment to avoid warnings
    sma_period = min(20, n_rows)
    ema_period = min(20, n_rows)
    rsi_period = min(14, n_rows)
    macd_fast = min(12, n_rows)
    macd_slow = min(26, n_rows)
    macd_signal = min(9, n_rows)
    bb_period = min(20, n_rows)
    atr_period = min(14, n_rows)
    stoch_k = min(14, n_rows)
    stoch_d = min(3, n_rows)

    out = compute_sma(out, period=sma_period)
    out = compute_ema(out, period=ema_period)
    out = compute_rsi(out, period=rsi_period)
    out = compute_macd(out, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    out = compute_bollinger_bands(out, period=bb_period)
    out = compute_atr(out, period=atr_period)
    out = compute_stochastic(out, k_period=stoch_k, d_period=stoch_d)

    # Cleanup without deleting index
    out = out.dropna(how="any")   # preserves date index

    return out


# ---------------------------------------------------------
# Quick test
# ---------------------------------------------------------
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
    print(df_ind.tail())
