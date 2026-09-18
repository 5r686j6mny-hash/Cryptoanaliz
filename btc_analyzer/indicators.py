"""Расчёт технических индикаторов для анализа BTC."""
from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    result = 100 - (100 / (1 + rs))
    return result.fillna(100).where(avg_loss != 0, 100).where(avg_gain != 0, 0).astype(float)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram})


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower})


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к df колонки со всеми индикаторами, используемыми в анализе."""
    out = df.copy()

    out["sma_20"] = sma(out["close"], 20)
    out["sma_50"] = sma(out["close"], 50)
    out["sma_200"] = sma(out["close"], 200)

    out["ema_12"] = ema(out["close"], 12)
    out["ema_26"] = ema(out["close"], 26)
    out["ema_50"] = ema(out["close"], 50)

    out["rsi_14"] = rsi(out["close"], 14)

    macd_df = macd(out["close"])
    out = pd.concat([out, macd_df], axis=1)

    bb_df = bollinger_bands(out["close"])
    out = pd.concat([out, bb_df], axis=1)

    out["atr_14"] = atr(out)
    out["volume_sma_20"] = sma(out["volume"], 20)

    return out
