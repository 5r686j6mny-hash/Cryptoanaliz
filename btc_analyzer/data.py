"""Загрузка и нормализация OHLCV-данных из CSV-файла."""
from __future__ import annotations

import pandas as pd

# Возможные варианты названий колонок в исходном CSV (без учёта регистра)
COLUMN_ALIASES = {
    "date": ["date", "time", "timestamp", "datetime", "open_time"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "price", "close_price", "adj_close"],
    "volume": ["volume", "vol", "volume_btc", "volumefrom", "volumeto"],
}


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def load_ohlcv(path: str) -> pd.DataFrame:
    """Читает CSV с историческими данными и приводит колонки к единому виду.

    Ожидаются (в любом регистре и порядке) колонки даты, open, high, low,
    close и, желательно, volume. Отсутствующий volume заполняется нулями,
    отсутствующий high/low берётся из close.
    """
    df = pd.read_csv(path)
    columns = list(df.columns)

    resolved = {}
    for target, aliases in COLUMN_ALIASES.items():
        found = _find_column(columns, aliases)
        if found is not None:
            resolved[target] = found

    if "date" not in resolved or "close" not in resolved:
        raise ValueError(
            "Не удалось найти обязательные колонки 'date' и 'close' в CSV. "
            f"Найденные колонки: {columns}"
        )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[resolved["date"]], utc=False, errors="coerce")
    out["close"] = pd.to_numeric(df[resolved["close"]], errors="coerce")
    out["open"] = pd.to_numeric(df[resolved.get("open", resolved["close"])], errors="coerce") if "open" in resolved else out["close"]
    out["high"] = pd.to_numeric(df[resolved.get("high", resolved["close"])], errors="coerce") if "high" in resolved else out[["open", "close"]].max(axis=1)
    out["low"] = pd.to_numeric(df[resolved.get("low", resolved["close"])], errors="coerce") if "low" in resolved else out[["open", "close"]].min(axis=1)
    out["volume"] = pd.to_numeric(df[resolved["volume"]], errors="coerce") if "volume" in resolved else 0.0

    out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("После очистки данных не осталось ни одной валидной строки.")
    return out
