"""Расчёт 'скорости роста' цены — изменения цены в USDT за фиксированное
скользящее окно времени (сколько долларов цена прошла за N баров)."""
from __future__ import annotations

import pandas as pd


def rolling_price_speed(close: pd.Series, window_bars: int) -> pd.Series:
    """Изменение цены в USDT между текущим баром и баром `window_bars` назад.

    Положительное значение — рост на столько-то USDT за окно, отрицательное —
    падение. Это и есть 'скорость' роста/падения цены в единицах USDT/окно.
    """
    return close - close.shift(window_bars)
