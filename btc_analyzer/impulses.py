"""Детекция вертикальных импульсов роста цены (zigzag), расчёт их скорости
на разных горизонтах и сравнение с фоновым (боковым) рынком."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Impulse:
    start_idx: int
    end_idx: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    start_price: float
    end_price: float

    @property
    def magnitude(self) -> float:
        return self.end_price - self.start_price

    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60.0


def zigzag_pivots(close: pd.Series, threshold: float) -> list[tuple[int, float, str]]:
    """Классический zigzag: возвращает список (позиция, цена, 'H'/'L')
    подтверждённых разворотных точек — цена должна пройти `threshold`
    USDT против текущего тренда, чтобы зафиксировать разворот."""
    prices = close.to_numpy()
    n = len(prices)
    if n == 0:
        return []

    pivots: list[tuple[int, float, str]] = []
    trend = 0  # 0 = ещё не определён, 1 = вверх, -1 = вниз
    extreme_i = 0
    extreme_p = prices[0]

    for i in range(1, n):
        p = prices[i]
        if trend == 0:
            if p - extreme_p >= threshold:
                pivots.append((extreme_i, extreme_p, "L"))
                trend = 1
                extreme_i, extreme_p = i, p
            elif extreme_p - p >= threshold:
                pivots.append((extreme_i, extreme_p, "H"))
                trend = -1
                extreme_i, extreme_p = i, p
            elif p > extreme_p:
                extreme_i, extreme_p = i, p
            elif p < extreme_p:
                extreme_i, extreme_p = i, p
        elif trend == 1:
            if p > extreme_p:
                extreme_i, extreme_p = i, p
            elif extreme_p - p >= threshold:
                pivots.append((extreme_i, extreme_p, "H"))
                trend = -1
                extreme_i, extreme_p = i, p
        else:  # trend == -1
            if p < extreme_p:
                extreme_i, extreme_p = i, p
            elif p - extreme_p >= threshold:
                pivots.append((extreme_i, extreme_p, "L"))
                trend = 1
                extreme_i, extreme_p = i, p

    pivots.append((extreme_i, extreme_p, "H" if trend == 1 else "L" if trend == -1 else "?"))
    return pivots


def detect_up_impulses(df: pd.DataFrame, min_magnitude: float, zigzag_threshold: float | None = None) -> list[Impulse]:
    """Находит восходящие "ноги" zigzag с амплитудой >= min_magnitude USDT.

    zigzag_threshold — минимальный разворот для фильтра шума в самом
    zigzag-алгоритме; по умолчанию min_magnitude / 6 (эмпирически хватает,
    чтобы не терять составные импульсы из-за мелких зубцов).
    """
    if zigzag_threshold is None:
        zigzag_threshold = max(min_magnitude / 6.0, 50.0)

    pivots = zigzag_pivots(df["close"], zigzag_threshold)

    impulses: list[Impulse] = []
    for (i0, p0, t0), (i1, p1, t1) in zip(pivots[:-1], pivots[1:]):
        if t0 == "L" and t1 == "H" and (p1 - p0) >= min_magnitude:
            impulses.append(
                Impulse(
                    start_idx=i0,
                    end_idx=i1,
                    start_time=df["date"].iloc[i0],
                    end_time=df["date"].iloc[i1],
                    start_price=p0,
                    end_price=p1,
                )
            )
    return impulses


def speed_at_horizons(df: pd.DataFrame, start_idx: int, horizons_bars: dict[str, int]) -> dict[str, float]:
    """Изменение цены (USDT) от start_idx через каждый горизонт (в барах),
    если данных достаточно; иначе NaN."""
    close = df["close"]
    n = len(close)
    out = {}
    base = close.iloc[start_idx]
    for label, bars in horizons_bars.items():
        idx = start_idx + bars
        out[label] = float(close.iloc[idx] - base) if idx < n else float("nan")
    return out


def impulses_to_frame(df: pd.DataFrame, impulses: list[Impulse], horizons_bars: dict[str, int]) -> pd.DataFrame:
    rows = []
    for imp in impulses:
        row = {
            "start_time": imp.start_time,
            "end_time": imp.end_time,
            "start_price": imp.start_price,
            "end_price": imp.end_price,
            "magnitude_usdt": imp.magnitude,
            "duration_min": imp.duration_minutes,
            "steepness_usdt_per_min": imp.magnitude / max(imp.duration_minutes, 1e-9),
        }
        row.update({f"speed_{k}_usdt": v for k, v in speed_at_horizons(df, imp.start_idx, horizons_bars).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_speed_stats(df: pd.DataFrame, horizons_bars: dict[str, int]) -> pd.DataFrame:
    """Статистика скорости (изменение цены за горизонт) по ВСЕМ барам
    датасета — характеристика 'обычного' рынка для сравнения с импульсами."""
    close = df["close"]
    rows = []
    for label, bars in horizons_bars.items():
        diff = close.diff(bars).dropna()
        rows.append(
            {
                "horizon": label,
                "mean_abs_usdt": diff.abs().mean(),
                "median_abs_usdt": diff.abs().median(),
                "std_usdt": diff.std(),
                "p95_abs_usdt": diff.abs().quantile(0.95),
                "p99_abs_usdt": diff.abs().quantile(0.99),
                "max_usdt": diff.max(),
            }
        )
    return pd.DataFrame(rows).set_index("horizon")


def find_acceleration_threshold(
    df: pd.DataFrame,
    move_threshold: float,
    speed_bars: int,
    lookahead_bars: int,
    candidate_thresholds: np.ndarray,
    min_occurrences: int = 15,
) -> pd.DataFrame:
    """Для набора кандидатов-порогов скорости (изменение цены за
    speed_bars баров) считает: сколько раз порог был достигнут/превышен,
    и в какой доле случаев в течение следующих lookahead_bars баров цена
    выросла ещё как минимум на move_threshold USDT от текущей точки.
    """
    close = df["close"].to_numpy()
    n = len(close)
    speed = pd.Series(close).diff(speed_bars).to_numpy()

    # Будущий максимум роста в окне lookahead_bars вперёд от каждой точки
    future_gain = np.full(n, np.nan)
    for i in range(n - 1):
        end = min(i + 1 + lookahead_bars, n)
        if end > i + 1:
            future_gain[i] = close[i + 1 : end].max() - close[i]

    base_rate = float(np.nanmean(future_gain >= move_threshold))

    rows = []
    for th in candidate_thresholds:
        mask = speed >= th
        occurrences = int(np.nansum(mask))
        if occurrences == 0:
            continue
        hits = int(np.nansum(mask & (future_gain >= move_threshold)))
        hit_rate = hits / occurrences if occurrences else float("nan")
        rows.append(
            {
                "threshold_usdt": th,
                "occurrences": occurrences,
                "hits": hits,
                "hit_rate": hit_rate,
                "base_rate": base_rate,
                "lift_x": hit_rate / base_rate if base_rate else float("nan"),
            }
        )

    result = pd.DataFrame(rows)
    result = result[result["occurrences"] >= min_occurrences].reset_index(drop=True)
    return result
