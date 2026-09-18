"""Формирование бычьих/медвежьих сигналов на основе индикаторов."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Signal:
    name: str
    direction: str  # "bull", "bear" или "neutral"
    detail: str


def _cross_signal(prev_fast, prev_slow, cur_fast, cur_slow, bull_name, bear_name) -> Signal | None:
    if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(cur_fast) or pd.isna(cur_slow):
        return None
    if prev_fast <= prev_slow and cur_fast > cur_slow:
        return Signal(bull_name, "bull", "быстрая линия пересекла медленную снизу вверх")
    if prev_fast >= prev_slow and cur_fast < cur_slow:
        return Signal(bear_name, "bear", "быстрая линия пересекла медленную сверху вниз")
    return None


def evaluate_latest(df: pd.DataFrame) -> list[Signal]:
    """Возвращает список сигналов, актуальных на последней строке df."""
    if len(df) < 2:
        return []

    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals: list[Signal] = []

    # Golden Cross / Death Cross (SMA50 vs SMA200)
    cross = _cross_signal(prev["sma_50"], prev["sma_200"], last["sma_50"], last["sma_200"], "Golden Cross (SMA50/SMA200)", "Death Cross (SMA50/SMA200)")
    if cross:
        signals.append(cross)

    # MACD crossover
    cross = _cross_signal(prev["macd"], prev["macd_signal"], last["macd"], last["macd_signal"], "MACD бычье пересечение", "MACD медвежье пересечение")
    if cross:
        signals.append(cross)

    # RSI перекупленность/перепроданность
    rsi_val = last["rsi_14"]
    if pd.notna(rsi_val):
        if rsi_val >= 70:
            signals.append(Signal("RSI перекуплен", "bear", f"RSI={rsi_val:.1f} >= 70, возможна коррекция"))
        elif rsi_val <= 30:
            signals.append(Signal("RSI перепродан", "bull", f"RSI={rsi_val:.1f} <= 30, возможен отскок"))

    # Цена относительно полос Боллинджера
    close, bb_upper, bb_lower = last["close"], last["bb_upper"], last["bb_lower"]
    if pd.notna(bb_upper) and close >= bb_upper:
        signals.append(Signal("Пробой верхней полосы Боллинджера", "bear", "цена у/выше верхней полосы, риск отката"))
    elif pd.notna(bb_lower) and close <= bb_lower:
        signals.append(Signal("Пробой нижней полосы Боллинджера", "bull", "цена у/ниже нижней полосы, риск отскока"))

    # Тренд по расположению цены относительно SMA200
    sma200 = last["sma_200"]
    if pd.notna(sma200):
        if close > sma200:
            signals.append(Signal("Цена выше SMA200", "bull", "долгосрочный тренд восходящий"))
        else:
            signals.append(Signal("Цена ниже SMA200", "bear", "долгосрочный тренд нисходящий"))

    # Всплеск объёма
    vol, vol_sma = last["volume"], last["volume_sma_20"]
    if pd.notna(vol_sma) and vol_sma > 0 and vol >= 2 * vol_sma:
        direction = "bull" if last["close"] >= prev["close"] else "bear"
        signals.append(Signal("Аномальный объём", direction, f"объём в {vol / vol_sma:.1f}x выше среднего за 20 периодов"))

    return signals


def overall_sentiment(signals: list[Signal]) -> tuple[str, float]:
    """Считает общий скор настроения рынка на основе списка сигналов."""
    if not signals:
        return "нейтрально", 0.0

    score = 0
    for s in signals:
        if s.direction == "bull":
            score += 1
        elif s.direction == "bear":
            score -= 1

    normalized = score / len(signals)
    if normalized >= 0.3:
        label = "бычий рынок"
    elif normalized <= -0.3:
        label = "медвежий рынок"
    else:
        label = "нейтрально / смешанные сигналы"
    return label, normalized
