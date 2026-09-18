"""Вывод отчёта в консоль и построение графика."""
from __future__ import annotations

import pandas as pd

from .signals import Signal, overall_sentiment


def print_summary(df: pd.DataFrame, signals: list[Signal]) -> None:
    last = df.iloc[-1]
    print("=" * 60)
    print(f"Дата последней свечи : {last['date'].date()}")
    print(f"Цена закрытия        : {last['close']:.2f}")
    print(f"SMA20 / SMA50 / SMA200 : {last['sma_20']:.2f} / {last['sma_50']:.2f} / {last['sma_200']:.2f}")
    print(f"RSI(14)               : {last['rsi_14']:.1f}")
    print(f"MACD / Signal / Hist   : {last['macd']:.2f} / {last['macd_signal']:.2f} / {last['macd_hist']:.2f}")
    print(f"Bollinger (нижняя/средняя/верхняя): {last['bb_lower']:.2f} / {last['bb_mid']:.2f} / {last['bb_upper']:.2f}")
    print(f"ATR(14)               : {last['atr_14']:.2f}")
    print("-" * 60)

    if signals:
        print("Обнаруженные сигналы:")
        for s in signals:
            arrow = "▲" if s.direction == "bull" else "▼" if s.direction == "bear" else "•"
            print(f"  {arrow} [{s.direction.upper():7}] {s.name} — {s.detail}")
    else:
        print("Явных сигналов на последней свече не обнаружено.")

    label, score = overall_sentiment(signals)
    print("-" * 60)
    print(f"Итоговое настроение рынка: {label} (score={score:+.2f})")
    print("=" * 60)


def plot_chart(df: pd.DataFrame, output_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

    ax_price, ax_rsi, ax_macd = axes

    ax_price.plot(df["date"], df["close"], label="Close", color="black", linewidth=1)
    ax_price.plot(df["date"], df["sma_50"], label="SMA50", color="orange", linewidth=1)
    ax_price.plot(df["date"], df["sma_200"], label="SMA200", color="blue", linewidth=1)
    ax_price.fill_between(df["date"], df["bb_lower"], df["bb_upper"], color="gray", alpha=0.15, label="Bollinger Bands")
    ax_price.set_title("BTC — цена и скользящие средние")
    ax_price.legend(loc="upper left")
    ax_price.grid(alpha=0.3)

    ax_rsi.plot(df["date"], df["rsi_14"], color="purple", linewidth=1)
    ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.8)
    ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.8)
    ax_rsi.set_ylabel("RSI(14)")
    ax_rsi.grid(alpha=0.3)

    ax_macd.plot(df["date"], df["macd"], label="MACD", color="blue", linewidth=1)
    ax_macd.plot(df["date"], df["macd_signal"], label="Signal", color="red", linewidth=1)
    ax_macd.bar(df["date"], df["macd_hist"], label="Histogram", color="gray", alpha=0.5)
    ax_macd.set_ylabel("MACD")
    ax_macd.legend(loc="upper left")
    ax_macd.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
