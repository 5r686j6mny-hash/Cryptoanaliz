#!/usr/bin/env python3
"""CLI: свечной график с наложенной 'скоростью роста' цены BTC в USDT.

Скорость = изменение цены закрытия (в USDT) за скользящее окно из N баров.
На панель скорости наносятся горизонтальные ориентиры (по умолчанию
+-3000 и +-5000 USDT), чтобы визуально отмечать быстрые движения рынка.

Пример:
    python plot_speed_chart.py --input btcusdt_15m.parquet --days 45 \
        --window-bars 24 --thresholds 3000 5000 --output speed_chart.png
"""
from __future__ import annotations

import argparse

import pandas as pd

from btc_analyzer.data import load_ohlcv
from btc_analyzer.speed_chart import plot_candles_with_speed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Свечной график BTC с наложенной скоростью роста цены (USDT/окно)")
    parser.add_argument("--input", "-i", required=True, help="CSV или Parquet с историческими OHLCV-данными")
    parser.add_argument("--output", "-o", default="speed_chart.png", help="Путь для сохранения графика (PNG)")
    parser.add_argument("--days", type=int, default=45, help="Сколько последних дней показать (0 = весь файл)")
    parser.add_argument("--window-bars", type=int, default=24, help="Окно расчёта скорости, в барах (по умолчанию 24 бара = 6ч при 15m свечах)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[3000.0, 5000.0], help="Уровни-ориентиры для скорости, USDT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_ohlcv(args.input)

    if args.days > 0:
        end = df["date"].max()
        start = end - pd.Timedelta(days=args.days)
        df = df[df["date"] >= start].reset_index(drop=True)

    speed = plot_candles_with_speed(
        df,
        args.output,
        window_bars=args.window_bars,
        thresholds=tuple(args.thresholds),
    )

    print(f"Период: {df['date'].min()} — {df['date'].max()} ({len(df)} баров)")
    print(f"Окно скорости: {args.window_bars} баров, ориентиры: {args.thresholds} USDT")
    print("Статистика скорости (USDT за окно):")
    print(speed.describe())
    print(f"\nГрафик сохранён в {args.output}")


if __name__ == "__main__":
    main()
