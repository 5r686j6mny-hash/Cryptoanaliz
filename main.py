#!/usr/bin/env python3
"""CLI для анализа исторических данных BTC: индикаторы + бычьи/медвежьи сигналы.

Пример:
    python main.py --input data.csv --output result.csv --plot chart.png
"""
from __future__ import annotations

import argparse

from btc_analyzer.data import load_ohlcv
from btc_analyzer.indicators import add_all_indicators
from btc_analyzer.report import plot_chart, print_summary
from btc_analyzer.signals import evaluate_latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Анализ BTC по CSV с историческими данными (OHLCV)")
    parser.add_argument("--input", "-i", required=True, help="Путь к CSV-файлу с историческими данными")
    parser.add_argument("--output", "-o", default=None, help="Путь для сохранения CSV с рассчитанными индикаторами")
    parser.add_argument("--plot", "-p", default=None, help="Путь для сохранения графика (PNG)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_ohlcv(args.input)
    df = add_all_indicators(df)

    signals = evaluate_latest(df)
    print_summary(df, signals)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nИндикаторы сохранены в {args.output}")

    if args.plot:
        plot_chart(df, args.plot)
        print(f"График сохранён в {args.plot}")


if __name__ == "__main__":
    main()
