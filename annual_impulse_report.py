#!/usr/bin/env python3
"""Годовой отчёт по BTCUSDT: детальный график цены за год с наложенной
скоростью роста (по 15m барам), разметка восходящих импульсов >= порога,
расчёт скорости импульсов на нескольких горизонтах, сравнение с фоновым
рынком, поиск порога-предвестника и увеличенные графики самых заметных
эпизодов.

Пример:
    python annual_impulse_report.py --input btcusdt_15m.parquet \
        --days 365 --min-impulse 3000 --out-dir report_out --top-n 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from btc_analyzer.data import load_ohlcv
from btc_analyzer.impulses import (
    baseline_speed_stats,
    detect_up_impulses,
    find_acceleration_threshold,
    impulses_to_frame,
)
from btc_analyzer.speed_chart import plot_candles_with_speed
from btc_analyzer.velocity import rolling_price_speed

HORIZONS_BARS = {"15m": 1, "30m": 2, "60m": 4, "120m": 8}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Годовой анализ вертикальных импульсов BTCUSDT")
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--min-impulse", type=float, default=3000.0)
    p.add_argument("--overview-window-bars", type=int, default=4, help="Окно (в барах) для красной линии скорости на общем графике")
    p.add_argument("--out-dir", default="impulse_report")
    p.add_argument("--top-n", type=int, default=6, help="Сколько самых заметных импульсов показать крупным планом")
    return p.parse_args()


def make_overview_chart(df: pd.DataFrame, impulses_df: pd.DataFrame, window_bars: int, out_path: str) -> None:
    fig, ax_price = plt.subplots(figsize=(20, 9))
    ax_price.plot(df["date"], df["close"], color="black", linewidth=0.7, label="BTCUSDT close")
    ax_price.set_ylabel("Цена, USDT")
    ax_price.set_title(f"BTCUSDT за год (15m) — цена и скорость роста, импульсы ≥ {impulses_df.attrs.get('min_impulse', ''):,.0f} USDT" if "min_impulse" in impulses_df.attrs else "BTCUSDT за год (15m) — цена и скорость роста")

    speed = rolling_price_speed(df["close"], window_bars)
    ax_speed = ax_price.twinx()
    ax_speed.plot(df["date"], speed, color="red", alpha=0.55, linewidth=0.6, label=f"Скорость, USDT/{window_bars} бар(а)")
    ax_speed.set_ylabel(f"Скорость, USDT / {window_bars * 15} мин", color="red")
    ax_speed.tick_params(axis="y", colors="red")
    ax_speed.axhline(0, color="red", alpha=0.2, linewidth=0.5)

    for _, row in impulses_df.iterrows():
        ax_price.axvspan(row["start_time"], row["end_time"], color="green", alpha=0.15)
        ax_price.scatter([row["start_time"]], [row["start_price"]], color="green", marker="^", s=40, zorder=5)

    ax_price.xaxis.set_major_locator(mdates.MonthLocator())
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    lines1, labels1 = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_speed.get_legend_handles_labels()
    ax_price.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_ohlcv(args.input)
    end = df["date"].max()
    start = end - pd.Timedelta(days=args.days)
    df = df[df["date"] >= start].reset_index(drop=True)
    print(f"Период анализа: {df['date'].min()} — {df['date'].max()} ({len(df)} баров по 15м)")

    # 1-3: детекция импульсов + общий график
    impulses = detect_up_impulses(df, min_magnitude=args.min_impulse)
    print(f"Найдено восходящих импульсов >= {args.min_impulse:.0f} USDT: {len(impulses)}")

    impulses_df = impulses_to_frame(df, impulses, HORIZONS_BARS)
    impulses_df.attrs["min_impulse"] = args.min_impulse
    impulses_csv = out_dir / "impulses.csv"
    impulses_df.to_csv(impulses_csv, index=False)
    print(f"Таблица импульсов сохранена: {impulses_csv}")

    overview_path = out_dir / "annual_overview.png"
    make_overview_chart(df, impulses_df, args.overview_window_bars, str(overview_path))
    print(f"Общий график сохранён: {overview_path}")

    # 5: сравнение скорости импульсов с фоновым рынком
    baseline = baseline_speed_stats(df, HORIZONS_BARS)
    comparison_rows = []
    for label in HORIZONS_BARS:
        col = f"speed_{label}_usdt"
        impulse_mean = impulses_df[col].abs().mean()
        base_mean = baseline.loc[label, "mean_abs_usdt"]
        base_p99 = baseline.loc[label, "p99_abs_usdt"]
        comparison_rows.append(
            {
                "horizon": label,
                "impulse_mean_abs_usdt": impulse_mean,
                "market_mean_abs_usdt": base_mean,
                "market_p99_abs_usdt": base_p99,
                "impulse_vs_market_mean_x": impulse_mean / base_mean if base_mean else float("nan"),
                "impulse_vs_market_p99_x": impulse_mean / base_p99 if base_p99 else float("nan"),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).set_index("horizon")
    comparison_csv = out_dir / "impulse_vs_market.csv"
    comparison_df.to_csv(comparison_csv)
    print("\nСравнение скорости импульсов с фоновым рынком:")
    print(comparison_df.to_string(float_format=lambda x: f"{x:,.1f}"))

    # 6: поиск порога-предвестника
    candidates = np.arange(200, 2600, 100)
    thresh_df = find_acceleration_threshold(
        df,
        move_threshold=args.min_impulse,
        speed_bars=HORIZONS_BARS["30m"],
        lookahead_bars=16,  # 4 часа вперёд
        candidate_thresholds=candidates,
        min_occurrences=15,
    )
    thresh_csv = out_dir / "acceleration_threshold.csv"
    thresh_df.to_csv(thresh_csv, index=False)
    print(f"\nТаблица порогов-предвестников сохранена: {thresh_csv}")
    if not thresh_df.empty:
        base_rate = thresh_df["base_rate"].iloc[0]
        print(f"Безусловная вероятность роста на {args.min_impulse:.0f}+ USDT в течение ближайших 4ч (любой момент): {base_rate*100:.2f}%")
        reliable = thresh_df[thresh_df["occurrences"] >= 50]
        if not reliable.empty:
            best = reliable.loc[reliable["hit_rate"].idxmax()]
            print(
                f"Лучший порог (скорость за 30м, надёжная выборка >=50 набл.): {best['threshold_usdt']:.0f} USDT, "
                f"hit-rate={best['hit_rate']*100:.1f}% ({int(best['hits'])}/{int(best['occurrences'])}), "
                f"лифт x{best['lift_x']:.1f} к базовой вероятности"
            )

    # 7: увеличенные графики топ-N САМЫХ ВЕРТИКАЛЬНЫХ импульсов (по крутизне
    # USDT/мин, а не по абсолютной величине — иначе в топ попадают растянутые
    # многочасовые ралли вместо резких скачков)
    top_impulses = sorted(impulses, key=lambda imp: imp.magnitude / max(imp.duration_minutes, 1e-9), reverse=True)[: args.top_n]
    for i, imp in enumerate(top_impulses, start=1):
        pad = pd.Timedelta(hours=2)
        window = df[(df["date"] >= imp.start_time - pad) & (df["date"] <= imp.end_time + pad)].reset_index(drop=True)
        if len(window) < 5:
            continue
        speed_bars = max(HORIZONS_BARS["30m"], 2)
        fname = out_dir / f"impulse_{i:02d}_{imp.start_time.strftime('%Y%m%d_%H%M')}.png"
        plot_candles_with_speed(
            window,
            str(fname),
            window_bars=speed_bars,
            thresholds=(args.min_impulse,),
            title=f"Импульс #{i}: {imp.start_time:%Y-%m-%d %H:%M} → {imp.end_time:%H:%M}, +{imp.magnitude:,.0f} USDT за {imp.duration_minutes:.0f} мин",
        )
        print(f"Увеличенный график импульса #{i} сохранён: {fname}")


if __name__ == "__main__":
    main()
