"""Свечной график (mplfinance) с наложенной 'скоростью роста' цены —
изменением цены в USDT за фиксированное скользящее окно, с ориентирами
на уровнях 3000 и 5000 USDT."""
from __future__ import annotations

import mplfinance as mpf
import pandas as pd

from .velocity import rolling_price_speed


def plot_candles_with_speed(
    df: pd.DataFrame,
    output_path: str,
    window_bars: int,
    thresholds: tuple[float, ...] = (3000.0, 5000.0),
    title: str = "BTCUSDT — 15m свечи и скорость роста цены",
) -> pd.Series:
    """Строит свечной график df (колонки date/open/high/low/close/volume) с
    отдельной панелью 'скорости' под ним и сохраняет в output_path.

    Возвращает рассчитанный ряд скорости (USDT за window_bars баров).
    """
    ohlc = df.set_index("date")[["open", "high", "low", "close", "volume"]]

    speed = rolling_price_speed(df["close"], window_bars)
    speed.index = ohlc.index

    plots = [
        mpf.make_addplot(
            speed,
            panel=1,
            color="dodgerblue",
            ylabel=f"Скорость, USDT/{window_bars} баров",
            width=1.2,
        )
    ]
    for t in thresholds:
        for level, color in ((t, "green"), (-t, "red")):
            level_series = pd.Series(level, index=ohlc.index)
            plots.append(
                mpf.make_addplot(level_series, panel=1, color=color, linestyle="--", width=0.8, secondary_y=False)
            )

    mpf.plot(
        ohlc,
        type="candle",
        style="yahoo",
        addplot=plots,
        panel_ratios=(3, 1.3),
        volume=False,
        title=title,
        ylabel="Цена, USDT",
        figsize=(16, 9),
        warn_too_much_data=len(ohlc) + 1,
        savefig=dict(fname=output_path, dpi=150),
    )

    return speed
