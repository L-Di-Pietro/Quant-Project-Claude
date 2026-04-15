"""
ma_crossover.py — Moving Average Crossover Strategy.

A classic trend-following strategy:
    * Go LONG when the short-period moving average crosses *above* the
      long-period moving average.
    * EXIT the position when the short MA crosses *below* the long MA.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from backtester.events import MarketEvent, SignalEvent, SignalDirection
from backtester.strategy import Strategy


class MovingAverageCrossoverStrategy(Strategy):
    """
    Parameters
    ----------
    data_handler : DataHandler
        Reference to the data handler.
    symbol : str
        The single symbol to trade.
    short_window : int
        Look-back for the fast moving average (default 50).
    long_window : int
        Look-back for the slow moving average (default 200).
    """

    def __init__(
        self,
        data_handler,
        symbol: str = "AAPL",
        short_window: int = 50,
        long_window: int = 200,
    ) -> None:
        self.data_handler = data_handler
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self._in_position = False

    def calculate_signals(self, event: MarketEvent) -> Optional[SignalEvent]:
        """
        Emit a LONG signal when the fast MA crosses above the slow MA,
        and an EXIT signal on the reverse crossover.
        """
        bars = self.data_handler.get_latest_bars_values(
            self.symbol, "Adj Close", n=self.long_window + 1
        )
        if len(bars) < self.long_window:
            return None  # Not enough data yet

        short_ma = np.mean(bars[-self.short_window:])
        long_ma = np.mean(bars[-self.long_window:])
        prev_short = np.mean(bars[-self.short_window - 1 : -1])
        prev_long = np.mean(bars[-self.long_window - 1 : -1])

        dt_stamp = self.data_handler.get_latest_bar_datetime(self.symbol)

        # Crossover: short crosses above long
        if short_ma > long_ma and prev_short <= prev_long and not self._in_position:
            self._in_position = True
            return SignalEvent(
                symbol=self.symbol,
                datetime=dt_stamp,
                direction=SignalDirection.LONG,
                strength=1.0,
            )
        # Crossunder: short crosses below long
        elif short_ma < long_ma and prev_short >= prev_long and self._in_position:
            self._in_position = False
            return SignalEvent(
                symbol=self.symbol,
                datetime=dt_stamp,
                direction=SignalDirection.EXIT,
                strength=1.0,
            )

        return None
