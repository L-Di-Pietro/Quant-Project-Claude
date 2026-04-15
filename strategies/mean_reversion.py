"""
mean_reversion.py — Bollinger-Band Mean Reversion Strategy.

A mean-reversion strategy based on Bollinger Bands:
    * Go LONG when the price drops *below* the lower Bollinger Band
      (oversold condition).
    * EXIT the position when the price rises *above* the upper Bollinger
      Band (overbought, take profit) or when it reverts to the mean.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from backtester.events import MarketEvent, SignalEvent, SignalDirection
from backtester.strategy import Strategy


class MeanReversionBollingerStrategy(Strategy):
    """
    Parameters
    ----------
    data_handler : DataHandler
        Reference to the data handler.
    symbol : str
        The single symbol to trade.
    lookback : int
        Window for the moving average and standard deviation (default 20).
    num_std : float
        Number of standard deviations for the bands (default 2.0).
    """

    def __init__(
        self,
        data_handler,
        symbol: str = "AAPL",
        lookback: int = 20,
        num_std: float = 2.0,
    ) -> None:
        self.data_handler = data_handler
        self.symbol = symbol
        self.lookback = lookback
        self.num_std = num_std
        self._in_position = False

    def calculate_signals(self, event: MarketEvent) -> Optional[SignalEvent]:
        """
        Emit a LONG when price < lower band, EXIT when price > upper band
        or reverts to the mean.
        """
        bars = self.data_handler.get_latest_bars_values(
            self.symbol, "Adj Close", n=self.lookback + 1
        )
        if len(bars) < self.lookback:
            return None  # Not enough data

        window = bars[-self.lookback:]
        mean = np.mean(window)
        std = np.std(window, ddof=1)

        upper_band = mean + self.num_std * std
        lower_band = mean - self.num_std * std

        current_price = bars[-1]
        dt_stamp = self.data_handler.get_latest_bar_datetime(self.symbol)

        # Entry: price below lower band → oversold → go LONG
        if current_price < lower_band and not self._in_position:
            self._in_position = True
            return SignalEvent(
                symbol=self.symbol,
                datetime=dt_stamp,
                direction=SignalDirection.LONG,
                strength=1.0,
            )
        # Exit: price above upper band OR reverts to mean
        elif current_price > upper_band and self._in_position:
            self._in_position = False
            return SignalEvent(
                symbol=self.symbol,
                datetime=dt_stamp,
                direction=SignalDirection.EXIT,
                strength=1.0,
            )

        return None
