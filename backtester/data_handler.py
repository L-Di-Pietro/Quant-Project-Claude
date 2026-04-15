"""
data_handler.py — Historical Data Handler with strict no-look-ahead enforcement.

The DataHandler is responsible for feeding market data into the event loop
one bar at a time, ensuring no future data is ever accessible to the strategy.
"""

from __future__ import annotations

import os
import datetime as dt
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtester.events import MarketEvent


class DataHandler(ABC):
    """
    Abstract base class for all data handlers.

    A DataHandler must:
    1. Provide ``symbol_list`` — the tickers it manages.
    2. Yield data chronologically via ``update_bars()``.
    3. Never expose data past the current simulation timestamp.
    """

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> pd.Series:
        """Return the most recent bar for *symbol*."""
        raise NotImplementedError

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Return the last *n* bars for *symbol*."""
        raise NotImplementedError

    @abstractmethod
    def get_latest_bar_datetime(self, symbol: str) -> dt.datetime:
        """Return the datetime of the latest bar."""
        raise NotImplementedError

    @abstractmethod
    def get_latest_bar_value(self, symbol: str, val_type: str) -> float:
        """Return a single field (e.g. ``'Adj Close'``) for the latest bar."""
        raise NotImplementedError

    @abstractmethod
    def get_latest_bars_values(
        self, symbol: str, val_type: str, n: int = 1
    ) -> np.ndarray:
        """Return the last *n* values of *val_type* as a numpy array."""
        raise NotImplementedError

    @abstractmethod
    def update_bars(self) -> Optional[MarketEvent]:
        """Advance by one bar for every symbol.  Return a MarketEvent or None
        if the data is exhausted."""
        raise NotImplementedError

    @property
    @abstractmethod
    def continue_backtest(self) -> bool:
        """``True`` while there is still data to process."""
        raise NotImplementedError


class HistoricParquetDataHandler(DataHandler):
    """
    Reads historical OHLCV data from Parquet files and replays it
    bar-by-bar into the event loop.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing ``<SYMBOL>.parquet`` files.
    symbol_list : list[str]
        Tickers to load.
    start_date : str or None
        Optional start date filter (ISO format).
    end_date : str or None
        Optional end date filter (ISO format).
    """

    def __init__(
        self,
        data_dir: str,
        symbol_list: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        self.data_dir = data_dir
        self.symbol_list = symbol_list
        self.start_date = start_date
        self.end_date = end_date

        # Internal state
        self._symbol_data: Dict[str, pd.DataFrame] = {}
        self._latest_symbol_data: Dict[str, List[pd.Series]] = {}
        self._generators: Dict[str, any] = {}
        self._continue_backtest = True

        self._load_data()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        """Read parquet files, align indices, and create row generators."""
        combined_index: Optional[pd.DatetimeIndex] = None

        for symbol in self.symbol_list:
            path = os.path.join(self.data_dir, f"{symbol}.parquet")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Data file not found: {path}. "
                    f"Run the download_data.py script first."
                )
            df = pd.read_parquet(path)

            # Ensure a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)
                else:
                    df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Date filtering
            if self.start_date:
                df = df.loc[self.start_date:]
            if self.end_date:
                df = df.loc[:self.end_date]

            self._symbol_data[symbol] = df
            self._latest_symbol_data[symbol] = []

            if combined_index is None:
                combined_index = df.index
            else:
                combined_index = combined_index.union(df.index)

        # Re-index each symbol to the union of dates — forward fill gaps
        for symbol in self.symbol_list:
            self._symbol_data[symbol] = (
                self._symbol_data[symbol]
                .reindex(combined_index, method="ffill")
                .dropna()
            )
            self._generators[symbol] = self._symbol_data[symbol].iterrows()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    @property
    def continue_backtest(self) -> bool:
        return self._continue_backtest

    def get_latest_bar(self, symbol: str) -> pd.Series:
        """Return the most recent bar for *symbol*."""
        bars = self._latest_symbol_data[symbol]
        if not bars:
            raise ValueError(f"No bars available yet for {symbol}")
        return bars[-1]

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Return the last *n* bars as a DataFrame."""
        bars = self._latest_symbol_data[symbol]
        if not bars:
            raise ValueError(f"No bars available yet for {symbol}")
        return pd.DataFrame(bars[-n:])

    def get_latest_bar_datetime(self, symbol: str) -> dt.datetime:
        """Return the datetime of the last-delivered bar."""
        bar = self.get_latest_bar(symbol)
        return bar.name  # The index label of the Series

    def get_latest_bar_value(self, symbol: str, val_type: str) -> float:
        """Return a single field for the latest bar."""
        bar = self.get_latest_bar(symbol)
        return float(bar[val_type])

    def get_latest_bars_values(
        self, symbol: str, val_type: str, n: int = 1
    ) -> np.ndarray:
        """Return the last *n* values of a given field as a numpy array."""
        bars_df = self.get_latest_bars(symbol, n)
        return bars_df[val_type].values

    def update_bars(self) -> Optional[MarketEvent]:
        """
        Advance every symbol by one bar.  If any symbol's data is exhausted,
        the backtest is flagged to stop.

        Returns
        -------
        MarketEvent or None
        """
        for symbol in self.symbol_list:
            try:
                idx, row = next(self._generators[symbol])
                row.name = idx  # Preserve the datetime index
                self._latest_symbol_data[symbol].append(row)
            except StopIteration:
                self._continue_backtest = False
                return None
        return MarketEvent()
