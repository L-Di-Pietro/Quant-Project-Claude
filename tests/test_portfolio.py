"""
test_portfolio.py — Unit tests for the NaivePortfolio.
"""

import os
import numpy as np
import pandas as pd
import pytest

from backtester.data_handler import HistoricParquetDataHandler
from backtester.events import (
    FillEvent,
    OrderDirection,
    SignalDirection,
    SignalEvent,
)
from backtester.portfolio import NaivePortfolio


@pytest.fixture
def data_handler(tmp_path):
    """Create a data handler with sample data."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(50) * 0.5)
    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, 50),
        },
        index=dates,
    )
    df.index.name = "Date"
    df.to_parquet(os.path.join(str(tmp_path), "TEST.parquet"))
    dh = HistoricParquetDataHandler(
        data_dir=str(tmp_path), symbol_list=["TEST"]
    )
    # Advance a few bars so data is available
    for _ in range(5):
        dh.update_bars()
    return dh


class TestNaivePortfolio:
    def test_initial_state(self, data_handler):
        port = NaivePortfolio(data_handler, initial_capital=100_000)
        assert port.current_holdings["cash"] == 100_000
        assert port.current_holdings["total"] == 100_000
        assert port.current_positions["TEST"] == 0

    def test_generate_long_order(self, data_handler):
        port = NaivePortfolio(data_handler)
        signal = SignalEvent(symbol="TEST", direction=SignalDirection.LONG)
        order = port.generate_order(signal)
        assert order is not None
        assert order.direction == OrderDirection.BUY
        assert order.quantity == 100

    def test_generate_exit_order_when_flat(self, data_handler):
        port = NaivePortfolio(data_handler)
        signal = SignalEvent(symbol="TEST", direction=SignalDirection.EXIT)
        order = port.generate_order(signal)
        assert order is None  # Can't exit if flat

    def test_update_fill(self, data_handler):
        port = NaivePortfolio(data_handler, initial_capital=100_000)
        fill = FillEvent(
            symbol="TEST",
            quantity=100,
            direction=OrderDirection.BUY,
            fill_cost=10_000.0,
            commission=1.0,
        )
        port.update_fill(fill)
        assert port.current_positions["TEST"] == 100
        assert port.current_holdings["cash"] < 100_000

    def test_equity_curve(self, data_handler):
        port = NaivePortfolio(data_handler)
        port.update_timeindex()
        ec = port.get_equity_curve()
        assert not ec.empty
        assert "total" in ec.columns
