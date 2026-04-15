"""
test_data_handler.py — Unit tests for the HistoricParquetDataHandler.

Creates a temporary parquet file, loads it, and verifies that
no look-ahead bias occurs.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from backtester.data_handler import HistoricParquetDataHandler


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a minimal parquet file for testing."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, 100),
        },
        index=dates,
    )
    df.index.name = "Date"
    path = os.path.join(str(tmp_path), "TEST.parquet")
    df.to_parquet(path)
    return str(tmp_path)


class TestHistoricParquetDataHandler:
    def test_load_and_update(self, sample_data_dir):
        dh = HistoricParquetDataHandler(
            data_dir=sample_data_dir, symbol_list=["TEST"]
        )
        assert dh.continue_backtest is True

        # First bar
        event = dh.update_bars()
        assert event is not None
        bar = dh.get_latest_bar("TEST")
        assert "Adj Close" in bar.index

    def test_no_look_ahead(self, sample_data_dir):
        """Only one bar should be visible after one update."""
        dh = HistoricParquetDataHandler(
            data_dir=sample_data_dir, symbol_list=["TEST"]
        )
        dh.update_bars()
        bars = dh.get_latest_bars("TEST", n=999)
        assert len(bars) == 1  # Only the first bar

    def test_exhaustion(self, sample_data_dir):
        dh = HistoricParquetDataHandler(
            data_dir=sample_data_dir, symbol_list=["TEST"]
        )
        count = 0
        while dh.continue_backtest:
            ev = dh.update_bars()
            if ev is None:
                break
            count += 1
        assert count == 100

    def test_latest_bar_value(self, sample_data_dir):
        dh = HistoricParquetDataHandler(
            data_dir=sample_data_dir, symbol_list=["TEST"]
        )
        dh.update_bars()
        val = dh.get_latest_bar_value("TEST", "Adj Close")
        assert isinstance(val, float)

    def test_latest_bars_values(self, sample_data_dir):
        dh = HistoricParquetDataHandler(
            data_dir=sample_data_dir, symbol_list=["TEST"]
        )
        for _ in range(10):
            dh.update_bars()
        vals = dh.get_latest_bars_values("TEST", "Adj Close", n=5)
        assert len(vals) == 5
