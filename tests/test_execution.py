"""
test_execution.py — Unit tests for the SimulatedExecutionHandler.
"""

import os
import numpy as np
import pandas as pd
import pytest

from backtester.data_handler import HistoricParquetDataHandler
from backtester.events import OrderDirection, OrderEvent, OrderType
from backtester.execution_handler import SimulatedExecutionHandler


@pytest.fixture
def data_handler(tmp_path):
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    close = np.array([100.0, 101.0, 102.0, 101.5, 103.0,
                      104.0, 103.5, 105.0, 106.0, 107.0])
    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": [1_000_000] * 10,
        },
        index=dates,
    )
    df.index.name = "Date"
    df.to_parquet(os.path.join(str(tmp_path), "TEST.parquet"))
    dh = HistoricParquetDataHandler(
        data_dir=str(tmp_path), symbol_list=["TEST"]
    )
    dh.update_bars()
    return dh


class TestSimulatedExecutionHandler:
    def test_fixed_commission(self, data_handler):
        handler = SimulatedExecutionHandler(
            commission_type="fixed", commission_value=5.0,
            slippage_type="fixed", slippage_value=0.0,
        )
        order = OrderEvent(
            symbol="TEST", order_type=OrderType.MARKET,
            quantity=100, direction=OrderDirection.BUY,
        )
        fill = handler.execute_order(order, data_handler)
        assert fill is not None
        assert fill.commission == 5.0

    def test_percentage_commission(self, data_handler):
        handler = SimulatedExecutionHandler(
            commission_type="percentage", commission_value=0.001,
            slippage_type="fixed", slippage_value=0.0,
        )
        order = OrderEvent(
            symbol="TEST", order_type=OrderType.MARKET,
            quantity=100, direction=OrderDirection.BUY,
        )
        fill = handler.execute_order(order, data_handler)
        assert fill is not None
        expected_commission = abs(fill.fill_cost) * 0.001
        assert abs(fill.commission - expected_commission) < 0.01

    def test_slippage_buy(self, data_handler):
        handler = SimulatedExecutionHandler(
            commission_type="fixed", commission_value=0.0,
            slippage_type="spread_pct", slippage_value=0.1,
        )
        order = OrderEvent(
            symbol="TEST", order_type=OrderType.MARKET,
            quantity=100, direction=OrderDirection.BUY,
        )
        fill = handler.execute_order(order, data_handler)
        assert fill is not None
        # Buy-side slippage should increase fill cost
        bar = data_handler.get_latest_bar("TEST")
        raw_cost = float(bar["Adj Close"]) * 100
        assert fill.fill_cost > raw_cost

    def test_slippage_sell(self, data_handler):
        handler = SimulatedExecutionHandler(
            commission_type="fixed", commission_value=0.0,
            slippage_type="spread_pct", slippage_value=0.1,
        )
        order = OrderEvent(
            symbol="TEST", order_type=OrderType.MARKET,
            quantity=100, direction=OrderDirection.SELL,
        )
        fill = handler.execute_order(order, data_handler)
        assert fill is not None
        bar = data_handler.get_latest_bar("TEST")
        raw_cost = float(bar["Adj Close"]) * 100
        assert fill.fill_cost < raw_cost

    def test_zero_quantity_rejected(self, data_handler):
        handler = SimulatedExecutionHandler()
        order = OrderEvent(
            symbol="TEST", order_type=OrderType.MARKET,
            quantity=0, direction=OrderDirection.BUY,
        )
        fill = handler.execute_order(order, data_handler)
        assert fill is None
