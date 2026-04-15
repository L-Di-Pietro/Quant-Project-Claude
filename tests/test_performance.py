"""
test_performance.py — Unit tests for the PerformanceManager.
"""

import numpy as np
import pandas as pd
import pytest

from backtester.performance import PerformanceManager


@pytest.fixture
def sample_equity_curve():
    """Create a synthetic equity curve for testing."""
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01
    total = 100_000 * np.cumprod(1 + returns)
    df = pd.DataFrame({"total": total, "returns": returns}, index=dates)
    return df


@pytest.fixture
def sample_trade_log():
    """Create a synthetic trade log."""
    return pd.DataFrame(
        [
            {"datetime": "2020-01-10", "symbol": "AAPL", "direction": "BUY",
             "quantity": 100, "fill_cost": 10_000.0, "commission": 1.0, "slippage": 2.0},
            {"datetime": "2020-02-10", "symbol": "AAPL", "direction": "SELL",
             "quantity": 100, "fill_cost": 10_500.0, "commission": 1.0, "slippage": 2.0},
            {"datetime": "2020-03-01", "symbol": "AAPL", "direction": "BUY",
             "quantity": 100, "fill_cost": 9_800.0, "commission": 1.0, "slippage": 2.0},
            {"datetime": "2020-04-01", "symbol": "AAPL", "direction": "SELL",
             "quantity": 100, "fill_cost": 9_500.0, "commission": 1.0, "slippage": 2.0},
        ]
    )


class TestPerformanceManager:
    def test_cumulative_return(self, sample_equity_curve):
        pm = PerformanceManager(sample_equity_curve)
        cr = pm.cumulative_return()
        expected = (sample_equity_curve["total"].iloc[-1] /
                    sample_equity_curve["total"].iloc[0]) - 1
        assert abs(cr - expected) < 1e-8

    def test_annualized_return(self, sample_equity_curve):
        pm = PerformanceManager(sample_equity_curve)
        ar = pm.annualized_return()
        assert isinstance(ar, float)

    def test_sharpe_ratio(self, sample_equity_curve):
        pm = PerformanceManager(sample_equity_curve)
        sr = pm.sharpe_ratio()
        assert isinstance(sr, float)

    def test_sortino_ratio(self, sample_equity_curve):
        pm = PerformanceManager(sample_equity_curve)
        sortino = pm.sortino_ratio()
        assert isinstance(sortino, float)

    def test_max_drawdown(self, sample_equity_curve):
        pm = PerformanceManager(sample_equity_curve)
        mdd = pm.max_drawdown()
        assert mdd <= 0  # Drawdown is negative

    def test_max_drawdown_duration(self, sample_equity_curve):
        pm = PerformanceManager(sample_equity_curve)
        dur = pm.max_drawdown_duration()
        assert dur >= 0

    def test_win_rate(self, sample_equity_curve, sample_trade_log):
        pm = PerformanceManager(sample_equity_curve, sample_trade_log)
        wr = pm.win_rate()
        assert 0 <= wr <= 1

    def test_profit_factor(self, sample_equity_curve, sample_trade_log):
        pm = PerformanceManager(sample_equity_curve, sample_trade_log)
        pf = pm.profit_factor()
        assert isinstance(pf, float)

    def test_summary_keys(self, sample_equity_curve, sample_trade_log):
        pm = PerformanceManager(sample_equity_curve, sample_trade_log)
        s = pm.summary()
        expected_keys = [
            "Cumulative Return", "Annualized Return", "Sharpe Ratio",
            "Sortino Ratio", "Information Ratio", "Max Drawdown",
            "Max Drawdown Duration (bars)", "Win Rate", "Profit Factor",
            "Avg Win", "Avg Loss", "Total Trades",
        ]
        for key in expected_keys:
            assert key in s, f"Missing key: {key}"
