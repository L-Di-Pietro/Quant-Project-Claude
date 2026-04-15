"""
Backtester — Event-Driven Backtesting Framework for Quantitative Strategies.

This package provides a modular, event-driven architecture for simulating
algorithmic trading strategies on historical financial data.
"""

__version__ = "1.0.0"

from backtester.events import (
    Event,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
)
from backtester.data_handler import HistoricParquetDataHandler
from backtester.strategy import Strategy
from backtester.portfolio import NaivePortfolio
from backtester.execution_handler import SimulatedExecutionHandler
from backtester.performance import PerformanceManager
from backtester.tearsheet import TearSheetGenerator
from backtester.optimizer import GridSearchOptimizer, RandomSearchOptimizer
from backtester.engine import BacktestEngine
