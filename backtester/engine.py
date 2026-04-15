"""
engine.py — Central event-loop orchestrator.

Wires together all modules (DataHandler, Strategy, Portfolio,
ExecutionHandler) and runs the main event-driven simulation.
"""

from __future__ import annotations

import queue
from typing import Any, Dict, Optional

from backtester.events import (
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from backtester.data_handler import DataHandler
from backtester.strategy import Strategy
from backtester.portfolio import NaivePortfolio
from backtester.execution_handler import SimulatedExecutionHandler
from backtester.performance import PerformanceManager
from backtester.tearsheet import TearSheetGenerator


class BacktestEngine:
    """
    Main back-testing engine.

    It creates the event queue and loops through market data, dispatching
    events to the appropriate handler.

    Parameters
    ----------
    data_handler : DataHandler
        Pre-configured data handler instance.
    strategy : Strategy
        Pre-configured strategy instance.
    portfolio_kwargs : dict
        Keyword arguments forwarded to ``NaivePortfolio``
        (e.g. ``initial_capital``, ``order_quantity``).
    execution_kwargs : dict
        Keyword arguments forwarded to ``SimulatedExecutionHandler``
        (e.g. ``commission_type``, ``slippage_value``).
    """

    def __init__(
        self,
        data_handler: DataHandler,
        strategy: Strategy,
        portfolio_kwargs: Optional[Dict[str, Any]] = None,
        execution_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data_handler = data_handler
        self.strategy = strategy

        p_kw = portfolio_kwargs or {}
        e_kw = execution_kwargs or {}

        self.portfolio = NaivePortfolio(data_handler=data_handler, **p_kw)
        self.execution_handler = SimulatedExecutionHandler(**e_kw)

        self.events: queue.Queue = queue.Queue()
        self._ran = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute the back-test.

        The outer loop advances the data by one bar, placing a
        ``MarketEvent`` on the queue.  The inner loop drains the queue,
        dispatching each event:

        MarketEvent  → Strategy.calculate_signals  → SignalEvent
        SignalEvent  → Portfolio.generate_order     → OrderEvent
        OrderEvent   → ExecutionHandler.execute     → FillEvent
        FillEvent    → Portfolio.update_fill
        """
        while self.data_handler.continue_backtest:
            market_event = self.data_handler.update_bars()
            if market_event is not None:
                self.events.put(market_event)

            # Process all events generated in this time-step
            while not self.events.empty():
                event = self.events.get(block=False)

                if event.event_type == EventType.MARKET:
                    # 1. Update portfolio valuation
                    self.portfolio.update_timeindex()
                    # 2. Let strategy react
                    signal = self.strategy.calculate_signals(event)
                    if signal is not None:
                        self.events.put(signal)

                elif event.event_type == EventType.SIGNAL:
                    order = self.portfolio.generate_order(event)
                    if order is not None:
                        self.events.put(order)

                elif event.event_type == EventType.ORDER:
                    fill = self.execution_handler.execute_order(
                        event, self.data_handler
                    )
                    if fill is not None:
                        self.events.put(fill)

                elif event.event_type == EventType.FILL:
                    self.portfolio.update_fill(event)

        self._ran = True

    # ------------------------------------------------------------------
    # Post-run accessors
    # ------------------------------------------------------------------
    def get_equity_curve(self):
        """Return the equity curve DataFrame."""
        return self.portfolio.get_equity_curve()

    def get_trade_log(self):
        """Return the trade log DataFrame."""
        return self.portfolio.get_trade_log()

    def get_performance_summary(self) -> Dict[str, float]:
        """Return the performance metrics as a dictionary."""
        ec = self.get_equity_curve()
        tl = self.get_trade_log()
        pm = PerformanceManager(ec, tl)
        return pm.summary()

    def generate_tearsheet(
        self, output_path: str = "tearsheet.html", strategy_name: str = "Strategy"
    ) -> str:
        """
        Generate the HTML tear sheet and return the file path.
        """
        ec = self.get_equity_curve()
        tl = self.get_trade_log()
        pm = PerformanceManager(ec, tl)
        gen = TearSheetGenerator(pm, strategy_name=strategy_name)
        return gen.generate(output_path)
