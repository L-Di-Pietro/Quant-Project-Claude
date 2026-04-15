"""
execution_handler.py — Simulated broker execution with slippage and commissions.

The ExecutionHandler receives ``OrderEvent``s and returns ``FillEvent``s.
It models transaction costs to prevent unrealistic strategy performance.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Optional

from backtester.events import FillEvent, OrderDirection, OrderEvent


class ExecutionHandler(ABC):
    """Abstract execution handler."""

    @abstractmethod
    def execute_order(self, event: OrderEvent, data_handler) -> Optional[FillEvent]:
        raise NotImplementedError


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulated broker that fills every market order with realistic
    slippage and commission modelling.

    Parameters
    ----------
    commission_type : str
        ``"fixed"`` — flat fee per order, or ``"percentage"`` — % of
        traded notional.
    commission_value : float
        Dollar amount (if fixed) or fraction (if percentage, e.g. 0.001
        for 10 bps).
    slippage_type : str
        ``"fixed"`` — fixed cents per share, or ``"spread_pct"`` —
        fraction of the bar's High-Low range.
    slippage_value : float
        Dollar amount per share (if fixed) or fraction (if spread_pct,
        e.g. 0.1 for 10 % of the range).
    """

    def __init__(
        self,
        commission_type: str = "fixed",
        commission_value: float = 1.0,
        slippage_type: str = "spread_pct",
        slippage_value: float = 0.05,
    ) -> None:
        self.commission_type = commission_type
        self.commission_value = commission_value
        self.slippage_type = slippage_type
        self.slippage_value = slippage_value

    # ------------------------------------------------------------------
    # Commission calculation
    # ------------------------------------------------------------------
    def _calculate_commission(self, fill_cost: float) -> float:
        """Return the commission for a given fill cost."""
        if self.commission_type == "percentage":
            return abs(fill_cost) * self.commission_value
        return self.commission_value  # fixed

    # ------------------------------------------------------------------
    # Slippage calculation
    # ------------------------------------------------------------------
    def _calculate_slippage(
        self, price: float, high: float, low: float, direction: OrderDirection
    ) -> float:
        """
        Calculate per-share slippage.

        For a BUY the slippage is *added* to the price; for a SELL it is
        *subtracted*.
        """
        if self.slippage_type == "spread_pct":
            spread = high - low
            slip = spread * self.slippage_value
        else:
            slip = self.slippage_value

        return slip if direction == OrderDirection.BUY else -slip

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    def execute_order(
        self, event: OrderEvent, data_handler
    ) -> Optional[FillEvent]:
        """
        Fill a market order at the latest bar's close price, adjusted
        for slippage, and attach commission.
        """
        if event.quantity <= 0:
            return None

        symbol = event.symbol
        try:
            bar = data_handler.get_latest_bar(symbol)
        except ValueError:
            return None

        close_price = float(bar["Adj Close"])
        high_price = float(bar["High"])
        low_price = float(bar["Low"])
        timeindex = bar.name

        # Slippage-adjusted fill price
        per_share_slippage = self._calculate_slippage(
            close_price, high_price, low_price, event.direction
        )
        fill_price = close_price + per_share_slippage
        fill_cost = fill_price * event.quantity

        # Commission
        commission = self._calculate_commission(fill_cost)
        total_slippage = abs(per_share_slippage) * event.quantity

        return FillEvent(
            timeindex=timeindex,
            symbol=symbol,
            exchange="BACKTEST",
            quantity=event.quantity,
            direction=event.direction,
            fill_cost=fill_cost,
            commission=commission,
            slippage=total_slippage,
        )
