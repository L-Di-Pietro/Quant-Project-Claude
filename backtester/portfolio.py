"""
portfolio.py — Portfolio management, position tracking, and equity curve.

The Portfolio is the "accountant" of the system.  It:
    1. Receives ``SignalEvent``s and converts them into ``OrderEvent``s
       (position sizing).
    2. Receives ``FillEvent``s and updates internal holdings/positions.
    3. Tracks the equity curve over time.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtester.events import (
    FillEvent,
    OrderDirection,
    OrderEvent,
    OrderType,
    SignalDirection,
    SignalEvent,
)


class NaivePortfolio:
    """
    A straightforward portfolio that converts every signal into a
    fixed-quantity market order and keeps a running equity curve.

    Parameters
    ----------
    data_handler : DataHandler
        Reference to the data handler (to look up current prices).
    initial_capital : float
        Starting cash balance (default 100 000).
    order_quantity : int
        Default number of shares per order (default 100).
    """

    def __init__(
        self,
        data_handler,
        initial_capital: float = 100_000.0,
        order_quantity: int = 100,
    ) -> None:
        self.data_handler = data_handler
        self.initial_capital = initial_capital
        self.order_quantity = order_quantity

        self.symbol_list: List[str] = data_handler.symbol_list

        # current_positions: symbol → number of units held (+ long, − short)
        self.current_positions: Dict[str, int] = {
            s: 0 for s in self.symbol_list
        }
        # current_holdings: cash, total equity, individual market values
        self.current_holdings: Dict[str, float] = self._construct_initial_holdings()

        # Historical records (one row per bar)
        self.all_positions: List[Dict] = []
        self.all_holdings: List[Dict] = []

        # Trade log
        self.trades: List[Dict] = []

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _construct_initial_holdings(self) -> Dict[str, float]:
        holdings: Dict[str, float] = {s: 0.0 for s in self.symbol_list}
        holdings["cash"] = self.initial_capital
        holdings["commission"] = 0.0
        holdings["total"] = self.initial_capital
        return holdings

    # ------------------------------------------------------------------
    # Market update — recalculate holdings at current prices
    # ------------------------------------------------------------------
    def update_timeindex(self) -> None:
        """
        Called on every ``MarketEvent`` to snapshot positions and
        mark-to-market the portfolio.
        """
        # Snapshot positions
        pos_snapshot: Dict[str, object] = {}
        try:
            dt_stamp = self.data_handler.get_latest_bar_datetime(
                self.symbol_list[0]
            )
            pos_snapshot["datetime"] = dt_stamp
        except (IndexError, ValueError):
            return

        for s in self.symbol_list:
            pos_snapshot[s] = self.current_positions[s]
        self.all_positions.append(pos_snapshot)

        # Snapshot holdings (mark-to-market)
        hold_snapshot: Dict[str, float] = {"datetime": dt_stamp}
        hold_snapshot["cash"] = self.current_holdings["cash"]
        hold_snapshot["commission"] = self.current_holdings["commission"]

        total = self.current_holdings["cash"]
        for s in self.symbol_list:
            try:
                market_value = (
                    self.current_positions[s]
                    * self.data_handler.get_latest_bar_value(s, "Adj Close")
                )
            except (KeyError, ValueError):
                market_value = 0.0
            hold_snapshot[s] = market_value
            total += market_value
        hold_snapshot["total"] = total
        self.all_holdings.append(hold_snapshot)

    # ------------------------------------------------------------------
    # Signal → Order
    # ------------------------------------------------------------------
    def generate_order(self, signal: SignalEvent) -> Optional[OrderEvent]:
        """
        Convert a ``SignalEvent`` into an ``OrderEvent``.

        Simple sizing logic:
        - LONG  → BUY  ``order_quantity`` shares (if flat or short).
        - SHORT → SELL ``order_quantity`` shares (if flat or long).
        - EXIT  → close the current position.
        """
        symbol = signal.symbol
        current_qty = self.current_positions.get(symbol, 0)

        if signal.direction == SignalDirection.LONG and current_qty == 0:
            return OrderEvent(
                symbol=symbol,
                order_type=OrderType.MARKET,
                quantity=self.order_quantity,
                direction=OrderDirection.BUY,
            )
        elif signal.direction == SignalDirection.SHORT and current_qty == 0:
            return OrderEvent(
                symbol=symbol,
                order_type=OrderType.MARKET,
                quantity=self.order_quantity,
                direction=OrderDirection.SELL,
            )
        elif signal.direction == SignalDirection.EXIT and current_qty != 0:
            direction = (
                OrderDirection.SELL if current_qty > 0 else OrderDirection.BUY
            )
            return OrderEvent(
                symbol=symbol,
                order_type=OrderType.MARKET,
                quantity=abs(current_qty),
                direction=direction,
            )
        return None

    # ------------------------------------------------------------------
    # Fill → update positions / holdings
    # ------------------------------------------------------------------
    def update_fill(self, fill: FillEvent) -> None:
        """
        Update positions and cash in response to a ``FillEvent``.
        """
        direction_sign = 1 if fill.direction == OrderDirection.BUY else -1

        # Update position count
        self.current_positions[fill.symbol] += direction_sign * fill.quantity

        # Update cash (cost basis + commission)
        self.current_holdings[fill.symbol] += direction_sign * fill.fill_cost
        self.current_holdings["commission"] += fill.commission
        self.current_holdings["cash"] -= (
            direction_sign * fill.fill_cost + fill.commission
        )

        # Record the trade
        self.trades.append(
            {
                "datetime": fill.timeindex,
                "symbol": fill.symbol,
                "direction": fill.direction.value,
                "quantity": fill.quantity,
                "fill_cost": fill.fill_cost,
                "commission": fill.commission,
                "slippage": fill.slippage,
            }
        )

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------
    def get_equity_curve(self) -> pd.DataFrame:
        """Return a DataFrame with the portfolio equity curve."""
        if not self.all_holdings:
            return pd.DataFrame()
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index("datetime", inplace=True)
        curve["returns"] = curve["total"].pct_change().fillna(0.0)
        curve["equity_curve"] = (1.0 + curve["returns"]).cumprod()
        return curve

    def get_trade_log(self) -> pd.DataFrame:
        """Return the trade log as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
