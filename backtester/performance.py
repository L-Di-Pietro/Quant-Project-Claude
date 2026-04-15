"""
performance.py — PerformanceManager: quantitative risk and return metrics.

Computes the full set of metrics required by the project outline:
    - Cumulative & annualized returns
    - Sharpe Ratio, Sortino Ratio, Information Ratio
    - Maximum Drawdown & drawdown duration
    - Trade statistics (win rate, profit factor, avg win/loss)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class PerformanceManager:
    """
    Calculate performance metrics from an equity curve DataFrame
    (as produced by ``NaivePortfolio.get_equity_curve()``).

    Parameters
    ----------
    equity_curve : pd.DataFrame
        Must contain at minimum a ``"total"`` column and ``"returns"``
        column, indexed by datetime.
    trade_log : pd.DataFrame
        Trade-level data with ``fill_cost``, ``direction``, etc.
    risk_free_rate : float
        Annualised risk-free rate (default 0.02 = 2 %).
    periods_per_year : int
        Number of trading periods per year (default 252 for daily).
    """

    def __init__(
        self,
        equity_curve: pd.DataFrame,
        trade_log: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> None:
        self.equity_curve = equity_curve
        self.trade_log = trade_log if trade_log is not None else pd.DataFrame()
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------
    def cumulative_return(self) -> float:
        """Total cumulative return over the backtest period."""
        total = self.equity_curve["total"]
        return (total.iloc[-1] / total.iloc[0]) - 1.0

    def annualized_return(self) -> float:
        """Compound annual growth rate (CAGR)."""
        total = self.equity_curve["total"]
        n_periods = len(total)
        if n_periods <= 1:
            return 0.0
        years = n_periods / self.periods_per_year
        return (total.iloc[-1] / total.iloc[0]) ** (1.0 / years) - 1.0

    # ------------------------------------------------------------------
    # Risk-adjusted metrics
    # ------------------------------------------------------------------
    def sharpe_ratio(self) -> float:
        """
        Annualised Sharpe Ratio.

        SR = sqrt(N) * (mean(R) - Rf/N) / std(R)
        """
        returns = self.equity_curve["returns"]
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess = returns - rf_per_period
        if excess.std() == 0:
            return 0.0
        return float(
            np.sqrt(self.periods_per_year) * excess.mean() / excess.std()
        )

    def sortino_ratio(self) -> float:
        """
        Annualised Sortino Ratio — penalises only downside deviation.
        """
        returns = self.equity_curve["returns"]
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess = returns - rf_per_period
        downside = excess[excess < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        return float(
            np.sqrt(self.periods_per_year) * excess.mean() / downside.std()
        )

    def information_ratio(self, benchmark_returns: Optional[pd.Series] = None) -> float:
        """
        Information Ratio vs. a benchmark.  If no benchmark is supplied,
        uses zero (equivalent to Sharpe without risk-free subtraction).
        """
        returns = self.equity_curve["returns"]
        if benchmark_returns is None:
            benchmark_returns = pd.Series(0.0, index=returns.index)
        active = returns - benchmark_returns
        if active.std() == 0:
            return 0.0
        return float(
            np.sqrt(self.periods_per_year) * active.mean() / active.std()
        )

    # ------------------------------------------------------------------
    # Drawdown analysis
    # ------------------------------------------------------------------
    def drawdown_series(self) -> pd.DataFrame:
        """
        Compute running drawdown, duration, and high-water mark.

        Returns a DataFrame with columns:
            ``equity``, ``hwm`` (high-water mark), ``drawdown``, ``duration``.
        """
        equity = self.equity_curve["total"]
        hwm = equity.cummax()
        drawdown = (equity - hwm) / hwm
        # Duration: how many bars since last high-water mark
        duration = pd.Series(0, index=equity.index, dtype=int)
        counter = 0
        for i, (eq, hw) in enumerate(zip(equity, hwm)):
            if eq >= hw:
                counter = 0
            else:
                counter += 1
            duration.iloc[i] = counter

        return pd.DataFrame(
            {
                "equity": equity,
                "hwm": hwm,
                "drawdown": drawdown,
                "duration": duration,
            }
        )

    def max_drawdown(self) -> float:
        """Maximum drawdown as a negative fraction (e.g. -0.15 = −15 %)."""
        dd = self.drawdown_series()["drawdown"]
        return float(dd.min())

    def max_drawdown_duration(self) -> int:
        """Longest drawdown in number of bars."""
        dur = self.drawdown_series()["duration"]
        return int(dur.max())

    # ------------------------------------------------------------------
    # Trade statistics
    # ------------------------------------------------------------------
    def _compute_round_trip_pnl(self) -> pd.Series:
        """
        Pair up BUY/SELL fills per symbol to compute round-trip P&L.
        Returns a Series of P&L values for each closed trade.
        """
        if self.trade_log.empty:
            return pd.Series(dtype=float)

        pnl_list = []
        open_trades: Dict[str, list] = {}

        for _, row in self.trade_log.iterrows():
            sym = row["symbol"]
            cost = row["fill_cost"]
            comm = row["commission"]
            direction = row["direction"]

            if sym not in open_trades:
                open_trades[sym] = []

            if not open_trades[sym]:
                # Open a new position
                open_trades[sym].append(
                    {"direction": direction, "cost": cost, "commission": comm}
                )
            else:
                prev = open_trades[sym][-1]
                if prev["direction"] != direction:
                    # Closing trade
                    if prev["direction"] == "BUY":
                        pnl = cost - prev["cost"] - prev["commission"] - comm
                    else:
                        pnl = prev["cost"] - cost - prev["commission"] - comm
                    pnl_list.append(pnl)
                    open_trades[sym].pop()
                else:
                    open_trades[sym].append(
                        {"direction": direction, "cost": cost, "commission": comm}
                    )

        return pd.Series(pnl_list, dtype=float)

    def win_rate(self) -> float:
        """Fraction of round-trip trades that were profitable."""
        pnl = self._compute_round_trip_pnl()
        if pnl.empty:
            return 0.0
        return float((pnl > 0).sum() / len(pnl))

    def profit_factor(self) -> float:
        """Gross profits divided by gross losses."""
        pnl = self._compute_round_trip_pnl()
        if pnl.empty:
            return 0.0
        gross_profit = pnl[pnl > 0].sum()
        gross_loss = abs(pnl[pnl < 0].sum())
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    def average_win(self) -> float:
        """Average P&L of winning trades."""
        pnl = self._compute_round_trip_pnl()
        wins = pnl[pnl > 0]
        return float(wins.mean()) if len(wins) > 0 else 0.0

    def average_loss(self) -> float:
        """Average P&L of losing trades (returned as a negative number)."""
        pnl = self._compute_round_trip_pnl()
        losses = pnl[pnl < 0]
        return float(losses.mean()) if len(losses) > 0 else 0.0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, float]:
        """Return all metrics as a dictionary."""
        return {
            "Cumulative Return": self.cumulative_return(),
            "Annualized Return": self.annualized_return(),
            "Sharpe Ratio": self.sharpe_ratio(),
            "Sortino Ratio": self.sortino_ratio(),
            "Information Ratio": self.information_ratio(),
            "Max Drawdown": self.max_drawdown(),
            "Max Drawdown Duration (bars)": self.max_drawdown_duration(),
            "Win Rate": self.win_rate(),
            "Profit Factor": self.profit_factor(),
            "Avg Win": self.average_win(),
            "Avg Loss": self.average_loss(),
            "Total Trades": len(self._compute_round_trip_pnl()),
        }
