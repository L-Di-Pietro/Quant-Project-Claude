"""
strategy.py — Abstract Strategy base class.

Every concrete strategy inherits from ``Strategy`` and implements
``calculate_signals()``, which is called on each ``MarketEvent``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from backtester.events import MarketEvent, SignalEvent


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Sub-classes must implement :meth:`calculate_signals`, which receives
    a :class:`MarketEvent` and optionally returns a :class:`SignalEvent`.
    """

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> Optional[SignalEvent]:
        """
        Analyse the latest market data and, if appropriate, return a
        ``SignalEvent`` indicating a desired trade direction.

        Parameters
        ----------
        event : MarketEvent
            The market event that triggered this call.

        Returns
        -------
        SignalEvent or None
        """
        raise NotImplementedError
