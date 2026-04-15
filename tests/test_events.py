"""
test_events.py — Unit tests for the Event class hierarchy.
"""

import datetime as dt

from backtester.events import (
    Event,
    EventType,
    FillEvent,
    MarketEvent,
    OrderDirection,
    OrderEvent,
    OrderType,
    SignalDirection,
    SignalEvent,
)


class TestMarketEvent:
    def test_event_type(self):
        e = MarketEvent()
        assert e.event_type == EventType.MARKET


class TestSignalEvent:
    def test_default_values(self):
        e = SignalEvent(symbol="AAPL", direction=SignalDirection.LONG)
        assert e.event_type == EventType.SIGNAL
        assert e.symbol == "AAPL"
        assert e.direction == SignalDirection.LONG
        assert e.strength == 1.0

    def test_custom_strength(self):
        e = SignalEvent(symbol="MSFT", direction=SignalDirection.SHORT, strength=0.5)
        assert e.strength == 0.5
        assert e.direction == SignalDirection.SHORT


class TestOrderEvent:
    def test_creation(self):
        o = OrderEvent(
            symbol="GOOG",
            order_type=OrderType.MARKET,
            quantity=50,
            direction=OrderDirection.BUY,
        )
        assert o.event_type == EventType.ORDER
        assert o.symbol == "GOOG"
        assert o.quantity == 50
        assert o.direction == OrderDirection.BUY

    def test_repr(self):
        o = OrderEvent(symbol="AAPL", quantity=10, direction=OrderDirection.SELL)
        assert "AAPL" in repr(o)
        assert "SELL" in repr(o)


class TestFillEvent:
    def test_creation(self):
        f = FillEvent(
            symbol="AAPL",
            quantity=100,
            direction=OrderDirection.BUY,
            fill_cost=15000.0,
            commission=1.0,
            slippage=5.0,
        )
        assert f.event_type == EventType.FILL
        assert f.fill_cost == 15000.0
        assert f.commission == 1.0
        assert f.slippage == 5.0
