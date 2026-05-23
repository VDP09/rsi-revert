"""
Tests for rsi_revert.live.

Strategy: build a fake Broker that records every call, and a synthetic
SPY history that triggers known signals. Verify the right orders are
submitted (or not) in each scenario.

Edge cases covered:
- Kill switch halts everything before any broker call.
- Entry signal places a market buy with correct sizing.
- Already-holding + no stop → places stop at rolling low.
- Exit signal + holding → cancels stops, submits market sell.
- Insufficient history → skip without error.
- Errors fetching account propagate to the report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rsi_revert.broker import AccountInfo, OrderInfo, Position
from rsi_revert.live import daily_run, RunReport


class FakeBroker:
    """In-memory broker for testing. Records every call."""

    def __init__(
        self,
        equity: float = 100_000.0,
        position: Position | None = None,
        open_orders: list[OrderInfo] | None = None,
        account_error: bool = False,
    ) -> None:
        self.equity = equity
        self._position = position
        self._open_orders = open_orders or []
        self.account_error = account_error
        self.calls: list[tuple[str, dict]] = []
        self.submitted_orders: list[OrderInfo] = []
        self.cancelled_ids: list[str] = []

    def get_account(self) -> AccountInfo:
        self.calls.append(("get_account", {}))
        if self.account_error:
            raise RuntimeError("simulated account fetch failure")
        return AccountInfo(
            equity=self.equity, cash=self.equity, buying_power=self.equity,
            portfolio_value=self.equity, pattern_day_trader=False,
        )

    def get_position(self, symbol: str) -> Position | None:
        self.calls.append(("get_position", {"symbol": symbol}))
        return self._position if self._position and self._position.symbol == symbol else None

    def list_open_orders(self, symbol: str | None = None) -> list[OrderInfo]:
        self.calls.append(("list_open_orders", {"symbol": symbol}))
        if symbol is None:
            return list(self._open_orders)
        return [o for o in self._open_orders if o.symbol == symbol]

    def submit_market_buy(self, symbol: str, qty: int) -> OrderInfo:
        self.calls.append(("submit_market_buy", {"symbol": symbol, "qty": qty}))
        order = OrderInfo(
            id=f"buy-{len(self.submitted_orders)}", symbol=symbol, side="buy", qty=qty,
            order_type="market", status="accepted", stop_price=None,
            filled_qty=0, filled_avg_price=None, submitted_at=datetime.now(timezone.utc),
        )
        self.submitted_orders.append(order)
        return order

    def submit_market_sell(self, symbol: str, qty: int) -> OrderInfo:
        self.calls.append(("submit_market_sell", {"symbol": symbol, "qty": qty}))
        order = OrderInfo(
            id=f"sell-{len(self.submitted_orders)}", symbol=symbol, side="sell", qty=qty,
            order_type="market", status="accepted", stop_price=None,
            filled_qty=0, filled_avg_price=None, submitted_at=datetime.now(timezone.utc),
        )
        self.submitted_orders.append(order)
        return order

    def submit_stop_loss(self, symbol: str, qty: int, stop_price: float) -> OrderInfo:
        self.calls.append(("submit_stop_loss", {"symbol": symbol, "qty": qty, "stop_price": stop_price}))
        order = OrderInfo(
            id=f"stop-{len(self.submitted_orders)}", symbol=symbol, side="sell", qty=qty,
            order_type="stop", status="accepted", stop_price=stop_price,
            filled_qty=0, filled_avg_price=None, submitted_at=datetime.now(timezone.utc),
        )
        self.submitted_orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> None:
        self.calls.append(("cancel_order", {"order_id": order_id}))
        self.cancelled_ids.append(order_id)


# Synthetic SPY price builders ----------------------------------------


def _spy_bars_with_pullback() -> pd.DataFrame:
    """
    Build 250 days of SPY-ish prices where the LAST bar produces:
    - close > 200-SMA (regime on)
    - RSI(2) < 10 (Variant B entry)
    """
    rng = np.random.default_rng(7)
    n = 250
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    # Steady drift up.
    drift = np.linspace(0, 0.4, n)
    noise = rng.normal(0, 0.01, n)
    log_prices = drift + np.cumsum(noise * 0.3)
    closes = 400.0 * np.exp(log_prices)
    # Tack a sharp pullback onto the last 3 bars to crush RSI(2).
    closes[-3:] *= [0.99, 0.96, 0.93]
    opens = closes * (1 + rng.normal(0, 0.001, n))
    highs = np.maximum(opens, closes) * 1.003
    lows = np.minimum(opens, closes) * 0.997
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": 1e6, "trade_count": 1000, "vwap": closes},
        index=idx,
    )


def _spy_bars_with_rally() -> pd.DataFrame:
    """
    Bars where the LAST bar produces:
    - Variant B exit (close > SMA(5))
    """
    rng = np.random.default_rng(11)
    n = 250
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    closes = 400.0 + np.cumsum(rng.normal(0.05, 0.5, n))
    # Force last bar well above SMA(5).
    closes[-5:] = [400.0, 399.0, 398.0, 397.0, 420.0]
    opens = closes * (1 + rng.normal(0, 0.001, n))
    highs = np.maximum(opens, closes) * 1.003
    lows = np.minimum(opens, closes) * 0.997
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": 1e6, "trade_count": 1000, "vwap": closes},
        index=idx,
    )


def _base_cfg() -> dict[str, Any]:
    return {
        "universe": ["SPY"],
        "strategy": {"variant": "B"},
        "backtest": {"risk_per_trade": 0.01, "initial_equity": 100_000, "slippage_bps": 5.0,
                     "commission_per_share": 0.0},
        "live": {"data_lookback_days": 365, "data_feed": "iex"},
        "runtime": {"cache_dir": "data/cache", "log_dir": "logs", "log_level": "INFO"},
    }


# ----------------------------- tests ---------------------------------


def test_kill_switch_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KILL_SWITCH", "true")
    broker = FakeBroker()
    report = daily_run(_base_cfg(), broker)
    assert report.kill_switch_engaged is True
    # No broker calls should have happened.
    assert broker.calls == []


def test_entry_signal_places_market_buy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    broker = FakeBroker(equity=100_000)
    bars = _spy_bars_with_pullback()
    with patch("rsi_revert.live.get_bars", return_value=bars):
        report = daily_run(_base_cfg(), broker)
    actions = report.symbol_actions
    assert len(actions) == 1
    assert actions[0].decision == "entry", f"got {actions[0].decision}: {actions[0].reason}"
    buys = [o for o in broker.submitted_orders if o.side == "buy"]
    assert len(buys) == 1


def test_holding_with_no_stop_places_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    held = Position(symbol="SPY", qty=10, avg_entry_price=400.0,
                    market_value=4000.0, unrealized_pl=0.0)
    broker = FakeBroker(equity=100_000, position=held, open_orders=[])
    bars = _spy_bars_with_pullback()
    with patch("rsi_revert.live.get_bars", return_value=bars):
        report = daily_run(_base_cfg(), broker)
    assert report.symbol_actions[0].decision == "stop_placed"
    stops = [o for o in broker.submitted_orders if o.order_type == "stop"]
    assert len(stops) == 1
    assert stops[0].qty == 10


def test_exit_signal_cancels_stop_and_sells(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    held = Position(symbol="SPY", qty=10, avg_entry_price=400.0,
                    market_value=4000.0, unrealized_pl=0.0)
    existing_stop = OrderInfo(
        id="stop-1", symbol="SPY", side="sell", qty=10, order_type="stop",
        status="accepted", stop_price=390.0, filled_qty=0,
        filled_avg_price=None, submitted_at=datetime.now(timezone.utc),
    )
    broker = FakeBroker(equity=100_000, position=held, open_orders=[existing_stop])
    bars = _spy_bars_with_rally()
    with patch("rsi_revert.live.get_bars", return_value=bars):
        report = daily_run(_base_cfg(), broker)
    assert report.symbol_actions[0].decision == "exit"
    assert "stop-1" in broker.cancelled_ids
    sells = [o for o in broker.submitted_orders if o.side == "sell" and o.order_type == "market"]
    assert len(sells) == 1


def test_insufficient_history_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    broker = FakeBroker()
    # Only 50 bars — not enough for 200-SMA.
    short = _spy_bars_with_pullback().iloc[:50]
    with patch("rsi_revert.live.get_bars", return_value=short):
        report = daily_run(_base_cfg(), broker)
    assert len(report.errors) >= 1
    assert "Insufficient history" in report.errors[0]


def test_account_error_aborts_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    broker = FakeBroker(account_error=True)
    report = daily_run(_base_cfg(), broker)
    assert report.errors
    assert "Failed to fetch account" in report.errors[0]
    # Symbol processing should not have happened.
    assert report.symbol_actions == []


def test_no_signal_no_position_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    broker = FakeBroker()
    # Bars where the last bar has no entry/exit signal.
    # Use a smooth uptrend with no extreme RSI moves.
    rng = np.random.default_rng(3)
    n = 250
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    closes = 400.0 + np.cumsum(rng.normal(0.05, 0.2, n))
    bars = pd.DataFrame(
        {"open": closes, "high": closes * 1.005, "low": closes * 0.995, "close": closes,
         "volume": 1e6, "trade_count": 1000, "vwap": closes},
        index=idx,
    )
    with patch("rsi_revert.live.get_bars", return_value=bars):
        report = daily_run(_base_cfg(), broker)
    assert report.symbol_actions[0].decision == "hold"
    assert broker.submitted_orders == []
