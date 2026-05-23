"""
Tests for rsi_revert.broker.

Strategy: replace the TradingClient attribute on the Broker instance
with a MagicMock. We're testing our adapter layer (correct argument
mapping, error translation), not Alpaca's SDK behavior.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from alpaca.common.exceptions import APIError

from rsi_revert.broker import Broker


@pytest.fixture
def broker() -> Broker:
    """A Broker with the underlying client mocked out."""
    with patch.dict("os.environ", {
        "ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test", "ALPACA_PAPER": "true",
    }):
        # Patch TradingClient so we don't actually try to connect.
        with patch("rsi_revert.broker.TradingClient") as tc_cls:
            b = Broker()
            b._client = MagicMock()
            return b


def _fake_alpaca_account() -> SimpleNamespace:
    return SimpleNamespace(
        equity="100000.00", cash="50000.00", buying_power="100000.00",
        portfolio_value="100000.00", pattern_day_trader=False,
    )


def _fake_alpaca_position(symbol: str = "SPY", qty: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol, qty=str(qty), avg_entry_price="450.00",
        market_value=str(qty * 460.0), unrealized_pl=str(qty * 10.0),
    )


def _fake_alpaca_order(
    order_id: str = "abc-123",
    symbol: str = "SPY",
    side: str = "buy",
    qty: int = 10,
    order_type: str = "market",
    stop_price: float | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=order_id, symbol=symbol, side=side, qty=str(qty),
        order_type=order_type, status="accepted",
        stop_price=str(stop_price) if stop_price else None,
        filled_qty="0", filled_avg_price=None,
        submitted_at=datetime.now(timezone.utc),
    )


def test_get_account_translates(broker: Broker) -> None:
    broker._client.get_account.return_value = _fake_alpaca_account()
    acct = broker.get_account()
    assert acct.equity == 100_000.0
    assert acct.cash == 50_000.0
    assert acct.buying_power == 100_000.0
    assert acct.pattern_day_trader is False


def test_get_position_returns_none_when_flat(broker: Broker) -> None:
    broker._client.get_open_position.side_effect = APIError('{"message":"position does not exist"}')
    assert broker.get_position("SPY") is None


def test_get_position_returns_data(broker: Broker) -> None:
    broker._client.get_open_position.return_value = _fake_alpaca_position("SPY", 100)
    pos = broker.get_position("SPY")
    assert pos is not None
    assert pos.symbol == "SPY"
    assert pos.qty == 100
    assert pos.avg_entry_price == 450.0


def test_submit_market_buy_rejects_non_positive_qty(broker: Broker) -> None:
    with pytest.raises(ValueError, match="qty must be positive"):
        broker.submit_market_buy("SPY", 0)


def test_submit_market_buy_calls_client(broker: Broker) -> None:
    broker._client.submit_order.return_value = _fake_alpaca_order(side="buy", qty=10)
    result = broker.submit_market_buy("SPY", 10)
    assert broker._client.submit_order.call_count == 1
    assert result.side == "buy"
    assert result.qty == 10


def test_submit_stop_loss_rejects_non_positive(broker: Broker) -> None:
    with pytest.raises(ValueError, match="stop_price must be positive"):
        broker.submit_stop_loss("SPY", 10, 0.0)
    with pytest.raises(ValueError, match="qty must be positive"):
        broker.submit_stop_loss("SPY", 0, 100.0)


def test_submit_stop_loss_rounds_price(broker: Broker) -> None:
    broker._client.submit_order.return_value = _fake_alpaca_order(
        order_type="stop", side="sell", stop_price=123.46,
    )
    broker.submit_stop_loss("SPY", 10, 123.4567)
    # Inspect the StopOrderRequest passed to submit_order.
    submitted = broker._client.submit_order.call_args.args[0]
    assert float(submitted.stop_price) == 123.46


def test_cancel_order_swallows_already_cancelled(broker: Broker) -> None:
    broker._client.cancel_order_by_id.side_effect = APIError('{"message":"Order not cancellable"}')
    broker.cancel_order("abc")  # should not raise


def test_cancel_order_propagates_other_errors(broker: Broker) -> None:
    broker._client.cancel_order_by_id.side_effect = APIError('{"message":"internal server error"}')
    with pytest.raises(APIError):
        broker.cancel_order("abc")


def test_list_open_orders_filters_by_symbol(broker: Broker) -> None:
    broker._client.get_orders.return_value = [
        _fake_alpaca_order("a", "SPY", "buy"),
        _fake_alpaca_order("b", "SPY", "sell", order_type="stop", stop_price=400.0),
    ]
    orders = broker.list_open_orders("SPY")
    assert len(orders) == 2
    # GetOrdersRequest should have been passed with symbols=["SPY"]
    call = broker._client.get_orders.call_args
    req = call.kwargs.get("filter") or call.args[0]
    assert req.symbols == ["SPY"]


def test_broker_defaults_to_paper() -> None:
    with patch.dict("os.environ", {
        "ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
    }, clear=False):
        # Make sure ALPACA_PAPER is not set
        import os
        os.environ.pop("ALPACA_PAPER", None)
        with patch("rsi_revert.broker.TradingClient") as tc:
            b = Broker()
            assert b.paper is True
            # TradingClient should have been called with paper=True.
            assert tc.call_args.kwargs["paper"] is True


def test_broker_missing_keys_raises() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
            Broker()
