"""
Alpaca broker wrapper.

Provides a narrow, testable interface around alpaca-py's TradingClient.
Wrapping the SDK has three benefits:
1. Retries + timeouts uniformly applied to every API call.
2. Easy to mock for tests — we don't depend on Alpaca's class hierarchy.
3. If we ever switch brokers (Tradier, IBKR), the swap is local.

Defaults to PAPER trading. Live trading requires explicit opt-in via
the ALPACA_PAPER environment variable being set to "false". This is
the safest possible default.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    StopOrderRequest,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Plain-data return types. Using dataclasses (rather than passing the
# SDK's own objects around) means the rest of the system never imports
# alpaca-py — only broker.py does. This is good isolation.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AccountInfo:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    pattern_day_trader: bool


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: int  # signed; negative = short. We only go long, so should be >= 0.
    avg_entry_price: float
    market_value: float
    unrealized_pl: float


@dataclass(frozen=True)
class OrderInfo:
    id: str
    symbol: str
    side: str            # "buy" or "sell"
    qty: int
    order_type: str      # "market" or "stop"
    status: str          # alpaca status string
    stop_price: float | None
    filled_qty: int
    filled_avg_price: float | None
    submitted_at: datetime | None


# ---------------------------------------------------------------------


def _is_paper() -> bool:
    """
    Paper mode unless ALPACA_PAPER is explicitly 'false'.

    Defaulting to paper is the right safety posture: a misconfigured
    GitHub secret should never accidentally route real orders.
    """
    raw = os.environ.get("ALPACA_PAPER", "true").strip().lower()
    return raw not in ("false", "0", "no", "off")


class Broker:
    """Thin wrapper over Alpaca TradingClient. All methods retry on transient errors."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("ALPACA_API_KEY")
        secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set "
                "(env vars or constructor args)."
            )
        paper_mode = paper if paper is not None else _is_paper()
        self._client = TradingClient(api_key, secret_key, paper=paper_mode)
        self.paper = paper_mode
        if not paper_mode:
            logger.warning("Broker initialized in LIVE mode. Real money is at risk.")
        else:
            logger.info("Broker initialized in PAPER mode.")

    # ---- account ----------------------------------------------------

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((APIError, OSError, ConnectionError, TimeoutError)),
        reraise=True,
    )
    def get_account(self) -> AccountInfo:
        """Fetch account snapshot."""
        a = self._client.get_account()
        return AccountInfo(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            portfolio_value=float(a.portfolio_value),
            pattern_day_trader=bool(a.pattern_day_trader),
        )

    # ---- market clock ----------------------------------------------

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((APIError, OSError, ConnectionError, TimeoutError)),
        reraise=True,
    )
    def is_market_open(self) -> bool:
        """Is the market currently open? (For pre-flight sanity in live runs.)"""
        clock = self._client.get_clock()
        return bool(clock.is_open)

    # ---- positions --------------------------------------------------

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((APIError, OSError, ConnectionError, TimeoutError)),
        reraise=True,
    )
    def get_position(self, symbol: str) -> Position | None:
        """Return position for symbol, or None if flat."""
        try:
            p = self._client.get_open_position(symbol)
        except APIError as exc:
            # Alpaca returns HTTP 404 with code 40410000 when no position exists.
            # Some SDK versions stringify this as "position does not exist".
            msg = str(exc).lower()
            if "404" in msg or "position does not exist" in msg or "not found" in msg:
                return None
            raise
        return Position(
            symbol=p.symbol,
            qty=int(float(p.qty)),
            avg_entry_price=float(p.avg_entry_price),
            market_value=float(p.market_value),
            unrealized_pl=float(p.unrealized_pl),
        )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((APIError, OSError, ConnectionError, TimeoutError)),
        reraise=True,
    )
    def list_positions(self) -> list[Position]:
        """All open positions."""
        positions = self._client.get_all_positions()
        return [
            Position(
                symbol=p.symbol,
                qty=int(float(p.qty)),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
            )
            for p in positions
        ]

    # ---- orders -----------------------------------------------------

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((APIError, OSError, ConnectionError, TimeoutError)),
        reraise=True,
    )
    def list_open_orders(self, symbol: str | None = None) -> list[OrderInfo]:
        """All open (not yet filled, cancelled, or rejected) orders."""
        req = GetOrdersRequest(
            status="open",
            symbols=[symbol] if symbol else None,
        )
        orders = self._client.get_orders(filter=req)
        return [_to_order_info(o) for o in orders]

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((APIError, OSError, ConnectionError, TimeoutError)),
        reraise=True,
    )
    def submit_market_buy(self, symbol: str, qty: int) -> OrderInfo:
        """
        Submit a market BUY for `qty` shares.

        Uses DAY time-in-force, which is queued for the next session open
        if submitted outside market hours — exactly what we want for an
        EOD strategy that wants to fill at next open.
        """
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        logger.info("Submitted market BUY %s x%d (order_id=%s)", symbol, qty, order.id)
        return _to_order_info(order)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((APIError, OSError, ConnectionError, TimeoutError)),
        reraise=True,
    )
    def submit_market_sell(self, symbol: str, qty: int) -> OrderInfo:
        """Market SELL — used for signal-based exits (stops are placed as stop orders)."""
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        logger.info("Submitted market SELL %s x%d (order_id=%s)", symbol, qty, order.id)
        return _to_order_info(order)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((APIError, OSError, ConnectionError, TimeoutError)),
        reraise=True,
    )
    def submit_stop_loss(self, symbol: str, qty: int, stop_price: float) -> OrderInfo:
        """
        Submit a resting stop-loss SELL.

        Uses GTC (good-til-cancelled) so the stop survives across sessions
        until either the stop hits or we cancel it on a signal-based exit.
        """
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if stop_price <= 0:
            raise ValueError(f"stop_price must be positive, got {stop_price}")
        req = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=round(stop_price, 2),
        )
        order = self._client.submit_order(req)
        logger.info(
            "Submitted stop-loss SELL %s x%d @ $%.2f (order_id=%s)",
            symbol, qty, stop_price, order.id,
        )
        return _to_order_info(order)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((APIError, OSError, ConnectionError, TimeoutError)),
        reraise=True,
    )
    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order. Idempotent — already-cancelled orders are not an error."""
        try:
            self._client.cancel_order_by_id(order_id)
            logger.info("Cancelled order %s", order_id)
        except APIError as exc:
            msg = str(exc).lower()
            if "not cancellable" in msg or "already" in msg or "422" in msg:
                logger.debug("Order %s already terminal, no cancel needed", order_id)
                return
            raise


def _to_order_info(o) -> OrderInfo:
    """Adapt an alpaca Order object to our dataclass."""
    return OrderInfo(
        id=str(o.id),
        symbol=o.symbol,
        side=str(o.side).lower().replace("orderside.", ""),
        qty=int(float(o.qty)) if o.qty else 0,
        order_type=str(o.order_type).lower().replace("ordertype.", ""),
        status=str(o.status).lower().replace("orderstatus.", ""),
        stop_price=float(o.stop_price) if o.stop_price else None,
        filled_qty=int(float(o.filled_qty)) if o.filled_qty else 0,
        filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
        submitted_at=o.submitted_at,
    )
