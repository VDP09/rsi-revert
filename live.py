"""
Daily paper-trading loop.

Designed to run once per day after the US market closes (e.g. 21:05 UTC
weekdays via GitHub Actions cron). The full pipeline:

  1. Check kill switch → if engaged, log and exit.
  2. Fetch latest bars for each symbol.
  3. Compute regime filter on SPY.
  4. Generate signals for the configured variant.
  5. Read current positions and open orders from the broker.
  6. Reconcile: place/cancel orders as the strategy dictates.
  7. Return a structured RunReport for the reporting module.

Idempotency
-----------
This function reads state from the broker (positions, orders) each run,
not from local files. Re-running the same day is safe: it will detect
existing positions and orders and only place what's missing.

Reconciliation rules (per symbol, evaluated in order)
-----------------------------------------------------
- If exit signal fired today AND we have a position:
    - Cancel any open stop orders for this symbol.
    - Submit a market SELL for the full position.
- Elif we have a position AND no open stop order exists:
    - Submit a stop order at today's rolling 10-day low.
    - (This catches the case where yesterday's BUY filled but no stop
      has been placed yet.)
- Elif entry signal fired today AND we have no position AND no pending buy:
    - Calculate share count from risk sizing using current equity.
    - Submit a market BUY.
    - (Stop will be placed on the NEXT run, after the BUY fills.)

We deliberately don't try to place the stop in the same run as the buy
because we don't yet know the fill price; we'd be guessing. Placing it
on the next run, after Alpaca reports avg_entry_price, is more accurate.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from rsi_revert.broker import Broker, OrderInfo, Position
from rsi_revert.data import get_bars
from rsi_revert.signals import (
    VariantParams,
    compute_regime_filter,
    generate_signals,
)
from rsi_revert.utils import check_kill_switch

logger = logging.getLogger(__name__)


@dataclass
class SymbolAction:
    """One symbol's outcome from a daily run."""
    symbol: str
    decision: str  # "entry", "exit", "stop_placed", "hold", "skip", "error"
    reason: str
    position_before: Position | None = None
    order_submitted: OrderInfo | None = None
    error: str | None = None
    rsi: float | None = None
    close: float | None = None
    regime_ok: bool | None = None


@dataclass
class RunReport:
    """Output of one daily run."""
    timestamp: datetime
    kill_switch_engaged: bool
    account_equity: float | None = None
    account_cash: float | None = None
    symbol_actions: list[SymbolAction] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False


def daily_run(
    cfg: dict[str, Any],
    broker: Broker,
    *,
    dry_run: bool = False,
) -> RunReport:
    """
    Run one daily trading cycle.

    Parameters
    ----------
    cfg : dict
        Parsed config dict from utils.load_config.
    broker : Broker
        Injected broker instance. Tests pass a mock; production passes
        a real Broker bound to Alpaca paper credentials.
    dry_run : bool, default False
        If True, all decision logic runs normally but order-submission
        calls (buy/sell/stop/cancel) are skipped. The RunReport still
        shows decisions and would-be order details, so you can verify
        the strategy before committing real (paper or live) orders.

    Returns
    -------
    RunReport
    """
    report = RunReport(timestamp=datetime.now(timezone.utc), kill_switch_engaged=False)
    report.dry_run = dry_run

    if check_kill_switch():
        logger.warning("KILL_SWITCH engaged — exiting without trading.")
        report.kill_switch_engaged = True
        return report

    universe: list[str] = cfg["universe"]
    variant = _variant_from_cfg(cfg)
    lookback_days: int = cfg["live"].get("data_lookback_days", 365)
    cache_dir = cfg["runtime"].get("cache_dir", "data/cache")
    risk_per_trade: float = cfg["backtest"]["risk_per_trade"]

    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    # We need enough history for the 200-SMA regime filter.
    # data_lookback_days should be >= ~260 (200-SMA + buffer).
    start = end - pd.Timedelta(days=lookback_days)

    # 1. Fetch account snapshot (needed for sizing).
    try:
        account = broker.get_account()
        report.account_equity = account.equity
        report.account_cash = account.cash
        logger.info(
            "Account: equity=$%.0f cash=$%.0f buying_power=$%.0f",
            account.equity, account.cash, account.buying_power,
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to fetch account: {exc}"
        logger.error(msg)
        report.errors.append(msg)
        return report

    # 2. Fetch SPY bars (used for regime + signals; reused below).
    try:
        spy_bars = get_bars("SPY", start, end, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to fetch SPY bars: {exc}"
        logger.error(msg)
        report.errors.append(msg)
        return report

    if len(spy_bars) < 200:
        msg = (
            f"Insufficient history for regime filter: {len(spy_bars)} bars, "
            f"need >=200. Skipping run."
        )
        logger.warning(msg)
        report.errors.append(msg)
        return report

    regime = compute_regime_filter(spy_bars, sma_period=200)

    # 3. Per-symbol processing.
    for symbol in universe:
        try:
            action = _process_symbol(
                symbol=symbol,
                cfg=cfg,
                broker=broker,
                variant=variant,
                spy_bars=spy_bars,
                regime=regime,
                start=start,
                end=end,
                account_equity=account.equity,
                account_buying_power=account.buying_power,
                risk_per_trade=risk_per_trade,
                cache_dir=cache_dir,
                dry_run=dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error processing %s", symbol)
            action = SymbolAction(
                symbol=symbol,
                decision="error",
                reason="unhandled exception",
                error=str(exc),
            )
            report.errors.append(f"{symbol}: {exc}")
        report.symbol_actions.append(action)

    return report


def _variant_from_cfg(cfg: dict[str, Any]) -> VariantParams:
    """Local re-import to avoid circular import with utils."""
    from rsi_revert.utils import get_variant_from_config
    return get_variant_from_config(cfg)


def _process_symbol(
    symbol: str,
    cfg: dict[str, Any],
    broker: Broker,
    variant: VariantParams,
    spy_bars: pd.DataFrame,
    regime: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    account_equity: float,
    account_buying_power: float,
    risk_per_trade: float,
    cache_dir: str,
    dry_run: bool = False,
) -> SymbolAction:
    """Decide and submit orders for one symbol.

    If dry_run=True, decision logic still runs and the SymbolAction is
    populated normally, but no broker write methods are called. The
    `order_submitted` field will be None; the `reason` is prefixed
    with [DRY RUN] so logs are unambiguous.
    """
    # Fetch this symbol's bars. If symbol is SPY we already have them.
    if symbol == "SPY":
        bars = spy_bars
        symbol_regime = regime
    else:
        bars = get_bars(symbol, start, end, cache_dir=cache_dir)
        if not bars.index.equals(spy_bars.index):
            # Trading-day alignment for non-SPY symbols: reindex regime
            # to the symbol's bar dates. Missing days → False (no entry).
            symbol_regime = regime.reindex(bars.index).fillna(False).astype(bool)
        else:
            symbol_regime = regime

    if len(bars) < max(200, variant.rsi_period + 1):
        return SymbolAction(
            symbol=symbol, decision="skip",
            reason=f"insufficient history ({len(bars)} bars)",
        )

    signals = generate_signals(bars, variant, symbol_regime)
    last = signals.iloc[-1]
    last_entry: bool = bool(last["entry_signal"])
    last_exit: bool = bool(last["exit_signal"])
    last_close: float = float(last["close"])
    last_rsi: float = float(last["rsi"]) if not pd.isna(last["rsi"]) else float("nan")
    last_rolling_low: float = float(last["rolling_low"]) if not pd.isna(last["rolling_low"]) else float("nan")
    last_regime: bool = bool(last["regime_ok"])

    common_fields = dict(
        symbol=symbol, rsi=last_rsi, close=last_close, regime_ok=last_regime,
    )

    position = broker.get_position(symbol)
    open_orders = broker.list_open_orders(symbol)
    open_stops = [o for o in open_orders if o.order_type == "stop" and o.side == "sell"]
    open_buys = [o for o in open_orders if o.order_type == "market" and o.side == "buy"]

    # Rule 1: exit signal + position held → sell.
    if last_exit and position is not None and position.qty > 0:
        if dry_run:
            return SymbolAction(
                decision="exit",
                reason=f"[DRY RUN] would exit_signal at RSI={last_rsi:.1f}, sell {position.qty} shares",
                position_before=position,
                **common_fields,
            )
        for stop in open_stops:
            broker.cancel_order(stop.id)
        order = broker.submit_market_sell(symbol, position.qty)
        return SymbolAction(
            decision="exit",
            reason=f"exit_signal at RSI={last_rsi:.1f}",
            position_before=position,
            order_submitted=order,
            **common_fields,
        )

    # Rule 2: position held + no stop pending → place stop.
    if position is not None and position.qty > 0 and not open_stops:
        if math.isnan(last_rolling_low) or last_rolling_low <= 0:
            return SymbolAction(
                decision="skip",
                reason="cannot place stop — rolling_low NaN/non-positive",
                position_before=position,
                **common_fields,
            )
        if dry_run:
            return SymbolAction(
                decision="stop_placed",
                reason=f"[DRY RUN] would place stop @ ${last_rolling_low:.2f} for {position.qty} shares",
                position_before=position,
                **common_fields,
            )
        order = broker.submit_stop_loss(symbol, position.qty, last_rolling_low)
        return SymbolAction(
            decision="stop_placed",
            reason=f"stop @ ${last_rolling_low:.2f}",
            position_before=position,
            order_submitted=order,
            **common_fields,
        )

    # Rule 3: entry signal + flat + no pending buy → buy.
    if last_entry and position is None and not open_buys:
        if math.isnan(last_rolling_low) or last_rolling_low <= 0:
            return SymbolAction(
                decision="skip",
                reason="cannot size trade — rolling_low NaN/non-positive",
                **common_fields,
            )
        risk_per_share = last_close - last_rolling_low
        if risk_per_share <= 0:
            return SymbolAction(
                decision="skip",
                reason=f"stop ({last_rolling_low:.2f}) at or above close ({last_close:.2f})",
                **common_fields,
            )
        target_risk = risk_per_trade * account_equity
        shares = int(math.floor(target_risk / risk_per_share))
        # Cap by buying power (passed in from top-level call to avoid duplicate API hit).
        max_affordable = int(math.floor(account_buying_power / last_close))
        shares = min(shares, max_affordable)
        if shares <= 0:
            return SymbolAction(
                decision="skip",
                reason=f"sized to 0 shares (risk={target_risk:.0f}, risk_per_share={risk_per_share:.2f})",
                **common_fields,
            )
        if dry_run:
            return SymbolAction(
                decision="entry",
                reason=f"[DRY RUN] would entry_signal at RSI={last_rsi:.1f}, buy {shares} shares (risk_per_share=${risk_per_share:.2f})",
                **common_fields,
            )
        order = broker.submit_market_buy(symbol, shares)
        return SymbolAction(
            decision="entry",
            reason=f"entry_signal at RSI={last_rsi:.1f}, {shares} shares",
            order_submitted=order,
            **common_fields,
        )

    # Otherwise: hold (or already-pending orders mean no action).
    if position is not None and position.qty > 0:
        reason = "holding"
        if open_stops:
            reason = f"holding with stop @ ${open_stops[0].stop_price:.2f}"
        return SymbolAction(
            decision="hold", reason=reason, position_before=position, **common_fields,
        )
    return SymbolAction(
        decision="hold",
        reason=(
            f"flat, no entry signal "
            f"(RSI={last_rsi:.1f}, regime={'on' if last_regime else 'off'})"
        ),
        **common_fields,
    )
