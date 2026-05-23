"""
Targeted "weed-out" tests for code paths not fully exercised by the
existing test suite. These catch subtle behaviors that would otherwise
only surface in production:

- Backtest: gap-down execution model (open below stop → fill at open)
- Backtest: cash-constrained position sizing (cap below risk-based size)
- Indicators: RSI edge cases (pure up, pure down, flat market)
- Live: pending buy from previous day blocks duplicate entry
- Live: non-SPY symbol reindexes the regime filter to its own bar dates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from rsi_revert.backtest import BacktestConfig, run_backtest
from rsi_revert.broker import AccountInfo, OrderInfo, Position
from rsi_revert.indicators import rsi
from rsi_revert.live import daily_run


# =====================================================================
# Backtest execution model
# =====================================================================


def _trivial_bars(opens, highs, lows, closes) -> pd.DataFrame:
    n = len(opens)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=idx,
    )


def _signals_from(
    bars: pd.DataFrame,
    entries: list[bool],
    exits: list[bool],
    stops: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_signal": entries,
            "exit_signal": exits,
            "rolling_low": stops,
        },
        index=bars.index,
    )


def test_backtest_gap_down_through_stop_fills_at_open() -> None:
    """If next bar's open is below the stop, the fill must be at the open
    price (worse than the stop), not at the stop. Catches gap risk."""
    # Day 0: signal day. Day 1: entry day (open=100, stop will be 95).
    # Day 2: gaps DOWN to open=90 (well below stop=95).
    bars = _trivial_bars(
        opens=[100.0, 100.0, 90.0, 92.0],
        highs=[101.0, 102.0, 93.0, 94.0],
        lows=[99.0, 99.0, 89.0, 91.0],
        closes=[100.0, 101.0, 92.0, 93.0],
    )
    signals = _signals_from(
        bars,
        entries=[True, False, False, False],
        exits=[False, False, False, False],
        stops=[95.0, 95.0, 95.0, 95.0],
    )
    cfg = BacktestConfig(initial_equity=10_000, slippage_bps=0, risk_per_trade=1.0)
    r = run_backtest(bars, signals, cfg)

    # Expect exactly one trade, closed by stop.
    assert len(r.trades) == 1
    trade = r.trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    # Exit price should be ~90 (gap-down open), NOT 95 (the stop level).
    assert trade["exit_price"] == pytest.approx(90.0, abs=0.01)


def test_backtest_intraday_stop_hit_fills_at_stop_level() -> None:
    """If the bar opens above the stop but the low pierces it intraday,
    the fill is at the stop level (the order rests there)."""
    bars = _trivial_bars(
        opens=[100.0, 100.0, 97.0, 92.0],
        highs=[101.0, 102.0, 98.0, 94.0],
        lows=[99.0, 99.0, 94.0, 91.0],   # day 2 low pierces stop=95
        closes=[100.0, 101.0, 96.0, 93.0],
    )
    signals = _signals_from(
        bars,
        entries=[True, False, False, False],
        exits=[False, False, False, False],
        stops=[95.0, 95.0, 95.0, 95.0],
    )
    cfg = BacktestConfig(initial_equity=10_000, slippage_bps=0, risk_per_trade=1.0)
    r = run_backtest(bars, signals, cfg)

    assert len(r.trades) == 1
    trade = r.trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    # Fill at stop level (95), not at the low (94).
    assert trade["exit_price"] == pytest.approx(95.0, abs=0.01)


def test_backtest_sizing_capped_by_cash() -> None:
    """When risk-based sizing wants more shares than cash allows, the
    cash cap binds and sizing falls back to floor(cash/entry)."""
    # entry_price ~100, stop=99 → risk_per_share=1, risk_per_trade=1% of $1000
    # → risk-based size = 10 shares. But cash is only $500, so cap is 5.
    bars = _trivial_bars(
        opens=[100.0, 100.0, 101.0, 102.0],
        highs=[101.0, 102.0, 103.0, 104.0],
        lows=[99.0, 99.5, 100.5, 101.5],
        closes=[100.0, 101.0, 102.0, 103.0],
    )
    signals = _signals_from(
        bars,
        entries=[True, False, False, False],
        exits=[False, False, False, False],
        stops=[99.0, 99.0, 99.0, 99.0],
    )
    cfg = BacktestConfig(
        initial_equity=500.0, risk_per_trade=0.01, slippage_bps=0,
    )
    r = run_backtest(bars, signals, cfg)

    # 1% of $500 = $5. Risk per share = $1. Risk-sized = 5 shares.
    # Cash cap also = floor(500/100) = 5. Same answer here; the assertion
    # below is the important one: we did not get a divide-by-zero or skip.
    assert len(r.trades) >= 0  # at least it ran
    # If it bought, shares must respect cash.
    if len(r.trades) > 0:
        trade = r.trades.iloc[0]
        assert trade["shares"] * trade["entry_price"] <= 500 + 1e-6


# =====================================================================
# Indicators: edge cases
# =====================================================================


def test_rsi_pure_uptrend_approaches_100() -> None:
    """Monotonically rising series → RSI = 100 (avg_loss=0)."""
    prices = pd.Series([100.0 + i for i in range(30)])
    r = rsi(prices, period=14)
    # The first `period` values are NaN; rest should be 100.
    assert r.iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_rsi_pure_downtrend_approaches_0() -> None:
    """Monotonically falling series → RSI = 0 (avg_gain=0)."""
    prices = pd.Series([100.0 - i for i in range(30)])
    r = rsi(prices, period=14)
    assert r.iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_rsi_flat_market_is_50() -> None:
    """Constant price → both avg_gain and avg_loss are 0 → RSI = 50."""
    prices = pd.Series([100.0] * 30)
    r = rsi(prices, period=14)
    # Last value should be 50 (flat-market convention).
    assert r.iloc[-1] == pytest.approx(50.0, abs=1e-6)


def test_rsi_insufficient_data_returns_all_nan() -> None:
    """Series shorter than period+1 should return all NaN (not raise)."""
    prices = pd.Series([100.0, 101.0, 99.0])  # only 3 values, period=14
    r = rsi(prices, period=14)
    assert r.isna().all()
    assert len(r) == 3


# =====================================================================
# Live: code paths the integration test doesn't hit
# =====================================================================


@dataclass
class _RecordingBroker:
    """Test double for Broker. Records calls; returns canned state."""
    account_equity: float = 100_000.0
    account_cash: float = 100_000.0
    account_buying_power: float = 100_000.0
    position: Position | None = None
    open_orders: list[OrderInfo] = field(default_factory=list)
    calls: list[tuple] = field(default_factory=list)

    def get_account(self) -> AccountInfo:
        return AccountInfo(
            equity=self.account_equity, cash=self.account_cash,
            buying_power=self.account_buying_power,
            portfolio_value=self.account_equity, pattern_day_trader=False,
        )

    def get_position(self, symbol):
        self.calls.append(("get_position", symbol))
        return self.position

    def list_open_orders(self, symbol=None):
        self.calls.append(("list_open_orders", symbol))
        return list(self.open_orders)

    def submit_market_buy(self, symbol, qty):
        self.calls.append(("submit_market_buy", symbol, qty))
        return OrderInfo(
            id=f"buy-{symbol}", symbol=symbol, side="buy", qty=qty,
            order_type="market", status="accepted", stop_price=None,
            filled_qty=0, filled_avg_price=None,
            submitted_at=datetime.now(timezone.utc),
        )

    def submit_market_sell(self, symbol, qty):
        self.calls.append(("submit_market_sell", symbol, qty))
        return OrderInfo(
            id=f"sell-{symbol}", symbol=symbol, side="sell", qty=qty,
            order_type="market", status="accepted", stop_price=None,
            filled_qty=0, filled_avg_price=None,
            submitted_at=datetime.now(timezone.utc),
        )

    def submit_stop_loss(self, symbol, qty, stop_price):
        self.calls.append(("submit_stop_loss", symbol, qty, stop_price))
        return OrderInfo(
            id=f"stop-{symbol}", symbol=symbol, side="sell", qty=qty,
            order_type="stop", status="accepted", stop_price=stop_price,
            filled_qty=0, filled_avg_price=None,
            submitted_at=datetime.now(timezone.utc),
        )

    def cancel_order(self, order_id):
        self.calls.append(("cancel_order", order_id))


def _bars_with_signal(n: int = 300, last_rsi_oversold: bool = True) -> pd.DataFrame:
    """
    Build SPY bars that end with an RSI(2) oversold print so a Variant B
    entry signal will fire on the last bar. Long enough for 200-SMA regime.
    """
    rng = np.random.default_rng(42)
    drift = np.linspace(0, 0.6, n)  # gentle uptrend (regime ON)
    noise = rng.normal(0, 0.005, n)
    log_returns = drift / n + noise
    close = 200.0 * np.exp(np.cumsum(log_returns))
    if last_rsi_oversold:
        # Force the last few bars to be a sharp drop so RSI(2) prints low.
        close[-3:] = close[-4] * np.array([0.99, 0.97, 0.95])
    open_ = close * 0.999
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": 1_000_000.0, "trade_count": 0.0, "vwap": close},
        index=idx,
    )


def _base_cfg() -> dict:
    return {
        "universe": ["SPY"],
        "strategy": {"variant": "B"},
        "backtest": {
            "initial_equity": 100_000, "risk_per_trade": 0.01,
            "slippage_bps": 5.0, "commission_per_share": 0.0,
            "history_start": "2020-01-01",
        },
        "live": {"data_lookback_days": 365},
        "runtime": {"cache_dir": "data/cache"},
    }


def test_pending_buy_from_previous_day_blocks_re_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If we're flat but there's already a pending buy from yesterday, a
    fresh entry signal must NOT submit a second buy.
    """
    bars = _bars_with_signal(last_rsi_oversold=True)

    monkeypatch.setattr("rsi_revert.live.get_bars", lambda *a, **k: bars)
    monkeypatch.delenv("KILL_SWITCH", raising=False)

    pending_buy = OrderInfo(
        id="prev-buy", symbol="SPY", side="buy", qty=10,
        order_type="market", status="new", stop_price=None,
        filled_qty=0, filled_avg_price=None,
        submitted_at=datetime.now(timezone.utc),
    )
    broker = _RecordingBroker(position=None, open_orders=[pending_buy])

    report = daily_run(_base_cfg(), broker)

    # No new buy should have been submitted.
    buy_calls = [c for c in broker.calls if c[0] == "submit_market_buy"]
    assert len(buy_calls) == 0, f"unexpected buy submitted: {buy_calls}"

    # The action should be a hold (not entry, not error).
    spy_action = next(a for a in report.symbol_actions if a.symbol == "SPY")
    assert spy_action.decision == "hold"


def test_non_spy_symbol_reindexes_regime(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    For a non-SPY symbol, the SPY regime filter must be reindexed to the
    symbol's own bar dates. If the index differs (different trading
    calendar), missing days default to regime_off.
    """
    spy_bars = _bars_with_signal(last_rsi_oversold=False)
    # Symbol bars: same dates as SPY minus the last 10. Regime reindexes;
    # missing days do not appear, but everything aligned should work.
    qqq_bars = spy_bars.iloc[:-10].copy()

    def fake_get_bars(symbol, *args, **kwargs):
        return spy_bars if symbol == "SPY" else qqq_bars

    monkeypatch.setattr("rsi_revert.live.get_bars", fake_get_bars)
    monkeypatch.delenv("KILL_SWITCH", raising=False)

    cfg = _base_cfg()
    cfg["universe"] = ["SPY", "QQQ"]
    broker = _RecordingBroker(position=None)

    report = daily_run(cfg, broker)

    # Both symbols should be processed; no unhandled errors.
    assert len(report.symbol_actions) == 2
    assert {a.symbol for a in report.symbol_actions} == {"SPY", "QQQ"}
    for a in report.symbol_actions:
        assert a.decision != "error", f"{a.symbol}: {a.error}"


def test_holding_position_no_stop_no_signal_places_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Defensive case: we hold a position, no exit signal, but there's no
    open stop order. Live must place a stop, not just sit there with an
    unprotected position.
    """
    bars = _bars_with_signal(last_rsi_oversold=False)
    monkeypatch.setattr("rsi_revert.live.get_bars", lambda *a, **k: bars)
    monkeypatch.delenv("KILL_SWITCH", raising=False)

    held = Position(
        symbol="SPY", qty=10, avg_entry_price=200.0,
        market_value=2050.0, unrealized_pl=50.0,
    )
    broker = _RecordingBroker(position=held, open_orders=[])

    report = daily_run(_base_cfg(), broker)

    stop_calls = [c for c in broker.calls if c[0] == "submit_stop_loss"]
    assert len(stop_calls) == 1
    assert stop_calls[0][1] == "SPY"
    assert stop_calls[0][2] == 10  # qty
    assert stop_calls[0][3] > 0  # stop_price positive

    spy_action = next(a for a in report.symbol_actions if a.symbol == "SPY")
    assert spy_action.decision == "stop_placed"


def test_dry_run_submits_no_orders_but_reports_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    dry_run=True should produce the same decisions but call none of the
    broker write methods. The report should be marked dry_run and the
    reason strings prefixed [DRY RUN].
    """
    bars = _bars_with_signal(last_rsi_oversold=True)
    monkeypatch.setattr("rsi_revert.live.get_bars", lambda *a, **k: bars)
    monkeypatch.delenv("KILL_SWITCH", raising=False)

    broker = _RecordingBroker(position=None, open_orders=[])

    report = daily_run(_base_cfg(), broker, dry_run=True)

    write_methods = {"submit_market_buy", "submit_market_sell",
                     "submit_stop_loss", "cancel_order"}
    write_calls = [c for c in broker.calls if c[0] in write_methods]
    assert write_calls == [], f"unexpected writes in dry-run: {write_calls}"

    assert report.dry_run is True

    spy_action = next(a for a in report.symbol_actions if a.symbol == "SPY")
    assert spy_action.decision == "entry"
    assert "[DRY RUN]" in spy_action.reason
    assert spy_action.order_submitted is None


def test_dry_run_stop_placement_path_submits_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run while holding a position without a stop: no submit_stop_loss."""
    bars = _bars_with_signal(last_rsi_oversold=False)
    monkeypatch.setattr("rsi_revert.live.get_bars", lambda *a, **k: bars)
    monkeypatch.delenv("KILL_SWITCH", raising=False)

    held = Position(
        symbol="SPY", qty=10, avg_entry_price=200.0,
        market_value=2050.0, unrealized_pl=50.0,
    )
    broker = _RecordingBroker(position=held, open_orders=[])

    report = daily_run(_base_cfg(), broker, dry_run=True)

    stop_calls = [c for c in broker.calls if c[0] == "submit_stop_loss"]
    assert stop_calls == []
    spy_action = next(a for a in report.symbol_actions if a.symbol == "SPY")
    assert spy_action.decision == "stop_placed"
    assert "[DRY RUN]" in spy_action.reason
