"""
Integration tests.

These mock only the network boundary (Alpaca's HTTP calls) — everything
else runs for real. They catch wiring bugs that unit tests miss.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rsi_revert.backtest import BacktestConfig, buy_and_hold, run_backtest
from rsi_revert.broker import AccountInfo, Broker, Position
from rsi_revert.live import daily_run
from rsi_revert.report import format_backtest_comparison, format_run_report
from rsi_revert.signals import (
    VARIANT_A,
    VARIANT_B,
    compute_regime_filter,
    generate_signals,
)
from rsi_revert.walkforward import WalkForwardConfig, walk_forward


def _realistic_spy_bars(years: float = 8.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic SPY-like daily bars with drift, vol, and pullbacks."""
    rng = np.random.default_rng(seed)
    n = int(round(years * 252))
    idx = pd.date_range("2017-01-02", periods=n, freq="B")
    # Geometric Brownian motion with ~10% drift, 18% vol.
    daily_returns = rng.normal(loc=0.0004, scale=0.012, size=n)
    # Inject some pullbacks for trade variety.
    for shock_day in (200, 500, 800, 1100, 1400):
        if shock_day < n:
            daily_returns[shock_day:shock_day + 5] -= 0.015
    close = 250.0 * np.exp(np.cumsum(daily_returns))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": 1e7, "trade_count": 1e4, "vwap": close},
        index=idx,
    )


# ---------------------------- backtest path ----------------------------


def test_full_backtest_pipeline_runs_and_produces_reasonable_results() -> None:
    bars = _realistic_spy_bars(8.0)
    regime = compute_regime_filter(bars, sma_period=200)

    bt_cfg = BacktestConfig(initial_equity=100_000, risk_per_trade=0.01, slippage_bps=5.0)

    # Both variants should execute end-to-end without errors.
    for variant in (VARIANT_A, VARIANT_B):
        sig = generate_signals(bars, variant, regime)
        result = run_backtest(bars, sig, bt_cfg, label=variant.name)
        assert result.equity_curve.iloc[0] == pytest.approx(100_000)
        # Sanity: final equity should be > 0 (didn't blow up).
        assert result.equity_curve.iloc[-1] > 0
        # Sharpe in a sane range.
        assert -5.0 < result.metrics["sharpe"] < 5.0
        # Max DD between -100% and 0%.
        assert -1.0 <= result.metrics["max_drawdown"] <= 0.0

    # Buy and hold benchmark works.
    bh = buy_and_hold(bars, bt_cfg)
    assert bh.metrics["n_trades"] == 0  # excluded from win-rate (end_of_data)


def test_full_walkforward_runs_and_reports_degradation() -> None:
    bars = _realistic_spy_bars(8.0)
    regime = compute_regime_filter(bars, sma_period=200)
    bt_cfg = BacktestConfig(initial_equity=100_000)
    wf_cfg = WalkForwardConfig(train_years=3, test_years=1, step_years=1)

    result = walk_forward(bars, [VARIANT_A, VARIANT_B], regime, bt_cfg, wf_cfg)
    assert len(result.windows) >= 3
    # Degradation keys present.
    assert "sharpe_degradation" in result.degradation
    assert "cagr_degradation" in result.degradation
    # Stitched curve has values for every test bar.
    assert result.stitched_test_equity.iloc[0] == pytest.approx(100_000, rel=0.01)


def test_format_backtest_comparison_produces_readable_output() -> None:
    bars = _realistic_spy_bars(5.0)
    regime = compute_regime_filter(bars, sma_period=200)
    sig = generate_signals(bars, VARIANT_B, regime)
    strat = run_backtest(bars, sig, BacktestConfig(), label="VariantB")
    bh = buy_and_hold(bars, BacktestConfig(), label="BuyHold")
    out = format_backtest_comparison(strat, bh)
    assert "VariantB" in out.upper() or "VARIANTB" in out.upper()
    assert "BUYHOLD" in out.upper() or "BUY" in out.upper()
    assert "Sharpe" in out
    assert "Max drawdown" in out


# ---------------------------- live path ----------------------------


def _cfg() -> dict:
    return {
        "universe": ["SPY"],
        "strategy": {"variant": "B"},
        "backtest": {"risk_per_trade": 0.01, "initial_equity": 100_000,
                     "slippage_bps": 5.0, "commission_per_share": 0.0},
        "live": {"data_lookback_days": 365},
        "runtime": {"cache_dir": "data/cache"},
    }


class _NetworkOnlyBroker:
    """Broker stub that only mocks network calls but uses real logic."""

    def __init__(self) -> None:
        self.calls = []
        self.submitted = []
        self._position: Position | None = None
        self._orders: list = []

    def get_account(self):
        return AccountInfo(
            equity=100_000.0, cash=100_000.0, buying_power=100_000.0,
            portfolio_value=100_000.0, pattern_day_trader=False,
        )

    def get_position(self, symbol):
        return self._position

    def list_open_orders(self, symbol=None):
        return list(self._orders)

    def submit_market_buy(self, symbol, qty):
        from rsi_revert.broker import OrderInfo
        order = OrderInfo(
            id=f"buy-{len(self.submitted)}", symbol=symbol, side="buy", qty=qty,
            order_type="market", status="accepted", stop_price=None,
            filled_qty=0, filled_avg_price=None, submitted_at=datetime.now(timezone.utc),
        )
        self.submitted.append(("buy", order))
        return order

    def submit_market_sell(self, symbol, qty):
        from rsi_revert.broker import OrderInfo
        order = OrderInfo(
            id=f"sell-{len(self.submitted)}", symbol=symbol, side="sell", qty=qty,
            order_type="market", status="accepted", stop_price=None,
            filled_qty=0, filled_avg_price=None, submitted_at=datetime.now(timezone.utc),
        )
        self.submitted.append(("sell", order))
        return order

    def submit_stop_loss(self, symbol, qty, stop_price):
        from rsi_revert.broker import OrderInfo
        order = OrderInfo(
            id=f"stop-{len(self.submitted)}", symbol=symbol, side="sell", qty=qty,
            order_type="stop", status="accepted", stop_price=stop_price,
            filled_qty=0, filled_avg_price=None, submitted_at=datetime.now(timezone.utc),
        )
        self.submitted.append(("stop", order))
        return order

    def cancel_order(self, order_id):
        self.calls.append(("cancel", order_id))


def test_daily_run_produces_formattable_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole live pipeline runs and the report formatter handles its output."""
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    bars = _realistic_spy_bars(2.0)  # enough for 200-SMA
    broker = _NetworkOnlyBroker()
    with patch("rsi_revert.live.get_bars", return_value=bars):
        report = daily_run(_cfg(), broker)
    # Should produce a clean report without errors.
    assert report.account_equity == 100_000.0
    assert len(report.symbol_actions) == 1
    # The formatter handles the output without raising.
    text = format_run_report(report)
    assert "SPY" in text
    assert "ACCOUNT" in text
    # Formatter right-pads inside the $ field, so the literal substring is "100,000".
    assert "100,000" in text


def test_no_lookahead_in_live_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Live signal at bar t must be identical when computed on a truncated
    series ending at t. This is the live-equivalent of the backtest test.
    """
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    full_bars = _realistic_spy_bars(3.0)
    regime_full = compute_regime_filter(full_bars, sma_period=200)
    sig_full = generate_signals(full_bars, VARIANT_B, regime_full)

    # Last signal on the full series.
    full_last = sig_full.iloc[-1]

    # Now truncate by one bar and recompute — the second-to-last bar
    # should be unchanged.
    truncated = full_bars.iloc[:-1]
    regime_trunc = compute_regime_filter(truncated, sma_period=200)
    sig_trunc = generate_signals(truncated, VARIANT_B, regime_trunc)
    trunc_last = sig_trunc.iloc[-1]

    # These should be identical: same bar, same window of history.
    assert sig_full.iloc[-2]["entry_signal"] == trunc_last["entry_signal"]
    assert sig_full.iloc[-2]["exit_signal"] == trunc_last["exit_signal"]
    if not pd.isna(sig_full.iloc[-2]["rsi"]) and not pd.isna(trunc_last["rsi"]):
        assert sig_full.iloc[-2]["rsi"] == pytest.approx(trunc_last["rsi"], abs=1e-9)


def test_public_api_imports() -> None:
    """The package __init__ exposes the documented public surface."""
    import rsi_revert
    assert hasattr(rsi_revert, "__version__")
    assert hasattr(rsi_revert, "VARIANT_A")
    assert hasattr(rsi_revert, "VARIANT_B")
    assert hasattr(rsi_revert, "run_backtest")
    assert hasattr(rsi_revert, "walk_forward")
