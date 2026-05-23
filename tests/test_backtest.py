"""
Tests for rsi_revert.backtest.

Covers:
- Slippage applied correctly on buy/sell.
- Position sizing math.
- Stop-loss firing on gap-down (open <= stop) and intraday (low <= stop).
- Signal-based exit fires at next open.
- End-of-data closure marked correctly and excluded from win rate.
- Buy-and-hold benchmark sanity.
- Metrics: total_return, CAGR, drawdown computed correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsi_revert.backtest import (
    BacktestConfig,
    buy_and_hold,
    compute_metrics,
    run_backtest,
)


def _bars(opens, highs, lows, closes, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(opens), freq="B")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=idx,
    )


def _signals(idx, entries=None, exits=None, stops=None):
    """Build a signals frame with given entry/exit booleans and stop refs."""
    n = len(idx)
    entries = entries if entries is not None else [False] * n
    exits = exits if exits is not None else [False] * n
    stops = stops if stops is not None else [float("nan")] * n
    return pd.DataFrame(
        {
            "entry_signal": entries,
            "exit_signal": exits,
            "rolling_low": stops,
        },
        index=idx,
    )


# --------------- happy path: single profitable trade ---------------


def test_single_winning_trade() -> None:
    # 5 bars. Entry signal on bar 1 (executes at open of bar 2 = 100).
    # Exit signal on bar 3 (executes at open of bar 4 = 110).
    bars = _bars(
        opens=[99, 100, 105, 110, 112],
        highs=[100, 106, 108, 113, 113],
        lows=[98, 99, 103, 109, 111],
        closes=[99.5, 105, 107, 111, 112],
    )
    sig = _signals(
        bars.index,
        entries=[True, False, False, False, False],  # fires day 0 → exec day 1
        exits=[False, False, True, False, False],    # fires day 2 → exec day 3
        stops=[95.0, 95.0, 95.0, 95.0, 95.0],
    )
    cfg = BacktestConfig(initial_equity=100_000, risk_per_trade=0.01, slippage_bps=0)
    res = run_backtest(bars, sig, cfg)

    assert len(res.trades) == 1
    trade = res.trades.iloc[0]
    assert trade["exit_reason"] == "signal"
    # Entry at open of bar 1 = 100, exit at open of bar 3 = 110.
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["exit_price"] == pytest.approx(110.0)
    # risk_per_share = 100 - 95 = 5. target_risk = 0.01*100000 = 1000. shares = 200.
    assert trade["shares"] == 200
    assert trade["pnl"] == pytest.approx(200 * 10)
    assert res.metrics["win_rate"] == 1.0
    assert res.metrics["n_trades"] == 1


# --------------- slippage ---------------


def test_slippage_applied_to_buy_and_sell() -> None:
    bars = _bars(
        opens=[100, 100, 100, 100],
        highs=[101, 101, 101, 101],
        lows=[99, 99, 99, 99],
        closes=[100, 100, 100, 100],
    )
    sig = _signals(
        bars.index,
        entries=[True, False, False, False],
        exits=[False, False, True, False],
        stops=[90.0] * 4,
    )
    cfg = BacktestConfig(initial_equity=100_000, slippage_bps=50)  # 0.5% slippage
    res = run_backtest(bars, sig, cfg)
    trade = res.trades.iloc[0]
    # Buy: 100 * 1.005 = 100.5; Sell: 100 * 0.995 = 99.5
    assert trade["entry_price"] == pytest.approx(100.5)
    assert trade["exit_price"] == pytest.approx(99.5)
    assert trade["pnl"] < 0  # slippage eats the round trip


# --------------- stop loss ---------------


def test_stop_fires_on_intraday_hit() -> None:
    # Enter at bar 1 open = 100, stop = 95. Bar 2 has low = 94 → stop hit.
    bars = _bars(
        opens=[100, 100, 96, 90],
        highs=[100, 101, 97, 91],
        lows=[99, 99, 94, 89],
        closes=[100, 100, 96, 90],
    )
    sig = _signals(
        bars.index,
        entries=[True, False, False, False],
        exits=[False] * 4,
        stops=[95.0, 95.0, 95.0, 95.0],
    )
    cfg = BacktestConfig(initial_equity=100_000, slippage_bps=0)
    res = run_backtest(bars, sig, cfg)
    trade = res.trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    # Open (96) > stop (95), so we fill AT the stop level.
    assert trade["exit_price"] == pytest.approx(95.0)


def test_stop_fires_on_gap_down() -> None:
    # Bar 2 opens at 92, below the 95 stop → fill at open (worse).
    bars = _bars(
        opens=[100, 100, 92, 90],
        highs=[100, 101, 93, 91],
        lows=[99, 99, 91, 89],
        closes=[100, 100, 92, 90],
    )
    sig = _signals(
        bars.index,
        entries=[True, False, False, False],
        exits=[False] * 4,
        stops=[95.0] * 4,
    )
    cfg = BacktestConfig(initial_equity=100_000, slippage_bps=0)
    res = run_backtest(bars, sig, cfg)
    trade = res.trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == pytest.approx(92.0)  # fills at the gapped-down open


def test_stop_takes_priority_over_signal_exit() -> None:
    # Both stop and exit_signal would trigger at bar 2. Stop should win
    # (resting order convention).
    bars = _bars(
        opens=[100, 100, 92, 90],
        highs=[100, 101, 93, 91],
        lows=[99, 99, 91, 89],
        closes=[100, 100, 92, 90],
    )
    sig = _signals(
        bars.index,
        entries=[True, False, False, False],
        exits=[False, True, False, False],  # signal fires bar 1 → exec bar 2
        stops=[95.0] * 4,
    )
    cfg = BacktestConfig(initial_equity=100_000, slippage_bps=0)
    res = run_backtest(bars, sig, cfg)
    assert res.trades.iloc[0]["exit_reason"] == "stop"


# --------------- no entry when conditions unmet ---------------


def test_no_entry_when_stop_above_entry() -> None:
    bars = _bars(
        opens=[100, 100, 100, 100],
        highs=[101] * 4, lows=[99] * 4, closes=[100] * 4,
    )
    sig = _signals(
        bars.index,
        entries=[True, False, False, False],
        exits=[False] * 4,
        stops=[105.0] * 4,  # stop ABOVE entry — pathological
    )
    cfg = BacktestConfig(initial_equity=100_000, slippage_bps=0)
    res = run_backtest(bars, sig, cfg)
    assert len(res.trades) == 0


def test_no_entry_when_insufficient_cash() -> None:
    # Risk window = $1, allowed risk = $1000 → 1000 shares wanted.
    # But each share costs $100 → need $100k cash. Equity = $50, so:
    # cash cap = floor(50 / 100) = 0 shares. Skip.
    bars = _bars(
        opens=[100, 100, 100, 100],
        highs=[101] * 4, lows=[99] * 4, closes=[100] * 4,
    )
    sig = _signals(
        bars.index,
        entries=[True, False, False, False],
        exits=[False] * 4,
        stops=[99.0] * 4,
    )
    cfg = BacktestConfig(initial_equity=50.0, slippage_bps=0)
    res = run_backtest(bars, sig, cfg)
    assert len(res.trades) == 0


# --------------- end of data ---------------


def test_open_position_closed_at_end_of_data() -> None:
    bars = _bars(
        opens=[100, 100, 100, 100],
        highs=[101] * 4, lows=[99] * 4, closes=[100, 100, 100, 105],
    )
    sig = _signals(
        bars.index,
        entries=[True, False, False, False],
        exits=[False] * 4,
        stops=[95.0] * 4,
    )
    cfg = BacktestConfig(initial_equity=100_000, slippage_bps=0)
    res = run_backtest(bars, sig, cfg)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["exit_reason"] == "end_of_data"
    # end_of_data trades excluded from win-rate denominator.
    assert res.metrics["n_trades"] == 0
    assert res.metrics["win_rate"] == 0.0


# --------------- buy and hold ---------------


def test_buy_and_hold_basic() -> None:
    bars = _bars(
        opens=[100, 100, 100, 100],
        highs=[101] * 4, lows=[99] * 4, closes=[100, 105, 110, 120],
    )
    cfg = BacktestConfig(initial_equity=100_000, slippage_bps=0)
    res = buy_and_hold(bars, cfg)
    # 1000 shares bought at 100, sold at 120. P&L = 20,000.
    assert res.trades.iloc[0]["shares"] == 1000
    assert res.metrics["total_return"] == pytest.approx(0.20)


# --------------- metrics ---------------


def test_metrics_drawdown_computed_correctly() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    equity = pd.Series([100, 120, 80, 90, 110], index=idx, dtype=float)
    metrics = compute_metrics(equity, pd.DataFrame(), BacktestConfig())
    # Peak 120 → trough 80 = -33.33% drawdown
    assert metrics["max_drawdown"] == pytest.approx(-0.3333, abs=1e-3)
    assert metrics["total_return"] == pytest.approx(0.10)


def test_metrics_handle_empty_trades() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    equity = pd.Series([100_000] * 5, index=idx, dtype=float)
    metrics = compute_metrics(equity, pd.DataFrame(), BacktestConfig())
    assert metrics["n_trades"] == 0
    assert metrics["win_rate"] == 0.0
    assert metrics["sharpe"] == 0.0
