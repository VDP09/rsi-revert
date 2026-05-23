"""
Tests for rsi_revert.signals.

Critical properties tested:
- Entry signal fires when RSI is below threshold AND regime is OK.
- Entry signal is suppressed when regime is False, regardless of RSI.
- Variant A exits on RSI > 70.
- Variant B exits on close > SMA(5).
- No look-ahead: signal[t] is unchanged when we truncate the input at t.
- Schema integrity: required columns present, dtypes correct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsi_revert.signals import (
    VARIANT_A,
    VARIANT_B,
    VariantParams,
    compute_regime_filter,
    generate_signals,
)


def _make_bars(closes: list[float], lows: list[float] | None = None) -> pd.DataFrame:
    """Build a minimal bars dataframe from close prices."""
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    if lows is None:
        # Assume low = close - 0.5 if not provided.
        lows = [c - 0.5 for c in closes]
    return pd.DataFrame({"close": closes, "low": lows}, index=idx)


# ----------------------- variant params -----------------------


def test_variant_params_validates_exit_config() -> None:
    with pytest.raises(ValueError, match="rsi_exit_threshold"):
        VariantParams(
            name="bad", rsi_period=14, rsi_entry_threshold=30.0,
            exit_kind="rsi_threshold",  # missing rsi_exit_threshold
        )
    with pytest.raises(ValueError, match="exit_sma_period"):
        VariantParams(
            name="bad", rsi_period=2, rsi_entry_threshold=10.0,
            exit_kind="sma_cross",  # missing exit_sma_period
        )


# ----------------------- regime filter -----------------------


def test_regime_filter_basic() -> None:
    # Prices below their long-term mean → regime False; above → True.
    closes = [100.0] * 200 + [110.0] * 50  # 200 days flat, then jumps up
    bars = _make_bars(closes)
    regime = compute_regime_filter(bars, sma_period=200)
    # First 199 rows: SMA undefined → regime False (via fillna).
    assert not regime.iloc[:199].any()
    # Row 199: SMA = 100, close = 100 → not strictly greater → False.
    assert not regime.iloc[199]
    # After jump: close 110 > SMA which is still mostly ~100 → True.
    assert regime.iloc[200:].all()


# ----------------------- signal generation -----------------------


def test_generate_signals_variant_a_fires_on_oversold() -> None:
    # 50 bars trending up to seed the 200-SMA... actually we need more
    # for the 200-SMA. Skip the regime check by forcing regime_ok=True.
    # Then construct a sharp drop that pushes RSI(14) below 30.
    n = 100
    closes = [100.0 + i * 0.1 for i in range(60)] + [100.0 - i * 1.0 for i in range(40)]
    bars = _make_bars(closes)
    regime = pd.Series(True, index=bars.index)
    sig = generate_signals(bars, VARIANT_A, regime)

    # At least one entry should fire during the sharp drop.
    assert sig["entry_signal"].any()
    # Where entry fires, RSI must be < 30.
    fired = sig[sig["entry_signal"]]
    assert (fired["rsi"] < 30.0).all()


def test_generate_signals_variant_b_fires_more_often_than_a() -> None:
    # Connors RSI(2)<10 is a much more sensitive trigger than RSI(14)<30
    # on the same price series. Verify that property on a noisy walk.
    rng = np.random.default_rng(7)
    closes = 100.0 + np.cumsum(rng.standard_normal(500) * 0.5)
    bars = _make_bars(closes.tolist())
    regime = pd.Series(True, index=bars.index)
    sig_a = generate_signals(bars, VARIANT_A, regime)
    sig_b = generate_signals(bars, VARIANT_B, regime)
    assert sig_b["entry_signal"].sum() > sig_a["entry_signal"].sum()


def test_regime_filter_suppresses_entries() -> None:
    # Build a series where Variant A would fire, then verify that
    # forcing regime_ok=False blocks all entries.
    closes = [100.0 + i * 0.1 for i in range(60)] + [100.0 - i * 1.0 for i in range(40)]
    bars = _make_bars(closes)
    regime_off = pd.Series(False, index=bars.index)
    sig = generate_signals(bars, VARIANT_A, regime_off)
    assert not sig["entry_signal"].any()


def test_variant_a_exit_is_rsi_above_70() -> None:
    # Sharp rally pushes RSI(14) > 70.
    closes = [100.0 - i * 0.1 for i in range(20)] + [100.0 + i * 2.0 for i in range(40)]
    bars = _make_bars(closes)
    regime = pd.Series(True, index=bars.index)
    sig = generate_signals(bars, VARIANT_A, regime)
    fired = sig[sig["exit_signal"]]
    assert len(fired) > 0
    assert (fired["rsi"] > 70.0).all()


def test_variant_b_exit_is_close_above_sma5() -> None:
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
    bars = _make_bars(closes)
    regime = pd.Series(True, index=bars.index)
    sig = generate_signals(bars, VARIANT_B, regime)
    # Where exit_signal is True, close must exceed the trailing SMA(5).
    from rsi_revert.indicators import sma
    sma5 = sma(bars["close"], period=5)
    fired = sig[sig["exit_signal"]]
    for ts in fired.index:
        assert bars.loc[ts, "close"] > sma5.loc[ts]


def test_no_lookahead_bias() -> None:
    # Critical property: signal[t] depends only on data up to t.
    # If we truncate the input at t, signal[t] must be unchanged.
    rng = np.random.default_rng(42)
    closes = (100.0 + np.cumsum(rng.standard_normal(300) * 0.3)).tolist()
    bars = _make_bars(closes)
    regime = pd.Series(True, index=bars.index)
    full = generate_signals(bars, VARIANT_A, regime)

    # Truncate at several points and verify earlier signals are stable.
    for cut in (100, 150, 200, 250):
        truncated_bars = bars.iloc[:cut]
        truncated_regime = regime.iloc[:cut]
        truncated_sig = generate_signals(truncated_bars, VARIANT_A, truncated_regime)
        # Compare every column at every shared timestamp.
        pd.testing.assert_frame_equal(
            full.iloc[:cut],
            truncated_sig,
            check_dtype=False,
        )


def test_schema() -> None:
    bars = _make_bars([100.0] * 50)
    regime = pd.Series(True, index=bars.index)
    sig = generate_signals(bars, VARIANT_A, regime)
    expected = {"entry_signal", "exit_signal", "close", "rsi", "rolling_low", "regime_ok"}
    assert set(sig.columns) == expected
    assert sig["entry_signal"].dtype == bool
    assert sig["exit_signal"].dtype == bool
    assert sig["regime_ok"].dtype == bool


def test_misaligned_regime_raises() -> None:
    bars = _make_bars([100.0] * 50)
    wrong_regime = pd.Series(True, index=pd.date_range("2025-01-01", periods=50, freq="B"))
    with pytest.raises(ValueError, match="must match"):
        generate_signals(bars, VARIANT_A, wrong_regime)


def test_missing_columns_raises() -> None:
    bad = pd.DataFrame({"close": [100.0] * 50}, index=pd.date_range("2024-01-01", periods=50, freq="B"))
    regime = pd.Series(True, index=bad.index)
    with pytest.raises(KeyError, match="low"):
        generate_signals(bad, VARIANT_A, regime)
