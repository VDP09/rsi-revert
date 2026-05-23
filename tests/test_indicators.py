"""
Tests for rsi_revert.indicators.

Strategy: test mathematical properties (RSI=100 on monotonic up, RSI=0 on
monotonic down, RSI=50 on flat) plus a hand-computed reference for RSI(2)
on a small series. This catches the most common implementation bugs:
- using SMA instead of Wilder smoothing
- off-by-one in the seed
- wrong handling of the first delta (which is NaN)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsi_revert.indicators import rsi, sma, rolling_low


# --------------------------- SMA ---------------------------


def test_sma_basic() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(s, period=3)
    # First two are NaN, then means of (1,2,3), (2,3,4), (3,4,5)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(3.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_sma_period_one_is_identity() -> None:
    s = pd.Series([10.0, 20.0, 30.0])
    result = sma(s, period=1)
    pd.testing.assert_series_equal(result, s, check_names=False)


def test_sma_rejects_invalid_period() -> None:
    with pytest.raises(ValueError):
        sma(pd.Series([1.0]), period=0)


# ----------------------- rolling_low -----------------------


def test_rolling_low_basic() -> None:
    s = pd.Series([5.0, 3.0, 4.0, 2.0, 6.0])
    result = rolling_low(s, period=3)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 3.0  # min(5,3,4)
    assert result.iloc[3] == 2.0  # min(3,4,2)
    assert result.iloc[4] == 2.0  # min(4,2,6)


def test_rolling_low_includes_current_bar() -> None:
    # Stop loss logic depends on this: the current bar IS in the window.
    s = pd.Series([10.0, 10.0, 10.0, 1.0])
    result = rolling_low(s, period=3)
    assert result.iloc[3] == 1.0  # current bar's low wins


# --------------------------- RSI ---------------------------


def test_rsi_monotonic_up_is_100() -> None:
    # Strictly increasing prices → all gains, no losses → RSI = 100.
    s = pd.Series(np.arange(1.0, 50.0))
    result = rsi(s, period=14)
    # First 14 are NaN, rest should be 100.
    assert result.iloc[:14].isna().all()
    assert (result.iloc[14:] == 100.0).all()


def test_rsi_monotonic_down_is_0() -> None:
    s = pd.Series(np.arange(50.0, 1.0, -1.0))
    result = rsi(s, period=14)
    assert result.iloc[:14].isna().all()
    assert (result.iloc[14:] == 0.0).all()


def test_rsi_flat_series_is_50() -> None:
    # No movement at all → both avg_gain and avg_loss are 0 → RSI = 50.
    s = pd.Series([100.0] * 30)
    result = rsi(s, period=14)
    assert result.iloc[:14].isna().all()
    assert (result.iloc[14:] == 50.0).all()


def test_rsi_returns_all_nan_when_insufficient_data() -> None:
    s = pd.Series([1.0, 2.0, 3.0])
    result = rsi(s, period=14)
    assert result.isna().all()


def test_rsi_values_in_valid_range() -> None:
    rng = np.random.default_rng(42)
    # Random walk
    prices = pd.Series(100.0 + rng.standard_normal(500).cumsum())
    result = rsi(prices, period=14)
    valid = result.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_rsi_period_2_hand_computed() -> None:
    """
    Hand-computed RSI(2) on a 5-bar series.

    Prices:  10, 11, 10, 12, 11
    Deltas:   _, +1, -1, +2, -1
    Gains:    _,  1,  0,  2,  0
    Losses:   _,  0,  1,  0,  1

    Seed at index 2 (need 2 deltas):
      avg_gain[2] = mean(gains[1..2]) = mean(1, 0) = 0.5
      avg_loss[2] = mean(losses[1..2]) = mean(0, 1) = 0.5
      RS = 1.0, RSI = 50.0

    Index 3: avg_gain = (0.5 * 1 + 2) / 2 = 1.25
             avg_loss = (0.5 * 1 + 0) / 2 = 0.25
             RS = 5.0, RSI = 100 - 100/6 = 83.333...

    Index 4: avg_gain = (1.25 * 1 + 0) / 2 = 0.625
             avg_loss = (0.25 * 1 + 1) / 2 = 0.625
             RS = 1.0, RSI = 50.0
    """
    s = pd.Series([10.0, 11.0, 10.0, 12.0, 11.0])
    result = rsi(s, period=2)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(50.0)
    assert result.iloc[3] == pytest.approx(83.333333, abs=1e-4)
    assert result.iloc[4] == pytest.approx(50.0)


def test_rsi_preserves_index() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    s = pd.Series(np.linspace(100, 110, 30), index=idx)
    result = rsi(s, period=14)
    pd.testing.assert_index_equal(result.index, idx)


def test_rsi_rejects_invalid_period() -> None:
    with pytest.raises(ValueError):
        rsi(pd.Series([1.0, 2.0]), period=0)
