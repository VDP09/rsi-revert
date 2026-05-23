"""
Technical indicators for the trading system.

All functions are pure: same input → same output, no side effects, no IO.
They take a pandas Series (or DataFrame for `rolling_low`) and return a
Series of the same length, with NaNs at the start where there isn't
enough history to compute a value.

Why we implement RSI from scratch:
- `pandas-ta` is unmaintained and incompatible with numpy 2.x.
- RSI has two common implementations; we want Wilder's smoothing, which
  is what TradingView, Connors, and most charting platforms use. The
  "simple moving average" variant gives different values and would not
  match any reference numbers you look up.

Wilder's smoothing is equivalent to an EMA with alpha = 1/period, but
the first value is seeded as the SMA of the first `period` values
rather than the first value itself. We implement it explicitly so the
math is auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple moving average over `period` bars.

    Returns NaN for the first `period - 1` rows (not enough history).

    Parameters
    ----------
    series : pd.Series
        Numeric series, typically close prices.
    period : int
        Lookback window. Must be >= 1.

    Returns
    -------
    pd.Series
        Same index as input, named "sma_{period}".
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    result = series.rolling(window=period, min_periods=period).mean()
    result.name = f"sma_{period}"
    return result


def rolling_low(series: pd.Series, period: int) -> pd.Series:
    """
    Lowest value over the trailing `period` bars (inclusive of current bar).

    Used for stop-loss placement: stop = rolling_low(low, 10) at entry.
    The current bar is included, so on entry day the stop is the lowest
    low of the last 10 days including today.

    Returns NaN for the first `period - 1` rows.

    Parameters
    ----------
    series : pd.Series
        Numeric series, typically low prices.
    period : int
        Lookback window. Must be >= 1.

    Returns
    -------
    pd.Series
        Same index as input, named "low_{period}".
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    result = series.rolling(window=period, min_periods=period).min()
    result.name = f"low_{period}"
    return result


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing.

    The math:
        delta = price - price.shift(1)
        gain = max(delta, 0)
        loss = max(-delta, 0)
        avg_gain[0..period-1] = NaN
        avg_gain[period] = mean(gain[1..period])         # seed with SMA
        avg_gain[t]  = (avg_gain[t-1] * (period-1) + gain[t]) / period
        avg_loss analogous
        RS  = avg_gain / avg_loss
        RSI = 100 - 100 / (1 + RS)

    Edge cases:
    - If avg_loss == 0 (pure up move over the window), RS is +inf and
      RSI = 100. We handle this without raising.
    - First `period` rows are NaN — we need `period` deltas to seed.

    Parameters
    ----------
    series : pd.Series
        Price series, typically close.
    period : int
        Lookback. Default 14 (classic). Use 2 for Connors-style.

    Returns
    -------
    pd.Series
        Values in [0, 100], same index as input, named "rsi_{period}".
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if len(series) < period + 1:
        # Not enough data to compute a single value. Return all-NaN.
        out = pd.Series(np.nan, index=series.index, name=f"rsi_{period}")
        return out

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing via EMA with alpha = 1/period and adjust=False.
    # We seed with the SMA of the first `period` values explicitly to
    # match the textbook formulation exactly. pandas' ewm with adjust=False
    # alone would seed from the first value, which is not what Wilder
    # specifies.
    avg_gain = pd.Series(np.nan, index=series.index)
    avg_loss = pd.Series(np.nan, index=series.index)

    # First valid index: position `period` (0-indexed), because delta[0]
    # is NaN and we need `period` real deltas (positions 1..period).
    first_valid = period
    avg_gain.iloc[first_valid] = gain.iloc[1 : period + 1].mean()
    avg_loss.iloc[first_valid] = loss.iloc[1 : period + 1].mean()

    # Iteratively apply Wilder smoothing for the rest.
    # We use the recurrence: avg[t] = (avg[t-1] * (p-1) + x[t]) / p
    # This is a hot loop; for a few thousand bars it's fine. If we ever
    # do millions of bars per symbol, we'd vectorize via ewm().
    gain_values = gain.to_numpy()
    loss_values = loss.to_numpy()
    # copy=True ensures writable arrays; the default can return read-only
    # views depending on pandas/numpy versions.
    avg_gain_values = avg_gain.to_numpy(copy=True)
    avg_loss_values = avg_loss.to_numpy(copy=True)
    for t in range(first_valid + 1, len(series)):
        avg_gain_values[t] = (avg_gain_values[t - 1] * (period - 1) + gain_values[t]) / period
        avg_loss_values[t] = (avg_loss_values[t - 1] * (period - 1) + loss_values[t]) / period

    avg_gain = pd.Series(avg_gain_values, index=series.index)
    avg_loss = pd.Series(avg_loss_values, index=series.index)

    # Handle div-by-zero: where avg_loss == 0, RSI is 100.
    # Where both avg_gain and avg_loss are 0 (flat market), Wilder convention
    # is RSI = 50, but in practice this is so rare that defaulting to 100
    # via the +inf path is harmless. We explicitly set 50 to be correct.
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))

    # Flat-market case: both gain and loss zero → rs is 0/0 = NaN → out is NaN.
    # Replace those with 50.
    flat_mask = (avg_gain == 0) & (avg_loss == 0)
    out[flat_mask] = 50.0

    # Pure-up case: avg_loss == 0, avg_gain > 0 → rs = inf → out = 100. ✓
    # Pure-down: avg_gain == 0, avg_loss > 0 → rs = 0 → out = 0. ✓

    out.name = f"rsi_{period}"
    return out
