"""
Signal generation.

Pure function: given a price dataframe and a regime filter, return a
dataframe of trade signals. No state, no IO, no position tracking — that's
the backtester's job. This module just says "at the close of bar t,
conditions are met for an entry/exit."

Convention: signal at bar t is evaluated using data through bar t's close.
The backtester is responsible for executing on bar t+1 (next open or
next close, depending on its convention). This separation avoids
look-ahead bias by construction — signals are never computed using
future data.

Two variants are predefined as VARIANT_A and VARIANT_B constants. You
can also construct your own VariantParams for parameter sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from rsi_revert.indicators import rolling_low, rsi, sma


ExitKind = Literal["rsi_threshold", "sma_cross"]


@dataclass(frozen=True)
class VariantParams:
    """
    Strategy variant configuration.

    Attributes
    ----------
    name : str
        Human label for logs and reports (e.g. "A_rsi14", "B_connors").
    rsi_period : int
        Lookback for the RSI entry signal.
    rsi_entry_threshold : float
        Enter long when RSI crosses below this value (e.g. 30 or 10).
    exit_kind : {"rsi_threshold", "sma_cross"}
        How to exit:
        - "rsi_threshold": exit when RSI rises above `rsi_exit_threshold`.
        - "sma_cross": exit when close rises above SMA(`exit_sma_period`).
    rsi_exit_threshold : float | None
        Required when exit_kind == "rsi_threshold".
    exit_sma_period : int | None
        Required when exit_kind == "sma_cross".
    stop_lookback : int
        Window for the rolling low used as the initial stop.
    """

    name: str
    rsi_period: int
    rsi_entry_threshold: float
    exit_kind: ExitKind
    rsi_exit_threshold: float | None = None
    exit_sma_period: int | None = None
    stop_lookback: int = 10

    def __post_init__(self) -> None:
        if self.exit_kind == "rsi_threshold" and self.rsi_exit_threshold is None:
            raise ValueError("rsi_exit_threshold required when exit_kind='rsi_threshold'")
        if self.exit_kind == "sma_cross" and self.exit_sma_period is None:
            raise ValueError("exit_sma_period required when exit_kind='sma_cross'")


# Classic RSI(14) with 30/70 thresholds.
VARIANT_A = VariantParams(
    name="A_rsi14",
    rsi_period=14,
    rsi_entry_threshold=30.0,
    exit_kind="rsi_threshold",
    rsi_exit_threshold=70.0,
    stop_lookback=10,
)

# Connors-style RSI(2) with 5-day SMA exit.
VARIANT_B = VariantParams(
    name="B_connors",
    rsi_period=2,
    rsi_entry_threshold=10.0,
    exit_kind="sma_cross",
    exit_sma_period=5,
    stop_lookback=10,
)


def compute_regime_filter(bars: pd.DataFrame, sma_period: int = 200) -> pd.Series:
    """
    Boolean series: True when close > SMA(sma_period).

    Computed on whatever bars you pass — typically SPY, used as the
    market regime filter for every symbol traded.

    Parameters
    ----------
    bars : pd.DataFrame
        Must have a 'close' column.
    sma_period : int
        Default 200 (classic long-term trend filter).

    Returns
    -------
    pd.Series
        Boolean, same index as bars. NaN comparisons (early bars before
        the SMA is defined) resolve to False — entries blocked until we
        have enough history.
    """
    long_sma = sma(bars["close"], period=sma_period)
    # NaN > anything is False, but explicit fillna(False) avoids any
    # pandas surprises with future-warning behavior on boolean ops.
    regime = (bars["close"] > long_sma).fillna(False)
    regime.name = f"regime_above_sma{sma_period}"
    return regime


def generate_signals(
    bars: pd.DataFrame,
    params: VariantParams,
    regime_ok: pd.Series,
) -> pd.DataFrame:
    """
    Generate entry and exit signals for one symbol under one variant.

    Parameters
    ----------
    bars : pd.DataFrame
        Must have 'close' and 'low' columns. Index is the bar timestamp.
    params : VariantParams
        Variant configuration (RSI period, thresholds, exit rule, stop).
    regime_ok : pd.Series
        Boolean series aligned with `bars.index`. True when the market
        regime permits long entries. Typically computed once on SPY via
        `compute_regime_filter` and shared across symbols.

    Returns
    -------
    pd.DataFrame
        Same index as `bars`, with columns:
        - entry_signal (bool):  enter long at next bar's execution price
        - exit_signal (bool):   exit any open position at next execution
        - close (float):        passed through for backtester convenience
        - rsi (float):          the RSI value used (for auditing)
        - rolling_low (float):  stop reference at this bar
        - regime_ok (bool):     copied through for auditing

    Raises
    ------
    KeyError
        If `bars` is missing 'close' or 'low'.
    ValueError
        If `regime_ok.index` doesn't match `bars.index`.
    """
    for col in ("close", "low"):
        if col not in bars.columns:
            raise KeyError(f"bars must have a '{col}' column")
    if not bars.index.equals(regime_ok.index):
        raise ValueError("regime_ok.index must match bars.index")

    close = bars["close"]
    low = bars["low"]

    rsi_values = rsi(close, period=params.rsi_period)
    stop_ref = rolling_low(low, period=params.stop_lookback)

    # Entry: RSI below threshold AND regime permits long entries.
    # NaN < threshold is False, so early bars (before RSI is defined)
    # naturally produce no entry.
    entry_signal = (rsi_values < params.rsi_entry_threshold) & regime_ok
    entry_signal = entry_signal.fillna(False).astype(bool)

    # Exit: depends on variant.
    if params.exit_kind == "rsi_threshold":
        exit_signal = rsi_values > params.rsi_exit_threshold  # type: ignore[operator]
    else:  # sma_cross
        exit_sma = sma(close, period=params.exit_sma_period)  # type: ignore[arg-type]
        exit_signal = close > exit_sma
    exit_signal = exit_signal.fillna(False).astype(bool)

    return pd.DataFrame(
        {
            "entry_signal": entry_signal,
            "exit_signal": exit_signal,
            "close": close,
            "rsi": rsi_values,
            "rolling_low": stop_ref,
            "regime_ok": regime_ok.astype(bool),
        },
        index=bars.index,
    )
