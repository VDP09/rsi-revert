"""
Backtester.

Takes a signals dataframe (from rsi_revert.signals) and the underlying
bars, walks forward bar-by-bar, simulates trades, and returns:
- the equity curve (Series indexed by date)
- a trade log (DataFrame, one row per closed trade)
- performance metrics (CAGR, Sharpe, max drawdown, win rate, etc.)

Execution model
---------------
Signals are evaluated at the close of bar t. Execution happens at the
open of bar t+1. The backtester at iteration t therefore consults
signals.iloc[t-1] when deciding what to do at bars[t].open.

Within each bar, events fire in this order:
  1. Stop loss check (resting order, fires first if triggered)
       - If open[t] <= stop: filled at open (gap through stop)
       - Elif low[t] <= stop: filled at stop level (intraday hit)
  2. Signal exit (from the previous bar's exit_signal)
  3. Signal entry (from the previous bar's entry_signal)
  4. Mark-to-market at close[t] for the equity curve

Slippage is applied unfavorably: buys at price*(1+s), sells at price*(1-s).
Commission defaults to zero (Alpaca) but is configurable.

Position sizing
---------------
    risk_per_share = entry_price - stop_price
    shares = floor( (risk_per_trade * equity) / risk_per_share )
    shares = min(shares, floor(cash / entry_price))   # never over-leverage

If risk_per_share <= 0 (stop above entry — shouldn't happen, but be safe)
or shares == 0 (insufficient cash), the trade is skipped and logged.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    """
    Runtime configuration for the simulator.

    Attributes
    ----------
    initial_equity : float
        Starting cash in dollars.
    risk_per_trade : float
        Fraction of equity risked per trade (0.01 = 1%).
    slippage_bps : float
        Slippage in basis points (5.0 = 0.05% = 0.0005). Applied
        unfavorably to every fill.
    commission_per_share : float
        Per-share commission. Alpaca = 0.
    trading_days_per_year : int
        Used for CAGR and Sharpe annualization. 252 is standard.
    """

    initial_equity: float = 100_000.0
    risk_per_trade: float = 0.01
    slippage_bps: float = 5.0
    commission_per_share: float = 0.0
    trading_days_per_year: int = 252

    @property
    def slippage_fraction(self) -> float:
        return self.slippage_bps / 10_000.0


ExitReason = Literal["signal", "stop", "end_of_data"]


@dataclass
class _OpenPosition:
    """Internal state for a currently-held position."""
    entry_date: pd.Timestamp
    entry_price: float  # net of slippage
    shares: int
    stop_price: float


@dataclass
class BacktestResult:
    """
    Output of a backtest.

    Attributes
    ----------
    equity_curve : pd.Series
        Daily mark-to-market equity, indexed by bar timestamp.
    trades : pd.DataFrame
        One row per closed trade. Columns:
        entry_date, entry_price, exit_date, exit_price, shares,
        stop_price, exit_reason, pnl, return_pct, hold_days.
    metrics : dict
        Performance summary (see compute_metrics).
    config : BacktestConfig
        Configuration the backtest was run with.
    label : str
        Human label for plots and reports.
    """
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict
    config: BacktestConfig
    label: str = "strategy"


def _apply_slippage(price: float, side: Literal["buy", "sell"], frac: float) -> float:
    """Apply unfavorable slippage. Buys pay more, sells receive less."""
    if side == "buy":
        return price * (1.0 + frac)
    return price * (1.0 - frac)


def run_backtest(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    config: BacktestConfig | None = None,
    label: str = "strategy",
) -> BacktestResult:
    """
    Run the bar-by-bar simulation.

    Parameters
    ----------
    bars : pd.DataFrame
        Must contain 'open', 'high', 'low', 'close'. Index is the
        bar timestamp. Aligned with `signals`.
    signals : pd.DataFrame
        Output of rsi_revert.signals.generate_signals. Must contain
        'entry_signal', 'exit_signal', 'rolling_low'.
    config : BacktestConfig
        If None, uses defaults.
    label : str
        Identifier for reports (e.g. "VariantA_SPY").

    Returns
    -------
    BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    for col in ("open", "high", "low", "close"):
        if col not in bars.columns:
            raise KeyError(f"bars must have a '{col}' column")
    for col in ("entry_signal", "exit_signal", "rolling_low"):
        if col not in signals.columns:
            raise KeyError(f"signals must have a '{col}' column")
    if not bars.index.equals(signals.index):
        raise ValueError("bars.index must match signals.index")

    cash: float = config.initial_equity
    position: _OpenPosition | None = None
    trades: list[dict] = []
    equity_history: list[float] = []

    # Pre-extract arrays for speed; bar-by-bar Python is the slow path.
    opens = bars["open"].to_numpy()
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    closes = bars["close"].to_numpy()
    timestamps = bars.index

    entry_sig = signals["entry_signal"].to_numpy()
    exit_sig = signals["exit_signal"].to_numpy()
    stop_refs = signals["rolling_low"].to_numpy()

    slip = config.slippage_fraction
    n = len(bars)

    # Bar 0: no execution possible (no prior signal). Just mark equity.
    equity_history.append(cash)

    for t in range(1, n):
        bar_open = opens[t]
        bar_low = lows[t]
        bar_close = closes[t]
        ts = timestamps[t]

        # --- 1. Stop check (resting order) ---
        if position is not None:
            stop = position.stop_price
            if not math.isnan(stop):
                if bar_open <= stop:
                    # Gap-down through stop: fill at open, then slippage.
                    fill = _apply_slippage(bar_open, "sell", slip)
                    _close_trade(
                        position, ts, fill, "stop", trades, config
                    )
                    cash += fill * position.shares
                    cash -= config.commission_per_share * position.shares
                    position = None
                elif bar_low <= stop:
                    # Intraday hit: assume stop order fills at stop price.
                    fill = _apply_slippage(stop, "sell", slip)
                    _close_trade(
                        position, ts, fill, "stop", trades, config
                    )
                    cash += fill * position.shares
                    cash -= config.commission_per_share * position.shares
                    position = None

        # --- 2. Signal-based exit (from previous bar's signal) ---
        if position is not None and bool(exit_sig[t - 1]):
            fill = _apply_slippage(bar_open, "sell", slip)
            _close_trade(position, ts, fill, "signal", trades, config)
            cash += fill * position.shares
            cash -= config.commission_per_share * position.shares
            position = None

        # --- 3. Signal-based entry (from previous bar's signal) ---
        if position is None and bool(entry_sig[t - 1]):
            entry_fill = _apply_slippage(bar_open, "buy", slip)
            stop_level = stop_refs[t - 1]
            risk_per_share = entry_fill - stop_level

            if math.isnan(stop_level) or risk_per_share <= 0:
                # Pathological: stop above entry. Skip and log at debug.
                logger.debug(
                    "%s: skipping entry — bad stop (entry=%.2f stop=%.2f)",
                    ts, entry_fill, stop_level,
                )
            else:
                # Equity for sizing: yesterday's close-of-day equity.
                equity_for_sizing = equity_history[-1]
                target_risk = config.risk_per_trade * equity_for_sizing
                shares_by_risk = math.floor(target_risk / risk_per_share)
                shares_by_cash = math.floor(cash / entry_fill)
                shares = max(0, min(shares_by_risk, shares_by_cash))

                if shares > 0:
                    cost = entry_fill * shares + config.commission_per_share * shares
                    cash -= cost
                    position = _OpenPosition(
                        entry_date=ts,
                        entry_price=entry_fill,
                        shares=shares,
                        stop_price=stop_level,
                    )

        # --- 4. Mark-to-market ---
        mtm_equity = cash + (position.shares * bar_close if position else 0.0)
        equity_history.append(mtm_equity)

    # End of data: close any open position at the last close.
    if position is not None:
        last_close = closes[-1]
        fill = _apply_slippage(last_close, "sell", slip)
        _close_trade(position, timestamps[-1], fill, "end_of_data", trades, config)
        cash += fill * position.shares
        cash -= config.commission_per_share * position.shares
        # Update final equity entry (we already appended close MTM, now it's pure cash).
        equity_history[-1] = cash
        position = None

    equity_curve = pd.Series(equity_history, index=bars.index, name=label)
    trades_df = pd.DataFrame(trades) if trades else _empty_trades_df()
    metrics = compute_metrics(equity_curve, trades_df, config)

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades_df,
        metrics=metrics,
        config=config,
        label=label,
    )


def _close_trade(
    pos: _OpenPosition,
    exit_date: pd.Timestamp,
    exit_price: float,
    reason: ExitReason,
    trades: list[dict],
    config: BacktestConfig,
) -> None:
    """Record a closed trade. Does NOT mutate cash — caller does that."""
    gross_pnl = (exit_price - pos.entry_price) * pos.shares
    commission = config.commission_per_share * pos.shares * 2  # entry + exit
    pnl = gross_pnl - commission
    return_pct = (exit_price / pos.entry_price) - 1.0
    hold_days = (exit_date - pos.entry_date).days
    trades.append(
        {
            "entry_date": pos.entry_date,
            "entry_price": pos.entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "shares": pos.shares,
            "stop_price": pos.stop_price,
            "exit_reason": reason,
            "pnl": pnl,
            "return_pct": return_pct,
            "hold_days": hold_days,
        }
    )


def _empty_trades_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "entry_date", "entry_price", "exit_date", "exit_price",
            "shares", "stop_price", "exit_reason", "pnl", "return_pct", "hold_days",
        ]
    )


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    config: BacktestConfig,
) -> dict:
    """
    Compute standard performance metrics from an equity curve and trade log.

    Returns
    -------
    dict
        total_return, cagr, max_drawdown, sharpe, n_trades, win_rate,
        avg_win, avg_loss, profit_factor, avg_hold_days. All values are
        Python floats/ints (JSON-serializable).
    """
    initial = float(equity_curve.iloc[0])
    final = float(equity_curve.iloc[-1])
    total_return = final / initial - 1.0

    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (final / initial) ** (1.0 / years) - 1.0 if final > 0 else -1.0

    # Drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    max_drawdown = float(drawdown.min())

    # Sharpe (risk-free = 0 for simplicity; common backtest convention)
    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = float(
            math.sqrt(config.trading_days_per_year)
            * daily_returns.mean() / daily_returns.std()
        )
    else:
        sharpe = 0.0

    # Trade stats (exclude end_of_data closures from win-rate so the
    # number reflects actual strategy decisions).
    decision_trades = trades[trades["exit_reason"] != "end_of_data"] if len(trades) else trades
    n_trades = int(len(decision_trades))
    if n_trades > 0:
        wins = decision_trades[decision_trades["pnl"] > 0]
        losses = decision_trades[decision_trades["pnl"] <= 0]
        win_rate = float(len(wins) / n_trades)
        avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0
        total_wins = float(wins["pnl"].sum())
        total_losses_abs = float(abs(losses["pnl"].sum()))
        profit_factor = total_wins / total_losses_abs if total_losses_abs > 0 else float("inf")
        avg_hold_days = float(decision_trades["hold_days"].mean())
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        avg_hold_days = 0.0

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe),
        "n_trades": n_trades,
        "win_rate": float(win_rate),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_hold_days": avg_hold_days,
        "final_equity": final,
    }


def buy_and_hold(
    bars: pd.DataFrame,
    config: BacktestConfig | None = None,
    label: str = "buy_and_hold",
) -> BacktestResult:
    """
    Benchmark: invest the full initial equity at the first bar's open,
    hold to the last bar's close. Same slippage on entry/exit. No stops.
    """
    if config is None:
        config = BacktestConfig()

    slip = config.slippage_fraction
    first_open = float(bars["open"].iloc[0])
    last_close = float(bars["close"].iloc[-1])
    entry_fill = _apply_slippage(first_open, "buy", slip)
    exit_fill = _apply_slippage(last_close, "sell", slip)

    shares = math.floor(config.initial_equity / entry_fill)
    cash_leftover = config.initial_equity - shares * entry_fill

    # Daily mark-to-market.
    equity = cash_leftover + shares * bars["close"]
    # Apply exit slippage to the final bar's mark.
    equity.iloc[-1] = cash_leftover + shares * exit_fill
    equity.name = label

    trades_df = pd.DataFrame(
        [
            {
                "entry_date": bars.index[0],
                "entry_price": entry_fill,
                "exit_date": bars.index[-1],
                "exit_price": exit_fill,
                "shares": shares,
                "stop_price": float("nan"),
                "exit_reason": "end_of_data",
                "pnl": (exit_fill - entry_fill) * shares,
                "return_pct": (exit_fill / entry_fill) - 1.0,
                "hold_days": (bars.index[-1] - bars.index[0]).days,
            }
        ]
    )
    metrics = compute_metrics(equity, trades_df, config)
    return BacktestResult(equity, trades_df, metrics, config, label=label)


def format_metrics(metrics: dict) -> str:
    """Pretty-print metrics for console/log output."""
    lines = [
        f"  Total return:   {metrics['total_return']:>8.2%}",
        f"  CAGR:           {metrics['cagr']:>8.2%}",
        f"  Max drawdown:   {metrics['max_drawdown']:>8.2%}",
        f"  Sharpe:         {metrics['sharpe']:>8.2f}",
        f"  # trades:       {metrics['n_trades']:>8d}",
        f"  Win rate:       {metrics['win_rate']:>8.2%}",
        f"  Avg win:        ${metrics['avg_win']:>8,.0f}",
        f"  Avg loss:       ${metrics['avg_loss']:>8,.0f}",
        f"  Profit factor:  {metrics['profit_factor']:>8.2f}",
        f"  Avg hold days:  {metrics['avg_hold_days']:>8.1f}",
        f"  Final equity:   ${metrics['final_equity']:>10,.0f}",
    ]
    return "\n".join(lines)
