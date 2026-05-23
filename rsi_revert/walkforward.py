"""
Walk-forward validation.

Two modes:

1. **Stationarity check** (single VariantParams): run the same strategy on
   rolling train/test windows. Shows whether the fixed strategy works
   across different market regimes. If Variant A had a great 2010-2014
   and a terrible 2018-2022, you want to see that.

2. **Optimization walk-forward** (list of VariantParams): on each train
   window, pick the variant with the best metric (e.g. Sharpe). Apply
   that variant — and ONLY that variant — to the next test window.
   The gap between train and test metrics is the honest measure of
   overfitting: if train Sharpe averages 2.0 and test Sharpe averages
   0.3, the optimization is fitting noise.

The stitched test equity curve concatenates the out-of-sample test
segments end-to-end, with each test window starting from the previous
test window's final equity. This is the closest approximation to "what
would I have actually earned trading this system without future knowledge."

Lookback handling
-----------------
Signals are computed on the FULL bar series first, then sliced to each
window. This ensures the RSI/SMA values at the start of any window are
identical to what you'd compute given full history — there's no warmup
truncation. Slicing a pre-computed signals DataFrame is safe; computing
signals on a sliced window would discard early bars to warmup, biasing
results.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from rsi_revert.backtest import (
    BacktestConfig,
    compute_metrics,
    run_backtest,
)
from rsi_revert.signals import VariantParams, generate_signals

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardConfig:
    """
    Window configuration.

    Attributes
    ----------
    train_years : float
        Length of each training window in years.
    test_years : float
        Length of each test (out-of-sample) window in years.
    step_years : float
        How far forward to slide between consecutive windows. If
        step_years == test_years, test windows are non-overlapping and
        the stitched equity curve covers each calendar date exactly once.
    optimization_metric : str
        Which metric to maximize on the train window when optimizing.
        Default 'sharpe'. Any key in compute_metrics output works.
    """
    train_years: float = 4.0
    test_years: float = 1.0
    step_years: float = 1.0
    optimization_metric: str = "sharpe"


@dataclass
class WindowResult:
    """Results for one (train, test) window pair."""
    window_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: dict
    test_metrics: dict
    chosen_variant: str  # name of the VariantParams used on test


@dataclass
class WalkForwardResult:
    """Full walk-forward output."""
    windows: list[WindowResult]
    summary: pd.DataFrame  # one row per window, train+test metrics side-by-side
    stitched_test_equity: pd.Series  # compounded across test windows
    overall_test_metrics: dict  # metrics on the stitched curve as a whole
    degradation: dict  # train_mean - test_mean for key metrics


def make_windows(
    index: pd.DatetimeIndex,
    cfg: WalkForwardConfig,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Build a list of (train_start, train_end, test_start, test_end) tuples.

    Windows are anchored at the start of `index` and slide forward by
    `step_years` until the test window would extend past the end of the
    data. Boundaries snap to the nearest actual bar in `index` so we
    don't pick dates the market wasn't open.
    """
    if len(index) == 0:
        return []

    start = index[0]
    end = index[-1]
    train_delta = pd.Timedelta(days=int(round(cfg.train_years * 365.25)))
    test_delta = pd.Timedelta(days=int(round(cfg.test_years * 365.25)))
    step_delta = pd.Timedelta(days=int(round(cfg.step_years * 365.25)))

    windows = []
    train_start = start
    while True:
        train_end_target = train_start + train_delta
        test_start_target = train_end_target  # contiguous
        test_end_target = test_start_target + test_delta

        if test_end_target > end:
            break

        # Snap each boundary to the closest actual bar.
        train_start_candidates = index[index >= train_start]
        if len(train_start_candidates) == 0:
            break
        train_start_snap = train_start_candidates[0]
        # train_end: last bar <= target
        train_end_idx = index[index <= train_end_target]
        if len(train_end_idx) == 0:
            break
        train_end_snap = train_end_idx[-1]
        # test_start: first bar > train_end
        test_start_candidates = index[index > train_end_snap]
        if len(test_start_candidates) == 0:
            break
        test_start_snap = test_start_candidates[0]
        # test_end: last bar <= target
        test_end_candidates = index[index <= test_end_target]
        if len(test_end_candidates) == 0:
            break
        test_end_snap = test_end_candidates[-1]
        if test_end_snap <= test_start_snap:
            break

        windows.append((train_start_snap, train_end_snap, test_start_snap, test_end_snap))
        train_start = train_start + step_delta

    return windows


def walk_forward(
    bars: pd.DataFrame,
    params: VariantParams | Sequence[VariantParams],
    regime_ok: pd.Series,
    bt_config: BacktestConfig | None = None,
    wf_config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """
    Run walk-forward validation.

    Parameters
    ----------
    bars : pd.DataFrame
        Full bar history.
    params : VariantParams | Sequence[VariantParams]
        Single params → stationarity check.
        Multiple params → optimization on each train window.
    regime_ok : pd.Series
        Boolean regime filter aligned with bars.index.
    bt_config : BacktestConfig
        Backtest config. The initial_equity of the first test window
        comes from here; subsequent test windows compound forward.
    wf_config : WalkForwardConfig
        Window sizing and optimization config.

    Returns
    -------
    WalkForwardResult
    """
    if bt_config is None:
        bt_config = BacktestConfig()
    if wf_config is None:
        wf_config = WalkForwardConfig()

    # Normalize params to a list for uniform handling.
    if isinstance(params, VariantParams):
        param_list: list[VariantParams] = [params]
        optimizing = False
    else:
        param_list = list(params)
        if not param_list:
            raise ValueError("params list is empty")
        optimizing = len(param_list) > 1

    windows = make_windows(bars.index, wf_config)
    if not windows:
        raise ValueError(
            f"No valid windows. Need >={wf_config.train_years + wf_config.test_years:.1f} "
            f"years of data; got {(bars.index[-1] - bars.index[0]).days / 365.25:.1f}."
        )

    # Pre-compute signals for every candidate variant on the FULL series.
    # This is cheap and avoids any window-warmup issues.
    signals_by_variant: dict[str, pd.DataFrame] = {
        p.name: generate_signals(bars, p, regime_ok) for p in param_list
    }

    window_results: list[WindowResult] = []
    stitched_pieces: list[pd.Series] = []
    current_equity = bt_config.initial_equity

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        train_bars = bars.loc[tr_start:tr_end]

        # --- Pick the variant: either the single given one, or the best
        # on the train window by the configured metric.
        if optimizing:
            best_variant: VariantParams | None = None
            best_score = -math.inf
            best_train_metrics: dict = {}
            for cand in param_list:
                sig_train = signals_by_variant[cand.name].loc[tr_start:tr_end]
                # Always use the original initial_equity for the optimization
                # search so metrics are comparable across candidates.
                tr_result = run_backtest(
                    train_bars, sig_train,
                    BacktestConfig(
                        initial_equity=bt_config.initial_equity,
                        risk_per_trade=bt_config.risk_per_trade,
                        slippage_bps=bt_config.slippage_bps,
                        commission_per_share=bt_config.commission_per_share,
                        trading_days_per_year=bt_config.trading_days_per_year,
                    ),
                    label=f"train_{cand.name}",
                )
                score = tr_result.metrics.get(wf_config.optimization_metric, -math.inf)
                if score > best_score:
                    best_score = score
                    best_variant = cand
                    best_train_metrics = tr_result.metrics
            assert best_variant is not None
            chosen = best_variant
            train_metrics = best_train_metrics
        else:
            chosen = param_list[0]
            sig_train = signals_by_variant[chosen.name].loc[tr_start:tr_end]
            tr_result = run_backtest(
                train_bars, sig_train,
                BacktestConfig(
                    initial_equity=bt_config.initial_equity,
                    risk_per_trade=bt_config.risk_per_trade,
                    slippage_bps=bt_config.slippage_bps,
                    commission_per_share=bt_config.commission_per_share,
                    trading_days_per_year=bt_config.trading_days_per_year,
                ),
                label=f"train_{chosen.name}",
            )
            train_metrics = tr_result.metrics

        # --- Test: apply chosen variant to the test window, compounding
        # equity forward from the previous window's end.
        test_bars = bars.loc[te_start:te_end]
        sig_test = signals_by_variant[chosen.name].loc[te_start:te_end]
        te_result = run_backtest(
            test_bars, sig_test,
            BacktestConfig(
                initial_equity=current_equity,
                risk_per_trade=bt_config.risk_per_trade,
                slippage_bps=bt_config.slippage_bps,
                commission_per_share=bt_config.commission_per_share,
                trading_days_per_year=bt_config.trading_days_per_year,
            ),
            label=f"test_{chosen.name}_w{i}",
        )

        current_equity = float(te_result.equity_curve.iloc[-1])
        stitched_pieces.append(te_result.equity_curve)

        window_results.append(
            WindowResult(
                window_idx=i,
                train_start=tr_start,
                train_end=tr_end,
                test_start=te_start,
                test_end=te_end,
                train_metrics=train_metrics,
                test_metrics=te_result.metrics,
                chosen_variant=chosen.name,
            )
        )

        logger.info(
            "Window %d: train [%s, %s] sharpe=%.2f  →  test [%s, %s] sharpe=%.2f  variant=%s",
            i, tr_start.date(), tr_end.date(), train_metrics.get("sharpe", 0.0),
            te_start.date(), te_end.date(), te_result.metrics.get("sharpe", 0.0),
            chosen.name,
        )

    # Stitch test equity. Drop overlapping timestamps between adjacent pieces.
    stitched = stitched_pieces[0]
    for piece in stitched_pieces[1:]:
        non_overlap = piece[~piece.index.isin(stitched.index)]
        stitched = pd.concat([stitched, non_overlap])
    stitched.name = "walk_forward_test_equity"

    overall_test_metrics = compute_metrics(stitched, pd.DataFrame(), bt_config)
    summary = _build_summary(window_results)
    degradation = _compute_degradation(window_results)

    return WalkForwardResult(
        windows=window_results,
        summary=summary,
        stitched_test_equity=stitched,
        overall_test_metrics=overall_test_metrics,
        degradation=degradation,
    )


def _build_summary(windows: list[WindowResult]) -> pd.DataFrame:
    """Tidy DataFrame: one row per window with key metrics side-by-side."""
    rows = []
    for w in windows:
        rows.append({
            "window": w.window_idx,
            "train_start": w.train_start.date(),
            "train_end": w.train_end.date(),
            "test_start": w.test_start.date(),
            "test_end": w.test_end.date(),
            "variant": w.chosen_variant,
            "train_cagr": w.train_metrics.get("cagr"),
            "test_cagr": w.test_metrics.get("cagr"),
            "train_sharpe": w.train_metrics.get("sharpe"),
            "test_sharpe": w.test_metrics.get("sharpe"),
            "train_maxdd": w.train_metrics.get("max_drawdown"),
            "test_maxdd": w.test_metrics.get("max_drawdown"),
            "train_n_trades": w.train_metrics.get("n_trades"),
            "test_n_trades": w.test_metrics.get("n_trades"),
        })
    return pd.DataFrame(rows)


def _compute_degradation(windows: list[WindowResult]) -> dict:
    """
    Average train-vs-test gap for key metrics.

    Positive degradation = train was better than test (suggests overfitting).
    """
    if not windows:
        return {}
    keys = ("cagr", "sharpe", "max_drawdown")
    out = {}
    for k in keys:
        train_vals = [w.train_metrics.get(k, 0.0) for w in windows]
        test_vals = [w.test_metrics.get(k, 0.0) for w in windows]
        train_mean = sum(train_vals) / len(train_vals)
        test_mean = sum(test_vals) / len(test_vals)
        out[f"train_{k}_mean"] = float(train_mean)
        out[f"test_{k}_mean"] = float(test_mean)
        out[f"{k}_degradation"] = float(train_mean - test_mean)
    return out


def format_walkforward_summary(result: WalkForwardResult) -> str:
    """Pretty-print walk-forward results for console output."""
    lines = ["", "=" * 70, "WALK-FORWARD RESULTS", "=" * 70, ""]
    lines.append(result.summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    lines.extend(["", "Train vs test (means across windows):", ""])
    deg = result.degradation
    lines.append(f"  CAGR:    train {deg['train_cagr_mean']:.2%}  test {deg['test_cagr_mean']:.2%}  "
                 f"degradation {deg['cagr_degradation']:.2%}")
    lines.append(f"  Sharpe:  train {deg['train_sharpe_mean']:.2f}    test {deg['test_sharpe_mean']:.2f}    "
                 f"degradation {deg['sharpe_degradation']:.2f}")
    lines.append(f"  MaxDD:   train {deg['train_max_drawdown_mean']:.2%}  test {deg['test_max_drawdown_mean']:.2%}")
    lines.extend(["", "Stitched out-of-sample equity curve metrics:", ""])
    m = result.overall_test_metrics
    lines.append(f"  Total return: {m['total_return']:.2%}")
    lines.append(f"  CAGR:         {m['cagr']:.2%}")
    lines.append(f"  Max DD:       {m['max_drawdown']:.2%}")
    lines.append(f"  Sharpe:       {m['sharpe']:.2f}")
    lines.append("=" * 70)
    return "\n".join(lines)
