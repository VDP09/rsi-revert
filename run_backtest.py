#!/usr/bin/env python
"""
Backtest entry point.

Runs both variants of the strategy on the configured universe, plus
buy-and-hold benchmark, plus walk-forward validation. Writes:
- console summary
- trades CSV per variant
- equity-curve CSV per variant
- walk-forward summary CSV

Usage:
    python scripts/run_backtest.py [--config path/to/config.yaml]
                                   [--out-dir reports/]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from rsi_revert.backtest import BacktestConfig, buy_and_hold, run_backtest
from rsi_revert.data import get_bars
from rsi_revert.report import format_backtest_comparison
from rsi_revert.signals import (
    VARIANT_A,
    VARIANT_B,
    compute_regime_filter,
    generate_signals,
)
from rsi_revert.utils import load_config, setup_logging
from rsi_revert.walkforward import (
    WalkForwardConfig,
    format_walkforward_summary,
    walk_forward,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest runner")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(
        log_dir=cfg["runtime"].get("log_dir", "logs"),
        level=cfg["runtime"].get("log_level", "INFO"),
        name="run_backtest",
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bt_cfg = BacktestConfig(
        initial_equity=cfg["backtest"]["initial_equity"],
        risk_per_trade=cfg["backtest"]["risk_per_trade"],
        slippage_bps=cfg["backtest"]["slippage_bps"],
        commission_per_share=cfg["backtest"]["commission_per_share"],
    )
    history_start = cfg["backtest"]["history_start"]
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    # Single-symbol universe for now; first symbol is the trade target.
    symbol = cfg["universe"][0]
    logger.info("Fetching %s bars from %s to %s", symbol, history_start, end.date())
    bars = get_bars(symbol, history_start, end, cache_dir=cfg["runtime"]["cache_dir"])

    regime = compute_regime_filter(bars, sma_period=200)

    print()
    print("#" * 70)
    print(f"# rsi-revert backtest  —  {symbol}  {bars.index[0].date()} → {bars.index[-1].date()}")
    print("#" * 70)

    # Buy-and-hold benchmark.
    bh = buy_and_hold(bars, bt_cfg, label="buy_and_hold")

    # Each variant.
    for variant in (VARIANT_A, VARIANT_B):
        sig = generate_signals(bars, variant, regime)
        result = run_backtest(bars, sig, bt_cfg, label=variant.name)

        print()
        print(format_backtest_comparison(result, bh))

        # Persist trade log and equity curve.
        result.trades.to_csv(out_dir / f"trades_{variant.name}.csv", index=False)
        result.equity_curve.to_csv(out_dir / f"equity_{variant.name}.csv", header=True)
        logger.info(
            "%s: %d trades, CAGR=%.2f%%, MaxDD=%.2f%%, Sharpe=%.2f",
            variant.name, result.metrics["n_trades"],
            result.metrics["cagr"] * 100, result.metrics["max_drawdown"] * 100,
            result.metrics["sharpe"],
        )

    # Walk-forward (optimizer mode with both variants as candidates).
    wf_cfg = WalkForwardConfig(
        train_years=cfg.get("walk_forward", {}).get("train_years", 4),
        test_years=cfg.get("walk_forward", {}).get("test_years", 1),
        step_years=cfg.get("walk_forward", {}).get("step_years", 1),
    )
    try:
        wf = walk_forward(bars, [VARIANT_A, VARIANT_B], regime, bt_cfg, wf_cfg)
        print(format_walkforward_summary(wf))
        wf.summary.to_csv(out_dir / "walk_forward_summary.csv", index=False)
        wf.stitched_test_equity.to_csv(out_dir / "walk_forward_equity.csv", header=True)
    except ValueError as exc:
        logger.warning("Walk-forward skipped: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
