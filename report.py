"""
Reporting.

Takes a RunReport (or BacktestResult) and formats it for human reading.
Currently console/log output; email and other channels can be added by
calling these formatters and piping the string to whatever transport.
"""

from __future__ import annotations

import math

from rsi_revert.backtest import BacktestResult, format_metrics
from rsi_revert.live import RunReport


def format_run_report(report: RunReport) -> str:
    """
    Format a RunReport as a human-readable string for console/log output.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"RSI-REVERT DAILY RUN — {report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
    if report.dry_run:
        lines.append("*** DRY RUN MODE — no orders submitted ***")
    lines.append("=" * 70)

    if report.kill_switch_engaged:
        lines.append("")
        lines.append("⚠  KILL_SWITCH ENGAGED — no trading performed.")
        lines.append("=" * 70)
        return "\n".join(lines)

    # Account snapshot.
    if report.account_equity is not None:
        lines.append("")
        lines.append("ACCOUNT")
        lines.append("-" * 70)
        lines.append(f"  Equity:        ${report.account_equity:>12,.2f}")
        if report.account_cash is not None:
            lines.append(f"  Cash:          ${report.account_cash:>12,.2f}")

    # Per-symbol decisions.
    lines.append("")
    lines.append("SYMBOL ACTIONS")
    lines.append("-" * 70)
    if not report.symbol_actions:
        lines.append("  (none)")
    else:
        # Header row.
        lines.append(f"  {'Symbol':<8}{'Decision':<14}{'RSI':>8}  {'Close':>10}  Reason")
        for a in report.symbol_actions:
            rsi_str = (
                f"{a.rsi:>8.1f}"
                if a.rsi is not None and not math.isnan(a.rsi)
                else f"{'—':>8}"
            )
            close_str = (
                f"${a.close:>9,.2f}"
                if a.close is not None
                else f"{'—':>10}"
            )
            lines.append(f"  {a.symbol:<8}{a.decision:<14}{rsi_str}  {close_str}  {a.reason}")
            if a.position_before is not None:
                lines.append(
                    f"           Position before: {a.position_before.qty} sh "
                    f"@ avg ${a.position_before.avg_entry_price:.2f} "
                    f"(unrealized P&L ${a.position_before.unrealized_pl:+,.2f})"
                )
            if a.order_submitted is not None:
                o = a.order_submitted
                stop_str = f" stop=${o.stop_price:.2f}" if o.stop_price else ""
                lines.append(
                    f"           Order: {o.side} {o.qty} {o.symbol} "
                    f"[{o.order_type}{stop_str}] id={o.id}"
                )
            if a.error:
                lines.append(f"           ERROR: {a.error}")

    # Errors.
    if report.errors:
        lines.append("")
        lines.append("ERRORS")
        lines.append("-" * 70)
        for err in report.errors:
            lines.append(f"  • {err}")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_backtest_comparison(
    strategy: BacktestResult,
    benchmark: BacktestResult,
) -> str:
    """
    Side-by-side comparison of a strategy vs a benchmark backtest.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"BACKTEST: {strategy.label}  vs  {benchmark.label}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(strategy.label.upper())
    lines.append("-" * 70)
    lines.append(format_metrics(strategy.metrics))
    lines.append("")
    lines.append(benchmark.label.upper())
    lines.append("-" * 70)
    lines.append(format_metrics(benchmark.metrics))
    lines.append("=" * 70)
    return "\n".join(lines)
