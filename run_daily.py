#!/usr/bin/env python
"""
Daily paper-trading entry point.

Intended to be invoked once per day after the US market closes, e.g.
from a GitHub Actions cron job at 21:05 UTC weekdays. Steps:

  1. Load config.
  2. Set up logging (rotating file + stderr).
  3. Build a Broker bound to Alpaca paper credentials.
  4. Call daily_run().
  5. Print a formatted RunReport.
  6. Exit non-zero if there were errors (so CI marks the run failed
     and emails you).

Usage:
    python scripts/run_daily.py [--config path/to/config.yaml] [--dry-run]

The --dry-run flag runs the full decision pipeline but submits no orders.
Use this to monitor the system's behavior against your paper account
without committing trades.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsi_revert.broker import Broker
from rsi_revert.live import daily_run
from rsi_revert.report import format_run_report
from rsi_revert.utils import load_config, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily paper-trade runner")
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to YAML config (default: config/config.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run full decision pipeline but submit no orders. The report "
             "will show what would have been traded.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(
        log_dir=cfg["runtime"].get("log_dir", "logs"),
        level=cfg["runtime"].get("log_level", "INFO"),
        name="run_daily",
    )

    mode = "DRY RUN (no orders will be submitted)" if args.dry_run else "LIVE (orders will be submitted)"
    logger.info("===== rsi-revert daily run starting — %s =====", mode)
    try:
        broker = Broker()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Broker init failed")
        print(f"FATAL: Broker init failed: {exc}", file=sys.stderr)
        return 2

    try:
        report = daily_run(cfg, broker, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        logger.exception("daily_run raised an unhandled exception")
        print(f"FATAL: daily_run failed: {exc}", file=sys.stderr)
        return 3

    print(format_run_report(report))
    logger.info("===== rsi-revert daily run complete =====")

    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
