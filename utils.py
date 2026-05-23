"""
Shared utilities: config loading, logger setup, kill switch.

The config is a single YAML file. Sensitive values (API keys) come from
environment variables — never the config file. The .env.example file
documents which env vars are required.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import yaml

from rsi_revert.signals import VARIANT_A, VARIANT_B, VariantParams


DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load and minimally validate the YAML config.

    Raises FileNotFoundError if the file is missing — fail loudly so
    misconfigured CI runs don't silently use defaults.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. "
            f"See config/config.yaml in the repo for the expected format."
        )
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse as a mapping")
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """Surface the most obvious config mistakes early."""
    required_sections = ("universe", "strategy", "backtest", "live", "runtime")
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Config missing required section: {section}")
    if not isinstance(cfg["universe"], list) or not cfg["universe"]:
        raise ValueError("config.universe must be a non-empty list of symbols")
    variant = cfg["strategy"].get("variant", "").upper()
    if variant not in ("A", "B"):
        raise ValueError(f"config.strategy.variant must be 'A' or 'B', got {variant!r}")


def get_variant_from_config(cfg: dict[str, Any]) -> VariantParams:
    """Map config.strategy.variant to a VariantParams instance."""
    variant = cfg["strategy"]["variant"].upper()
    return VARIANT_A if variant == "A" else VARIANT_B


def setup_logging(
    log_dir: Path | str = "logs",
    level: str = "INFO",
    *,
    name: str | None = None,
) -> logging.Logger:
    """
    Configure root logger with rotating file output + stderr stream.

    Returns the root logger so callers can use it directly, but also
    configures it globally so module-level loggers inherit the handlers.

    File handler rotates at 5 MB, keeps 5 backups — bounded disk use
    for long-running cron deployments.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "rsi_revert.log"
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    # Reset handlers if setup is called twice (common in tests).
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper()))

    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    return logging.getLogger(name) if name else root


def check_kill_switch() -> bool:
    """
    Check whether the kill switch is engaged.

    Reads KILL_SWITCH environment variable. Any truthy value
    ('true', '1', 'yes' — case-insensitive) means STOP trading.
    Anything else (including unset) means proceed normally.

    In GitHub Actions, set this via repo Variables → KILL_SWITCH=true
    to halt the daily trader without a code push.
    """
    raw = os.environ.get("KILL_SWITCH", "").strip().lower()
    return raw in ("true", "1", "yes", "on")
