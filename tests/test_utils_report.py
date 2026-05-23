"""Tests for utils.py and report.py."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from rsi_revert.broker import OrderInfo, Position
from rsi_revert.live import RunReport, SymbolAction
from rsi_revert.report import format_run_report
from rsi_revert.signals import VARIANT_A, VARIANT_B
from rsi_revert.utils import check_kill_switch, get_variant_from_config, load_config


# --------------------------- utils ---------------------------


def test_load_config_round_trip(tmp_path: Path) -> None:
    cfg_text = """
universe:
  - SPY
strategy:
  variant: B
backtest:
  initial_equity: 100000
  risk_per_trade: 0.01
  slippage_bps: 5.0
  commission_per_share: 0.0
  history_start: "2010-01-01"
live:
  data_lookback_days: 365
runtime:
  cache_dir: data/cache
  log_dir: logs
  log_level: INFO
"""
    p = tmp_path / "config.yaml"
    p.write_text(cfg_text)
    cfg = load_config(p)
    assert cfg["universe"] == ["SPY"]
    assert cfg["strategy"]["variant"] == "B"


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent/config.yaml")


def test_load_config_rejects_invalid_variant(tmp_path: Path) -> None:
    cfg_text = """
universe: [SPY]
strategy:
  variant: Z
backtest: {initial_equity: 100000, risk_per_trade: 0.01, slippage_bps: 5.0, commission_per_share: 0.0}
live: {data_lookback_days: 365}
runtime: {cache_dir: data/cache}
"""
    p = tmp_path / "config.yaml"
    p.write_text(cfg_text)
    with pytest.raises(ValueError, match="variant"):
        load_config(p)


def test_load_config_rejects_empty_universe(tmp_path: Path) -> None:
    cfg_text = """
universe: []
strategy: {variant: A}
backtest: {initial_equity: 100000, risk_per_trade: 0.01, slippage_bps: 5.0, commission_per_share: 0.0}
live: {data_lookback_days: 365}
runtime: {cache_dir: data/cache}
"""
    p = tmp_path / "config.yaml"
    p.write_text(cfg_text)
    with pytest.raises(ValueError, match="universe"):
        load_config(p)


def test_get_variant_from_config_maps_correctly() -> None:
    assert get_variant_from_config({"strategy": {"variant": "A"}}).name == VARIANT_A.name
    assert get_variant_from_config({"strategy": {"variant": "B"}}).name == VARIANT_B.name
    assert get_variant_from_config({"strategy": {"variant": "b"}}).name == VARIANT_B.name


def test_kill_switch_truthy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for val in ("true", "True", "TRUE", "1", "yes", "on"):
        monkeypatch.setenv("KILL_SWITCH", val)
        assert check_kill_switch() is True


def test_kill_switch_falsy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for val in ("false", "0", "no", "off", "", "random"):
        monkeypatch.setenv("KILL_SWITCH", val)
        assert check_kill_switch() is False


def test_kill_switch_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    assert check_kill_switch() is False


# --------------------------- report ---------------------------


def test_format_run_report_kill_switch() -> None:
    rep = RunReport(timestamp=datetime.now(timezone.utc), kill_switch_engaged=True)
    out = format_run_report(rep)
    assert "KILL_SWITCH" in out
    assert "no trading" in out.lower()


def test_format_run_report_with_actions() -> None:
    pos = Position(symbol="SPY", qty=100, avg_entry_price=450.0,
                   market_value=46000.0, unrealized_pl=1000.0)
    order = OrderInfo(
        id="x1", symbol="SPY", side="sell", qty=100, order_type="market",
        status="accepted", stop_price=None, filled_qty=0,
        filled_avg_price=None, submitted_at=datetime.now(timezone.utc),
    )
    action = SymbolAction(
        symbol="SPY", decision="exit", reason="exit_signal at RSI=75.2",
        position_before=pos, order_submitted=order, rsi=75.2, close=460.0, regime_ok=True,
    )
    rep = RunReport(
        timestamp=datetime.now(timezone.utc), kill_switch_engaged=False,
        account_equity=100_000, account_cash=53_900,
        symbol_actions=[action],
    )
    out = format_run_report(rep)
    assert "SPY" in out
    assert "exit" in out
    assert "75.2" in out
    assert "460.00" in out
    assert "1,000" in out  # unrealized PL appears as $+1,000.00


def test_format_run_report_with_errors() -> None:
    rep = RunReport(
        timestamp=datetime.now(timezone.utc), kill_switch_engaged=False,
        errors=["Failed to fetch SPY bars: network down"],
    )
    out = format_run_report(rep)
    assert "ERRORS" in out
    assert "network down" in out
