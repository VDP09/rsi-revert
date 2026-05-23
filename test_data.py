"""
Tests for rsi_revert.data.

These mock the Alpaca client so they run in CI without credentials and
without hitting the network. The point is to verify cache logic, not
that Alpaca returns correct data.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from rsi_revert import data as data_mod


def _make_bars(start: str, end: str) -> pd.DataFrame:
    """Build a fake bar dataframe with the canonical schema."""
    idx = pd.date_range(start, end, freq="B")  # business days
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000.0,
            "trade_count": 1000.0,
            "vwap": 100.25,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def test_get_bars_cold_cache_fetches_full_range(tmp_path: Path) -> None:
    fake = _make_bars("2024-01-02", "2024-01-31")
    with patch.object(data_mod, "_fetch_from_alpaca", return_value=fake) as mock_fetch:
        result = data_mod.get_bars("SPY", "2024-01-01", "2024-01-31", cache_dir=tmp_path)
    assert mock_fetch.call_count == 1
    assert len(result) == len(fake)
    assert (tmp_path / "SPY.parquet").exists()


def test_get_bars_warm_cache_no_refetch_when_in_range(tmp_path: Path) -> None:
    fake = _make_bars("2024-01-02", "2024-01-31")
    with patch.object(data_mod, "_fetch_from_alpaca", return_value=fake):
        data_mod.get_bars("SPY", "2024-01-01", "2024-01-31", cache_dir=tmp_path)

    # Second call inside cached range should NOT trigger fetch.
    with patch.object(data_mod, "_fetch_from_alpaca") as mock_fetch:
        result = data_mod.get_bars("SPY", "2024-01-05", "2024-01-20", cache_dir=tmp_path)
    assert mock_fetch.call_count == 0
    assert len(result) > 0


def test_get_bars_warm_cache_fetches_tail_only(tmp_path: Path) -> None:
    initial = _make_bars("2024-01-02", "2024-01-15")
    with patch.object(data_mod, "_fetch_from_alpaca", return_value=initial):
        data_mod.get_bars("SPY", "2024-01-02", "2024-01-15", cache_dir=tmp_path)

    # Request extends past cache — should only fetch the gap.
    # Start must not be before cache_min, or we trigger refetch-from-start.
    tail = _make_bars("2024-01-16", "2024-01-31")
    with patch.object(data_mod, "_fetch_from_alpaca", return_value=tail) as mock_fetch:
        result = data_mod.get_bars("SPY", "2024-01-02", "2024-01-31", cache_dir=tmp_path)
    assert mock_fetch.call_count == 1
    # The fetch should have started from 2024-01-16, not 2024-01-01.
    call_args = mock_fetch.call_args
    fetch_start = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs["start"]
    assert pd.Timestamp(fetch_start).date() == pd.Timestamp("2024-01-16").date()
    assert len(result) == len(initial) + len(tail)


def test_get_bars_force_refresh_bypasses_cache(tmp_path: Path) -> None:
    fake = _make_bars("2024-01-02", "2024-01-15")
    with patch.object(data_mod, "_fetch_from_alpaca", return_value=fake):
        data_mod.get_bars("SPY", "2024-01-01", "2024-01-15", cache_dir=tmp_path)

    with patch.object(data_mod, "_fetch_from_alpaca", return_value=fake) as mock_fetch:
        data_mod.get_bars(
            "SPY", "2024-01-01", "2024-01-15", cache_dir=tmp_path, force_refresh=True
        )
    assert mock_fetch.call_count == 1


def test_get_bars_rejects_inverted_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="after end"):
        data_mod.get_bars("SPY", "2024-02-01", "2024-01-01", cache_dir=tmp_path)


def test_get_bars_handles_empty_response(tmp_path: Path) -> None:
    empty = pd.DataFrame(columns=data_mod.BAR_COLUMNS)
    with patch.object(data_mod, "_fetch_from_alpaca", return_value=empty):
        result = data_mod.get_bars("XYZ", "2024-01-01", "2024-01-31", cache_dir=tmp_path)
    assert result.empty
    # Cache should NOT be written for empty results.
    assert not (tmp_path / "XYZ.parquet").exists()


def test_missing_credentials_raises() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
            data_mod._get_client()
