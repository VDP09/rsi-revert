"""
rsi-revert — RSI mean-reversion trader with trend filter and walk-forward validation.
"""

__version__ = "0.1.0"

from rsi_revert.signals import VARIANT_A, VARIANT_B, VariantParams
from rsi_revert.backtest import BacktestConfig, BacktestResult, run_backtest, buy_and_hold
from rsi_revert.walkforward import WalkForwardConfig, WalkForwardResult, walk_forward

__all__ = [
    "__version__",
    "VARIANT_A",
    "VARIANT_B",
    "VariantParams",
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "buy_and_hold",
    "WalkForwardConfig",
    "WalkForwardResult",
    "walk_forward",
]
