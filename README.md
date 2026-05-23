# rsi-revert

An RSI mean-reversion trader with a trend filter, walk-forward validation, and Alpaca paper trading on GitHub Actions.

## What it does

Buys oversold dips inside a long-term uptrend, exits when the bounce arrives. Two variants:

- **Variant A — classic**: enter when RSI(14) < 30, exit when RSI(14) > 70.
- **Variant B — Connors**: enter when RSI(2) < 10, exit when close > 5-day SMA.

Both gate entries on SPY > 200-SMA so the strategy only trades dip-buys inside confirmed uptrends, not during bear markets. Stops at the 10-day low. Position sizing is risk-based — 1% of equity per trade, sized off the stop distance.

The system was built to be honest: every metric reported is computed on out-of-sample data via walk-forward validation, slippage is modeled at 5 bps per fill, and the backtester compares against buy-and-hold so a strategy that doesn't beat passive ownership shows up as such.

## Project layout

```
rsi-revert/
├── .github/workflows/
│   ├── ci.yml                # tests on every push
│   ├── backtest.yml          # manual backtest, uploads CSV reports
│   └── daily-trade.yml       # 21:05 UTC weekdays — paper trading
├── config/
│   ├── config.yaml           # strategy + runtime parameters
│   └── .env.example          # required environment variables
├── rsi_revert/
│   ├── data.py               # Alpaca bar fetcher + parquet cache
│   ├── indicators.py         # RSI (Wilder), SMA, rolling low
│   ├── signals.py            # variant definitions + signal generator
│   ├── backtest.py           # simulator + metrics
│   ├── walkforward.py        # rolling train/test validation
│   ├── broker.py             # Alpaca trading wrapper (retries, paper-safe)
│   ├── live.py               # daily paper-trading loop
│   ├── report.py             # output formatting
│   └── utils.py              # config, logging, kill switch
├── scripts/
│   ├── run_daily.py          # daily-trade entry point
│   └── run_backtest.py       # backtest entry point
├── tests/                    # unit + integration tests (no API calls)
├── data/cache/               # parquet bar cache (gitignored)
├── logs/                     # rotating log files (gitignored)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

### Local development

1. Clone the repo and create a Python 3.12 venv.
2. `pip install -r requirements.txt && pip install -e .`
3. Get paper-trading credentials from [Alpaca](https://app.alpaca.markets/paper/dashboard/overview).
4. Copy `config/.env.example` to `.env` and fill in `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.
5. Run tests to verify the install: `pytest tests/`
6. Run a backtest: `python scripts/run_backtest.py`. Output goes to `reports/`.

### GitHub Actions deployment

1. Push this repo to GitHub.
2. Settings → Secrets and variables → Actions:
   - Add **Secrets**: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`.
   - Add **Variables**: `ALPACA_PAPER=true`, `KILL_SWITCH=false`.
3. Settings → Actions → Workflow permissions: enable "Read and write permissions" so the daily workflow can commit cache updates (only if you choose to commit cache; default uses Actions cache instead).
4. The `daily-trade` workflow will fire every weekday at 21:05 UTC.
5. Trigger a backtest manually from the Actions tab any time.

### Kill switch

To halt trading immediately without a code push:
- Go to Settings → Secrets and variables → Actions → Variables.
- Edit `KILL_SWITCH` to `true`.
- The next daily run will see this and exit before any orders.

## How it works

### Signal flow (one day)

1. Cron triggers at 21:05 UTC weekdays.
2. `run_daily.py` loads config and constructs a `Broker`.
3. `daily_run()` checks the kill switch, then fetches account info and the last ~1 year of SPY bars.
4. Computes the regime filter (SPY > 200-SMA) and entry/exit signals for the configured variant.
5. For each symbol in the universe:
   - If exit signal and we hold the symbol → cancel stops, submit market sell.
   - If we hold the symbol and no stop is open → submit a stop-loss at the rolling 10-day low.
   - If entry signal and we're flat → submit a market buy sized by 1% risk.
   - Otherwise hold.
6. Format and print a `RunReport`. Errors → exit code 1 → GitHub emails you.

Market orders submitted after-hours are queued by Alpaca for the next session's open. Stop orders are GTC and execute intraday if hit.

### Why walk-forward matters

A backtest that uses every available day to tune parameters tells you almost nothing about future performance — you've fit noise. The walk-forward validator (in `walkforward.py`) trains on 4-year windows and evaluates on the next unseen 1-year window. The **degradation** column in its output is the punchline: if train Sharpe averages 1.5 and test averages 0.2, you're overfit. The stitched test equity curve is the closest honest approximation of "what would I have actually earned trading this without future knowledge."

### Safety defaults

- `ALPACA_PAPER` defaults to `true`. The `Broker` reads this on every initialization. To switch to live trading you must explicitly set `ALPACA_PAPER=false` — a misconfigured secret can't accidentally route real orders.
- The kill switch (`KILL_SWITCH=true`) short-circuits the whole daily run before any broker call.
- Stop-loss orders are real Alpaca stop orders, not just records in a local database — they survive if our process or GitHub Actions ever goes down.

## Configuration

Edit `config/config.yaml`. Key knobs:

| Field | Default | Notes |
|---|---|---|
| `universe` | `[SPY]` | List of tickers to trade. Regime filter is always SPY. |
| `strategy.variant` | `B` | `A` (RSI14) or `B` (RSI2, Connors). |
| `backtest.risk_per_trade` | `0.01` | Fraction of equity risked per trade. |
| `backtest.slippage_bps` | `5.0` | Per-fill slippage in basis points. |
| `backtest.history_start` | `"2005-01-01"` | Backtest lookback. |
| `walk_forward.train_years` | `4` | Length of in-sample window. |
| `walk_forward.test_years` | `1` | Length of out-of-sample window. |
| `live.data_lookback_days` | `365` | Days of history fetched for daily signal. |

## Testing

```bash
pytest tests/
```

All tests mock the Alpaca API — no credentials needed. CI runs them on every push.

## License

MIT.
