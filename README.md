# binance-grid-backtest

A small, fast and deliberately honest backtester for **long-only ladder grids**
on 1-minute candles. It was written to answer one question with data instead
of opinion: *does a classic "buy the dips, sell one step up" grid make money on
Binance USDT-M futures once real fees and stops are included?*

The answer, over 48 months and the 25 most liquid symbols, was **no**, and the
repository keeps that result on purpose. The value here is the method: a
simulator that is a pure function over a price array, unit tests on synthetic
paths that pin down fee accounting and the slot invariant, and a loader that
runs the whole study in about two minutes on a 16 GB laptop.

## What it does

```
python -m gridlab.cli --root D:/data/binance_1m \
    --months julio22 agosto22 ... junio26 \
    --vol-months setiembre22 marzo23 setiembre23 marzo24 setiembre24 marzo25 setiembre25 junio26 \
    --start 2022-07-01 --top 25 --spacings 0.005 0.01 0.015 0.02 --out report.txt
```

1. **Rank by volume** using a handful of sampled months (one per semester), so
   the universe of 500+ symbols is scanned once and cheaply.
2. **Load only the top-N symbols**, two columns per Parquet file, in parallel.
   Timestamps are a column, not the index, which makes `columns=[...]` reads
   about 5x faster.
3. **Sweep grid spacings** and print cycles, stops, net per month and a
   monthly breakdown with worst/best day and annualised Sharpe.

### Strategy modelled

* `max_open` BUY LIMIT orders below price at -1x, -2x, -3x `spacing`.
* Each fill places a SELL LIMIT one `spacing` above the entry.
* Price stop at `price_stop_k * spacing` below entry; time stop after
  `max_bars` bars; both close at market (taker fee).
* BUY orders deeper than `stale_k * spacing` are cancelled, and the grid
  **re-seeds every bar** so that `positions + resting_buys == max_open`.
  That invariant is asserted inside the hot loop and covered by a regression
  test: an earlier version silently stopped trading after rallies because
  stale BUYs were never replaced.
* Fees: 0.02 % maker on resting limits, 0.05 % taker on market closes
  (0.04 % per full cycle, 0.07 % per stop).

## Result on real data (July 2022 to June 2026)

Top 25 USDT-M perpetuals by volume, `pos_size=$5`, `max_open=3`,
time stop 4 h, price stop 1x spacing:

| Spacing | $/cycle | Cycles | Stops | Cycles per stop | Net per month |
|--------:|--------:|-------:|------:|----------------:|--------------:|
| 0.50 % | 0.0230 | 339,764 | 542,556 | 0.6 | -$308 |
| 1.00 % | 0.0480 | 112,098 | 169,602 | 0.7 | -$128 |
| 1.50 % | 0.0730 | 52,660 | 87,988 | 0.6 | -$75 |
| 2.00 % | 0.0980 | 28,550 | 55,930 | 0.5 | -$51 |

Best case: 47 of 48 months negative, 881 of 1,434 days negative, worst day
-$293 on $375 of deployed capital. A follow-up sweep with fewer and looser
stops converged to roughly zero with most of the capital trapped in
inventory, before funding costs. The grid *defers* losses; it does not avoid
them. The strategy was retired on that evidence.

## Install and test

```
pip install -r requirements.txt
pytest -q
```

The tests need no market data. They cover: flat price does nothing, one dip
and recovery books exactly one cycle with the right fee, price stop and time
stop fire where expected, a crash fills at most `max_open` levels, the slot
invariant holds on a 20,000-bar random walk, the grid keeps trading after a
rally, and the event log sums to the total PnL.

## Data layout

```
<root>/parquets_<month_tag>/<SYMBOL>.parquet
columns: open_time_local, open, high, low, close, volume, ...
```

Any source works as long as each file has `open_time_local`, `close` and
`volume`. The original dataset was built from Binance Vision monthly dumps.

## Layout

```
gridlab/sim.py    pure simulator + fee helpers
gridlab/data.py   volume ranking and selective Parquet loading
gridlab/cli.py    command line runner and report
tests/test_sim.py synthetic-path unit tests
```

MIT License.
