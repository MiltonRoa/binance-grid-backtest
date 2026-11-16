# -*- coding: utf-8 -*-
"""Command-line runner: volume ranking -> selective load -> spacing sweep -> report.

Example::

    python -m gridlab.cli --root "D:/data/binance_1m" --months julio22 agosto22 ... \
        --vol-months setiembre22 marzo23 --top 25 --spacings 0.005 0.01 0.02

Months are folder tags (``parquets_<tag>``); pass them in chronological order.
Output goes to stdout and, with ``--out``, to a text file as well.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .data import load_top_symbols, month_dirs, rank_by_volume
from .sim import GridParams, cycle_gross, simulate


class Tee:
    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ladder-grid backtest over 1m Parquet candles")
    ap.add_argument("--root", required=True, help="folder containing parquets_<month> dirs")
    ap.add_argument("--months", nargs="+", required=True, help="month tags for the test window")
    ap.add_argument("--vol-months", nargs="+", required=True, help="month tags sampled for volume ranking")
    ap.add_argument("--start", default=None, help="ignore bars before this timestamp (YYYY-MM-DD)")
    ap.add_argument("--top", type=int, default=25, help="number of symbols to trade")
    ap.add_argument("--spacings", nargs="+", type=float, default=[0.005, 0.01, 0.015, 0.02])
    ap.add_argument("--pos-size", type=float, default=5.0)
    ap.add_argument("--max-open", type=int, default=3)
    ap.add_argument("--max-bars", type=int, default=240)
    ap.add_argument("--price-stop-k", type=float, default=1.0)
    ap.add_argument("--stale-k", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="also write the report to this file")
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    a = parse_args(argv)
    if a.out:
        sys.stdout = Tee(Path(a.out))
    t0 = time.time()
    root = Path(a.root)
    dirs = month_dirs(root, a.months)
    vol_dirs = month_dirs(root, a.vol_months)
    if not dirs or not vol_dirs:
        print("no data folders found under", root)
        return 2
    start = pd.Timestamp(a.start) if a.start else None
    n_months = len(dirs)

    print(f"Ranking symbols by volume over {len(vol_dirs)} sampled months...")
    ranked = rank_by_volume(vol_dirs, workers=a.workers)
    print(f"  {len(ranked)} symbols ranked in {time.time()-t0:.0f}s")

    t1 = time.time()
    syms, closes, times = load_top_symbols(ranked, dirs, a.top, start, workers=a.workers)
    print(f"Loaded {len(syms)} symbols x {n_months} months in {time.time()-t1:.0f}s")
    print(f"Top: {', '.join(syms[:8])} ...\n")

    print(f"Params: max_open={a.max_open} | time_stop={a.max_bars} bars | "
          f"price_stop={a.price_stop_k}x spacing | pos_size=${a.pos_size}\n")
    print("=" * 75)
    print(f"  {'Spacing':>8}  {'$/cycle':>8}  {'Cycles':>9}  {'Stops':>8}  {'C/Stop':>7}  {'NET/month':>10}")
    print("=" * 75)

    best_net, best_sp = -float("inf"), None
    for sp in a.spacings:
        params = GridParams(spacing=sp, pos_size=a.pos_size, max_open=a.max_open,
                            max_bars=a.max_bars, price_stop_k=a.price_stop_k, stale_k=a.stale_k)
        tp = tc = ts = 0
        for s in syms:
            r = simulate(closes[s], params)
            tp += r.pnl
            tc += r.cycles
            ts += r.stops
        net = tp / n_months
        flag = "  <- best" if net > best_net else ""
        if net > best_net:
            best_net, best_sp = net, sp
        print(f"  {sp*100:>7.2f}%  ${cycle_gross(params):>8.4f}  {tc:>9,}  {ts:>8,}  "
              f"{tc/max(ts,1):>7.1f}  ${net:>+9.2f}{flag}")
    print("=" * 75)

    capital = a.top * a.max_open * a.pos_size
    print(f"\nBest spacing: {best_sp*100:.2f}%  NET/month ${best_net:+.2f}  "
          f"capital ${capital:.0f}  ROI/month {best_net/capital*100:+.1f}%\n")

    # Monthly breakdown for the best spacing.
    params = GridParams(spacing=best_sp, pos_size=a.pos_size, max_open=a.max_open,
                        max_bars=a.max_bars, price_stop_k=a.price_stop_k, stale_k=a.stale_k)
    rows = []
    for s in syms:
        r = simulate(closes[s], params, track_events=True)
        t = times[s]
        rows.extend((t[i], pnl, s) for i, pnl, kind in r.events if kind != "settlement")
    if rows:
        ev = pd.DataFrame(rows, columns=["time", "pnl", "symbol"])
        ev["month"] = ev["time"].dt.to_period("M").astype(str)
        ev["date"] = ev["time"].dt.date
        dm = ev.groupby("month")["pnl"].sum()
        dp = ev.groupby("date")["pnl"].sum()
        print(f"  {'Month':<10}  {'NET':>10}  {'Worst day':>10}  {'Best day':>10}")
        print(f"  {'-'*46}")
        for m in sorted(dm.index):
            d = ev[ev["month"] == m].groupby("date")["pnl"].sum()
            print(f"  {m:<10}  {dm[m]:>+10.2f}  {d.min():>+10.2f}  {d.max():>+10.2f}")
        sharpe = dp.mean() / dp.std() * np.sqrt(252) if dp.std() > 0 else 0.0
        print(f"\n  Negative days  : {(dp < 0).sum()}/{len(dp)}")
        print(f"  Negative months: {(dm < 0).sum()}/{len(dm)}")
        print(f"  Annual Sharpe  : {sharpe:.2f}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
