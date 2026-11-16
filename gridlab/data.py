# -*- coding: utf-8 -*-
"""Selective loading of 1-minute candles stored as monthly Parquet folders.

Expected layout (one folder per month, one file per symbol)::

    <root>/parquets_<month_tag>/<SYMBOL>.parquet

Each Parquet file has at least the columns ``open_time_local`` (timestamp),
``close`` and ``volume``.  Timestamps are a *column*, not the index, which is
why ``columns=[...]`` is passed to ``read_parquet``: reading two columns out of
ten is roughly 5x faster and keeps RAM flat even with 13,000 files on disk.

The loader never reads the whole universe.  It first ranks symbols by quote
volume using a handful of sampled months, then loads only the top-N symbols
for the full test window.  On the original dataset (529 symbols, 48 months)
that brought a full run from "does not fit in 16 GB" to about two minutes.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def month_dirs(root: Path, month_tags: Iterable[str]) -> List[Path]:
    """Return existing ``parquets_<tag>`` folders under ``root`` in order."""
    out = []
    for tag in month_tags:
        d = Path(root) / f"parquets_{tag}"
        if d.exists():
            out.append(d)
    return out


def scan_symbol_volume(sym: str, vol_dirs: List[Path]) -> Tuple[str, float]:
    """Average quote volume (close * volume) of ``sym`` over sampled months."""
    vals = []
    for d in vol_dirs:
        p = d / f"{sym}.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["close", "volume"])
        except Exception:  # corrupt or partial file: skip, do not abort
            continue
        if len(df):
            vals.append(float((df["close"] * df["volume"]).mean()))
    return sym, (float(np.mean(vals)) if vals else 0.0)


def rank_by_volume(vol_dirs: List[Path], exclude: Iterable[str] = ("BTCUSDT",),
                   workers: int = 8) -> List[str]:
    """Rank every symbol present in the sampled folders by mean quote volume."""
    syms = set()
    for d in vol_dirs:
        syms.update(p.stem for p in d.glob("*.parquet"))
    syms.difference_update(exclude)
    with ThreadPoolExecutor(workers) as ex:
        vol = dict(ex.map(lambda s: scan_symbol_volume(s, vol_dirs), sorted(syms)))
    return sorted(vol, key=lambda s: vol[s], reverse=True)


def load_closes(sym: str, dirs: List[Path], start: Optional[pd.Timestamp] = None
                ) -> Tuple[Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
    """Full close series of ``sym`` across ``dirs``; ``(None, None)`` if absent."""
    frames = []
    for d in dirs:
        p = d / f"{sym}.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["open_time_local", "close"])
        except Exception:
            continue
        if len(df):
            frames.append(df)
    if not frames:
        return None, None
    df = pd.concat(frames, ignore_index=True)
    df["open_time_local"] = pd.to_datetime(df["open_time_local"], errors="coerce")
    df = df.dropna(subset=["open_time_local", "close"])
    if start is not None:
        df = df[df["open_time_local"] >= start]
    df = (df.sort_values("open_time_local")
            .drop_duplicates(subset="open_time_local", keep="first"))
    return df["close"].to_numpy(np.float64), pd.DatetimeIndex(df["open_time_local"])


def load_top_symbols(ranked: List[str], dirs: List[Path], n: int,
                     start: Optional[pd.Timestamp] = None, min_bars: int = 1000,
                     workers: int = 8, margin: int = 10
                     ) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, pd.DatetimeIndex]]:
    """Load the first ``n`` ranked symbols that have at least ``min_bars`` bars."""
    candidates = ranked[: n + margin]
    with ThreadPoolExecutor(workers) as ex:
        loaded = dict(zip(candidates,
                          ex.map(lambda s: load_closes(s, dirs, start), candidates)))
    syms, closes, times = [], {}, {}
    for s in candidates:
        c, t = loaded[s]
        if c is not None and len(c) >= min_bars:
            syms.append(s)
            closes[s] = c
            times[s] = t
        if len(syms) == n:
            break
    return syms, closes, times
