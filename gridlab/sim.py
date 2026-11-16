# -*- coding: utf-8 -*-
"""Bar-by-bar simulator for a long-only ladder grid.

Strategy
--------
* At start, place ``max_open`` BUY LIMIT orders below price at
  ``-1x, -2x, ... spacing``.
* Each BUY fill places a SELL LIMIT at ``fill_price * (1 + spacing)``.
* Each SELL fill books a cycle; the freed slot is re-seeded with a new BUY
  below the current price.
* Price stop: a position is closed at market if price falls
  ``price_stop_k * spacing`` below its entry.
* Time stop: a position is closed at market after ``max_bars`` bars.
* Stale cancel: BUY orders further than ``stale_k * spacing`` below price are
  cancelled (so a crash does not fill every level at once).
* Invariant (the bug fix that motivated this version): at the end of every bar
  ``open_positions + resting_buys == max_open``.

Fees follow Binance USDT-M futures: maker 0.02 % on resting limits, taker
0.05 % on market closes.  A full limit/limit cycle therefore costs 0.04 % round
trip; a stop costs 0.07 %.

The hot loop uses plain Python floats and cached aggregates (``min_sell``,
``max_entry``, ``min_bar``, ``max_buy``, ``min_buy``) so that the common case
of "nothing happened this bar" is O(1).  On a typical laptop it processes
about 10 million bars per minute per symbol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

INF = float("inf")


@dataclass(frozen=True)
class GridParams:
    spacing: float = 0.01          # grid step as a fraction of price (1 % = 0.01)
    pos_size: float = 5.0          # notional per position in quote currency
    max_open: int = 3              # simultaneous positions per symbol
    max_bars: int = 240            # time stop in bars (240 x 1m = 4 h)
    price_stop_k: float = 1.0      # close if loss >= price_stop_k * spacing
    stale_k: float = 4.0           # cancel BUYs deeper than stale_k * spacing
    fee_limit: float = 0.0002      # maker fee (0.02 %)
    fee_market: float = 0.0005     # taker fee (0.05 %)

    def __post_init__(self) -> None:
        if not 0 < self.spacing < 1:
            raise ValueError("spacing must be in (0, 1)")
        if self.max_open < 1:
            raise ValueError("max_open must be >= 1")
        if self.max_bars < 1:
            raise ValueError("max_bars must be >= 1")


@dataclass
class SimResult:
    pnl: float
    cycles: int
    stops: int
    events: Optional[List[Tuple[int, float, str]]]  # (bar_index, pnl, kind)


def simulate(closes: Sequence[float], params: GridParams,
             track_events: bool = False) -> SimResult:
    """Run the ladder grid over a sequence of close prices.

    Parameters
    ----------
    closes:
        Close prices, one per bar, chronological.
    params:
        Strategy parameters.
    track_events:
        When True, every realised PnL event is recorded as
        ``(bar_index, net_pnl, kind)`` with kind in ``{"cycle", "stop",
        "settlement"}``.  Settlement events use ``bar_index = len(closes) - 1``.
    """
    px = [float(c) for c in closes]
    n = len(px)
    if n < 2:
        return SimResult(0.0, 0, 0, [] if track_events else None)

    p = params
    spacing = p.spacing
    pos_size = p.pos_size
    max_open = p.max_open
    max_bars = p.max_bars
    stop_pct = p.price_stop_k * spacing
    stale_mult = 1.0 - p.stale_k * spacing
    fee_cycle = 2.0 * p.fee_limit * pos_size
    fee_stop = (p.fee_limit + p.fee_market) * pos_size

    total_pnl = 0.0
    n_cycles = 0
    n_stops = 0
    events: Optional[List[Tuple[int, float, str]]] = [] if track_events else None

    # Resting BUY orders (prices) and open positions as parallel lists.
    start = px[0]
    buy_orders = [start * (1.0 - spacing * (k + 1)) for k in range(max_open)]
    max_buy = buy_orders[0]
    min_buy = buy_orders[-1]

    p_entry: List[float] = []
    p_sell: List[float] = []
    p_bar: List[int] = []
    min_sell = INF
    max_entry = -INF
    min_bar = INF

    for i in range(1, n):
        price = px[i]

        # SELL fills: every resting SELL is above the previous close, so the
        # first touch fills it.
        if price >= min_sell:
            k = 0
            while k < len(p_sell):
                if p_sell[k] <= price:
                    entry = p_entry[k]
                    net = (p_sell[k] - entry) / entry * pos_size - fee_cycle
                    total_pnl += net
                    n_cycles += 1
                    if events is not None:
                        events.append((i, net, "cycle"))
                    del p_entry[k], p_sell[k], p_bar[k]
                else:
                    k += 1
            min_sell = min(p_sell) if p_sell else INF
            max_entry = max(p_entry) if p_entry else -INF
            min_bar = min(p_bar) if p_bar else INF

        # BUY fills: every resting BUY is below the previous close.
        if price <= max_buy:
            k = 0
            while k < len(buy_orders):
                if price <= buy_orders[k] and len(p_entry) < max_open:
                    b = buy_orders.pop(k)
                    p_entry.append(b)
                    p_sell.append(b * (1.0 + spacing))
                    p_bar.append(i)
                else:
                    k += 1
            max_buy = max(buy_orders) if buy_orders else -INF
            min_buy = min(buy_orders) if buy_orders else INF
            min_sell = min(p_sell)
            max_entry = max(p_entry)
            min_bar = min(p_bar)

        # Price stop / time stop (guarded by cached aggregates).
        if p_entry and (price <= max_entry * (1.0 - stop_pct)
                        or (i - min_bar) >= max_bars):
            k = 0
            while k < len(p_entry):
                entry = p_entry[k]
                if (i - p_bar[k]) >= max_bars or (entry - price) / entry >= stop_pct:
                    net = (price - entry) / entry * pos_size - fee_stop
                    total_pnl += net
                    n_stops += 1
                    if events is not None:
                        events.append((i, net, "stop"))
                    del p_entry[k], p_sell[k], p_bar[k]
                else:
                    k += 1
            min_sell = min(p_sell) if p_sell else INF
            max_entry = max(p_entry) if p_entry else -INF
            min_bar = min(p_bar) if p_bar else INF

        # Stale cancel.
        if min_buy < price * stale_mult:
            stale_limit = price * stale_mult
            buy_orders = [b for b in buy_orders if b >= stale_limit]
            max_buy = max(buy_orders) if buy_orders else -INF
            min_buy = min(buy_orders) if buy_orders else INF

        # Re-seed: keep max_open slots active at all times.
        deficit = max_open - len(p_entry) - len(buy_orders)
        if deficit > 0:
            for k in range(deficit):
                buy_orders.append(price * (1.0 - spacing * (k + 1)))
            max_buy = max(buy_orders)
            min_buy = min(buy_orders)

        assert len(p_entry) + len(buy_orders) == max_open, "slot invariant broken"

    # Settlement: close whatever is still open at the last price.
    final = px[-1]
    for entry in p_entry:
        net = (final - entry) / entry * pos_size - fee_stop
        total_pnl += net
        if events is not None:
            events.append((n - 1, net, "settlement"))

    return SimResult(total_pnl, n_cycles, n_stops, events)


def cycle_gross(params: GridParams) -> float:
    """Net profit of one complete limit/limit cycle, before any stop."""
    return (params.spacing - 2.0 * params.fee_limit) * params.pos_size


def stop_loss_typical(params: GridParams) -> float:
    """Approximate loss of a price stop triggered exactly at the threshold."""
    return (params.price_stop_k * params.spacing + params.fee_limit
            + params.fee_market) * params.pos_size
