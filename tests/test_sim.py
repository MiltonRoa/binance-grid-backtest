# -*- coding: utf-8 -*-
"""Unit tests for the ladder-grid simulator on synthetic price paths.

No market data is needed; the tests pin down the accounting and the slot
invariant that caused the original v7 bug (grid dying after a rally).
"""
import math

import numpy as np
import pytest

from gridlab import GridParams, simulate
from gridlab.sim import cycle_gross, stop_loss_typical


def test_flat_price_does_nothing():
    r = simulate([100.0] * 500, GridParams(spacing=0.01))
    assert r.pnl == 0.0 and r.cycles == 0 and r.stops == 0


def test_one_dip_and_recovery_books_exactly_one_cycle():
    p = GridParams(spacing=0.01, max_bars=10_000)
    # 100 -> 99 (fills the -1% BUY at 99.0) -> 100 (SELL target 99.99 filled)
    closes = [100.0, 99.0, 100.0]
    r = simulate(closes, p, track_events=True)
    assert r.cycles == 1 and r.stops == 0
    assert math.isclose(r.pnl, cycle_gross(p), rel_tol=1e-9)
    assert r.events == [(2, pytest.approx(cycle_gross(p)), "cycle")]


def test_price_stop_closes_at_market_with_taker_fee():
    p = GridParams(spacing=0.01, price_stop_k=1.0, max_bars=10_000)
    # BUY at 99 fills on bar 1; price then falls to 98.005 (-1.005% from entry) -> stop.
    # 98.005 stays above the next resting BUY (98.0), so no second fill happens.
    closes = [100.0, 99.0, 98.005, 98.005]
    r = simulate(closes, p, track_events=True)
    assert r.stops == 1 and r.cycles == 0
    assert [e[2] for e in r.events] == ["stop"]
    expected = (98.005 - 99.0) / 99.0 * p.pos_size - (p.fee_limit + p.fee_market) * p.pos_size
    assert math.isclose(r.pnl, expected, rel_tol=1e-9)
    assert abs(r.pnl) <= stop_loss_typical(p) * 1.05


def test_time_stop_fires_after_max_bars():
    p = GridParams(spacing=0.01, max_bars=5, price_stop_k=50.0)
    closes = [100.0, 99.0] + [99.2] * 10   # entry at 99, never reaches 99.99 nor the price stop
    r = simulate(closes, p, track_events=True)
    assert r.stops == 1
    bar = r.events[0][0]
    assert bar == 1 + p.max_bars


def test_crash_fills_at_most_max_open_and_stale_cancels_deep_levels():
    p = GridParams(spacing=0.01, max_open=3, stale_k=4.0, max_bars=10_000, price_stop_k=100.0)
    closes = [100.0, 80.0, 80.0, 80.0]      # -20 % in one bar
    r = simulate(closes, p, track_events=True)
    # All three resting BUYs (99, 98, 97) fill on the crash bar; nothing else can.
    assert r.cycles == 0 and r.stops == 0
    assert len([e for e in r.events if e[2] == "settlement"]) == 3


def test_slot_invariant_holds_on_random_walk():
    rng = np.random.default_rng(7)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, 20_000)))
    # simulate() asserts positions + resting buys == max_open after every bar.
    r = simulate(closes, GridParams(spacing=0.01), track_events=True)
    assert r.cycles + r.stops > 0
    settlement = [e for e in r.events if e[2] == "settlement"]
    assert len(settlement) <= 3


def test_grid_survives_rally_and_keeps_trading():
    """Regression for the v7 bug: after a strong rally the grid must re-seed."""
    p = GridParams(spacing=0.01, max_bars=10_000)
    up = list(np.linspace(100, 150, 200))             # rally: every BUY goes stale
    chop = [150.0, 148.0, 150.0] * 5                   # dips of -1.33 % then recovery
    r = simulate(up + chop, p)
    assert r.cycles >= 5, "grid did not re-seed after the rally"


def test_sum_of_events_equals_total_pnl():
    rng = np.random.default_rng(11)
    closes = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.003, 5_000)))
    r = simulate(closes, GridParams(spacing=0.015), track_events=True)
    assert math.isclose(sum(e[1] for e in r.events), r.pnl, rel_tol=1e-9, abs_tol=1e-9)


def test_params_validation():
    with pytest.raises(ValueError):
        GridParams(spacing=0.0)
    with pytest.raises(ValueError):
        GridParams(max_open=0)
