# -*- coding: utf-8 -*-
"""Extra check: a one-bar crash fills exactly max_open levels and no more."""
from gridlab import GridParams, simulate


def test_crash_fill_count_matches_max_open():
    for max_open in (1, 2, 3, 5):
        p = GridParams(spacing=0.01, max_open=max_open, stale_k=4.0,
                       max_bars=10_000, price_stop_k=100.0)
        r = simulate([100.0, 80.0, 80.0], p, track_events=True)
        settled = [e for e in r.events if e[2] == "settlement"]
        assert len(settled) == max_open
