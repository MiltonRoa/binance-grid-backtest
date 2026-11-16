"""gridlab: a fast, honest backtester for ladder-grid strategies on 1-minute candles.

The package was extracted from a research script that ran over 48 months of
Binance USDT-M futures data (529 symbols, 1-minute bars stored as monthly
Parquet files).  The simulator is a pure function over a close-price array, so
it can be unit-tested on synthetic paths and reused on any data source.
"""

from .sim import GridParams, simulate

__all__ = ["GridParams", "simulate"]
__version__ = "1.0.0"
