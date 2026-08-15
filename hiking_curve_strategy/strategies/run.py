"""
Entry point: run the signal-driven hiking cycle payer backtest.

Instrument: DGS2 duration-approximated returns (price-only, no carry/roll).
Signal:     FRED spread-based first-hike detector (signal_market.detect_signal).
Exit:       oracle last-hike date − 30 trading days (same as benchmark.py).

Usage:
    python hiking_curve_strategy/run.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths  # noqa: F401 — registers core/, benchmark/, utils/ on sys.path
from strategies.signal_trade_dgs import main as signal_trade_dgs_main


def main():
    signal_trade_dgs_main()


if __name__ == "__main__":
    main()
