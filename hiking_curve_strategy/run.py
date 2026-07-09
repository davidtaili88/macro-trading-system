"""
Entry point: run the signal-driven hiking cycle payer backtest.

Instrument: DGS2 duration-approximated returns (price-only, no carry/roll).
Signal:     FRED spread-based first-hike detector (signal_market.detect_signal).
Exit:       oracle last-hike date − 30 trading days (same as benchmark.py).

To add ZT futures later, swap fetch_carryless_dgs2_returns for fetch_zt +
compute_returns in data.py, and add roll cost deduction in backtest.calc_strat_ret.

Usage:
    python hiking_curve_strategy/run.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from signal_market import main as signal_market_main


def main():
    signal_market_main()


if __name__ == "__main__":
    main()
