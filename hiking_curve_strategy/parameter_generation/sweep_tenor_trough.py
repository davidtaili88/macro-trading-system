"""
TENOR SWEEP — does the post-last-hike TROUGH-THEN-RALLY hold across the front end,
not just at the book's 2Y benchmark?

WHY THIS MATTERS (the instrument-aging problem):
  The book (Willer) benchmarks a CONSTANT-MATURITY 2Y and shows 2Y returns trough around
  the last hike, then rally (its exit signal). But a HELD instrument — a payer swap — AGES:
  a 2Y payer becomes a ~1Y payer as it's held, so at exit it is marked against a SHORTER
  tenor's rate. For the book's exit to transfer to the aged instrument, the SHORTER tenors
  must ALSO trough around the last hike and rally after. This sweep verifies that.

  It also bounds the ROLL policy for long cycles: a cycle longer than the swap tenor
  (e.g. 2015-2018, ~3yr) FORCES a roll — a 2Y swap entered early matures before the last
  hike. Rolling keeps the instrument's REMAINING tenor bounded. The safe rule that falls
  out (hindsight-free): roll often enough that remaining tenor never drops below the
  shortest tenor where the rally still holds — read that boundary off this sweep.

METHOD: anchor on ORACLE last-hike dates (ground truth from the target series), and for
each tenor take the duration-approx daily return r = -D*Δy, then average the CUMULATIVE
return path in a [-WIN, +WIN] trading-day window around each last hike. Report where each
tenor TROUGHS (min cumulative = yields peak = the exit point) and whether it rallies after.

Run:  cd hiking_curve_strategy && PYTHONPATH=. python3 parameter_generation/sweep_tenor_trough.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths  # noqa: F401
from utils.fred_utils import fetch_fred_dataframe
from signal_logic import derive_fed_hike_cycles, _load_signal_data

DATA_START = "1981-01-01"
WIN = 60   # trading days each side of the last hike

# tenor -> (FRED id or ('interp', a, b) for a synthetic point, approx modified duration in yrs).
# 1.5Y has no CMT series, so interpolate DGS1/DGS2. Duration ~= tenor is fine here: we care
# about the SHAPE/trough location, not absolute magnitude (which just scales with D).
TENORS = [
    ("1M",   ("DGS1MO",),        1 / 12),
    ("6M",   ("DGS6MO",),        0.5),
    ("1Y",   ("DGS1",),          1.0),
    ("1.5Y", ("interp", "DGS1", "DGS2"), 1.5),
    ("2Y",   ("DGS2",),          2.0),
]

RALLY_MIN_TENOR_NOTE = (
    "Roll the swap so REMAINING tenor never drops below the shortest tenor that still "
    "rallies (read from the table). E.g. if the rally holds down to 6M, a 2Y swap must be "
    "rolled at least every ~18mo (2.0 - 0.5). Quarterly/semiannual rolls sit well inside."
)


def _load(api_key):
    # Fetch each series on its OWN maximal calendar. fetch_fred_dataframe inner-joins and
    # dropna's ACROSS columns, so bundling DGS1MO (starts 2001) with 6M-2Y (1981+) would
    # truncate everything to 2001 and lose the 1995/2000 cycles. Fetch each id separately
    # (one-series call) so each tenor keeps every cycle it actually covers.
    def one(sid):
        return fetch_fred_dataframe(api_key, {"x": sid}, DATA_START, fill_method="ffill")["x"] / 100
    y1, y2 = one("DGS1"), one("DGS2")
    common = y1.index.intersection(y2.index)   # 1.5Y interp needs both (both 1980+, no loss)
    y1_5 = (0.5 * y1.loc[common] + 0.5 * y2.loc[common])
    return {
        "1M":   one("DGS1MO"),
        "6M":   one("DGS6MO"),
        "1Y":   y1,
        "1.5Y": y1_5,
        "2Y":   y2,
    }


def _event_avg(ret, anchors, win):
    """Average cumulative-return path in [-win, +win] td around each anchor date."""
    paths = []
    for lh in anchors:
        if lh not in ret.index:
            pos = ret.index.searchsorted(lh)
            if pos >= len(ret.index):
                continue
            lh = ret.index[pos]
        i = ret.index.get_loc(lh)
        if i - win < 0 or i + win >= len(ret.index):
            continue
        paths.append(np.nancumsum(ret.iloc[i - win:i + win + 1].values))
    if not paths:
        return None, 0
    return np.vstack(paths).mean(axis=0), len(paths)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    _, fed_target = _load_signal_data(api_key, start="1982-10-01")
    cycles = derive_fed_hike_cycles(fed_target, start_year=1990, min_hikes=2)
    anchors = [c["last_hike"] for c in cycles]
    print(f"Last-hike anchors ({len(anchors)} cycles): "
          f"{[d.date().isoformat() for d in anchors]}\n")

    series = _load(api_key)
    x = np.arange(-WIN, WIN + 1)

    # cumulative-path table (every 10 td)
    names = [t[0] for t in TENORS]
    print(f"{'td_from_last_hike':>18}" + "".join(f"{n:>9}" for n in names))
    ev = {}
    ncov = {}
    for name, _, D in TENORS:
        ret = (-D * series[name].diff()).rename(name)   # each tenor on its own calendar
        ev[name], ncov[name] = _event_avg(ret, anchors, WIN)
    for j in range(0, len(x), 10):
        row = []
        for name in names:
            row.append(f"{ev[name][j]*100:+8.2f}" if ev[name] is not None else "     n/a")
        print(f"{x[j]:>18}" + "".join(row))

    # trough location + rally verdict
    print("\nTROUGH (min cumulative return = yields peak = exit point) per tenor:")
    print(f"{'tenor':>6} {'trough_td':>10} {'rallies?':>9} {'ret@+60':>9} {'n_cyc':>6}")
    rally_ok = []
    for name, _, _ in TENORS:
        e = ev[name]
        if e is None:
            print(f"{name:>6} {'n/a':>10}")
            continue
        t_trough = int(x[np.argmin(e)])
        end = e[-1]
        rallied = e[-1] > e[np.argmin(e)] + 1e-9 and abs(t_trough) <= 15  # troughs near LH AND rises after
        print(f"{name:>6} {t_trough:>+10d} {'YES' if rallied else 'no':>9} "
              f"{end*100:>+8.2f}% {ncov[name]:>6}")
        if rallied:
            rally_ok.append(name)

    print()
    print(f"Rally-holds tenors (exit transfers here): {rally_ok}")
    print("=> " + RALLY_MIN_TENOR_NOTE)
    print("\nNOTE: shape/trough location is the signal; absolute magnitude just scales with D.")
    print("      1M has data only from 2001 (thinner), and is the expected exception — a 1M")
    print("      instrument is too short to price EXPECTED cuts, so it does not rally.")


if __name__ == "__main__":
    main()
