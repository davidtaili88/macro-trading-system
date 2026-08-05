"""
AND-GATE FEASIBILITY: does the curve slope separate REAL tops from MID-CYCLE FALSE ALARMS
at every moment the maturity-ratio exit WANTS to fire — not just at 2005?

The idea we're vetting: exit = (ratio exit fires) AND (curve shape confirms late-cycle).
That only helps if, ACROSS ALL HISTORY, the curve says "hold" at the mid-cycle ratio
false-alarms and "ok" at the real tops. If the curve looks the same in both, the AND-gate
buys nothing (or blocks good exits).

METHOD (no per-date cherry-picking):
  1. Reconstruct the RAW ratio  r(t) = smoothed_spread_1yr(t) / cum_bp_hiked_since_cycle_start(t)
     exactly as detect_signal computes it, on every in-cycle day with cum >= FLOOR.
  2. Find every day the ratio is "firing" (r < RATIO_EXIT_THRESHOLD) — these are all the
     moments the exit wants to trigger.
  3. Label each firing day by distance to that cycle's TRUE last hike:
        REAL    = within +/- 63td of the true last hike (a legitimate exit window)
        EARLY   = more than 63td BEFORE the true last hike (a FALSE alarm — the 2005 disease)
  4. Compare the CURVE SLOPE (10y-2y, 10y-3m) distribution across REAL vs EARLY firing days.
     AND-gate works iff the curve is systematically HIGHER (less inverted / steeper) on EARLY
     days than on REAL days — i.e. a curve-flatness threshold could veto the early ones while
     letting the real ones through.

Reports the two distributions and the overlap. If they overlap heavily, the AND-gate fails.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "hiking_curve_strategy"))
import _paths  # noqa: F401
from signal_logic import (
    _load_signal_data, detect_signal, signal_to_cycles, FED_TARGET_MOVE_FLOOR,
    EXIT_SMOOTH_WINDOW_DAYS, RATIO_EXIT_THRESHOLD, RATIO_EXIT_FLOOR_BP,
)

TRUE_LAST_HIKES = {
    "1994-95": "1995-02-01", "1999-00": "2000-05-16", "2004-06": "2006-06-29",
    "2015-18": "2018-12-20", "2022-23": "2023-07-27",
}
REAL_WINDOW_TD = 63   # within +/-63td of true last hike = a legitimate exit window


def _fetch(api_key, series, index):
    from fredapi import Fred
    fred = Fred(api_key=api_key)
    s = fred.get_series(series, observation_start="1976-01-01")
    s.index = pd.to_datetime(s.index)
    return s.reindex(index).astype(float)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    ds, fed_target = _load_signal_data(api_key, start="1976-01-01")
    spread_1yr = ds["spread_1yr_bp"]
    target_change = fed_target.diff()

    # cum bp hiked since each hiking-cycle start (in_cycle logic mirrors detect_signal)
    cum_hikes_bp = (target_change.clip(lower=0) * 10000).cumsum()
    spread_exit = spread_1yr.rolling(EXIT_SMOOTH_WINDOW_DAYS, min_periods=1).mean()

    curves = {
        "10y-2y": (_fetch(api_key, "DGS10", ds.index) - _fetch(api_key, "DGS2",   ds.index)) * 100,
        "10y-3m": (_fetch(api_key, "DGS10", ds.index) - _fetch(api_key, "DGS3MO", ds.index)) * 100,
    }

    # reconstruct in-cycle + cum-since-cycle-start using the strategy's episodes' first_hike
    # as cycle anchors (close enough to detect_signal's in_cycle-start for this scan)
    cycles = signal_to_cycles(detect_signal(api_key, start="1976-01-01"))

    # build ratio firing days, labelled, per true cycle
    real_rows, early_rows = [], []
    for lbl, lh in TRUE_LAST_HIKES.items():
        lh = pd.Timestamp(lh)
        # cycle span: from ~1yr before last hike out to the last hike (the hiking phase)
        # anchor cum at the first in-cycle day: use 400td before last hike as a generous start
        span = ds.index[(ds.index >= lh - pd.Timedelta(days=800)) & (ds.index <= lh + pd.Timedelta(days=30))]
        if not len(span):
            continue
        cum0 = cum_hikes_bp[span[0]]
        for d in span:
            cum = cum_hikes_bp[d] - cum0
            if cum < RATIO_EXIT_FLOOR_BP:
                continue
            r = spread_exit[d] / cum
            if r < RATIO_EXIT_THRESHOLD:                 # ratio wants to fire here
                lag = np.busday_count(d.date(), lh.date())   # +ve = before last hike
                rec = {"cycle": lbl, "date": d.date(), "ratio": r, "lag": lag}
                for name, sl in curves.items():
                    rec[name] = sl[d] if d in sl.index else np.nan
                (real_rows if abs(lag) <= REAL_WINDOW_TD else
                 (early_rows if lag > REAL_WINDOW_TD else real_rows)).append(rec)

    real = pd.DataFrame(real_rows)
    early = pd.DataFrame(early_rows)

    print("=" * 90)
    print(f"RATIO-EXIT FIRING DAYS (r < {RATIO_EXIT_THRESHOLD}), labelled by distance to true last hike")
    print(f"  REAL  = within +/-{REAL_WINDOW_TD}td of last hike (legit exit window)")
    print(f"  EARLY = >{REAL_WINDOW_TD}td BEFORE last hike (false alarm — the 2005 disease)")
    print("=" * 90)
    print(f"  firing days total: REAL={len(real)}   EARLY={len(early)}")

    for name in curves:
        print("\n" + "-" * 90)
        print(f"CURVE = {name}   (bp;  AND-gate works iff EARLY days are systematically HIGHER/steeper)")
        print("-" * 90)
        for tag, dfx in [("EARLY (false alarms)", early), ("REAL  (legit tops)", real)]:
            if len(dfx) and dfx[name].notna().any():
                s = dfx[name].dropna()
                print(f"  {tag:<22}  n={len(s):>4}   median={s.median():>+6.0f}   "
                      f"25/75={s.quantile(.25):>+5.0f}/{s.quantile(.75):>+5.0f}   "
                      f"min/max={s.min():>+5.0f}/{s.max():>+5.0f}")
            else:
                print(f"  {tag:<22}  (no data)")
        # separation check: is there a threshold T s.t. EARLY mostly > T and REAL mostly < T?
        if len(real) and len(early):
            e, r = early[name].dropna(), real[name].dropna()
            if len(e) and len(r):
                # best single threshold by simple midpoint of medians
                T = (e.median() + r.median()) / 2
                early_blocked = (e > T).mean() * 100
                real_kept     = (r < T).mean() * 100
                print(f"  candidate veto threshold T={T:+.0f} bp:  "
                      f"blocks {early_blocked:.0f}% of EARLY false alarms, "
                      f"keeps {real_kept:.0f}% of REAL exits")

    # per-cycle EARLY firing (which cycles have the false-alarm disease + curve then)
    print("\n" + "=" * 90)
    print("PER-CYCLE: earliest ratio false-alarm and the curve slope THAT day")
    print("=" * 90)
    if len(early):
        for lbl in TRUE_LAST_HIKES:
            sub = early[early["cycle"] == lbl]
            if len(sub):
                first = sub.sort_values("lag", ascending=False).iloc[0]  # most-early
                cv = "  ".join(f"{n}={first[n]:+.0f}" for n in curves)
                print(f"  {lbl}: first false-alarm {first['date']} (lag +{first['lag']}td)   {cv}")

    print("\nVerdict: if EARLY and REAL curve distributions OVERLAP heavily (a threshold can't")
    print("block early without also killing real exits), the AND-gate fails — the curve doesn't")
    print("separate them in general, and 2005 was a lucky single case.")


if __name__ == "__main__":
    main()
