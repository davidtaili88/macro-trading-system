"""
Does the TERM-STRUCTURE curve slope (10y-2y, 10y-3m) work as a leading indicator for the
end of a hiking cycle — where DGS1-DFF (front-end spread) could not?

CHOSEN BY THEORY, BEFORE looking at 2005: the 10y-2y / 10y-3m slope is the canonical
"Fed is late/near done" gauge. It carries the LONG end, which holds cycle-maturity
information the FRONT end (DGS1-DFF) mechanically compresses away in a measured cycle
(where DFF ratchets under a rising 1yr yield — see yield_slope_vs_spread_slope.py). It is
NOT just another function of DFF.

We test ONE mechanism across ALL cycles (not "does it rescue 2005"):

  (M1) LEAD TIME: does the curve slope trough / cross into inversion BEFORE the true last
       hike, consistently across cycles? A useful late-cycle signal leads the top by a
       stable-ish margin. We report, per cycle, the slope level at the last hike and the
       date it first inverts (crosses < 0) within the cycle.

  (M2) THE 2005 DISCRIMINATION: at 2005-09-02 (front-end spread FALSELY said exit), what is
       the curve slope doing? If it says "late cycle / near done but not yet" (low/inverting
       but the cycle plainly still running), that's consistent with HOLDING — the opposite
       of the front-end's false exit.

  (M3) FALSE-FIRE CHECK: is the curve slope ALSO low/negative during quiet holds and early
       cycle, where it must NOT signal an imminent top? An indicator that's inverted all the
       time is useless. We report its distribution on quiet-hold days vs at true tops.

Curve slopes from FRED: DGS10, DGS2, DGS3MO (all constant-maturity). All bp.
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
)

HOLD_MIN_DAYS = 126
TRIM_DAYS     = 30

TRUE_LAST_HIKES = {
    "1994-95": "1995-02-01",
    "1999-00": "2000-05-16",
    "2004-06": "2006-06-29",
    "2015-18": "2018-12-20",
    "2022-23": "2023-07-27",
}
# cycle START (first hike) to scope the "first inversion within cycle" search
CYCLE_START = {
    "1994-95": "1994-02-04",
    "1999-00": "1999-06-30",
    "2004-06": "2004-06-30",
    "2015-18": "2015-12-16",
    "2022-23": "2022-03-16",
}
SUSPECT = "2005-09-02"


def _fetch(api_key, series, index):
    from fredapi import Fred
    fred = Fred(api_key=api_key)
    s = fred.get_series(series, observation_start="1976-01-01")
    s.index = pd.to_datetime(s.index)
    return s.reindex(index).astype(float)   # percent


def _find_flat_holds(fed_target):
    change = fed_target.diff(); active = change.abs() > FED_TARGET_MOVE_FLOOR
    idx = fed_target.index; holds, rs = [], None
    for i in range(len(idx)):
        if active.iloc[i]:
            if rs is not None:
                if i - rs >= HOLD_MIN_DAYS: holds.append((idx[rs], idx[i-1]))
                rs = None
        elif rs is None:
            rs = i
    if rs is not None and (len(idx) - rs) >= HOLD_MIN_DAYS: holds.append((idx[rs], idx[-1]))
    return holds


def _interior_mask(index, holds):
    m = pd.Series(False, index=index)
    for s, e in holds:
        i0, i1 = index.searchsorted(s), index.searchsorted(e)
        lo, hi = i0 + TRIM_DAYS, i1 - TRIM_DAYS
        if hi > lo: m.iloc[lo:hi+1] = True
    return m


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    ds, fed_target = _load_signal_data(api_key, start="1976-01-01")
    dgs10  = _fetch(api_key, "DGS10",  ds.index)
    dgs2   = _fetch(api_key, "DGS2",   ds.index)
    dgs3mo = _fetch(api_key, "DGS3MO", ds.index)

    slope_10_2  = (dgs10 - dgs2)   * 100   # bp
    slope_10_3m = (dgs10 - dgs3mo) * 100
    front = ds["spread_1yr_bp"]            # the front-end spread that fails

    curves = {"10y-2y": slope_10_2, "10y-3m": slope_10_3m}

    # === M1: lead time — slope at last hike + first inversion within cycle ====
    print("=" * 92)
    print("(M1) CURVE SLOPE AT THE TRUE LAST HIKE, and first inversion (<0) within the cycle")
    print("     (want: consistently LOW/inverted near the top, and inverting a stable lead BEFORE it)")
    print("=" * 92)
    print(f"{'cycle':<10}{'last hike':<13}{'10y-2y@top':>12}{'first inv 10y-2y':>20}{'lead(td)':>10}")
    for lbl, lh in TRUE_LAST_HIKES.items():
        lh = pd.Timestamp(lh); cs = pd.Timestamp(CYCLE_START[lbl])
        sl = slope_10_2
        idx = sl.index[sl.index <= lh]
        at_top = sl[idx[-1]] if len(idx) else np.nan
        within = sl[(sl.index >= cs) & (sl.index <= lh)]
        inv = within.index[within < 0]
        if len(inv):
            first_inv = inv[0]
            lead = np.busday_count(first_inv.date(), lh.date())
            print(f"{lbl:<10}{str(lh.date()):<13}{at_top:>12.0f}{str(first_inv.date()):>20}{lead:>10}")
        else:
            print(f"{lbl:<10}{str(lh.date()):<13}{at_top:>12.0f}{'(never inverts)':>20}{'--':>10}")

    # === M2: the 2005 discrimination =========================================
    print("\n" + "=" * 92)
    print(f"(M2) 2005 DISCRIMINATION — {SUSPECT}: front-end spread FALSELY said exit. What do curves say?")
    print("=" * 92)
    d = pd.Timestamp(SUSPECT)
    i = ds.index[ds.index <= d][-1]
    print(f"  front-end spread (DGS1-DFF): {front[i]:+.0f} bp   (this fired the false exit)")
    for name, sl in curves.items():
        j = sl.index[sl.index <= d][-1]
        prior_idx = sl.index[sl.index <= d - pd.Timedelta(days=90)]
        d63 = sl[j] - sl.loc[prior_idx[-1]] if len(prior_idx) else np.nan
        print(f"  {name:<8}: {sl[j]:+.0f} bp   (trailing ~63td change {d63:+.0f} bp)")
    print("  Interpretation: if the curve is already flat/inverted here, it says 'late cycle,")
    print("  Fed nearly done' — consistent with HOLDING through a top that's near but not yet,")
    print("  NOT the front-end's 'exit now'.")

    # === M3: false-fire distribution =========================================
    holds = _find_flat_holds(fed_target)
    interior = _interior_mask(ds.index, holds)
    cycles = signal_to_cycles(detect_signal(api_key, start="1976-01-01"))
    in_hik = pd.Series(False, index=ds.index)
    for c in cycles:
        in_hik.loc[(ds.index >= c["first_hike"]) & (ds.index <= c["last_hike"])] = True
    quiet = interior & (~in_hik)

    print("\n" + "=" * 92)
    print("(M3) IS THE CURVE INVERTED ALL THE TIME? distribution on quiet holds vs at tops")
    print("     (a useful signal is NOT usually inverted; inversion should concentrate near tops)")
    print("=" * 92)
    for name, sl in curves.items():
        q = sl[quiet].dropna()
        tops = [sl[sl.index[sl.index <= pd.Timestamp(v)][-1]] for v in TRUE_LAST_HIKES.values()
                if len(sl.index[sl.index <= pd.Timestamp(v)])]
        print(f"  {name}:  quiet-hold median {q.median():+.0f} bp  (25/75: {q.quantile(.25):+.0f}/{q.quantile(.75):+.0f})"
              f"   |  frac quiet days inverted: {(q<0).mean()*100:.0f}%")
        print(f"           slope at the 5 true tops: {[round(x) for x in tops]}")

    print("\nVerdict logic:")
    print("  WORKS if: (M1) it's consistently low/inverted near tops with a not-crazy-variable lead,")
    print("  (M2) it says late-cycle/hold at 2005 (not exit), and (M3) it is NOT chronically inverted")
    print("  in quiet/early periods. If inversion is everywhere or the lead swings wildly, it's not usable.")


if __name__ == "__main__":
    main()
