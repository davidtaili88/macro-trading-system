"""
Does the 2yr spread (DGS2 - DFF) carry the "path" information the 1yr spread loses —
staying STEADY through measured-cycle NOISE (where the 1yr wobbles and falsely trips the
exit) while STILL falling when a cycle GENUINELY ends?

This tests the user's exact hypothesis, in three parts:

  (Q1) NOISE periods (quiet holds + mid-measured-cycle): does spread_1yr wobble MORE than
       spread_2yr? If the 2yr is steadier here, it won't false-fire where the 1yr does.

  (Q2) REAL ENDS: at the true last hike of each cycle, do BOTH spreads fall together? A
       second signal is only useful if it AGREES with the 1yr when the cycle really ends
       (otherwise it would just block good exits too).

  (Q3) THE 2005 CASE specifically: at 2005-09-02, is the 2yr spread ALSO collapsing (=> a
       real signal we should have believed) or is it HOLDING (=> confirms noise, and the
       2yr would have kept us in)? This is the decisive one.

OVERFITTING / CIRCULARITY GUARD (the user's concern — we trade the 2yr):
  We must separate "2yr is a better SIGNAL" from "2yr is just a lower-variance clone of the
  1yr" (longer duration mechanically averages more curve, so it's smoother for free). So we
  report, alongside the levels:
    - the ratio of noise-period volatility (2yr vol / 1yr vol): if the 2yr's only advantage
      is that it's ~X% less volatile, that's the smoothing story, not a new-information story.
    - whether, at the 2005 dip, the 2yr's move is SMALLER than what its own noise vol would
      predict (i.e. it genuinely held) vs just scaled-down by the same factor everywhere.
  If the 2yr holds at 2005 by MORE than its blanket smoothing factor, that's real signal.
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
SUSPECT_EXIT = "2005-09-02"


def _fetch_dgs2_spread(api_key, index, dff):
    """spread_2yr_bp = (DGS2 - DFF)*100, aligned to the existing signal index.
    NOTE: inside _load_signal_data, dff/dgs1 are in PERCENT (e.g. 3.63), and the
    existing spread cols are (dgs1 - dff)*100. DGS2 from FRED is ALSO percent (4.04),
    so we keep it percent and do the identical *100 — do NOT decimalize one side."""
    from fredapi import Fred
    fred = Fred(api_key=api_key)
    dgs2 = fred.get_series("DGS2", observation_start="1976-01-01")
    dgs2.index = pd.to_datetime(dgs2.index)
    dgs2 = dgs2.reindex(index).astype(float)            # keep percent, like dff
    return (dgs2 - dff) * 100.0                          # -> bp, exactly like spread_1yr_bp


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
    sp1 = ds["spread_1yr_bp"]
    dff = ds["dff"]
    sp2 = _fetch_dgs2_spread(api_key, ds.index, dff)

    both = pd.DataFrame({"sp1": sp1, "sp2": sp2}).dropna()
    print(f"DGS2 spread available from {both.index[0].date()} "
          f"(DGS2 FRED history); overlap days: {len(both)}")

    # ---- Q1: noise-period volatility, 1yr vs 2yr --------------------------------
    holds = _find_flat_holds(fed_target)
    interior = _interior_mask(ds.index, holds)
    q = interior.reindex(both.index, fill_value=False)
    n1 = both.loc[q, "sp1"]; n2 = both.loc[q, "sp2"]
    # day-to-day wobble = std of daily changes on quiet days
    v1 = n1.diff().std(); v2 = n2.diff().std()
    # also the 21d-change spread (the thing the gate keyed on)
    w1 = both["sp1"].diff(21).reindex(n1.index).std()
    w2 = both["sp2"].diff(21).reindex(n2.index).std()
    print("\n" + "=" * 78)
    print("(Q1) NOISE-PERIOD WOBBLE (quiet-hold interior days)")
    print("=" * 78)
    print(f"  daily-change std   1yr: {v1:.1f} bp    2yr: {v2:.1f} bp    ratio 2yr/1yr: {v2/v1:.2f}")
    print(f"  21d-change  std    1yr: {w1:.1f} bp    2yr: {w2:.1f} bp    ratio 2yr/1yr: {w2/w1:.2f}")
    smoothing_factor = v2 / v1
    print(f"  => the 2yr is ~{(1-smoothing_factor)*100:.0f}% less noisy day-to-day. HOLD this factor:")
    print(f"     if 2yr merely SCALES 1yr's moves by ~{smoothing_factor:.2f} everywhere, it's just")
    print(f"     a smoother clone (no new info). It earns its keep only if it holds MORE than")
    print(f"     this at the 2005 dip while still falling as much as 1yr at real ends.")

    # ---- Q2: do BOTH fall at real ends? -----------------------------------------
    print("\n" + "=" * 78)
    print("(Q2) REAL ENDS — trailing 63td change in each spread at the true last hike")
    print("     (want BOTH clearly negative: the 2yr must AGREE when the cycle really ends)")
    print("=" * 78)
    print(f"  {'cycle':<10}{'last hike':<13}{'sp1 lvl':>9}{'sp2 lvl':>9}{'sp1 d63':>9}{'sp2 d63':>9}")
    d63_1 = both["sp1"].diff(63); d63_2 = both["sp2"].diff(63)
    for lbl, lh in TRUE_LAST_HIKES.items():
        lh = pd.Timestamp(lh)
        idx = both.index[both.index <= lh]
        if not len(idx):
            print(f"  {lbl:<10}{str(lh.date()):<13}   (no DGS2 data this early)")
            continue
        d = idx[-1]
        print(f"  {lbl:<10}{str(lh.date()):<13}{both.loc[d,'sp1']:>9.0f}{both.loc[d,'sp2']:>9.0f}"
              f"{d63_1[d]:>9.0f}{d63_2[d]:>9.0f}")

    # ---- Q3: the 2005 case ------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"(Q3) THE 2005 CASE — around {SUSPECT_EXIT} (the false exit)")
    print("=" * 78)
    d = pd.Timestamp(SUSPECT_EXIT)
    win = both.index[(both.index > d - pd.Timedelta(days=120)) & (both.index <= d + pd.Timedelta(days=40))]
    wk = both.reindex(win).resample("W").last()
    wk["sp1-sp2"] = wk["sp1"] - wk["sp2"]
    print(wk.to_string(float_format=lambda x: f"{x:.0f}"))
    idx = both.index[both.index <= d][-1]
    m21_1 = both["sp1"].diff(21)[idx]; m21_2 = both["sp2"].diff(21)[idx]
    m63_1 = d63_1[idx]; m63_2 = d63_2[idx]
    print(f"\n  AT {idx.date()}:")
    print(f"    spread_1yr : {both.loc[idx,'sp1']:+.0f} bp   21d move {m21_1:+.0f}   63d move {m63_1:+.0f}")
    print(f"    spread_2yr : {both.loc[idx,'sp2']:+.0f} bp   21d move {m21_2:+.0f}   63d move {m63_2:+.0f}")
    # smoothing-adjusted test: is the 2yr's 21d move smaller than smoothing_factor * 1yr move?
    expected_if_clone = smoothing_factor * m21_1
    print(f"\n  Smoothing-adjusted check (the overfitting guard):")
    print(f"    if 2yr were just a {smoothing_factor:.2f}x-smoother CLONE, its 21d move would be ~{expected_if_clone:+.0f} bp")
    print(f"    actual 2yr 21d move                                             = {m21_2:+.0f} bp")
    if abs(m21_2) < abs(expected_if_clone) - 2:
        print(f"    => 2yr held MORE than smoothing alone predicts: GENUINE steadiness (real signal).")
    elif abs(m21_2) > abs(expected_if_clone) + 2:
        print(f"    => 2yr moved MORE than the clone would: no help here.")
    else:
        print(f"    => 2yr move ~= smoothed clone: the advantage is JUST lower variance, not new info.")

    print("\nVerdict logic:")
    print("  2yr HELPS iff (Q3) it holds at 2005 by more than its blanket smoothing factor,")
    print("  AND (Q2) it still falls hard at the real ends. If it only wins by being smoother,")
    print("  that's the circularity trap — we'd get the same by just smoothing the 1yr more.")


if __name__ == "__main__":
    main()
