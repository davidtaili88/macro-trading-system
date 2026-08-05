"""
Test the MEDIAN-ROC GATE in front of the ratio exit.

Design being tested (David's synthesis):
    EXIT if:
        median_ROC(spread_1yr) has crossed a "rolling over" gate   (shock-robust)
      AND
        spread_1yr / cum_bp_hiked_since_cycle_start < RATIO_THRESHOLD (mature)
      OR
        old level backstop: spread_1yr_exit < -15 AND spread_3mo_exit < 0

Why the median gate: a transient shock (SVB, ~5 trading days) is a MINORITY of a
~21-day window, so it can't move the 21-day MEDIAN of the ROC. The gate therefore
stays shut through a shock -> the ratio is never even checked -> no spurious exit
and no re-entry-into-shock (the ep10 whipsaw). A GENUINE rollover (1995, 2015-18)
is broad-based across the month, moves the median, opens the gate -> real exit.

The gate ALSO addresses the Dec-2022 premature exit: the ratio alone fired there
because the denominator got large while the spread was still ~+45bp (not rolling
over). Requiring the ROC gate first means "much hiked" can no longer fire the exit
by itself — the spread must independently be rolling over.

Two ROC-gate definitions compared (pick from the data):
  A) 21d-median of DAILY ROC:  median over 21d of spread.diff(1)   (short-horizon)
  B) median of MONTH-OVER-MONTH ROC: median over Wd of spread.diff(21) (long-horizon)

Gate opens when the median ROC < ROC_GATE_BP (spread falling at least this fast, bp).
ROC_GATE_BP is set mildly negative so flat/noisy chop doesn't open the gate; it is
a coarse principled cutoff ("spread genuinely declining"), not fitted to P&L.

This is a DIAGNOSTIC — prints exit dates for CURRENT rule vs each gated variant.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "hiking_curve_strategy"))
import _paths  # noqa: F401 — registers core/, benchmark/, utils/ on sys.path
from signal_logic import (
    _load_signal_data, _last_cut_dates,
    HOLD_MONTHS, THRESHOLD_1YR_BP, THRESHOLD_3MO_BP,
    ENTRY_SMOOTH_WINDOW_DAYS, FALSE_PROMISE_THRESHOLD_1YR_BP,
    THRESHOLD_1YR_EXIT, THRESHOLD_3MO_EXIT,
    RATIO_EXIT_THRESHOLD, RATIO_EXIT_FLOOR_BP,
    FED_TARGET_MOVE_FLOOR,
)

TRUE_LAST = {
    "1993": None,
    "1994": "1995-02-01", "2004": "2006-06-29",
    "2015": "2018-12-20", "2022": "2023-07-26",
}

ROC_MEDIAN_WINDOW = 21   # trading days for the median of ROC (~1 month)
ROC_DIFF_LAG_B    = 21   # def B: month-over-month ROC horizon
ROC_GATE_BP       = -3   # bp: gate opens when median ROC below this (spread falling)


def _build_in_cycle(daily_spread, fed_target):
    last_cuts = _last_cut_dates(fed_target)
    target_change = fed_target.diff()
    is_cut = target_change < -FED_TARGET_MOVE_FLOOR
    in_cycle = pd.Series(False, index=daily_spread.index)
    for cut_date in last_cuts:
        hold_end = cut_date + pd.DateOffset(months=HOLD_MONTHS)
        for date in daily_spread.index[daily_spread.index >= hold_end]:
            if is_cut.get(date, False):
                break
            in_cycle[date] = True
    return in_cycle, target_change


def detect(daily_spread, fed_target, roc_gate=None, roc_series=None):
    """Replicate detect_signal's loop, optionally requiring roc_series<roc_gate
    (median-ROC gate) as a PRECONDITION for the ratio exit. roc_gate=None -> current rule."""
    in_cycle, target_change = _build_in_cycle(daily_spread, fed_target)
    spread_3mo = daily_spread["spread_3mo_bp"]; spread_1yr = daily_spread["spread_1yr_bp"]
    s3e = spread_3mo.rolling(ENTRY_SMOOTH_WINDOW_DAYS, min_periods=1).mean()
    s1e = spread_1yr.rolling(ENTRY_SMOOTH_WINDOW_DAYS, min_periods=1).mean()
    hike_dates = set(target_change.index[target_change > FED_TARGET_MOVE_FLOOR].tolist())
    EXIT_CAP = 100
    s1x = spread_1yr.rolling(5, min_periods=1).mean()
    s3x = spread_3mo.clip(-EXIT_CAP, EXIT_CAP).rolling(5, min_periods=1).mean()
    # target_change is decimal; *10000 converts decimal -> bp
    cum = (target_change.clip(lower=0) * 10000).cumsum()

    latched = False; hiked = False; was_in = False; cum0 = 0.0
    sig = pd.Series(False, index=daily_spread.index)
    for date in daily_spread.index:
        if not in_cycle[date]:
            latched = False; hiked = False; was_in = False
        else:
            if not was_in:
                cum0 = cum[date]; was_in = True
            if not latched:
                if s3e[date] > THRESHOLD_3MO_BP and s1e[date] > THRESHOLD_1YR_BP:
                    latched = True; hiked = False
                sig[date] = latched; continue
            if date in hike_dates:
                hiked = True
            cs = cum[date] - cum0
            # ROC gate: only allow the ratio exit when the spread is genuinely rolling over
            roc_ok = True if roc_gate is None else (roc_series[date] < roc_gate)
            if not hiked and s1e[date] <= FALSE_PROMISE_THRESHOLD_1YR_BP:
                latched = False
            elif roc_ok and cs >= RATIO_EXIT_FLOOR_BP and s1x[date] / cs < RATIO_EXIT_THRESHOLD:
                latched = False
            elif s1x[date] < THRESHOLD_1YR_EXIT and s3x[date] < THRESHOLD_3MO_EXIT:
                latched = False   # old level backstop (always active)
        sig[date] = latched
    return sig


def to_cycles(sig):
    out = []; ins = False; st = None; n = 0
    for d, v in sig.items():
        if v and not ins: st = d; ins = True
        elif not v and ins: n += 1; out.append((f"ep{n} {st.year}-{d.year}", st, d)); ins = False
    if ins: n += 1; out.append((f"ep{n} {st.year}-ongoing", st, sig.index[-1]))
    return out


def show(sig, label):
    print(f"\n{'='*66}\n  {label}\n{'='*66}")
    for lbl, st, ex in to_cycles(sig):
        print(f"  {lbl:16s} entry {st.date()}  exit {ex.date()}")


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")
    ds, ft = _load_signal_data(api_key, start="1976-01-01")
    s1 = ds["spread_1yr_bp"]

    # def A: 21d median of daily ROC
    rocA = s1.diff(1).rolling(ROC_MEDIAN_WINDOW, min_periods=5).median()
    # def B: median over window of month-over-month ROC
    rocB = s1.diff(ROC_DIFF_LAG_B).rolling(ROC_MEDIAN_WINDOW, min_periods=5).median()

    show(detect(ds, ft, roc_gate=None), "CURRENT (no ROC gate)")
    show(detect(ds, ft, roc_gate=ROC_GATE_BP, roc_series=rocA),
         f"GATE A: 21d-median of DAILY ROC  <  {ROC_GATE_BP}bp")
    show(detect(ds, ft, roc_gate=ROC_GATE_BP, roc_series=rocB),
         f"GATE B: 21d-median of diff(21) ROC  <  {ROC_GATE_BP}bp")

    print("\n" + "=" * 66)
    print("What to look for:")
    print("  - Does the 2022-23 whipsaw (current ep9+ep10) collapse back to ONE")
    print("    2022->2023 episode? (gate blocks the Dec-2022 premature exit)")
    print("  - Do the good exits (1995 ~Mar/Apr, 2018 ~Dec/Jan) SURVIVE, not far later?")
    print(f"  true last hikes: {TRUE_LAST}")


if __name__ == "__main__":
    main()
