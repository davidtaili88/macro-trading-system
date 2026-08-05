"""
Diagnostic for the SIMPLE level-ratio exit rule:

    ratio(t) = smoothed_spread(t) / cumulative_bp_hiked_since_entry(t)
    EXIT when ratio < THRESHOLD   (default 0.15)

Intuition (the "25/225 vs 25/100" rule): the numerator is how much hiking is
STILL priced; the denominator is how much has ALREADY been delivered. When
what's-still-priced shrinks to a small fraction of what's-been-done, the cycle is
mostly behind us and we leave. Early on, the same small priced amount is a large
fraction of the little that's been hiked -> we stay (it's a pause, not the end).

Why this is simpler than the ROC version: the ratio is bp/bp -> dimensionless, so
NO peak-normalization is needed; and there is NO differencing (diff/ROC), so no
minus-sign bookkeeping. Just: (light-smoothed spread) / (cumulative hikes).

Key property we're testing — does dividing by hikes fix the LEVEL EXIT's lag?
The old level exit waited for spread < -15bp (deeply negative = very late). This
rule can fire while the spread is still POSITIVE (e.g. +30bp / 225bp = 0.13 < 0.15),
which would be months earlier. So we measure, for each cycle:
   - the day the 0.15 rule first fires,
   - how many trading days that is vs the TRUE last hike (want: small, and NOT
     before the last hike by a lot = not a false early exit),
   - whether any MID-CYCLE PAUSE day dips below 0.15 (a false fire -> disqualifying).

Tested on BOTH tenors (1yr, 3mo). Spread lightly smoothed (5d) to kill single-day
DFF/holiday blips without adding lag.

THRESHOLD is a DISCRETIONARY, round choice (0.15), NOT fitted. We also print the
ratio at each true-end and each pause so you can SEE whether 0.15 sits in a gap
between the two, or whether a different round value (0.10, 0.20) would be needed —
and whether ANY single value separates them (if not, per the n=4 discipline, the
rule fails and we don't tune one in).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "hiking_curve_strategy"))
import _paths  # noqa: F401 — registers core/, benchmark/, utils/ on sys.path
from signal_logic import _load_signal_data, FED_TARGET_MOVE_FLOOR


CYCLES = [
    {"label": "1994-95", "entry": "1993-11-04", "last_hike": "1995-02-01"},
    {"label": "2004-06", "entry": "2004-06-03", "last_hike": "2006-06-29"},
    {"label": "2015-18", "entry": "2015-12-10", "last_hike": "2018-12-20"},
    {"label": "2022-23", "entry": "2022-02-04", "last_hike": "2023-07-27"},
]

SMOOTH_DAYS    = 5      # light smooth: kill DFF/holiday blips, no real lag
PAUSE_MIN_DAYS = 120    # >=4 months between hikes = a real pause
THRESHOLD      = 0.15   # discretionary round exit level (NOT fitted)
FLOOR_BP       = 25     # min cumulative hikes before ratio is defined (avoid /~0)


def _cum_hikes(fed_target: pd.Series) -> pd.Series:
    chg = fed_target.diff()
    # fed_target is decimal (0.045 = 4.5%); *10000 converts decimal -> bp
    return (chg.clip(lower=0) * 10000).cumsum()   # bp, monotone up


def _pause_windows(fed_target, entry, last):
    chg = fed_target.diff()
    hd = chg.index[chg > FED_TARGET_MOVE_FLOOR]
    hd = hd[(hd >= entry) & (hd <= last)]
    return [(a, b) for a, b in zip(hd[:-1], hd[1:]) if (b - a).days >= PAUSE_MIN_DAYS]


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    ds, fed_target = _load_signal_data(api_key, start="1976-01-01")
    cum_all = _cum_hikes(fed_target)
    tenors = {"1yr": ds["spread_1yr_bp"], "3mo": ds["spread_3mo_bp"]}

    for tlabel, spread in tenors.items():
        sm = spread.rolling(SMOOTH_DAYS, min_periods=1).mean()
        print("\n" + "#" * 84)
        print(f"#  TENOR = {tlabel}   ratio = smoothed_spread / cumulative_bp_hiked,  exit < {THRESHOLD}")
        print("#" * 84)

        for c in CYCLES:
            entry = pd.Timestamp(c["entry"]); last = pd.Timestamp(c["last_hike"])
            cum_since = (cum_all - cum_all.asof(entry)).clip(lower=0)
            cum_since = cum_since.reindex(sm.index).ffill()

            # ratio defined only once >= FLOOR_BP has been hiked
            win = (sm.index >= entry) & (sm.index <= last + pd.DateOffset(months=8))
            defined = cum_since >= FLOOR_BP
            ratio = (sm / cum_since).where(defined & win)

            # ---- where does the 0.15 rule first fire (from entry onward)? ----
            fire_region = ratio[(ratio.index >= entry) & defined]
            below = fire_region[fire_region < THRESHOLD]
            fire_day = below.index[0] if len(below) else None
            if fire_day is not None:
                lag_td = int((sm.index.searchsorted(fire_day) - sm.index.searchsorted(last)))
                fire_str = f"fires {fire_day.date()}  ({lag_td:+d} trading days vs last hike)"
            else:
                fire_str = "NEVER fires below threshold in-window"

            # ---- ratio at TRUE END window ----
            endm = (ratio.index >= last - pd.DateOffset(months=2)) & (ratio.index <= last + pd.DateOffset(months=1))
            end_med = ratio[endm].median()

            # ---- ratio during each PAUSE + does any pause day fire early? ----
            pauses = _pause_windows(fed_target, entry, last)
            pause_notes = []
            false_fire = False
            for a, b in pauses:
                pm = (ratio.index >= a) & (ratio.index <= b)
                sub = ratio[pm].dropna()
                if sub.empty:
                    continue
                dips = (sub < THRESHOLD).any()
                if dips:
                    false_fire = True
                first_dip = sub[sub < THRESHOLD].index[0].date() if dips else None
                pause_notes.append(
                    f"{a.date()}->{b.date()}: median={sub.median():.3f} min={sub.min():.3f}"
                    + (f"  *** DIPS <{THRESHOLD} on {first_dip} (FALSE FIRE)" if dips else "  (safe)")
                )

            print(f"\n  {c['label']}   last hike {last.date()}")
            print(f"     end-window ratio median = {end_med:.3f}    {fire_str}")
            if pause_notes:
                for pn in pause_notes:
                    print(f"     PAUSE  {pn}")
            else:
                print(f"     PAUSE  (none in this cycle)")
            if false_fire:
                print(f"     >>> WARNING: 0.15 rule FALSE-FIRES inside a pause in this cycle")

    print("\n" + "=" * 84)
    print("How to read this:")
    print(f"  GOOD  = fires a small number of trading days AFTER the last hike (lag cut vs")
    print(f"          the old -15bp exit, which was ~80-115 td late), and NO pause false-fire.")
    print(f"  BAD   = fires long before the last hike, or DIPS <{THRESHOLD} inside a pause")
    print(f"          (would have exited a still-live cycle).")
    print(f"  Compare end-window ratio (should be <~{THRESHOLD}) against pause min (should be")
    print(f"  >{THRESHOLD}). If a pause min is < an end ratio, 0.15 can't separate them -> the")
    print(f"  rule fails on that tenor and we do NOT tune the threshold into the overlap.")


if __name__ == "__main__":
    main()
