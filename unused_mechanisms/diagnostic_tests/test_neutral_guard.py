"""
DISTANCE-TO-NEUTRAL guard on the ratio exit — David's rule, tested honestly.

THE RULE (David's spec):
  1. At CYCLE START (first hike/entry), classify the regime by live fed funds vs neutral:
       normalization  = fed funds is meaningfully BELOW neutral  -> ARM the guard
       restrictive    = fed funds at/above neutral               -> guard OFF (original exit)
  2. In an ARMED (normalization) cycle, EXIT requires BOTH:
       (a) NEUTRAL condition: fed funds has essentially ARRIVED at neutral, i.e.
           fed_funds >= nominal_neutral - NEUTRAL_TOL_BP  (no more than ~15bp below), AND
       (b) the ORIGINAL ratio exit (the existing gate).
     -> the guard can only ever DELAY an exit, never cause one, so it cannot create new
        false exits; it can only fix the too-EARLY ones (2005/2017).
  3. Non-normalization cycle -> guard never arms -> original exit unchanged.

Neutral is NOMINAL: nominal_neutral = r*(LW real-time, one-sided) + expected inflation.
  - r*: Laubach-Williams REAL-TIME one-sided estimate, per-vintage (NO hindsight). Covers
        2005q1+ -> 2004-06 back half, 2015-18, 2022-23. Pre-2005 (1994, early-2004) has no
        real-time r*; per the theory those are restrictive/unclassifiable and the guard would
        be off there anyway, so the missing coverage is not a gap in what we test.
  - expected inflation: FRED EXPINF10YR (Cleveland, monthly, 1990+).

WHAT WE CHECK:
  - regime classification at each covered cycle's start (does it correctly call 2004-06 &
    2015-18 normalization, 2022-23 restrictive?)
  - at the ratio false-alarms (2005-09, 2017-06): does the NEUTRAL condition (b) BLOCK the
    exit (fed funds still well below neutral)?  -> the guard fixes the early exit
  - at the true tops (2006-06, 2018-12): does the neutral condition CLEAR (fed funds ~at
    neutral) so the AND-gate can fire?  -> the guard doesn't hold us too long
  - 2022-23: guard OFF (restrictive), original exit governs.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import urllib.request
import ssl
import io

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "hiking_curve_strategy"))
import _paths  # noqa: F401
from signal_logic import (
    _load_signal_data, FED_TARGET_MOVE_FLOOR,
    EXIT_SMOOTH_WINDOW_DAYS, RATIO_EXIT_THRESHOLD, RATIO_EXIT_FLOOR_BP,
)

NEUTRAL_TOL_BP    = 15    # exit allowed once fed funds >= neutral - 15bp (essentially arrived)
NORMALIZATION_BP  = 50    # cycle-start classed 'normalization' if fed funds < neutral - 50bp
LW_URL = ("https://www.newyorkfed.org/medialibrary/media/research/economists/"
          "williams/data/Laubach_Williams_real_time_estimates.xlsx")

CYCLES = {  # covered by real-time r* (>=2005q1)
    "2004-06": {"start": "2004-06-30", "false_alarm": "2005-09-02", "true_top": "2006-06-29"},
    "2015-18": {"start": "2015-12-16", "false_alarm": "2017-06-21", "true_top": "2018-12-20"},
    "2022-23": {"start": "2022-03-16", "false_alarm": "2022-12-20", "true_top": "2023-07-27"},
}


def _load_rstar_realtime():
    req = urllib.request.Request(LW_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context()).read()
    xl = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
    rows = []
    for sh in [s for s in xl.sheet_names if s and s[0] == "2" and "q" in s.lower()]:
        df = xl.parse(sh, header=5)
        df = df[df["Date"].notna()].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"].notna()]
        onesided = [c for c in df.columns if str(c).strip() == "rstar"][0]  # first = one-sided
        last = df.sort_values("Date").iloc[-1]
        rows.append({"asof": last["Date"], "rstar": float(last[onesided])})
    return pd.DataFrame(rows).set_index("asof").sort_index()["rstar"]


def _fetch_fred(api_key, series, index):
    from fredapi import Fred
    s = Fred(api_key=api_key).get_series(series, observation_start="1990-01-01")
    s.index = pd.to_datetime(s.index)
    return s.reindex(index, method="ffill").astype(float)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    ds, fed_target = _load_signal_data(api_key, start="1976-01-01")
    dff = ds["dff"]  # percent

    print("Downloading Laubach-Williams real-time r* (per-vintage, no hindsight)...")
    rstar = _load_rstar_realtime()                       # quarterly, percent, as-of dates
    rstar_d = rstar.reindex(ds.index, method="ffill")    # forward-fill to daily (as-of knowable)
    infl_exp = _fetch_fred(api_key, "EXPINF10YR", ds.index)
    nominal_neutral = rstar_d + infl_exp                 # nominal neutral fed funds

    # gap = neutral - fed funds (positive = below neutral, more to hike)
    gap = nominal_neutral - dff

    print(f"r* real-time coverage: {rstar.index[0].date()}..{rstar.index[-1].date()}")
    print(f"\n{'':<10}{'--- CYCLE START ---':^34}{'--- FALSE ALARM ---':^30}{'--- TRUE TOP ---':^30}")
    print(f"{'cycle':<10}{'ff':>7}{'neutral':>9}{'gap':>7}{'regime':>11}"
          f"{'ff':>7}{'neutral':>8}{'gap':>7}{'blocks?':>8}"
          f"{'ff':>7}{'neutral':>8}{'gap':>7}{'clears?':>8}")

    def at(d):
        d = pd.Timestamp(d)
        i = dff.index[dff.index <= d][-1]
        return dff[i], nominal_neutral[i], gap[i]

    tol = NEUTRAL_TOL_BP / 100.0        # bp -> percent
    norm = NORMALIZATION_BP / 100.0
    for lbl, c in CYCLES.items():
        ff0, nu0, g0 = at(c["start"])
        regime = "NORMALIZ" if g0 > norm else "restrict"
        ffa, nua, ga = at(c["false_alarm"])
        # neutral condition (b): exit allowed once ff >= neutral - tol, i.e. gap <= tol
        blocks = "BLOCK" if ga > tol else "allow"     # BLOCK the false exit = good
        fft, nut, gt = at(c["true_top"])
        clears = "CLEAR" if gt <= tol else "hold"     # CLEAR at the top = good
        # if restrictive regime, guard is OFF -> both columns become 'off'
        if regime != "NORMALIZ":
            blocks = clears = "off"
        print(f"{lbl:<10}{ff0:>7.2f}{nu0:>9.2f}{g0:>+7.2f}{regime:>11}"
              f"{ffa:>7.2f}{nua:>8.2f}{ga:>+7.2f}{blocks:>8}"
              f"{fft:>7.2f}{nut:>8.2f}{gt:>+7.2f}{clears:>8}")

    print(f"\n  guard params: normalization if start gap > {NORMALIZATION_BP}bp;  "
          f"exit-neutral-clear if gap <= {NEUTRAL_TOL_BP}bp")
    print("\n  WANT, per normalization cycle:")
    print("   - regime = NORMALIZ  (2004-06, 2015-18)")
    print("   - FALSE ALARM -> BLOCK  (fed funds still well below neutral -> guard vetoes early exit)")
    print("   - TRUE TOP    -> CLEAR  (fed funds ~at neutral -> AND-gate can fire, not held too long)")
    print("   - 2022-23 -> regime restrict, guard OFF (original exit governs; distance-to-neutral")
    print("     is the wrong question for a cycle that intends to go ABOVE neutral)")


if __name__ == "__main__":
    main()
