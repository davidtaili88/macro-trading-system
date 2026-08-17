"""
RETIRED (kept for provenance) — this derived the *bp* threshold of the OLD momentum gate,
a 21d median of the 21d two-point change in spread_1yr < ROC_GATE_BP. That gate was replaced
in signal_logic.py by a standardized trailing-OLS-slope t-stat (SLOPE_WINDOW / SLOPE_T_THRESHOLD),
whose WINDOW is derived by unused_mechanisms/derive_slope_window.py (AR(1) noise timescale) and
whose t-THRESHOLD is a plateau value (overfitting_tests/sweep_slope_t_threshold.py). The two-point-
diff estimator this file assumes (and the ROC_GATE_* constants it imports) no longer exist, so this
script will NOT run against current signal_logic — it is preserved only to document the noise-floor
methodology that motivated the replacement. See the SLOPE_WINDOW comment in signal_logic.py.

PARAMETER DERIVATION — ROC_GATE_BP (the median-ROC gate threshold on the ratio exit).

The gate opens (ratio exit allowed) when the gate statistic

    roc(t) = 21-day rolling MEDIAN of [ spread_1yr(t) - spread_1yr(t-21) ]
           = "the typical monthly change in the 1yr spread, over the last month"

falls below ROC_GATE_BP. Intuition: only allow the exit when the spread is GENUINELY
rolling over. For that to mean anything, the threshold must sit BELOW the noise floor of
roc when NOTHING is happening — otherwise the gate just fires on flat-market chop.

So we derive ROC_GATE_BP as a noise floor:

    ROC_GATE_BP ~ -k * sigma_flat        (k = 1 or 2)

where sigma_flat is the ROBUST spread (1.4826 * MAD) of the SAME gate statistic, measured
only on days when the Fed is on hold and the spread is just wobbling on no news.

WHY THIS ISN'T CHEATING (the null must be defined WITHOUT hindsight)
-------------------------------------------------------------------
The "flat / nothing happening" regime is defined purely from EXOGENOUS, real-time-knowable
facts — the FOMC target rate being unchanged — never from where the gate fires, from P&L,
or from the last hike of a hiking cycle (the very thing the gate detects). Specifically:

  1. A "hold" = a maximal run of >= HOLD_MIN_DAYS trading days with NO hike and NO cut in
     the FOMC target (fed_target.diff() ~ 0). Knowable the day it happens.
  2. We TRIM TRIM_DAYS off each end of every hold (policy just changed at the edges; the
     spread is still digesting the last move / starting to price the next), keeping only
     the calm interior — the user's "middle 3 months" idea, generalized.
  3. We EXCLUDE any hold day that overlaps a hiking episode the strategy already identified
     (signal_to_cycles), so the null can never contain a period the strategy trades. This
     is the user's explicit guard: the noise sample must be disjoint from the signal.

Every slice uses only FOMC hike/cut dates + calendar offsets. No last-hike, no rollover,
no P&L. That is what keeps the derived threshold honest.

WHAT WE REPORT
--------------
  - how many holds and how many total interior-days survive (so we can see whether
    sigma_flat rests on enough data to trust — if not, THAT is the honest finding)
  - sigma_flat = 1.4826 * MAD(roc on interior-hold days)
  - candidate thresholds -1*sigma_flat, -2*sigma_flat
  - where the live -3bp sits relative to them (in sigma units)

Run:  cd hiking_curve_strategy && PYTHONPATH=. python3 parameter_generation/derive_roc_gate_bp.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths  # noqa: F401 — registers core/, benchmark/, utils/ on sys.path

from signal_logic import (
    _load_signal_data, detect_signal, signal_to_cycles,
    FED_TARGET_MOVE_FLOOR, ROC_GATE_DIFF_LAG, ROC_GATE_MEDIAN_WINDOW, ROC_GATE_BP,
)


DATA_START   = "1982-10-01"   # earliest date all signal inputs exist (DFEDTAR starts 1982-09-27)
HOLD_MIN_DAYS = 126   # >= 6 months (trading days) of NO hike and NO cut = a "flat hold"
TRIM_DAYS     = 30    # ~6 weeks trimmed off EACH end of a hold (policy-transition edge)
MAD_TO_SIGMA  = 1.4826


def _gate_statistic(spread_1yr: pd.Series) -> pd.Series:
    """The EXACT ROC gate series used in detect_signal:
    21-day rolling median of the 21-day (monthly) change in the 1yr spread."""
    return spread_1yr.diff(ROC_GATE_DIFF_LAG) \
                     .rolling(ROC_GATE_MEDIAN_WINDOW, min_periods=5).median()


def _find_flat_holds(fed_target: pd.Series) -> list[tuple]:
    """Maximal runs of >= HOLD_MIN_DAYS consecutive days with NO hike and NO cut.

    Defined purely from the FOMC target: a day is 'active' if |target change| exceeds the
    1bp noise floor. A hold is a run with no active day. Returns [(start_ts, end_ts), ...].
    """
    change = fed_target.diff()
    active = change.abs() > FED_TARGET_MOVE_FLOOR   # True on a hike or cut day
    idx = fed_target.index

    holds = []
    run_start = None
    for i, ts in enumerate(idx):
        if active.iloc[i]:
            # policy moved: close any open run at the PREVIOUS day
            if run_start is not None:
                run_len = i - run_start
                if run_len >= HOLD_MIN_DAYS:
                    holds.append((idx[run_start], idx[i - 1]))
                run_start = None
        else:
            if run_start is None:
                run_start = i
    # tail run (hold that extends to the end of the data)
    if run_start is not None and (len(idx) - run_start) >= HOLD_MIN_DAYS:
        holds.append((idx[run_start], idx[-1]))
    return holds


def _interior_mask(index: pd.DatetimeIndex, holds: list[tuple]) -> pd.Series:
    """Boolean over `index`: True on the TRIMMED interior of each flat hold."""
    mask = pd.Series(False, index=index)
    for start, end in holds:
        # trim TRIM_DAYS trading days off each end using index positions
        i0 = index.searchsorted(start)
        i1 = index.searchsorted(end)
        lo, hi = i0 + TRIM_DAYS, i1 - TRIM_DAYS
        if hi > lo:
            mask.iloc[lo:hi + 1] = True
    return mask


def _cycle_mask(index: pd.DatetimeIndex, cycles: list[dict]) -> pd.Series:
    """Boolean over `index`: True on any day inside a strategy hiking episode
    [first_hike, last_hike]. These days are EXCLUDED from the flat-noise sample."""
    mask = pd.Series(False, index=index)
    for c in cycles:
        mask.loc[(index >= c["first_hike"]) & (index <= c["last_hike"])] = True
    return mask


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    print("Loading FRED spreads + FOMC target (from 1982)...")
    daily_spreads, fed_target = _load_signal_data(api_key, start=DATA_START)
    spread_1yr = daily_spreads["spread_1yr_bp"]

    print("Detecting the strategy's own hiking episodes (to EXCLUDE from the null)...")
    signal = detect_signal(api_key, start=DATA_START)
    cycles = signal_to_cycles(signal)

    # --- build the gate statistic and the flat-hold null ------------------------
    roc = _gate_statistic(spread_1yr)

    holds = _find_flat_holds(fed_target)
    interior = _interior_mask(spread_1yr.index, holds)
    in_hiking = _cycle_mask(spread_1yr.index, cycles)

    # the null sample: interior-of-flat-hold, NOT inside any hiking episode, roc defined
    null_mask = interior & (~in_hiking) & roc.notna()
    roc_flat = roc[null_mask].dropna()

    # --- report the SAMPLE first (does sigma_flat rest on enough data?) ----------
    print("\n" + "=" * 84)
    print("FLAT-HOLD NULL SAMPLE  (Fed on hold >= 6mo, interior trimmed, hiking episodes removed)")
    print("=" * 84)
    print(f"  flat holds found (>= {HOLD_MIN_DAYS}td, no hike/cut) : {len(holds)}")
    for start, end in holds:
        n_days = spread_1yr.index.searchsorted(end) - spread_1yr.index.searchsorted(start)
        print(f"      {start.date()} .. {end.date()}   ({n_days} td)")
    print(f"  interior days (after +/-{TRIM_DAYS}td trim)           : {int(interior.sum())}")
    print(f"  ... excluded for overlapping a hiking episode      : {int((interior & in_hiking).sum())}")
    print(f"  ... excluded for roc undefined (warm-up)           : "
          f"{int((interior & (~in_hiking) & roc.isna()).sum())}")
    print(f"  FINAL null-sample days used for sigma_flat          : {len(roc_flat)}")

    if len(roc_flat) < 30:
        print("\n  *** WARNING: fewer than ~30 effectively-usable days. And these are heavily")
        print("      autocorrelated (diff21 + rolling21 share ~20/21 of inputs), so the")
        print("      EFFECTIVE sample is far smaller still. Treat sigma_flat as a coarse")
        print("      scale, not a precise number. This thinness may itself be the finding.")

    # --- the noise floor ---------------------------------------------------------
    med   = float(np.median(roc_flat))
    mad   = float(np.median(np.abs(roc_flat - med)))
    sigma_flat = MAD_TO_SIGMA * mad

    print("\n" + "=" * 84)
    print("NOISE FLOOR OF THE GATE STATISTIC ON FLAT DAYS")
    print("=" * 84)
    print(f"  median roc on flat days   : {med:+.3f} bp   (should be ~0: no trend when flat)")
    print(f"  MAD                       : {mad:.3f} bp")
    print(f"  sigma_flat = 1.4826*MAD   : {sigma_flat:.3f} bp")
    print(f"  (for context) raw std     : {float(np.std(roc_flat)):.3f} bp   "
          f"[std is outlier-sensitive; sigma_flat is the robust one]")

    # --- candidate thresholds and where -3 lands ---------------------------------
    print("\n" + "=" * 84)
    print("CANDIDATE THRESHOLDS vs. THE LIVE VALUE")
    print("=" * 84)
    labels = {
        1: "one noise-unit of decline (loosest defensible floor)",
        2: "clearly beyond flat chop (balanced)",
        3: "deep in the tail (strictest; gate opens rarely)",
    }
    for k in (1, 2, 3):
        print(f"  -{k} * sigma_flat  = {-k * sigma_flat:+.2f} bp   ({labels[k]})")
    print()
    live_in_sigma = ROC_GATE_BP / sigma_flat if sigma_flat > 0 else float("nan")
    print(f"  LIVE  ROC_GATE_BP = {ROC_GATE_BP:+.1f} bp  =  {live_in_sigma:+.2f} * sigma_flat")
    print()
    print("  Reading it:")
    print("   -1 to -2 sigma  => the live value is a genuine noise floor: it fires only when")
    print("                      the spread declines faster than it ever does on flat days.")
    print("   ~0 sigma (tiny) => the gate is INSIDE the noise: it would fire on flat chop;")
    print("                      the value is too loose (or the gate is near-inert either way).")
    print("   beyond -3 sigma => the threshold is so deep the gate almost never opens; the")
    print("                      gate is effectively off.")
    print()
    print("NOTE: this derives a value from the FLAT-DAY noise of the gate's own statistic,")
    print("measured on FOMC-hold periods disjoint from every hiking episode. It fixes the")
    print("SCALE of a sensible threshold; it is not an out-of-sample validation of the gate.")


if __name__ == "__main__":
    main()
