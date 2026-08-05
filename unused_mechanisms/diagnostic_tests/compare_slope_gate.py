"""
Does a MEDIAN (Theil-Sen) SLOPE over the derived window beat the current two-point Gate B
ACROSS ALL HISTORY — not just on 2005/2019? A fix that only works on the bug window is just
overfitting by another name, so we score every candidate on the WHOLE record.

Three candidate gate statistics on spread_1yr (all reported bp/month for comparability):

  A. CURRENT Gate B   : rolling_median_W( spread(t) - spread(t - L) )   [two-point diff]
                        L,W = live ROC_GATE_DIFF_LAG (21), ROC_GATE_MEDIAN_WINDOW (21).
  B. OLS slope        : OLS slope of spread over trailing W days.
  C. THEIL-SEN slope  : MEDIAN of all pairwise slopes over trailing W days (robust).

W = 42 (derived in derive_slope_window.py from the noise floor, NOT from any exit date);
W = 63 reported as a longer-window robustness variant.

WHAT "THE FIT HOLDS" MEANS, EVERYWHERE
--------------------------------------
A gate on the ratio exit has two failure modes, and we score BOTH across all history:

  (1) FALSE FIRE on quiet holds — the gate says "pricing out" when the Fed is just on hold
      and the spread is wobbling. This is the 2005 bug generalized. We take every quiet-hold
      interior day (same mask as derive_slope_window.py) and report, per statistic, the
      fraction of days that breach the current open-threshold (< ROC_GATE_BP). LOWER = better
      (a good slope statistic almost never calls flat chop a rollover).

  (2) MISS a real rollover — the gate stays flat into a genuine end-of-cycle pricing-out.
      For every historical hiking cycle we report each statistic's reading in the ~3 months
      BEFORE the true last hike's rollover. We WANT it clearly negative there.

The point is the trade-off: the current gate's diff is noisy, so to avoid false fires it needs
a lenient threshold, which makes it miss/lag real rollovers (and yet it STILL false-fired in
2005). A robust slope should push the false-fire rate down WITHOUT giving up the real signal.
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
    ROC_GATE_DIFF_LAG, ROC_GATE_MEDIAN_WINDOW, ROC_GATE_BP,
)

W_MAIN    = 42
W_ROBUST  = 63
TD_PER_MO = 21
HOLD_MIN_DAYS = 126
TRIM_DAYS     = 30

# true last hikes of each historical hiking cycle (FOMC record) — for the rollover test.
# These are the actual last-hike dates, used ONLY to locate the rollover window to score;
# the gate never sees them.
TRUE_LAST_HIKES = {
    "1994-95": "1995-02-01",
    "2004-06": "2006-06-29",
    "2015-18": "2018-12-20",
    "2022-23": "2023-07-27",
    "1999-00": "2000-05-16",
}


def current_gate_b(spread):
    return spread.diff(ROC_GATE_DIFF_LAG).rolling(ROC_GATE_MEDIAN_WINDOW, min_periods=5).median()


def ols_slope(spread, W):
    t = np.arange(W, dtype=float); td = t - t.mean(); denom = np.dot(td, td)
    def _s(y):
        if np.isnan(y).any(): return np.nan
        return float(np.dot(td, y - y.mean()) / denom) * TD_PER_MO
    return spread.rolling(W).apply(_s, raw=True)


def theil_sen_slope(spread, W):
    ii, jj = np.triu_indices(W, k=1); dt = (jj - ii).astype(float)
    def _s(y):
        if np.isnan(y).any(): return np.nan
        return float(np.median((y[jj] - y[ii]) / dt)) * TD_PER_MO
    return spread.rolling(W).apply(_s, raw=True)


# --- quiet-hold mask (identical to derive_slope_window.py) -------------------
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


def _cycle_mask(index, cycles):
    m = pd.Series(False, index=index)
    for c in cycles:
        m.loc[(index >= c["first_hike"]) & (index <= c["last_hike"])] = True
    return m


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    ds, fed_target = _load_signal_data(api_key, start="1976-01-01")
    sp = ds["spread_1yr_bp"]
    cycles = signal_to_cycles(detect_signal(api_key, start="1976-01-01"))

    stats = {
        "A_gateB(diff21)":       current_gate_b(sp),
        f"B_ols_W{W_MAIN}":      ols_slope(sp, W_MAIN),
        f"C_theilsen_W{W_MAIN}": theil_sen_slope(sp, W_MAIN),
        f"C_theilsen_W{W_ROBUST}": theil_sen_slope(sp, W_ROBUST),
    }
    names = list(stats.keys())

    # === (1) FALSE-FIRE RATE ON QUIET HOLDS (all history) =====================
    holds = _find_flat_holds(fed_target)
    interior = _interior_mask(sp.index, holds)
    in_hik = _cycle_mask(sp.index, cycles)
    quiet = interior & (~in_hik)

    print("=" * 92)
    print(f"(1) FALSE-FIRE on QUIET HOLDS — fraction of quiet days breaching the open-thresh (< {ROC_GATE_BP} bp/mo)")
    print(f"    quiet interior days scored: {int(quiet.sum())}   (a GOOD gate ~never fires here — lower is better)")
    print("=" * 92)
    print(f"{'statistic':<22}{'false-fire %':>14}{'median|read|':>14}{'5th pctile':>14}{'min':>10}")
    for n in names:
        s = stats[n][quiet].dropna()
        ff = float((s < ROC_GATE_BP).mean()) * 100
        print(f"{n:<22}{ff:>13.1f}%{s.abs().median():>14.1f}{np.percentile(s,5):>14.1f}{s.min():>10.1f}")

    # === (2) REAL ROLLOVER — reading in the ~3mo before each true last hike ===
    print("\n" + "=" * 92)
    print("(2) REAL ROLLOVER — each statistic's MIN reading in the 63td window ENDING at the true last hike")
    print("    (a GOOD gate is clearly NEGATIVE here — it should catch the genuine pricing-out)")
    print("=" * 92)
    print(f"{'cycle':<12}{'true last hike':<16}" + "".join(f"{n:>22}" for n in names))
    for lbl, lh in TRUE_LAST_HIKES.items():
        lh = pd.Timestamp(lh)
        win = sp.index[(sp.index > lh - pd.Timedelta(days=130)) & (sp.index <= lh + pd.Timedelta(days=20))]
        line = f"{lbl:<12}{str(lh.date()):<16}"
        for n in names:
            v = stats[n].reindex(win).min()
            line += f"{'':>22}" if pd.isna(v) else f"{v:>22.1f}"
        print(line)

    # === (3) PER-CYCLE: where does each gate FIRST say 'pricing out' inside the cycle? ===
    print("\n" + "=" * 92)
    print(f"(3) PER-CYCLE FIRST BREACH (< {ROC_GATE_BP}) after entry — lag vs the true last hike")
    print("    negative lag = fires BEFORE last hike (early exit risk); we want it near/after the last hike")
    print("=" * 92)
    print(f"{'episode':<26}" + "".join(f"{n:>22}" for n in names))
    for c in cycles:
        e0, e1 = c["first_hike"], c["last_hike"]
        # match to a true last hike if this episode belongs to a known cycle
        lh = None
        for _, v in TRUE_LAST_HIKES.items():
            v = pd.Timestamp(v)
            if e0 <= v <= e1 + pd.Timedelta(days=400):
                lh = v; break
        win = (sp.index >= e0) & (sp.index <= e1)
        line = f"{c['label'][:25]:<26}"
        for n in names:
            s = stats[n][win]
            breach = s.index[s < ROC_GATE_BP]
            if not len(breach):
                line += f"{'never':>22}"
            else:
                d = breach[0]
                if lh is not None:
                    lag = np.busday_count(lh.date(), d.date())
                    line += f"{str(d.date())+f'({lag:+d}td)':>22}"
                else:
                    line += f"{str(d.date()):>22}"
        print(line)

    print("\nReading it:")
    print("  (1) lower false-fire % = the statistic rarely mistakes flat chop for a rollover.")
    print("  (2) more-negative = it still catches the real end-of-cycle pricing-out.")
    print("  (3) a gate that first breaches only NEAR/AFTER the true last hike (lag ~0 or +) is")
    print("      holding the position through the cycle; a big NEGATIVE lag = early-exit risk (2005 bug).")
    print("  The median (Theil-Sen) slope WINS iff it lowers (1) while keeping (2) negative and")
    print("  pushing (3)'s lags out of the deep-negative zone — ACROSS cycles, not just 2005/2019.")


if __name__ == "__main__":
    main()
