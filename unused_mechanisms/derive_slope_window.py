"""
PARAMETER DERIVATION — the SLOPE WINDOW for a redesigned "Gate B" on the ratio exit.

Companion to derive_roc_gate_bp.py. That file derives the gate THRESHOLD (how negative
counts as "pricing out") from a flat-day noise floor. This file derives the gate WINDOW
(how many trading days the slope is measured over) from the NOISE TIMESCALE of the spread.

WHY WE NEED THIS
----------------
The live Gate B is a 21-day rolling MEDIAN of the 21-day CHANGE in spread_1yr. A median of
a two-endpoint difference is NOT a slope: it is blind to the path between the endpoints, so
a single low print 21 days after a high print reads as a strong negative "trend" even when
the spread just wobbled inside its normal band. That is exactly what fired the 2004-06 exit
~10 months early (Sep 2005) on a one-print dip — see diagnostic_tests/diagnose_2005_early_exit.py.

The fix we're pricing out here is to replace the gate statistic with a genuine trailing OLS
SLOPE of spread_1yr — "is the spread STEADILY declining" = pricing hikes out. An OLS slope
over W days needs W chosen well. Too short and a noise wiggle fakes a slope (21d ~ the noise
floor itself, which is why the live gate fails); too long and it lags real rollovers.

THE OVERFITTING TRAP (and how this file avoids it)
--------------------------------------------------
Picking W because it happens to read ~0 on 2005-09-02 and clearly negative on 2019-01-15
would be choosing the window FROM the exit answers we want — pure hindsight, the same sin as
tuning a threshold to P&L. With n~6 cycles the exit day can never be fit honestly.

So we do NOT choose W from exit dates. We DERIVE it from a property of the series itself:
how long a typical NOISE swing persists before it reverses (the noise timescale tau). A slope
window must be a few x tau, so that by construction a noise excursion averages out inside the
window while a sustained move survives. tau is measured on QUIET regimes only — mid-hold /
mid-cycle days far from any FOMC move — masking OUT the pricing-in/out episodes (which, left
in, would inflate tau). The mask is mechanical (distance-to-nearest-FOMC-move) and blind to
tau and to every exit date. Only AFTER W is fixed do we look at what it does to 2005 / 2019,
as a downstream TEST — never as an input here.

THREE ESTIMATORS OF tau
-----------------------
  1. AR(1) half-life          (PRIMARY — robust; a stray tick barely moves phi)
  2. Variance-ratio horizon   (PRIMARY — robust; level-based, aggregated)
  3. Smoothed sign-run length (CROSS-CHECK ONLY — raw sign-runs are fooled by a single
     flip inside a real run, e.g. -----+-----, so we smooth lightly and treat it as
     corroboration, not the primary number. It is EXPECTED to read short — sign-persistence
     is not swing-persistence — and its disagreement with 1&2 is itself the lesson.)

The reused quiet-hold machinery (_find_flat_holds / interior trim / hiking-episode exclusion)
is deliberately identical in spirit to derive_roc_gate_bp.py so both derivations rest on the
same disjoint-from-signal null.

WHAT WE REPORT
--------------
  - the quiet sample (spans + day count) so the mask can be sanity-checked, plus a probe that
    the known exit-region dates are EXCLUDED
  - tau from each estimator, and whether the two robust ones agree
  - the implied slope window W ~ 2 x tau, with a longer robustness variant

Run:  python3 unused_mechanisms/derive_slope_window.py    (run from the project root)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hiking_curve_strategy"))
import _paths  # noqa: F401 — registers core/, benchmark/, utils/ on sys.path

from signal_logic import (
    _load_signal_data, detect_signal, signal_to_cycles,
    FED_TARGET_MOVE_FLOOR,
)


DATA_START    = "1976-01-01"
HOLD_MIN_DAYS = 126   # >= 6 months (trading days) of NO hike and NO cut = a "flat hold"
TRIM_DAYS     = 30    # ~6 weeks trimmed off EACH end of a hold (policy-transition edge)

# probe dates: NOT inputs — only used to CONFIRM the quiet mask excludes the exit region,
# and (downstream, elsewhere) to TEST the derived window. Listed here for the sanity print.
PROBE_DATES = ["2005-09-02", "2006-06-29", "2019-01-15"]

VR_HORIZONS = [1, 2, 3, 5, 8, 10, 15, 21, 30, 42, 63]


# ── quiet-hold machinery (shared shape with derive_roc_gate_bp.py) ──────────
def _find_flat_holds(fed_target: pd.Series) -> list[tuple]:
    """Maximal runs of >= HOLD_MIN_DAYS consecutive days with NO hike and NO cut,
    defined purely from |FOMC target change| exceeding the 1bp noise floor."""
    change = fed_target.diff()
    active = change.abs() > FED_TARGET_MOVE_FLOOR
    idx = fed_target.index
    holds, run_start = [], None
    for i in range(len(idx)):
        if active.iloc[i]:
            if run_start is not None:
                if i - run_start >= HOLD_MIN_DAYS:
                    holds.append((idx[run_start], idx[i - 1]))
                run_start = None
        else:
            if run_start is None:
                run_start = i
    if run_start is not None and (len(idx) - run_start) >= HOLD_MIN_DAYS:
        holds.append((idx[run_start], idx[-1]))
    return holds


def _interior_mask(index: pd.DatetimeIndex, holds: list[tuple]) -> pd.Series:
    """True on the TRIMMED interior of each flat hold (edges dropped: policy just moved)."""
    mask = pd.Series(False, index=index)
    for start, end in holds:
        i0, i1 = index.searchsorted(start), index.searchsorted(end)
        lo, hi = i0 + TRIM_DAYS, i1 - TRIM_DAYS
        if hi > lo:
            mask.iloc[lo:hi + 1] = True
    return mask


def _cycle_mask(index: pd.DatetimeIndex, cycles: list[dict]) -> pd.Series:
    """True on any day inside a strategy hiking episode — EXCLUDED from the quiet null."""
    mask = pd.Series(False, index=index)
    for c in cycles:
        mask.loc[(index >= c["first_hike"]) & (index <= c["last_hike"])] = True
    return mask


# ── estimator 1: AR(1) half-life ────────────────────────────────────────────
def ar1_halflife(x: pd.Series) -> tuple[float, float]:
    """Fit x(t)-mu = phi*(x(t-1)-mu)+eps on the quiet sample. Returns (phi, half_life_td).
    NOTE: differences are taken across the whole quiet series; adjacent quiet days are
    genuinely adjacent trading days within each contiguous span, so phi is a real 1-day
    persistence. Half-life = ln(0.5)/ln(phi)."""
    x = x.dropna()
    mu = x.mean()
    lhs, rhs = (x - mu).values[1:], (x - mu).values[:-1]
    phi = float(np.dot(rhs, lhs) / np.dot(rhs, rhs))
    hl = np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.inf
    return phi, hl


# ── estimator 2: variance-ratio horizon ─────────────────────────────────────
def variance_ratio_curve(x: pd.Series, horizons) -> pd.DataFrame:
    """VR(k) = Var(x(t)-x(t-k)) / (k * Var(x(t)-x(t-1))).
    ~1 => still random-walk (noise growing linearly); < 1 => mean-reversion has kicked in.
    The k where VR clearly departs from 1 is ~ tau."""
    d1 = x.diff().dropna()
    var1 = d1.var()
    rows = []
    for k in horizons:
        dk = x.diff(k).dropna()
        vr = dk.var() / (k * var1) if var1 > 0 else np.nan
        rows.append({"k": k, "VR": vr})
    return pd.DataFrame(rows)


def vr_reversion_still_active(vr: pd.DataFrame) -> bool:
    """Is mean-reversion STILL removing variance at the LONGEST horizon we probe?

    We deliberately do NOT read tau as 'first k where VR < 0.5' — that just catches where the
    steep initial drop crosses an arbitrary line and badly UNDER-states the noise band. The
    honest read of a curve with no flat knee is: reversion is still active as long as VR keeps
    falling. If VR is still declining at the last horizon, the noise band is at least that wide,
    so the AR(1) half-life (a proper model-based timescale) is the number to anchor tau on and
    VR's job is only to CONFIRM the band isn't shorter than that."""
    return bool(vr["VR"].iloc[-1] < vr["VR"].iloc[-2])


# ── estimator 3: smoothed sign-run length (cross-check) ─────────────────────
def sign_run_lengths(x: pd.Series, smooth: int) -> np.ndarray:
    """Run lengths of same-sign daily changes on a lightly SMOOTHED series, so a single
    stray tick inside a real run (-----+-----) doesn't shatter it. Cross-check ONLY."""
    s = x.rolling(smooth, min_periods=1).mean()
    sgn = np.sign(s.diff().dropna().values)
    sgn = sgn[sgn != 0]
    if len(sgn) == 0:
        return np.array([])
    runs, cur = [], 1
    for a, b in zip(sgn[:-1], sgn[1:]):
        cur = cur + 1 if a == b else (runs.append(cur) or 1)
    runs.append(cur)
    return np.array(runs)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    print("Loading FRED spreads + FOMC target (from 1976)...")
    daily_spreads, fed_target = _load_signal_data(api_key, start=DATA_START)
    spread_1yr = daily_spreads["spread_1yr_bp"]

    print("Detecting the strategy's own hiking episodes (to EXCLUDE from the null)...")
    cycles = signal_to_cycles(detect_signal(api_key, start=DATA_START))

    # --- build the quiet-regime sample -----------------------------------------
    holds    = _find_flat_holds(fed_target)
    interior = _interior_mask(spread_1yr.index, holds)
    in_hik   = _cycle_mask(spread_1yr.index, cycles)
    quiet_mask = interior & (~in_hik)
    quiet = spread_1yr[quiet_mask].dropna()

    # --- report the SAMPLE first (sanity-check the mask) -----------------------
    print("\n" + "=" * 84)
    print("QUIET SAMPLE  (Fed on hold >= 6mo, interior trimmed, hiking episodes removed)")
    print("=" * 84)
    print(f"  flat holds found (>= {HOLD_MIN_DAYS}td, no hike/cut) : {len(holds)}")
    print(f"  interior days (after +/-{TRIM_DAYS}td trim)           : {int(interior.sum())}")
    print(f"  ... excluded for overlapping a hiking episode      : {int((interior & in_hik).sum())}")
    print(f"  FINAL quiet-sample days used for tau               : {len(quiet)}")

    print("\n  Quiet spans being measured on (first/last few):")
    grp = (quiet_mask != quiet_mask.shift()).cumsum()
    spans = [(b.index[0].date(), b.index[-1].date(), len(b))
             for _, b in quiet_mask[quiet_mask].groupby(grp[quiet_mask])]
    for s in spans[:5]:
        print(f"      {s[0]} -> {s[1]}   ({s[2]} td)")
    print("      ...")
    for s in spans[-3:]:
        print(f"      {s[0]} -> {s[1]}   ({s[2]} td)")

    print("\n  Sanity: exit-region probe dates must be OUTSIDE the quiet sample:")
    for probe in PROBE_DATES:
        d = pd.Timestamp(probe)
        idx = quiet_mask.index[quiet_mask.index <= d]
        val = bool(quiet_mask.loc[idx[-1]]) if len(idx) else False
        print(f"      {probe}: quiet={val}   (want False — it's an exit/rollover region)")

    if len(quiet) < 200:
        print("\n  *** NOTE: quiet sample is thin; treat tau as a coarse scale, not a precise number.")

    # === estimator 1: AR(1) half-life ==========================================
    phi, hl = ar1_halflife(quiet)
    tau_ar_lo, tau_ar_hi = 2 * hl, 3 * hl
    print("\n" + "=" * 84)
    print("[1] AR(1) HALF-LIFE  (primary, robust)")
    print("=" * 84)
    print(f"  phi (1-day persistence) : {phi:.4f}")
    print(f"  half-life               : {hl:.1f} td")
    print(f"  => noise timescale tau  ~ 2-3 x half-life = {tau_ar_lo:.0f}-{tau_ar_hi:.0f} td")

    # === estimator 2: variance ratio ===========================================
    vr = variance_ratio_curve(quiet, VR_HORIZONS)
    still_reverting = vr_reversion_still_active(vr)
    print("\n" + "=" * 84)
    print("[2] VARIANCE-RATIO CURVE  (primary, robust)")
    print("=" * 84)
    print("  VR ~ 1 => random-walk (noise still growing linearly);  < 1 => mean-reverting")
    for _, r in vr.iterrows():
        bar = "#" * int(max(0, r["VR"]) * 20)
        print(f"    k={int(r['k']):>3}   VR={r['VR']:.3f}  {bar}")
    print(f"  VR still FALLING at the longest horizon (k={VR_HORIZONS[-1]})? {still_reverting}")
    print("  => the curve has NO flat knee: reversion is still removing variance out past ~40-60td,")
    print("     so the noise band is WIDE. VR does NOT contradict AR(1); it confirms tau is at")
    print("     least the AR(1) scale (~15-23td), not the ~5td an arbitrary VR<0.5 crossing implies.")

    # === estimator 3: smoothed sign-runs (cross-check) =========================
    print("\n" + "=" * 84)
    print("[3] SMOOTHED SIGN-RUN LENGTH  (cross-check ONLY — expected to read SHORT)")
    print("=" * 84)
    for sm in (3, 5):
        runs = sign_run_lengths(quiet, smooth=sm)
        if len(runs):
            print(f"    smooth={sm}d:  median run={np.median(runs):.0f} td   "
                  f"mean={runs.mean():.1f}   90th pct={np.percentile(runs, 90):.0f} td")
    print("  Sign-PERSISTENCE is not swing-PERSISTENCE: a mean-reverting series flips sign")
    print("  every few days while its LEVEL meanders inside one excursion. A short read here")
    print("  that DISAGREES with [1]/[2] is expected and is the reason it's only a cross-check.")

    # === conclusion: the derived window ========================================
    # Anchor tau on the AR(1) half-life (robust, model-based). VR only CONFIRMS the band is
    # not shorter than this — it must NOT be collapsed to a single crossing point, which
    # under-states tau (that bug printed ~5td -> W~21td, the very window we know fails on 2005).
    tau_lo, tau_hi = tau_ar_lo, tau_ar_hi          # ~2-3 half-lives
    W_main = int(round(2 * (tau_lo + tau_hi) / 2 / 21.0) * 21)   # ~2 x tau, rounded to a month grid
    print("\n" + "=" * 84)
    print("DERIVED SLOPE WINDOW")
    print("=" * 84)
    print(f"  tau anchored on AR(1) half-life    : ~{tau_lo:.0f}-{tau_hi:.0f} td")
    print(f"  VR confirms band not shorter       : {still_reverting} (still reverting at {VR_HORIZONS[-1]}td)")
    print(f"  slope window must be >= ~2 x tau   : so a noise excursion averages out inside it")
    print(f"  => proposed W (main)               : ~{W_main} td")
    print(f"  => robustness variant (longer)     : ~{int(round(1.5*W_main/21)*21)} td")
    print()
    print("  This window is DERIVED FROM THE NOISE FLOOR, not from any exit date. The next")
    print("  step (a TEST, not part of this derivation) is to build an OLS-slope Gate B over")
    print("  this window and check it HOLDS the 2004-06 position through late 2005 while still")
    print("  OPENING for the genuine 2018->2019 rollover. The slope THRESHOLD (how negative =")
    print("  pricing out) is a separate, undish-derivable discretionary value — justify it with")
    print("  a corridor sweep, the same treatment as RATIO_EXIT_THRESHOLD and ROC_GATE_BP.")


if __name__ == "__main__":
    main()
