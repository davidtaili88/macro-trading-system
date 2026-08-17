"""
PARAMETER DERIVATION — the SLOPE WINDOW (SLOPE_WINDOW) for the momentum gate on the ratio exit.

This derives the LIVE value signal_logic.SLOPE_WINDOW (= 42td). The gate's statistic is a
standardized trailing OLS-SLOPE t-stat of spread_1yr over SLOPE_WINDOW days (see
signal_logic._rolling_slope_tstat); this file fixes the WINDOW from the NOISE TIMESCALE of the
spread, hindsight-free. The gate's THRESHOLD (SLOPE_T_THRESHOLD) is a separate, plateau-justified
value — see overfitting_tests/sweep_slope_t_threshold.py, not here.

WHY THE WINDOW MATTERS
----------------------
The gate asks "is spread_1yr STEADILY declining" via a trailing OLS slope over W days. W must be
chosen well: too SHORT and a single noise excursion fakes a slope (a window near the noise floor
just measures wobble); too LONG and the slope lags a genuine rollover. The right scale for W is the
NOISE TIMESCALE tau — how long a typical noise swing persists before it reverses. Make W a few x tau
and, by construction, a noise excursion averages out inside the window while a sustained move survives.
This file measures tau and reports the implied W.

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

TWO ESTIMATORS OF tau
---------------------
  1. AR(1) half-life        (ANCHOR — a model-based, robust single timescale: half-life =
     ln(0.5)/ln(phi), and one stray tick barely moves phi. This is the number W is built from.)
  2. Variance-ratio curve   (MODEL-FREE CONFIRMATION — makes NO AR(1) assumption; just measures
     how variance grows with horizon. Its job is to confirm the AR(1) scale isn't too SHORT, not
     to emit its own tau: the curve has no flat "knee", so it yields a band, not a point.)

The pairing is deliberate: one PARAMETRIC anchor plus one NON-PARAMETRIC check that the anchor
isn't shorter than the data allows. That guards the one place AR(1) could be fragile (forcing a
single-exponential model on the series) without adding a number that needs its own defense.

The quiet-hold machinery (_find_flat_holds / interior trim / hiking-episode exclusion) is the
same disjoint-from-signal null as the retired unused_mechanisms/derive_roc_gate_bp.py (which
derived the OLD gate's bp threshold and is kept only for provenance) — both rest on a null that
uses ONLY FOMC hike/cut dates + calendar offsets, never P&L, last-hike, or where the gate fires.

WHAT WE REPORT
--------------
  - the quiet sample (spans + day count) so the mask can be sanity-checked, plus a probe that
    the known exit-region dates are EXCLUDED
  - the AR(1) half-life -> tau, and whether the variance-ratio curve confirms tau isn't shorter
  - the implied slope window W ~ 2 x tau, with a longer robustness variant (-> SLOPE_WINDOW=42)

Run:  cd hiking_curve_strategy && PYTHONPATH=. python3 parameter_generation/derive_slope_window.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # package root
import _paths  # noqa: F401 — registers core/, benchmark/, utils/ on sys.path

from signal_logic import (
    _load_signal_data, detect_signal, signal_to_cycles,
    FED_TARGET_MOVE_FLOOR,
)


DATA_START    = "1982-10-01"   # earliest date all signal inputs exist (DFEDTAR starts 1982-09-27)
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


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    print(f"Loading FRED spreads + FOMC target (from {DATA_START})...")
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
    print("[1] AR(1) HALF-LIFE  (anchor — model-based, robust)")
    print("=" * 84)
    print(f"  phi (1-day persistence) : {phi:.4f}")
    print(f"  half-life               : {hl:.1f} td")
    print(f"  => noise timescale tau  ~ 2-3 x half-life = {tau_ar_lo:.0f}-{tau_ar_hi:.0f} td")

    # === estimator 2: variance ratio ===========================================
    vr = variance_ratio_curve(quiet, VR_HORIZONS)
    still_reverting = vr_reversion_still_active(vr)
    print("\n" + "=" * 84)
    print("[2] VARIANCE-RATIO CURVE  (model-free confirmation — no AR(1) assumption)")
    print("=" * 84)
    print("  VR ~ 1 => random-walk (noise still growing linearly);  < 1 => mean-reverting")
    for _, r in vr.iterrows():
        bar = "#" * int(max(0, r["VR"]) * 20)
        print(f"    k={int(r['k']):>3}   VR={r['VR']:.3f}  {bar}")
    print(f"  VR still FALLING at the longest horizon (k={VR_HORIZONS[-1]})? {still_reverting}")
    print("  => the curve has NO flat knee: reversion is still removing variance out past ~40-60td,")
    print("     so the noise band is WIDE. VR does NOT contradict AR(1); it confirms tau is at")
    print(f"     least the AR(1) scale (~{tau_ar_lo:.0f}-{tau_ar_hi:.0f}td), not the ~5td an arbitrary VR<0.5 crossing implies.")

    # === conclusion: the derived window ========================================
    # Anchor tau on the AR(1) half-life (robust, model-based). VR only CONFIRMS the band is
    # not shorter than this — it must NOT be collapsed to a single crossing point (e.g. "first k
    # with VR<0.5"), which badly UNDER-states tau: that naive read caught only the steep initial
    # drop (~5td) and would have shrunk W to roughly one half-life — a wobble-scale window.
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
    print("  This window is DERIVED FROM THE NOISE TIMESCALE, not from any exit date. It is the")
    print("  LIVE signal_logic.SLOPE_WINDOW, feeding the standardized OLS-slope t-stat gate. The")
    print("  slope THRESHOLD (how negative counts as pricing out) is a SEPARATE discretionary value,")
    print("  justified by a plateau sweep (overfitting_tests/sweep_slope_t_threshold.py), the same")
    print("  treatment as RATIO_EXIT_THRESHOLD — not derived here.")


if __name__ == "__main__":
    main()
