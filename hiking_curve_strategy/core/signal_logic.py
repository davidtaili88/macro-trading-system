"""
Market-driven signal logic for the hiking cycle 2-year payer strategy.

Both entry AND exit are fully market-driven — no oracle FOMC dates used.
Instrument-agnostic: produces cycle windows (entry/exit dates) that the DGS2
run script (signal_trade_dgs.py) applies to its return series via
backtest.calc_strat_ret. The same windows also drive the side futures
tradability experiment.

Entry: first day BOTH spread conditions hold inside a confirmed post-easing hold:
  - 3mo spread (DGS3MO - DFF) > 12bp  → hike priced within ~3 months
  - 1yr spread (DGS1 - DFF)   > 50bp  → market broadly pricing a hiking cycle
  Both spreads are smoothed with a short rolling mean (ENTRY_SMOOTH_WINDOW_DAYS)
  before comparison — a single noisy dff print (thin holiday trading) can
  otherwise flip the entry latch on for weeks.

Exit: first day ANY of the following holds:
  - false-promise exit: no hike has occurred yet since entry, AND smoothed
    1yr spread has fallen back to <= FALSE_PROMISE_THRESHOLD_1YR_BP (well
    below the entry bar). Catches cycles where the market briefly priced a
    hike that never came (e.g. 1996: spread spiked to ~80bp on entry, no
    hike ever followed, spread collapsed within weeks) without misfiring on
    normal chop around the entry threshold mid-cycle — the re-arm bar is
    set well below THRESHOLD_1YR_BP specifically so ordinary oscillation in
    an intact cycle (which routinely dips below the entry level without the
    cycle being over) doesn't trip it. Disarmed permanently once a real hike
    lands, since at that point the cycle is confirmed and the crude level
    exit below takes over.
  - level exit: smoothed 1yr spread < THRESHOLD_1YR_EXIT AND smoothed 3mo spread
    < THRESHOLD_3MO_EXIT (both must hold) → market pricing net cuts across the
    curve. Both spreads are smoothed over EXIT_SMOOTH_WINDOW_DAYS, and the 3mo
    leg is capped at ±EXIT_CAP_BP before smoothing so single-day holiday/thin-
    market spikes can't trip it. Responds to sustained drift below the thresholds
    rather than requiring a run of unbroken consecutive days.
  This mirrors the book's observation that 2s rally strongly once the hiking
  cycle is priced out, and that the trough in bond returns (the best exit
  point) coincides with the market starting to price cuts — not with a
  specific FOMC date the trader couldn't have known in advance.

  NOTE: this fires well after the true last hike (empirically 44-111 trading
  days late vs. ORACLE_CYCLES — see check_exit.py). A hike-indexed exit
  (level/deceleration/drawdown-from-peak of spread_1yr, sampled at each FOMC
  hike) was tried and rejected: no single threshold works across cycles,
  since e.g. 2015-2018 sat at 47-63bp (a fresh cycle high two hikes prior)
  right through its true last hike, while 2004-2006 was already 80%+ off its
  peak by its true last hike. Revisit with a second signal (cumulative bp
  hiked, cycle-length prior, SEP/dot-plot data) rather than another
  single-spread-series threshold.

Signal latches: turns on at first threshold crossover, stays on until either
spread drops past its exit threshold. This prevents spurious re-triggers from
intraday spread oscillation around the entry level.

Data sources (all FRED, no subscription required):
  DFF        — daily effective fed funds rate
  DGS1       — 1yr constant-maturity Treasury yield
  DGS3MO     — 3mo constant-maturity Treasury yield
  DFEDTAR    — pre-2008 fed funds target rate
  DFEDTARL   — post-2008 fed funds target lower bound
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # package root, for utils/
from utils.fred_utils import fetch_fred_dataframe


def _rolling_slope_tstat(y: pd.Series, window: int) -> pd.Series:
    """t-statistic of the trailing OLS slope of `y`, per window of `window` days.

    For each trailing window we fit y ~ a + b*x on the centered day-index x, then
    standardize the slope by its own regression standard error:
        t = b / SE(b),   SE(b) = s / sqrt(Sxx)
    where s = sqrt( sum(resid^2) / (window-2) ) is the residual std (in-window noise)
    and Sxx = sum(x_centered^2). t is a one-sided test of H0: b = 0 (flat): it is how
    many standard errors the measured slope sits from flat. Unit-free and self-scaling.

    Computed in closed form from rolling sums (no per-window python loop), so it is
    exact and fast. Assumes iid residuals (the standard OLS SE); spread residuals are
    mildly autocorrelated, so SE is slightly optimistic and t runs a touch hot — a
    disclosed approximation, leaned against by the stricter -1.5 threshold. Returns NaN
    until `window` observations exist. See SLOPE_WINDOW / SLOPE_T_THRESHOLD below.
    """
    n = window
    x = np.arange(n, dtype=float)
    x -= x.mean()
    Sxx = float((x * x).sum())            # fixed window geometry (~n^3/12)
    dof = n - 2

    # rolling sums needed for slope + residual SS, all with a common window length n
    roll = y.rolling(n, min_periods=n)
    Sy   = roll.sum()
    Syy  = y.pow(2).rolling(n, min_periods=n).sum()
    # Sxy uses the SAME centered x on every window (x is fixed), so it's a weighted
    # rolling sum: sum(x_i * y_i) over the window. Do it via a dot with the kernel.
    Sxy = y.rolling(n, min_periods=n).apply(lambda v: float(np.dot(x, v)), raw=True)

    b = Sxy / Sxx                          # slope (Sx = 0 because x is centered)
    # residual sum of squares = Syy - Sy^2/n - b^2 * Sxx   (Sxy = b*Sxx)
    rss = Syy - Sy.pow(2) / n - b.pow(2) * Sxx
    s2 = rss / dof
    se = (s2 / Sxx).pow(0.5)
    t = b / se
    # se == 0 (a perfectly straight window) -> slope is exact; treat as ±inf by sign of b
    t = t.where(se != 0, other=np.sign(b) * np.inf)
    return t


# ── tuneable thresholds ────────────────────────────────────────────────────
HOLD_MONTHS        = 6    # months of no cuts before easing cycle is "done"
THRESHOLD_1YR_BP   = 50   # bp: market must price ≥ 2 hikes 1yr out
THRESHOLD_3MO_BP   = 12   # bp: entry — market must price hike within ~3 months

# entry spreads are smoothed with a short rolling mean before being compared to
# threshold: raw dff is the *effective* (traded) fed funds rate, not the target,
# and wobbles several bp day-to-day (worse around holidays/quarter-end thin
# trading) — a single noisy print can otherwise flip the entry latch on for
# weeks (e.g. 1996-08-13, 1997-12-24 in the DGS backtest were pure dff blips,
# not real hike pricing).
ENTRY_SMOOTH_WINDOW_DAYS = 5  # trading days: rolling mean window on entry spreads
EXIT_SMOOTH_WINDOW_DAYS  = 5  # trading days: rolling window on exit spreads (both 1yr and 3mo)
EXIT_SMOOTH_METHOD       = "mean"  

THRESHOLD_1YR_EXIT = -15  # bp: exit threshold for 1yr spread
THRESHOLD_3MO_EXIT = 0  # bp: exit threshold for 3mo spread

# false-promise exit: if no hike has landed yet since entry and smoothed spread_1yr
# falls back to this level, treat the entry as a failed conviction and exit early.
# Set well below THRESHOLD_1YR_BP (not equal to it) so normal chop around the entry
# level mid-cycle doesn't trip it — only a near-full reversal does. Uses spread_1yr
# only (the cleaner series); spread_3mo is not part of this check.
FALSE_PROMISE_THRESHOLD_1YR_BP = 25  # bp

# ── maturity-ratio exit ─────────────────────────────────────────────────────
# Exit when  smoothed_spread_1yr / cumulative_bp_hiked_since_CYCLE_START  <  RATIO_EXIT_THRESHOLD.
# Denominator is bp hiked since the HIKING CYCLE started (the day in_cycle first turns
# True), NOT since strategy entry/re-entry — see the cum_hikes_at_cycle_start anchoring in
# detect_signal, which keeps a pause-exit-then-re-arm counting from the cycle start so the
# denominator can't reset small on a second leg (the 2004-06 / 2015-18 late-exit bug).
# Numerator = hiking STILL priced; denominator = hiking ALREADY delivered. When
# what's-still-priced shrinks to a small fraction of what's-been-done, the cycle is
# mostly behind us and we leave — while the spread may still be POSITIVE, i.e. well
# before the old spread<-15bp level exit fires. This is what cuts the exit lag on
# the 1994-95 and 2015-18 cycles (final exit lag ~+28td / ~+5td vs the true last
# hike, down from ~80/110td). See diagnose_level_ratio.py / diagnose_reentry_cost.py.
#
# 0.10 is a DISCRETIONARY round value, NOT fitted to P&L — and it CANNOT be uniquely
# derived from this data (n~6 cycles): the cost of exiting too LATE is only definable
# relative to the last hike, which is unknowable in real time, so no honest hindsight-free
# method pins the value. What we CAN show (overfitting_tests/sweep_ratio_exit_threshold.py)
# is that 0.10 is a safe choice inside a measured corridor:
#   - the robust-z plateau test returns PLATEAU at 0.10 (the local step in/out of 0.10 is
#     an ordinary-sized wiggle in pooled P&L, not an outlier) — so the EXACT value inside
#     the corridor is not load-bearing; and
#   - the sweep shows a flat corridor ~[0.06, 0.13] where the strategy holds 8 clean
#     cycles and pooled P&L moves < ~0.6pp. BELOW ~0.06 the exit fires too weakly and the
#     2017-18 leg is lost (drops to 7 cycles); ABOVE ~0.13 it false-fires in mid-cycle
#     pauses, fragmenting cycles (9->15 as the threshold climbs) and pushing the 2015-18
#     leg negative. 0.10 sits mid-corridor.
# So the CORRIDOR EDGES are the load-bearing thing, not 0.10 itself; 0.10 is a round value
# comfortably inside it. Pause round-trips near 0.10 cost almost nothing (~+1.65% total
# forgone, re-entry re-arms via the entry latch — see diagnose_reentry_cost.py). Rests on
# n~6 cycles; the honest firm-up is MORE cycles (pre-1990 FRED history), not more compute.
#
# Gated by RATIO_EXIT_FLOOR_BP: the ratio is only evaluated once at least this much
# has been hiked since entry, so the denominator is never near zero (which would make
# the ratio explode and the exit meaningless in the first weeks of a cycle).
RATIO_EXIT_THRESHOLD = 0.10  # exit when spread_1yr / cum_bp_hiked drops below this
RATIO_EXIT_FLOOR_BP  = 25    # bp: don't evaluate the ratio until this much hiked since entry

# ── distance-to-neutral guard on the ratio exit ──────────────────────────────
# THEORY: a hiking cycle that is NORMALIZING (lifting policy back up to neutral from a
# stimulative level — e.g. 2004-06 from 1%, 2015-18 from 0%) is not over until fed funds
# actually REACHES neutral, regardless of what the spread says. The market's spread/ratio
# signal false-fires EARLY on these cycles (the 2005-09 / 2017-06 early exits) because in a
# telegraphed measured cycle the market under-prices the slow grind of remaining hikes — the
# ratio decays on the denominator (cum hiked) while the numerator (still-priced) stays small.
# Distance-to-neutral measures the missing information directly: how much further the Fed must
# climb to reach its own neutral rate.
#
# This guard is a ONE-SIDED VETO on the ratio exit: block the ratio exit while fed funds is still
# more than NEUTRAL_VETO_BP BELOW nominal neutral (r* + expected inflation). Because it is an
# AND on the ratio branch (both the ratio AND "we've climbed close enough to neutral" must
# hold), it can only ever DELAY a ratio exit, never cause one — so it cannot create new false
# exits, only fix the too-EARLY ones. It self-deactivates in RESTRICTIVE cycles: once fed
# funds is at/above neutral (2022-23, which intends to overshoot neutral), the veto is silent
# and the original exit governs — no explicit regime label needed, the sign of the gap does it.
#
# neutral is from utils.fred_utils.fetch_nominal_neutral, using the Laubach-Williams REAL-TIME
# (one-sided, per-vintage, no-hindsight) r*. Coverage starts 2005q1: for any day without an r*
# reading (pre-2005 cycles: 1994, 2000, early 2004) the neutral series is NaN and the veto is
# INACTIVE — those cycles fall back to the unguarded exit, exactly as before this guard existed.
#
# KNOWN RESIDUAL (disregarded per the current design): a normalization cycle that the Fed
# ABANDONS below neutral due to an external shock (2018-Q4 equity crash → Powell pivot; last
# hike 2018-12 at ~0.5% BELOW neutral) will be held ~a few months too long, because "arrived
# at neutral" never triggers. Detecting that forced-pause needs a shock signal calibrated on
# n=1 (only 2018 in-sample) — not built. The guard is shipped WITHOUT the shock override; the
# 2018 late-hold is an accepted, documented imperfection. See unused_mechanisms/diagnostic_tests/test_neutral_guard.py.
NEUTRAL_VETO_BP = 15   # bp below nominal neutral: block the ratio exit while gap exceeds this.

                       # exit allowed only once we've essentially reached neutral (gap <= 15bp).
                       # Holds the 2004-06 exit to ~Apr-2006 (recovering ~7 of the ~10mo the old
                       # ratio exit gave away in Sep-2005) at +17.3% pooled vs +14.1% unguarded.
                       # COST (accepted, per the disregard-2018 decision): 2015-18 ended ~51bp
                       # BELOW neutral (Fed abandoned normalization on the Q4-2018 equity crash),
                       # so arrival-at-neutral triggers late — that leg is held ~3mo past its true
                       # Dec-2018 top (exits ~Apr-2019). A looser threshold (40/60bp) shrinks that
                       # over-hold but also releases the 2005 veto early, forfeiting most of the
                       # 2004-06 recovery — the whole point of the guard. 15bp keeps the win.

# ── momentum gate on the ratio exit (standardized trailing OLS slope) ─────────
# The ratio exit above fires on the ratio's LEVEL alone — it can trip purely
# because the denominator (cum bp hiked) is large, even while the spread is still
# healthy. This gate adds a PRECONDITION: only allow the ratio exit when the spread
# is genuinely rolling over — i.e. the 1yr spread is reliably TRENDING DOWN, not just
# wobbling.
#
# HOW: fit a trailing ordinary-least-squares line to spread_1yr over SLOPE_WINDOW
# days and standardize its slope by the slope's own regression standard error:
#     t = slope / SE(slope) ,   SE(slope) = s / sqrt(Sxx)
# where s = residual std of the fit (the in-window noise) and Sxx = sum of squared
# centered day-indices. The gate opens when t < SLOPE_T_THRESHOLD. t is a one-sided
# test of H0: "true slope = 0 (flat)" — it asks whether the measured down-tilt is too
# steep to be flat-plus-noise. Because it is standardized it is UNIT-FREE and SELF-
# SCALING: t < -1.5 means "the decline is beyond ~1.5 standard errors of the fit" in
# ANY volatility regime, unlike a fixed bp cutoff that silently means different things
# in a calm vs. a chaotic cycle.
#
# SLOPE_WINDOW is DERIVED, not picked: parameter_generation/derive_slope_window.py fits an
# AR(1) to spread_1yr on quiet Fed-on-hold periods (disjoint from every trade — a hindsight-
# free null using only FOMC hike/cut dates) and gets 1-day persistence phi~0.93, half-life
# ~10td, so the noise timescale tau ~ 2-3 half-lives = ~21-31td; a variance-ratio curve
# confirms mean-reversion is still active past 63td (band not shorter). The slope window must
# span >= ~2*tau so a noise excursion averages out inside it -> W ~= 42td (63td robustness
# variant). A window down at the half-life scale (~10td, i.e. << 2*tau) just measures wobble:
# one lingering noise excursion fills it and reads as a trend.
#
# SLOPE_T_THRESHOLD is a DISCLOSED discretionary value inside a verified plateau (the
# REENTRY_BLOCK_BP / RATIO_EXIT_THRESHOLD template), NOT a p-value: the residuals are
# autocorrelated (so SE is optimistic and t runs slightly hot) and the gate is evaluated on
# thousands of OVERLAPPING daily windows, so a per-test significance level would not mean
# what it says. Instead -1.5 is a round "clearly beyond flat" scale, chosen on the STRICTER
# side of one SE specifically to lean against that autocorrelation hotness. It is non-load-
# bearing: at W=42, t<-1.0 / -1.5 / -2.0 give BYTE-IDENTICAL trades (7 cycles, 17.06%), so
# the exact cutoff does not matter — the plateau, not the number, is the justification.
SLOPE_WINDOW      = 42     # trading days: trailing OLS window (~2*tau; derive_slope_window.py)
SLOPE_T_THRESHOLD = -1.5   # gate opens (ratio exit allowed) when standardized slope t < this

# ── no re-entry into a MATURE cycle ─────────────────────────────────────────
# Once we have exited, only re-arm the entry latch within the same in_cycle span if
# the cycle is still YOUNG (cum bp hiked since cycle start < REENTRY_BLOCK_BP). Past
# this level the cycle is winding down: few hikes — and thus little payer edge —
# typically remain, so re-entering exposes us to shock whipsaws (e.g. re-arming in
# Mar-2023 straight into the SVB spread collapse: the spurious 2022-23 second episode)
# with too little expected upside to compensate. This is a RISK-ASYMMETRY re-entry
# gate, NOT a take-profit rule: it only blocks RE-arming, never the initial position,
# so it cannot cap a live winner. Below the level, enough cycle plausibly remains to
# justify re-entry (preserves the legitimate 2004-06 / 2015-18 second legs, which the
# sweep shows re-arm below 150bp — never blocked anywhere in the tested range).
#
# 350bp is a DISCRETIONARY value, but GAP-VERIFIED (not a needle threaded to P&L):
# overfitting_tests/sweep_spurious_reentry_threshold.py sweeps it 150..500bp and finds a
# BYTE-IDENTICAL plateau across 150..450bp — same 8 cycles, pooled P&L pinned at 14.07% —
# with the gate's ONLY effect being to block the spurious 2023 SVB re-entry. Above ~475bp
# that whipsaw returns (a 9th cycle, -1.78%, pooled drops to 12.04%). The legitimate
# 2004-06 / 2015-18 second legs are NOT blocked even at 150bp (they re-armed well below
# that), so the only live edge is the UPPER one (~450-475bp). 350 sits ~100bp below it —
# a round choice with comfortable headroom to the sole edge on our data.
# Rests on n=3 re-entry events (n=1 spurious). An absolute-bp proxy will still misjudge
# unusually large/small cycles, and no literature bp anchor exists (the book frames
# maturity as distance-to-last-hike, unknowable in real time) — hence discretionary, just
# now an EVIDENCED one. Firm-up = more cycles.
REENTRY_BLOCK_BP = 350   # bp hiked since cycle start past which we don't re-arm

# fed_target (FOMC target rate) is decimal (0.045 = 4.5%), like every other yield
# series in this project. A 25/50/75bp FOMC step is the smallest real move, so a
# 1bp noise floor safely separates genuine hikes/cuts from float noise without
# ever mistaking one for the other.
FED_TARGET_MOVE_FLOOR = 0.0001   # 1bp in decimal: noise floor for hike/cut detection

# ──────────────────────────────────────────────────────────────────────────

def _load_signal_data(fred_api_key: str, start: str = "1990-01-01") -> tuple[pd.DataFrame, pd.Series]:
    """Load FRED spread data and stitched fed funds target rate."""
    # fill_method=None keeps only dates where all three series have real observations,
    # avoiding stale Treasury yields paired with a live DFF on holidays
    daily_yields = fetch_fred_dataframe(
        fred_api_key,
        {"dff": "DFF", "dgs1": "DGS1", "dgs3mo": "DGS3MO"},
        start,
        fill_method=None,
    )

    # stitch the fed funds target: DFEDTAR (pre-2008, exact 25bp FOMC steps) then the
    # DFEDTARL lower bound (post-2008 ZLB), keeping the last value at the transition date
    target_pre  = fetch_fred_dataframe(fred_api_key, {"target": "DFEDTAR"},  start)["target"]
    target_post = fetch_fred_dataframe(fred_api_key, {"target": "DFEDTARL"}, start)["target"]
    fed_target  = pd.concat([target_pre, target_post]).sort_index()
    # FRED serves DFEDTAR/DFEDTARL as raw percentage points (4.5% -> 4.5); divide by 100
    # so fed_target is decimal like every other yield series in this project (0.045).
    fed_target  = fed_target / 100
    fed_target  = fed_target[~fed_target.index.duplicated(keep="last")]
    fed_target  = fed_target.reindex(daily_yields.index, method="ffill")

    daily_yields["spread_1yr_bp"] = (daily_yields["dgs1"]   - daily_yields["dff"]) * 100
    daily_yields["spread_3mo_bp"] = (daily_yields["dgs3mo"] - daily_yields["dff"]) * 100

    # drop rows where either spread jumped more than 50bp in a day — implausible moves
    # from holiday/thin-market FRED prints, not real repricing
    clean_mask = (daily_yields["spread_1yr_bp"].diff().abs() < 50) & \
                 (daily_yields["spread_3mo_bp"].diff().abs() < 50)
    daily_spreads      = daily_yields[clean_mask]
    fed_target = fed_target.reindex(daily_yields.index, method="ffill")

    return daily_spreads, fed_target

def _last_cut_dates(fed_target: pd.Series, hold_months: int = HOLD_MONTHS) -> list[pd.Timestamp]:
    """
    Return confirmed 'easing cycle done' dates: last cut before a hold of
    >= hold_months with no intervening hike.
    """
    # fed_target is decimal (0.045 = 4.5%), so a 25bp FOMC step shows up as a 0.0025 change
    change  = fed_target.diff()
    is_cut  = change < -FED_TARGET_MOVE_FLOOR
    dates   = fed_target.index.tolist()
    confirmed = []
    i = 0

    while i < len(dates):
        if not is_cut.iloc[i]:
            i += 1
            continue

        last_cut_idx = i
        # walk forward over the run of cuts, tracking the last one, until a hike appears
        j = i + 1
        while j < len(dates) and not change.iloc[j] > FED_TARGET_MOVE_FLOOR:
            if is_cut.iloc[j]:
                last_cut_idx = j
            j += 1

        # confirm the last cut only if it's followed by a hold of >= hold_months
        # with no hike; a hike before the hold elapses disqualifies it
        hold_days_required = hold_months * 21
        hold_count    = 0
        k = last_cut_idx + 1
        while k < len(dates):
            if change.iloc[k] > FED_TARGET_MOVE_FLOOR:
                break
            hold_count += 1
            if hold_count >= hold_days_required:
                confirmed.append(dates[last_cut_idx])
                break
            k += 1

        i = j

    return confirmed


def derive_fed_hike_cycles(fed_target: pd.Series,
                           start_year: int | None = None,
                           min_hikes: int = 1) -> list[dict]:
    """Derive raw Fed HIKING cycles straight from the FRED target-rate series —
    the ground-truth 'oracle' of what the Fed actually did, with NO strategy logic.

    A cycle is one contiguous in-cycle span (built from the SAME confirmed-last-cut
    → HOLD_MONTHS → next-cut boundaries detect_signal uses, so the segmentation is
    consistent across the package). Within each span the hikes are the target
    increases; first_hike / last_hike are the first / last such increase.

    Returns raw hike-date pairs — the -60/-30 oracle timing offsets are applied
    LATER by benchmark.oracle_windows(), NOT here. Each dict: {label, first_hike,
    last_hike}.

    start_year: keep only cycles whose first hike is in/after this year (None = all).
    min_hikes:  drop cycles with fewer than this many hikes (default 1 = keep all,
                including degenerate single-hike cycles like Mar-1997). Raise to 2+
                to exclude lone insurance hikes.
    """
    change     = fed_target.diff()
    is_cut     = change < -FED_TARGET_MOVE_FLOOR
    hike_dates = change.index[change > FED_TARGET_MOVE_FLOOR]
    idx        = fed_target.index

    # in-cycle spans: same last-cut → hold → next-cut construction as detect_signal
    in_cycle = pd.Series(False, index=idx)
    for cut_date in _last_cut_dates(fed_target):
        hold_end = cut_date + pd.DateOffset(months=HOLD_MONTHS)
        for date in idx[idx >= hold_end]:
            if is_cut.get(date, False):
                break
            in_cycle[date] = True

    cycles = []
    span_start = None
    prev = False
    for date, active in in_cycle.items():
        if active and not prev:
            span_start = date
        elif not active and prev:
            _append_cycle(cycles, hike_dates, span_start, date, min_hikes)
        prev = active
    if prev and span_start is not None:
        _append_cycle(cycles, hike_dates, span_start, idx[-1], min_hikes)

    if start_year is not None:
        cycles = [c for c in cycles if c["first_hike"].year >= start_year]
    return cycles


def _append_cycle(cycles: list, hike_dates, span_start, span_end, min_hikes) -> None:
    """Bracket the hikes inside [span_start, span_end] into one cycle dict, if it
    clears min_hikes. Helper for derive_fed_hike_cycles."""
    hk = hike_dates[(hike_dates >= span_start) & (hike_dates <= span_end)]
    if len(hk) < min_hikes:
        return
    first, last = hk[0], hk[-1]
    cycles.append({
        "label":      f"{first.year}-{last.year}",
        "first_hike": first,
        "last_hike":  last,
    })


def _neutral_veto(nominal_neutral, dff: pd.Series, date) -> bool:
    """Distance-to-neutral veto: True == VETO the ratio exit (fed funds still too far below neutral to be done).

    Vetoes when  nominal_neutral(date) - dff(date) > NEUTRAL_VETO_BP  (both percent; *100 = bp).
    Returns False (no veto) when the guard is off or neutral is unavailable that day (pre-2005
    r* coverage → NaN), so those days keep the original unguarded exit behaviour."""
    if nominal_neutral is None:
        return False
    nu = nominal_neutral.get(date)
    if nu is None or pd.isna(nu):
        return False
    gap_bp = (nu - dff[date]) * 100.0
    return gap_bp > NEUTRAL_VETO_BP


def detect_signal(fred_api_key: str, start: str = "1982-10-01",
                  neutral_guard: bool = True) -> pd.Series:
    """Returns a daily boolean Series — True when the payer signal is active.

    neutral_guard: when True, apply the distance-to-neutral veto on the ratio exit
    (block the ratio exit while fed funds is > NEUTRAL_VETO_BP below real-time nominal neutral).
    Inactive on any day without an r* reading (pre-2005), so pre-2005 cycles are unchanged.
    Set False to reproduce the pre-guard behaviour."""
    daily_spreads_rates, fed_target = _load_signal_data(fred_api_key, start=start)

    # nominal-neutral series for the distance-to-neutral veto (real-time LW r* + expected inflation), forward-filled
    # onto the signal index. NaN before 2005q1 r* coverage -> guard inactive those days. Loaded
    # once here; a network/parse failure degrades gracefully to "guard off" rather than crashing.
    nominal_neutral = None
    if neutral_guard:
        try:
            from utils.fred_utils import fetch_nominal_neutral
            nominal_neutral = fetch_nominal_neutral(fred_api_key, daily_spreads_rates.index)
        except Exception as e:
            print(f"  [neutral guard] disabled — could not load real-time neutral rate: {e}")
            nominal_neutral = None
    last_cuts         = _last_cut_dates(fed_target)

    # in_cycle is True on date D if and only if:
    #   1. there is a confirmed last-cut date L where D >= L + HOLD_MONTHS, AND
    #   2. the Fed has not cut since L (any cut after L closes the window immediately).
    # Using confirmed last-cut dates as anchors prevents mid-pause false opens (e.g. the
    # Apr-Oct 2008 gap triggered a window under the old hold-counter approach because the
    # hold counter elapsed just before the Oct 2008 cut — but Apr 2008 is not a confirmed
    # last-cut, so it never opens a window here).
    target_change = fed_target.diff()
    is_cut        = target_change < -FED_TARGET_MOVE_FLOOR
    in_cycle      = pd.Series(False, index=daily_spreads_rates.index)

    for cut_date in last_cuts:
        hold_end = cut_date + pd.DateOffset(months=HOLD_MONTHS)
        # Walk forward from hold_end; stop the moment any cut occurs after cut_date
        after_hold = daily_spreads_rates.index[daily_spreads_rates.index >= hold_end]
        for date in after_hold:
            if is_cut.get(date, False):
                break
            in_cycle[date] = True

    spread_3mo = daily_spreads_rates["spread_3mo_bp"]
    spread_1yr = daily_spreads_rates["spread_1yr_bp"]
    # entry spreads smoothed over a short window so a single noisy dff print (holiday/thin
    # trading) can't flip the latch on by itself — see ENTRY_SMOOTH_WINDOW_DAYS comment above
    spread_3mo_entry = spread_3mo.rolling(ENTRY_SMOOTH_WINDOW_DAYS, min_periods=1).mean()
    spread_1yr_entry = spread_1yr.rolling(ENTRY_SMOOTH_WINDOW_DAYS, min_periods=1).mean()

    # every individual hike date — used to disarm the false-promise exit once a real
    # hike confirms the cycle
    hike_dates = set(target_change.index[target_change > FED_TARGET_MOVE_FLOOR].tolist())

    # Exit spreads: both 1yr and 3mo smoothed with EXIT_SMOOTH_WINDOW_DAYS.
    # EXIT_SMOOTH_METHOD selects mean vs median. A MEDIAN over a ~1-month window is
    # robust to transient shock swings (e.g. the ~1-week SVB spread collapse in Mar
    # 2023): a shock that occupies a minority of the window can't move the median,
    # so it won't trip the exit — only a broad-based, persistent move does. A mean
    # over the same window WOULD get dragged down by the shock's extreme days.
    # 3mo is also capped at ±EXIT_CAP_BP before smoothing to bound outlier prints.
    EXIT_CAP_BP     = 100
    _roll1 = spread_1yr.rolling(EXIT_SMOOTH_WINDOW_DAYS, min_periods=1)
    _roll3 = spread_3mo.clip(lower=-EXIT_CAP_BP, upper=EXIT_CAP_BP) \
                       .rolling(EXIT_SMOOTH_WINDOW_DAYS, min_periods=1)
    if EXIT_SMOOTH_METHOD == "median":
        spread_1yr_exit = _roll1.median()
        spread_3mo_exit = _roll3.median()
    else:
        spread_1yr_exit = _roll1.mean()
        spread_3mo_exit = _roll3.mean()

    # cumulative bp hiked, running total across the whole index — we subtract the
    # value at the HIKING-CYCLE start to get "bp hiked so far this cycle" for the ratio.
    # target_change is decimal (fed_target is decimal), so *10000 converts decimal -> bp
    # (matching spread_1yr_bp/spread_3mo_bp and the bp-denominated thresholds below).
    cum_hikes_bp = (target_change.clip(lower=0) * 10000).cumsum()

    # momentum gate: standardized trailing OLS slope (t-stat) of spread_1yr over
    # SLOPE_WINDOW days. The ratio exit is only allowed to fire when this is below
    # SLOPE_T_THRESHOLD (the spread is reliably trending down, not just wobbling).
    # See the SLOPE_WINDOW / SLOPE_T_THRESHOLD comment block above for the derivation.
    slope_t_series = _rolling_slope_tstat(spread_1yr, SLOPE_WINDOW)

    latched = False
    hiked_since_entry = False    # whether a real hike has landed since this entry
    was_in_cycle = False         # was the previous day inside a hiking-cycle window?
    cum_hikes_at_cycle_start = 0.0  # cumulative-hikes reading captured at CYCLE start
    signal = pd.Series(False, index=daily_spreads_rates.index)
    for date in daily_spreads_rates.index:
        if not in_cycle[date]:
            latched = False
            hiked_since_entry = False
            was_in_cycle = False
        else:
            # Anchor the maturity denominator to the START of the hiking cycle (the day
            # in_cycle first turns True), NOT to strategy entry/re-entry. A pause exit
            # followed by a re-arm within the SAME in_cycle span must keep counting bp
            # hiked from the cycle start — otherwise the denominator resets small on the
            # second leg, the ratio inflates, and the exit fires too late (bug that hit
            # 2004-06 / 2015-18 second legs). Reset only happens above when in_cycle ends.
            if not was_in_cycle:
                cum_hikes_at_cycle_start = cum_hikes_bp[date]
                was_in_cycle = True

            if not latched:
                # Don't re-arm once the cycle is mature (too much hiked): re-entering a
                # winding-down cycle risks a shock whipsaw with little edge left. Below
                # REENTRY_BLOCK_BP the cycle is young enough to justify re-entry. The
                # FIRST entry of a cycle is unaffected (cum since start is ~0 then).
                cum_since_start = cum_hikes_bp[date] - cum_hikes_at_cycle_start
                if cum_since_start < REENTRY_BLOCK_BP and \
                        spread_3mo_entry[date] > THRESHOLD_3MO_BP and spread_1yr_entry[date] > THRESHOLD_1YR_BP:
                    latched = True
                    hiked_since_entry = False
                signal[date] = latched
                continue

            if date in hike_dates:
                hiked_since_entry = True

            cum_since_entry = cum_hikes_bp[date] - cum_hikes_at_cycle_start

            # false-promise exit: no hike delivered yet, and conviction has collapsed
            if not hiked_since_entry and spread_1yr_entry[date] <= FALSE_PROMISE_THRESHOLD_1YR_BP:
                latched = False
            # maturity-ratio exit: hiking-still-priced has shrunk to a small fraction
            # of hiking-already-done. Gated on FLOOR_BP so the denominator isn't ~0,
            # and on the ROC gate so the spread must be genuinely rolling over — the
            # ratio's level alone can't fire the exit. The distance-to-neutral veto
            # additionally BLOCKS this exit while fed funds is still well below neutral
            # (a normalization cycle that isn't done climbing) — see _neutral_veto.
            elif cum_since_entry >= RATIO_EXIT_FLOOR_BP and \
                    slope_t_series[date] < SLOPE_T_THRESHOLD and \
                    spread_1yr_exit[date] / cum_since_entry < RATIO_EXIT_THRESHOLD and \
                    not _neutral_veto(nominal_neutral, daily_spreads_rates["dff"], date):
                latched = False
            elif spread_1yr_exit[date] < THRESHOLD_1YR_EXIT and spread_3mo_exit[date] < THRESHOLD_3MO_EXIT:
                latched = False
        signal[date] = latched

    signal.name         = "first_hike_signal"
    return signal

def signal_to_cycles(signal: pd.Series) -> list[dict]:
    """
    Convert a boolean signal series into cycle dicts for backtest.calc_strat_ret.

    Each contiguous True episode becomes one cycle:
      first_hike = first True day  (signal-driven entry)
      last_hike  = first False day after the episode (signal-driven exit)

    No oracle FOMC dates. Entry and exit are purely what the market spreads say.
    Episodes still active at end of data are included with last_hike = last date.
    """
    cycles     = []
    in_episode = False
    episode_start: pd.Timestamp | None = None
    episode_num = 0

    for date, val in signal.items():
        if val and not in_episode:
            episode_start = date
            in_episode    = True
        elif not val and in_episode:
            episode_num += 1
            cycles.append({
                "label":      f"episode_{episode_num}  {episode_start.year}–{date.year}",
                "first_hike": episode_start,
                "last_hike":  date,
            })
            in_episode = False

    # episode still active at the end of the data: close it at the last available date
    if in_episode and episode_start is not None:
        episode_num += 1
        cycles.append({
            "label":      f"episode_{episode_num}  {episode_start.year}–ongoing",
            "first_hike": episode_start,
            "last_hike":  signal.index[-1],
        })

    return cycles

def stats_per_cycle(df: pd.DataFrame, ret: pd.Series, cycles: list[dict]) -> pd.DataFrame:
    """Per-cycle breakdown: entry/exit dates, days held, total compounded return.

    ret is unused here (strat_ret is pulled from df); it's kept only for API symmetry
    with _run so callers can pass the same arguments to both.
    """
    rows = []
    for c in cycles:
        # restrict to days inside this cycle's window where the position was short (-1)
        mask = (df.index >= c["first_hike"]) & (df.index <= c["last_hike"]) & (df["signal"] == -1)
        r    = df.loc[mask, "strat_ret"].dropna()
        cum  = (1 + r).prod() - 1
        rows.append({
            "cycle":       c["label"],
            "entry":       str(c["first_hike"].date()),
            "exit":        str(c["last_hike"].date()),
            "days_held":   len(r),
            "total_ret_%": round(cum * 100, 2),
        })
    return pd.DataFrame(rows).set_index("cycle")
