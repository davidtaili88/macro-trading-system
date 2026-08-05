"""
Diagnostic: is "lenient exit when the hiking cycle is mature" actually present
in the data — BEFORE we pick any threshold?

Hypothesis (from the exit-lag discussion):
    The spread rolls over (rate of change turns sustainably negative) AT / JUST
    BEFORE the true last hike, and this rollover is DISTINGUISHABLE from ordinary
    mid-cycle chop. If true, an exit on "sustained normalized ROC < T", with T
    relaxed as the cycle matures, can beat the lagging level exit.

Two things killed earlier ideas and we test for both here:
  1. SCALE. Spreads live at different heights across cycles (1994 peaked ~180bp,
     2015-18 ~65bp), so RAW bp/month ROC does not transfer. We therefore compute
     a NORMALIZED monthly ROC (see below) and check it is comparable across cycles.
  2. SEPARATION. Drawdown-from-peak and raw slope both false-fired mid-cycle
     because chop looks like a rollover. So the real test is statistical:
     is the ROC in the true-rollover window actually more negative than the ROC
     during the rest of the cycle? If not, no threshold saves us.

We do NOT fit a threshold here. This only asks whether the SHAPE exists. Fitting a
number to 4 cycles would be the overfitting the whole discussion has been avoiding.

ROC construction (the point of this file — get the statistic right):
  We want "moving average over ~a month of the rate of change", evaluated EVERY
  trading day, not one snapshot per month. So:
     sm    = spread.rolling(MA_DAYS).mean()   # ~1-month moving average of the spread
     roc   = sm.diff(ROC_LAG_DAYS)            # change vs ~1 month ago, in bp, DAILY
  This keeps all data (no month-snapshotting, which discarded ~20 days/month and
  made the ROC the noisy difference of two single-day endpoints). The monthly MA
  already suppresses daily jitter, and diff-over-a-month is itself insensitive to
  exact sample timing, so the series is very smooth. We deliberately do NOT add a
  further 5-day MA on top: with a monthly MA already applied, a 5-day smooth
  removes essentially nothing (the residual structure is month-scale, not daily)
  and only adds lag — the exact lag we are trying to cut. If month-scale wobble
  causes false fires the right tool is a persistence requirement (ROC below a
  level for N consecutive days), not more averaging; we test for that need rather
  than assume it.

Normalization choices (both reported — they answer different questions):
  roc_bp     : raw daily ROC (bp change vs ~1 month ago) of the month-MA spread.
  roc_freng  : roc_bp / (running peak of the spread since entry). Fraction of the
               cycle's own achieved height lost per month — dimensionless, so a
               "20% of peak per month" decline is comparable across cycles of very
               different spread heights. Peak-since-entry (not a fixed scale) is the
               natural denominator because it is what the payer trade actually
               captured.

Statistical test:
  Per cycle, split the in-cycle days into
     ROLLOVER window = [last_hike - W months, last_hike + rollover_tail]
     MID-CYCLE       = entry .. (last_hike - W months)
  and run a one-sided Mann-Whitney U (nonparametric: these series are fat-tailed
  and autocorrelated, so a t-test's normality assumption is unjustified) testing
  H1: rollover ROC < mid-cycle ROC. Report U, p, and rank-biserial effect size.
  With only ~4 cycles the pooled conclusion is weak by construction; we report
  each cycle separately and say so, rather than pretending 4 points give power.

Data: FRED spreads via signal_logic._load_signal_data (same inputs as the strategy).
TRUE last-hike dates are the oracle FOMC dates (benchmark.ORACLE_CYCLES) plus the
two pre-2002 cycles this project cares about (1994-95, and the 1994 twin is folded in).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "hiking_curve_strategy"))
import _paths  # noqa: F401 — registers core/, benchmark/, utils/ on sys.path
from signal_logic import _load_signal_data

try:
    from scipy.stats import mannwhitneyu
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# TRUE last-hike dates (oracle FOMC hindsight) and the strategy entry anchor.
# entry here is the market-signal entry from signal_to_cycles (the day the payer
# actually went on); last_hike is the real final FOMC hike. The gap between
# last_hike and where the current level-exit fires is the lag we are attacking.
CYCLES = [
    {"label": "1994-95", "entry": "1993-11-04", "last_hike": "1995-02-01"},
    {"label": "2004-06", "entry": "2004-06-03", "last_hike": "2006-06-29"},
    {"label": "2015-18", "entry": "2015-12-10", "last_hike": "2018-12-20"},
    {"label": "2022-23", "entry": "2022-02-04", "last_hike": "2023-07-27"},
]

ROLLOVER_WINDOW_MONTHS = 3   # how far before the last hike we call "the rollover"
ROLLOVER_TAIL_MONTHS   = 1   # a little after the last hike still counts as rollover
MA_DAYS                = 21  # ~1-month moving average on the spread (trading days)
ROC_LAG_DAYS           = 21  # ROC horizon: change vs ~1 month ago (trading days)


def _roc_series(spread_daily: pd.Series) -> pd.DataFrame:
    """~1-month moving-average of the spread + its DAILY month-over-month ROC.

    No month snapshotting and no extra smoothing on the ROC — see module docstring.
    Returned daily so an exit could fire on any trading day, not just month-starts.
    """
    sm     = spread_daily.rolling(MA_DAYS, min_periods=MA_DAYS // 2).mean()
    roc_bp = sm.diff(ROC_LAG_DAYS)          # bp change vs ~1 month ago (negative = falling)
    return pd.DataFrame({"spread": sm, "roc_bp": roc_bp})


def _analyze_cycle(c: dict, spread_daily: pd.Series) -> dict:
    entry     = pd.Timestamp(c["entry"])
    last_hike = pd.Timestamp(c["last_hike"])

    df = _roc_series(spread_daily)   # DAILY series now
    df = df[(df.index >= entry) & (df.index <= last_hike + pd.DateOffset(months=6))].copy()

    # normalize by running peak-since-entry so cycles of different spread height compare
    df["peak"]      = df["spread"].cummax()
    df["roc_freng"] = df["roc_bp"] / df["peak"].replace(0, np.nan)  # fraction of peak per month

    roll_lo = last_hike - pd.DateOffset(months=ROLLOVER_WINDOW_MONTHS)
    roll_hi = last_hike + pd.DateOffset(months=ROLLOVER_TAIL_MONTHS)

    # Test on WEEKLY-sampled points, not raw daily. The daily ROC is a 21d MA
    # diffed over 21d, so consecutive days share ~20/21 of their inputs — they are
    # massively autocorrelated. Running MWU on raw daily points would treat ~250
    # correlated days as 250 independent draws and report a fake-tiny p. Weekly
    # sampling cuts the worst of that overlap. Even this OVERSTATES power (weekly
    # points still overlap within the 21d windows); we say so and read p as
    # suggestive only.
    wk = df[["roc_freng"]].resample("W").last().dropna()
    is_roll = (wk.index >= roll_lo) & (wk.index <= roll_hi)
    is_mid  = (wk.index <  roll_lo)

    roll = wk.loc[is_roll, "roc_freng"].dropna()
    mid  = wk.loc[is_mid,  "roc_freng"].dropna()

    result = {
        "label": c["label"],
        "df": df,
        "roll_vals": roll.values,   # weekly-sampled, for the pooled test
        "mid_vals":  mid.values,
        "roll_median": roll.median() if len(roll) else np.nan,
        "mid_median":  mid.median()  if len(mid)  else np.nan,
        "n_roll": len(roll),
        "n_mid":  len(mid),
        "U": np.nan, "p": np.nan, "rank_biserial": np.nan,
    }

    # one-sided Mann-Whitney: H1 = rollover ROC is LESS (more negative) than mid-cycle
    if HAVE_SCIPY and len(roll) >= 2 and len(mid) >= 2:
        U, p = mannwhitneyu(roll, mid, alternative="less")
        result["U"] = U
        result["p"] = p
        # rank-biserial effect size: 1 - 2U/(n1*n2); +1 => rollover fully below mid
        result["rank_biserial"] = 1 - (2 * U) / (len(roll) * len(mid))
    return result


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    ds, _ = _load_signal_data(api_key, start="1976-01-01")
    spread = ds["spread_1yr_bp"]

    if not HAVE_SCIPY:
        print("WARNING: scipy not installed — statistical test skipped. `pip install scipy`\n")

    results = [_analyze_cycle(c, spread) for c in CYCLES]

    # 1) Per-cycle ROC trajectories, sampled MONTH-START for readability only
    #    (the underlying series and the test are daily/weekly — this is display).
    for r in results:
        print("=" * 72)
        print(f"{r['label']}   true last hike {[c['last_hike'] for c in CYCLES if c['label']==r['label']][0]}")
        show = r["df"][["spread", "roc_bp", "roc_freng"]].resample("MS").last().round(3)
        print(show.to_string())
        print()

    # 2) Statistical separation table
    print("=" * 72)
    print("SEPARATION TEST  —  normalized monthly ROC (fraction of peak / month), weekly-sampled")
    print("H1: rollover-window ROC is MORE NEGATIVE than mid-cycle ROC (one-sided MWU)")
    print("=" * 72)
    tbl = pd.DataFrame([{
        "cycle":         r["label"],
        "mid_median":    round(r["mid_median"], 4),
        "rollover_med":  round(r["roll_median"], 4),
        "separation":    round((r["mid_median"] - r["roll_median"]), 4),
        "n_mid":         r["n_mid"],
        "n_roll":        r["n_roll"],
        "MWU_p":         round(r["p"], 4) if not np.isnan(r["p"]) else np.nan,
        "effect(rbc)":   round(r["rank_biserial"], 3) if not np.isnan(r["rank_biserial"]) else np.nan,
    } for r in results]).set_index("cycle")
    print(tbl.to_string())
    print()
    print("Reading it:")
    print("  separation > 0 and MWU_p < ~0.10 => rollover IS distinguishable from chop")
    print("    in that cycle (with tiny n, treat p as suggestive, not proof).")
    print("  effect(rbc) near +1 => rollover ROC ranks almost entirely below mid-cycle.")
    print("  If separation is ~0 or negative in a cycle, no ROC threshold can time")
    print("    that cycle's exit — the rollover looks like the chop. Report honestly.")
    print()

    # 3) Pooled view — stack all cycles' WEEKLY roll/mid points, one combined MWU.
    #    Weak by construction (few cycles, weekly points still overlap) — flagged.
    if HAVE_SCIPY:
        pooled_roll = np.concatenate([r["roll_vals"] for r in results])
        pooled_mid  = np.concatenate([r["mid_vals"]  for r in results])
        if len(pooled_roll) >= 2 and len(pooled_mid) >= 2:
            U, p = mannwhitneyu(pooled_roll, pooled_mid, alternative="less")
            rbc = 1 - (2 * U) / (len(pooled_roll) * len(pooled_mid))
            print(f"POOLED (all cycles, weekly points stacked — NOTE: still overstates")
            print(f"  power, weekly points overlap and cycles unequal): rollover median "
                  f"{np.median(pooled_roll):.4f} vs mid {np.median(pooled_mid):.4f}, "
                  f"MWU p={p:.4f}, effect={rbc:.3f}")


if __name__ == "__main__":
    main()
