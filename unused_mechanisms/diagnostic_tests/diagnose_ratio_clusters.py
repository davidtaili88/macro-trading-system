"""
Diagnostic: does (numerator / cumulative-bp-hiked) SEPARATE mid-cycle pauses from
the true end of a hiking cycle — with a wide enough gap that ONE threshold is
robust rather than overfit?

This is the decision test for the exit rule discussed at length:
  - EXIT idea: a percentile-flagged spread down-turn "counts" only once the cycle
    is mature (a lot has been hiked). Written as a single ratio so there is one
    knob, not two.
  - The whole rule stands or falls on ONE empirical question: at the true end of
    each cycle, is the ratio in a DIFFERENT, non-overlapping band than during
    mid-cycle pauses? If the two clusters have a clean gap, a threshold anywhere
    in the gap works and the exact value doesn't matter (robust). If they overlap,
    NO threshold works and we must not tune one into existence (that would be
    fitting 4 cycles). See the long overfitting discussion: with n=4 the GAP is
    the result; the threshold is just something in it.

We compute THREE numerators, each ÷ cumulative bp hiked since entry, on TWO tenors:

  NUMERATORS (all normalized by peak-since-entry first, so cross-cycle comparable):
    level : current smoothed spread            -> "how much hiking still priced"  (LAGS)
    roc   : 1-month-MA rate of change (diff21)  -> "how fast spread is falling"    (LEADS)
    ddown : drawdown from cycle peak            -> "how far retraced from the high"(hybrid)

  DENOMINATOR:
    cumulative bp hiked since entry (from fed_target). Monotone maturity clock.
    Bounded via a soft cap so the 525bp (2022-23) vs 225bp (2015-18) scale gap
    doesn't make one global threshold mean different things per cycle — see
    MATURITY_CAP_BP. Also floored so we never divide by ~0 at cycle start.

  TENORS: 1yr (DGS1-DFF) and 3mo (DGS3MO-DFF). 3mo leads 1yr into the turn but is
    noisier; we print both so the ROC numerator can be judged on each.

EVENTS compared:
    PAUSE      : the middle of each >=4-month gap between consecutive hikes.
    TRUE_END   : the window [last_hike - 2mo, last_hike + 1mo].
  For each event we report the ratio's median over the event window. A good rule
  needs TRUE_END ratios and PAUSE ratios to fall in separated bands.

Nothing here is tuned to a target. We fix, by principle (not by fit):
    MA_DAYS / ROC_LAG = 21   (~1 month, FOMC cadence)
    PAUSE_MIN_DAYS    = 120  (~4 months = a real pause, not inter-meeting spacing)
    MATURITY_CAP_BP   = 150  (~6 hikes of 25bp = "clearly mature"; coarse, round)
  and then we LOOK at whether a gap exists. We do not search for the value that
  makes it work.
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

MA_DAYS         = 21    # ~1-month moving average (FOMC cadence)
ROC_LAG_DAYS    = 21    # ROC horizon: change vs ~1 month ago
PAUSE_MIN_DAYS  = 120   # >=4 months between hikes = a real pause
MATURITY_CAP_BP = 150   # soft cap on cumulative hikes (~6x25bp): "clearly mature"
MATURITY_FLOOR_BP = 25  # floor so we never divide by ~0 near cycle start
END_LO_MONTHS   = 2     # true-end window starts 2mo before last hike
END_HI_MONTHS   = 1     # ...and ends 1mo after


def _cum_hikes(fed_target: pd.Series) -> pd.Series:
    """Cumulative bp hiked (cumulative sum of positive target changes), in bp."""
    chg = fed_target.diff()
    # fed_target is decimal (0.045 = 4.5%); *10000 converts decimal -> bp
    hikes_bp = chg.clip(lower=0) * 10000        # only rises; bp
    return hikes_bp.cumsum()


def _numerators(spread: pd.Series) -> pd.DataFrame:
    """The three numerators, each normalized by peak-since (done per-cycle later)."""
    sm    = spread.rolling(MA_DAYS, min_periods=MA_DAYS // 2).mean()
    roc   = -sm.diff(ROC_LAG_DAYS)              # POSITIVE when spread is falling
    return pd.DataFrame({"sm": sm, "roc_fall": roc})


def _pause_windows(fed_target: pd.Series, entry, last) -> list[tuple]:
    """Middles of >=PAUSE_MIN_DAYS gaps between consecutive hikes within the cycle."""
    chg = fed_target.diff()
    hike_dates = chg.index[chg > FED_TARGET_MOVE_FLOOR]
    hike_dates = hike_dates[(hike_dates >= entry) & (hike_dates <= last)]
    pauses = []
    for a, b in zip(hike_dates[:-1], hike_dates[1:]):
        if (b - a).days >= PAUSE_MIN_DAYS:
            pauses.append((a, b))
    return pauses


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    ds, fed_target = _load_signal_data(api_key, start="1976-01-01")
    cum_hikes_all = _cum_hikes(fed_target)

    tenors = {"1yr": ds["spread_1yr_bp"], "3mo": ds["spread_3mo_bp"]}

    # collect one row per (cycle, event, tenor) with all three ratios
    rows = []
    for c in CYCLES:
        entry = pd.Timestamp(c["entry"]); last = pd.Timestamp(c["last_hike"])
        # cumulative hikes measured SINCE entry, bounded [floor, cap]
        cum_since = (cum_hikes_all - cum_hikes_all.asof(entry))
        maturity  = cum_since.clip(lower=0).clip(upper=MATURITY_CAP_BP)
        maturity  = maturity.clip(lower=MATURITY_FLOOR_BP)  # avoid /~0

        pauses = _pause_windows(fed_target, entry, last)
        end_lo = last - pd.DateOffset(months=END_LO_MONTHS)
        end_hi = last + pd.DateOffset(months=END_HI_MONTHS)
        events = [("PAUSE", a, b) for (a, b) in pauses] + [("TRUE_END", end_lo, end_hi)]

        for tlabel, spread in tenors.items():
            num = _numerators(spread)
            win_full = num[(num.index >= entry) & (num.index <= last + pd.DateOffset(months=6))].copy()
            peak = win_full["sm"].cummax()
            # normalized numerators (fraction of peak)
            lvl_n   = (win_full["sm"] / peak.replace(0, np.nan))
            roc_n   = (win_full["roc_fall"] / peak.replace(0, np.nan))
            ddown_n = ((peak - win_full["sm"]) / peak.replace(0, np.nan))  # positive retrace
            mat = maturity.reindex(win_full.index).ffill()

            ratios = pd.DataFrame({
                "level_ratio": lvl_n   / mat * 100,   # ×100 just for readable magnitudes
                "roc_ratio":   roc_n   / mat * 100,
                "ddown_ratio": ddown_n / mat * 100,
            })

            for ev, lo, hi in events:
                m = (ratios.index >= lo) & (ratios.index <= hi)
                sub = ratios[m]
                if sub.empty:
                    continue
                rows.append({
                    "cycle": c["label"], "event": ev, "tenor": tlabel,
                    "when":  f"{lo.date()}..{hi.date()}",
                    "level": round(sub["level_ratio"].median(), 3),
                    "roc":   round(sub["roc_ratio"].median(), 3),
                    "ddown": round(sub["ddown_ratio"].median(), 3),
                })

    df = pd.DataFrame(rows)

    # ---- 1) raw event table ----
    print("=" * 88)
    print("RATIO VALUE AT EACH EVENT   (numerator normalized ÷ bounded cumulative bp hiked)")
    print("higher ratio = bigger numerator relative to hikes done")
    print("=" * 88)
    print(df.to_string(index=False))
    print()

    # ---- 2) cluster separation: for each numerator × tenor, PAUSE vs TRUE_END ----
    print("=" * 88)
    print("CLUSTER SEPARATION  —  does one threshold split PAUSE from TRUE_END?")
    print("For the exit we want TRUE_END and PAUSE in DIFFERENT bands with a GAP between.")
    print("(roc: exit fires on HIGH roc_ratio=fast fall late in cycle; so want END > PAUSE)")
    print("(level/ddown: direction depends on numerator — we just report the two bands)")
    print("=" * 88)
    for num in ["level", "roc", "ddown"]:
        print(f"\n  numerator = {num.upper()}")
        for tlabel in ["1yr", "3mo"]:
            sub = df[df["tenor"] == tlabel]
            pause = sub[sub["event"] == "PAUSE"][num].dropna()
            end   = sub[sub["event"] == "TRUE_END"][num].dropna()
            if len(pause) == 0:
                pstr = "(no pause events)"
                gap  = np.nan
            else:
                pstr = f"[{pause.min():.3f}, {pause.max():.3f}] (n={len(pause)})"
                # gap between the clusters (positive = clean separation, END above PAUSE)
                gap = end.min() - pause.max()
            estr = f"[{end.min():.3f}, {end.max():.3f}] (n={len(end)})" if len(end) else "(none)"
            gapstr = f"gap(END.min - PAUSE.max) = {gap:+.3f}" if not np.isnan(gap) else "gap = n/a"
            verdict = ""
            if not np.isnan(gap):
                verdict = "  <-- CLEAN GAP" if gap > 0 else "  <-- OVERLAP (no robust threshold)"
            print(f"    {tlabel}:  PAUSE {pstr:32s}  END {estr:28s}  {gapstr}{verdict}")

    print()
    print("Reading it:")
    print("  A positive gap => PAUSE and END occupy separated bands; a threshold anywhere")
    print("    in the gap separates them and the exact value is not critical (ROBUST).")
    print("  A negative gap => the bands OVERLAP; no single threshold works. Per the n=4")
    print("    overfitting discipline, we do NOT tune a value into the overlap — the")
    print("    honest conclusion is that numerator×tenor does not carry the exit signal.")
    print("  NOTE: only 2015-18 has real pause events, so PAUSE cluster is essentially")
    print("    n=2. Treat any 'clean gap' as suggestive, not validated.")


if __name__ == "__main__":
    main()
