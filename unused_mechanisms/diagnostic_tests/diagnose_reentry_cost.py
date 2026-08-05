"""
Does a PAUSE false-fire actually cost us — given the signal RE-ENTERS when the
cycle resumes?

Reframe (per the key question): exiting during a pause is only bad to the extent
that (a) we stay flat through a chunk of the trade, and (b) we re-enter at a worse
price. If the payer trade barely moves during the pause-and-re-entry gap, a false
fire is cheap and the level-ratio rule is salvageable.

Rule tested:  ratio = smoothed_spread(1yr) / cumulative_bp_hiked_since_entry
              EXIT  when ratio < THRESHOLD (0.10 here, per request)
              RE-ENTER when ratio climbs back above THRESHOLD (simplest symmetric
                        re-arm; a real impl would reuse the entry latch, but this
                        isolates the ratio rule's own round-trips).

For each cycle we find every EXIT->REENTER gap and measure the FORGONE PAYER P&L
across it: the payer return we would have earned had we stayed short through the
gap. Positive forgone = the flat period cost us (spread rose / bond sold off while
we were out). Negative forgone = the gap actually SAVED us (bond rallied while we
were flat — a pause exit that dodged a drawdown).

Payer daily return proxy = -1 * (duration * daily yield change). We approximate
with the 2y (DGS1 as the short-rate driver is wrong; use the actual instrument
return if available). Here we use a simple duration-scaled yield-change proxy on
the 1yr spread's underlying DGS1 is NOT the 2y — so instead we proxy payer P&L by
the change in the SPREAD itself scaled to bp, which tracks the direction/size of
the payer edge during these windows (good enough to rank 'cheap' vs 'costly').
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
from data import fetch_dgs2_full_pnl


CYCLES = [
    {"label": "1994-95", "entry": "1993-11-04", "last_hike": "1995-02-01"},
    {"label": "2004-06", "entry": "2004-06-03", "last_hike": "2006-06-29"},
    {"label": "2015-18", "entry": "2015-12-10", "last_hike": "2018-12-20"},
    {"label": "2022-23", "entry": "2022-02-04", "last_hike": "2023-07-27"},
]

SMOOTH_DAYS = 5
FLOOR_BP    = 25
THRESHOLD   = 0.10   # per request: was 0.15


def _cum_hikes(fed_target):
    chg = fed_target.diff()
    # fed_target is decimal (0.045 = 4.5%); *10000 converts decimal -> bp
    return (chg.clip(lower=0) * 10000).cumsum()


def _segments(ratio_in_window: pd.Series, thr: float):
    """Return list of (exit_date, reenter_date_or_None) gaps where ratio < thr."""
    below = ratio_in_window < thr
    segs = []
    in_gap = False
    start = None
    for dt, b in below.items():
        if b and not in_gap:
            in_gap = True; start = dt
        elif not b and in_gap:
            in_gap = False; segs.append((start, dt))   # dt = first re-entry day
    if in_gap:
        segs.append((start, None))                      # never re-entered
    return segs


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    ds, fed_target = _load_signal_data(api_key, start="1976-01-01")
    cum_all = _cum_hikes(fed_target)
    sm1 = ds["spread_1yr_bp"].rolling(SMOOTH_DAYS, min_periods=1).mean()

    # real payer P&L: short DGS2 total return (price + carry). payer earns +ret when
    # bonds sell off. We negate buy-hold bond return to get the payer's daily P&L.
    pnl = fetch_dgs2_full_pnl(api_key, start="1976-01-01")
    payer_ret = (-pnl["total_ret"]).rename("payer_ret")   # payer = short the 2y

    print("#" * 84)
    print(f"#  RE-ENTRY COST of the level-ratio rule (1yr), threshold = {THRESHOLD}")
    print(f"#  payer P&L = short DGS2 total return.  forgone>0 => gap cost us; <0 => gap saved us")
    print("#" * 84)

    grand_forgone = 0.0
    for c in CYCLES:
        entry = pd.Timestamp(c["entry"]); last = pd.Timestamp(c["last_hike"])
        cum_since = (cum_all - cum_all.asof(entry)).clip(lower=0).reindex(sm1.index).ffill()
        win = (sm1.index >= entry) & (sm1.index <= last + pd.DateOffset(months=8))
        defined = cum_since >= FLOOR_BP
        ratio = (sm1 / cum_since).where(defined & win).dropna()

        segs = _segments(ratio, THRESHOLD)
        print(f"\n  {c['label']}   last hike {last.date()}   ({len(segs)} exit/re-entry gap(s))")
        if not segs:
            print("     never fires below threshold in window")
            continue

        cyc_forgone = 0.0
        for ex, re in segs:
            # classify: is this gap a mid-cycle PAUSE exit (re-entered, ends before
            # last hike) or the FINAL exit (no re-entry before cycle truly ends)?
            gap_ret = payer_ret[(payer_ret.index > ex) &
                                (payer_ret.index <= (re if re is not None else last + pd.DateOffset(months=8)))]
            forgone = (1 + gap_ret).prod() - 1        # payer return we skipped while flat
            days = len(gap_ret)
            if re is None:
                kind = "FINAL exit (no re-entry)"; re_str = "—"
            elif re <= last:
                kind = "MID-CYCLE pause exit -> RE-ENTERED"; re_str = str(re.date())
            else:
                kind = "exit near/after last hike"; re_str = str(re.date())
            print(f"     exit {ex.date()}  re-enter {re_str:12s}  {days:4d} td flat   "
                  f"forgone payer P&L = {forgone*100:+6.2f}%   [{kind}]")
            # only pause round-trips (re-entered before last hike) count as the
            # 'cost of a false fire'; the final exit is the intended behavior
            if re is not None and re <= last:
                cyc_forgone += forgone
        print(f"     --> total forgone across MID-CYCLE pause round-trips: {cyc_forgone*100:+.2f}%")
        grand_forgone += cyc_forgone

    print("\n" + "=" * 84)
    print(f"GRAND TOTAL forgone payer P&L across all mid-cycle pause round-trips: "
          f"{grand_forgone*100:+.2f}%")
    print("Interpretation:")
    print("  If this is small/negative, pause false-fires are CHEAP (we re-enter with")
    print("  little missed, or the flat period dodged a drawdown) -> the level-ratio")
    print("  rule is salvageable despite firing mid-cycle.")
    print("  If large positive, the round-trips bleed real P&L -> the rule is not saved")
    print("  by re-entry and we drop it.")


if __name__ == "__main__":
    main()
