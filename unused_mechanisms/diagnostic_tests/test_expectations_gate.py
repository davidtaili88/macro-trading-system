"""
TERM-PREMIUM-STRIPPED EXPECTATIONS as the exit gate — the FRED proxy for the fed funds
futures signal we couldn't get historically.

WHAT THIS MEASURES
------------------
Any yield = (avg expected future short rate) + (term premium). The expected-short-rate part
is exactly what fed funds FUTURES price = "where will fed funds average over the horizon" =
the 'hikes remaining' signal we want. The front-end spread DGS1-DFF failed partly because
DGS1 carries a TERM-PREMIUM blob that moves for non-Fed reasons. Futures are ~term-premium-
free; here we strip it EXPLICITLY using the Kim-Wright model on FRED:

    pure_expectations_1yr = THREEFY1 - THREEFYTP1        (fitted 1yr yield minus its term premium)
    priced_hikes_remaining = pure_expectations_1yr - DFF (percent pts; /0.25 = # of 25bp hikes)

If futures would have helped, it's BECAUSE of this term-premium removal, so this proxy tests
the futures hypothesis directly — on FULL history (Kim-Wright from 1990), covering 2004-06 and
1994, the pre-SEP cycles where the bug lives.

THE TEST (same bar the curve slope FAILED)
------------------------------------------
At every day the maturity-ratio exit WANTS to fire (raw ratio r < RATIO_EXIT_THRESHOLD),
label by distance to that cycle's TRUE last hike:
    REAL  = within +/-63td of last hike (legit exit)
    EARLY = >63td before last hike (false alarm — the 2005 disease)
AND-gate works iff priced_hikes_remaining is systematically HIGHER on EARLY days (Fed still
has hikes to go -> veto the exit) than on REAL days (Fed nearly done -> allow it), with a
threshold that blocks most EARLY without killing REAL. Reports both distributions + separation.
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
    _load_signal_data, FED_TARGET_MOVE_FLOOR,
    EXIT_SMOOTH_WINDOW_DAYS, RATIO_EXIT_THRESHOLD, RATIO_EXIT_FLOOR_BP,
)

TRUE_LAST_HIKES = {
    "1994-95": "1995-02-01", "1999-00": "2000-05-16", "2004-06": "2006-06-29",
    "2015-18": "2018-12-20", "2022-23": "2023-07-27",
}
REAL_WINDOW_TD = 63


def _fetch(api_key, series, index):
    from fredapi import Fred
    fred = Fred(api_key=api_key)
    s = fred.get_series(series, observation_start="1976-01-01")
    s.index = pd.to_datetime(s.index)
    return s.reindex(index).astype(float)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    ds, fed_target = _load_signal_data(api_key, start="1976-01-01")
    spread_1yr = ds["spread_1yr_bp"]
    dff = ds["dff"]                      # percent
    target_change = fed_target.diff()

    # --- pure-expectations 1yr rate (Kim-Wright), term premium stripped ---------
    fy1  = _fetch(api_key, "THREEFY1",   ds.index)   # fitted 1yr yield (percent)
    tp1  = _fetch(api_key, "THREEFYTP1", ds.index)   # 1yr term premium (percent)
    pure_exp_1yr = fy1 - tp1                          # expected avg short rate, 1yr horizon
    hikes_left_pp = pure_exp_1yr - dff               # percent pts still expected to be hiked
    hikes_left_n  = hikes_left_pp / 0.25             # in # of 25bp hikes

    # sanity: coverage
    cov = hikes_left_pp.dropna()
    print(f"Kim-Wright expectations coverage: {cov.index[0].date()}..{cov.index[-1].date()} ({len(cov)} days)")

    # --- reconstruct ratio-exit firing days, labelled -----------------------------
    cum_hikes_bp = (target_change.clip(lower=0) * 10000).cumsum()
    spread_exit  = spread_1yr.rolling(EXIT_SMOOTH_WINDOW_DAYS, min_periods=1).mean()

    real_rows, early_rows = [], []
    for lbl, lh in TRUE_LAST_HIKES.items():
        lh = pd.Timestamp(lh)
        span = ds.index[(ds.index >= lh - pd.Timedelta(days=800)) & (ds.index <= lh + pd.Timedelta(days=30))]
        if not len(span):
            continue
        cum0 = cum_hikes_bp[span[0]]
        for d in span:
            cum = cum_hikes_bp[d] - cum0
            if cum < RATIO_EXIT_FLOOR_BP:
                continue
            r = spread_exit[d] / cum
            if r < RATIO_EXIT_THRESHOLD:
                lag = np.busday_count(d.date(), lh.date())
                rec = {"cycle": lbl, "date": d.date(), "lag": lag,
                       "hikes_left_n": hikes_left_n[d] if d in hikes_left_n.index else np.nan}
                (real_rows if abs(lag) <= REAL_WINDOW_TD else
                 (early_rows if lag > REAL_WINDOW_TD else real_rows)).append(rec)

    real = pd.DataFrame(real_rows); early = pd.DataFrame(early_rows)

    print("\n" + "=" * 90)
    print(f"RATIO-EXIT FIRING DAYS, labelled  (REAL=+/-{REAL_WINDOW_TD}td of last hike, EARLY=>{REAL_WINDOW_TD}td before)")
    print("=" * 90)
    print(f"  firing days: REAL={len(real)}   EARLY={len(early)}")
    # how many have expectations data (Kim-Wright starts 1990 -> misses nothing here, but 2022 dff etc ok)
    print(f"  with expectations data: REAL={int(real['hikes_left_n'].notna().sum()) if len(real) else 0}"
          f"   EARLY={int(early['hikes_left_n'].notna().sum()) if len(early) else 0}")

    print("\n" + "-" * 90)
    print("PRICED HIKES REMAINING (# of 25bp hikes still expected, term-premium-stripped)")
    print("  AND-gate works iff EARLY (false alarms) has MORE hikes-left than REAL (legit tops)")
    print("-" * 90)
    for tag, dfx in [("EARLY (false alarms)", early), ("REAL  (legit tops)", real)]:
        s = dfx["hikes_left_n"].dropna() if len(dfx) else pd.Series([], dtype=float)
        if len(s):
            print(f"  {tag:<22} n={len(s):>4}  median={s.median():>+5.1f} hikes  "
                  f"25/75={s.quantile(.25):>+5.1f}/{s.quantile(.75):>+5.1f}  min/max={s.min():>+5.1f}/{s.max():>+5.1f}")
        else:
            print(f"  {tag:<22} (no data)")

    if len(real) and len(early):
        e = early["hikes_left_n"].dropna(); r = real["hikes_left_n"].dropna()
        if len(e) and len(r):
            T = (e.median() + r.median()) / 2
            eb = (e > T).mean() * 100; rk = (r < T).mean() * 100
            print(f"\n  candidate veto: allow exit only if hikes_left < {T:+.2f}:")
            print(f"     blocks {eb:.0f}% of EARLY false alarms   |   keeps {rk:.0f}% of REAL exits")
            print(f"     (want BOTH high; ~50/50 = the signal doesn't separate = clone/failure)")

    # --- per-cycle: hikes-left at the earliest false alarm vs at the true top -----
    print("\n" + "=" * 90)
    print("PER-CYCLE: hikes-left at earliest false-alarm  vs  at true last hike")
    print("  (want: clearly POSITIVE at false alarm [veto], ~0 at the top [allow])")
    print("=" * 90)
    for lbl, lh in TRUE_LAST_HIKES.items():
        lh = pd.Timestamp(lh)
        at_top = np.nan
        idx = hikes_left_n.index[hikes_left_n.index <= lh]
        if len(idx): at_top = hikes_left_n[idx[-1]]
        sub = early[early["cycle"] == lbl] if len(early) else pd.DataFrame()
        if len(sub):
            first = sub.sort_values("lag", ascending=False).iloc[0]
            print(f"  {lbl}: false-alarm {first['date']} (lag +{first['lag']}td)  hikes_left={first['hikes_left_n']:+.1f}"
                  f"   |  at top {lh.date()}  hikes_left={at_top:+.1f}")
        else:
            print(f"  {lbl}: (no early false alarm)                          |  at top {lh.date()}  hikes_left={at_top:+.1f}")

    print("\nVerdict: if EARLY hikes-left is clearly > REAL hikes-left with a clean threshold,")
    print("the term-premium-stripped expectations (== the futures signal) rescue the gate. If")
    print("they overlap (~50/50), futures would NOT have helped either — same signal, case closed.")


if __name__ == "__main__":
    main()
