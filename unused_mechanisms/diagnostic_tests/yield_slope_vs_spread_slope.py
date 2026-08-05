"""
Is the real signal the slope of the RAW YIELD (DGS1), not the slope of the SPREAD (DGS1-DFF)?

The user's insight: they'd been treating "spread falling" as "yield falling", assuming DFF ~ flat
so spread tracks yield. But in a MEASURED hiking cycle DFF is NOT flat — it ratchets up every
meeting — so the spread can fall purely because DFF rises UNDER a flat-or-rising yield. That
mechanical DFF drag is exactly what could make a healthy mid-cycle look like a rollover in
SPREAD space while the YIELD itself is doing nothing bearish.

So we DECOMPOSE, over the trailing window, at each key date:
    d(spread) = d(yield) - d(DFF)
and ask: at the 2005 false exit, was the spread's fall YIELD-driven (real: the market re-priced
the 1yr rate DOWN) or DFF-driven (mechanical: DFF climbed under a stable/rising yield)?

Then we TEST the yield slope as the gate across ALL cycles: does slope(DGS1) stay flat/up in
the 2005 wobble (correctly NOT a rollover) while going clearly negative at the real ends
(2018/2000/1994) — i.e. does it separate what the spread slope could not?

Yield slope via OLS over W=42 (the noise-derived window), bp/month, alongside the spread slope
for direct contrast.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "hiking_curve_strategy"))
import _paths  # noqa: F401
from signal_logic import _load_signal_data

W = 42
TD_PER_MO = 21

TRUE_LAST_HIKES = {
    "1994-95": "1995-02-01",
    "1999-00": "2000-05-16",
    "2004-06": "2006-06-29",
    "2015-18": "2018-12-20",
    "2022-23": "2023-07-27",
}
SUSPECT = "2005-09-02"


def ols_slope(x, W):
    t = np.arange(W, dtype=float); td = t - t.mean(); denom = np.dot(td, td)
    def _s(y):
        if np.isnan(y).any(): return np.nan
        return float(np.dot(td, y - y.mean()) / denom) * TD_PER_MO
    return x.rolling(W).apply(_s, raw=True)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    ds, _ = _load_signal_data(api_key, start="1976-01-01")
    # everything in bp: dgs1/dff are percent, *100 -> bp
    yld    = ds["dgs1"] * 100
    dff    = ds["dff"] * 100
    spread = ds["spread_1yr_bp"]

    slope_yld    = ols_slope(yld, W)
    slope_spread = ols_slope(spread, W)

    d63_y = yld.diff(63); d63_f = dff.diff(63); d63_s = spread.diff(63)
    d21_y = yld.diff(21); d21_f = dff.diff(21); d21_s = spread.diff(21)

    # === decomposition at each key date ======================================
    print("=" * 96)
    print("DECOMPOSITION  d(spread) = d(yield) - d(DFF)   over trailing 63td   (all bp)")
    print("  If a 'rollover' in spread is really DFF climbing under a flat/rising yield, the")
    print("  spread fall is MECHANICAL, not a real re-pricing. Real rollover = YIELD itself falls.")
    print("=" * 96)
    hdr = f"{'date':<26}{'d63_yield':>11}{'d63_DFF':>10}{'d63_spread':>12}{'slope_yld':>11}{'slope_spr':>11}"
    print(hdr)

    rows = [("2005-09-02  (FALSE exit)", SUSPECT)] + \
           [(f"{k}  (real end)", v) for k, v in TRUE_LAST_HIKES.items()]
    for label, d in rows:
        d = pd.Timestamp(d)
        idx = yld.index[yld.index <= d]
        if not len(idx):
            print(f"{label:<26}{'(no data)':>11}"); continue
        i = idx[-1]
        print(f"{label:<26}{d63_y[i]:>11.0f}{d63_f[i]:>10.0f}{d63_s[i]:>12.0f}"
              f"{slope_yld[i]:>11.1f}{slope_spread[i]:>11.1f}")

    # === weekly path around 2005 =============================================
    print("\n" + "=" * 96)
    print("2005 PATH — raw yield vs DFF vs spread (weekly, bp)")
    print("=" * 96)
    d = pd.Timestamp(SUSPECT)
    win = ds.index[(ds.index > d - pd.Timedelta(days=110)) & (ds.index <= d + pd.Timedelta(days=45))]
    wk = pd.DataFrame({"DGS1": yld, "DFF": dff, "spread": spread,
                       "slp_yld": slope_yld, "slp_spr": slope_spread}).reindex(win).resample("W").last()
    print(wk.to_string(float_format=lambda x: f"{x:.0f}"))

    # === the actual discrepancy, stated ======================================
    i = yld.index[yld.index <= d][-1]
    print("\n" + "=" * 96)
    print("THE DISCREPANCY AT 2005-09-02")
    print("=" * 96)
    print(f"  trailing 63td:  yield {d63_y[i]:+.0f} bp   DFF {d63_f[i]:+.0f} bp   spread {d63_s[i]:+.0f} bp")
    print(f"  -> the spread's {d63_s[i]:+.0f}bp 'fall' = yield {d63_y[i]:+.0f} MINUS DFF {d63_f[i]:+.0f}.")
    if d63_y[i] > 0:
        print(f"     The YIELD actually ROSE {d63_y[i]:+.0f}bp. The entire spread decline is the DFF")
        print(f"     ratchet (+{d63_f[i]:.0f}bp of hikes) outrunning a RISING yield. Nothing bearish in")
        print(f"     the yield at all -> the spread slope was reading a MECHANICAL artifact.")
    print(f"\n  yield slope (OLS W{W}):  {slope_yld[i]:+.1f} bp/mo   vs   spread slope: {slope_spread[i]:+.1f} bp/mo")


if __name__ == "__main__":
    main()
