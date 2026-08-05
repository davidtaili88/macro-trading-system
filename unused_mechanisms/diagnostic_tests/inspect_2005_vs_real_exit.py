"""
Context check: at the 2005-09-02 exit, HOW MUCH were yields/spreads actually falling,
and does that motion look like a GENUINE end-of-cycle rollover (2018-12 / 2000-05) or not?

Prints, around each exit-region date, the raw levels and the trailing moves of:
  - DGS1  (1yr Treasury yield)         — the traded tenor's yield
  - DFF   (effective fed funds)        — what's already been hiked to
  - spread_1yr = DGS1 - DFF (bp)       — hiking STILL priced
  - trailing 21/63td change in DGS1 and in spread_1yr

so we can eyeball: is the 2005 spread decline a big directional move (like a real
rollover) or a shallow wobble that only LOOKS like an exit through the ratio denominator?
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "hiking_curve_strategy"))
import _paths  # noqa: F401
from signal_logic import _load_signal_data

EVENTS = {
    "2005-09-02  (STRATEGY EXIT — the suspect one)": "2005-09-02",
    "2000-05-16  (real end: last hike 2000)":        "2000-05-16",
    "2018-12-20  (real end: last hike 2018)":        "2018-12-20",
}


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    ds, _ = _load_signal_data(api_key, start="1976-01-01")
    dgs1 = ds["dgs1"] * 100          # -> bp-ish (percent points *100 = bp of yield)
    dff  = ds["dff"] * 100
    sp   = ds["spread_1yr_bp"]

    d21_y, d63_y = dgs1.diff(21), dgs1.diff(63)
    d21_s, d63_s = sp.diff(21), sp.diff(63)

    for title, d in EVENTS.items():
        d = pd.Timestamp(d)
        print("\n" + "=" * 78)
        print(f"  {title}")
        print("=" * 78)
        # show the ~3 months leading in, weekly
        win = ds.index[(ds.index > d - pd.Timedelta(days=100)) & (ds.index <= d + pd.Timedelta(days=10))]
        wk = pd.DataFrame({
            "DGS1(bp)":   dgs1.reindex(win),
            "DFF(bp)":    dff.reindex(win),
            "spread(bp)": sp.reindex(win),
        }).resample("W").last()
        print(wk.to_string(float_format=lambda x: f"{x:.0f}"))
        # the moves AT the event day
        idx = ds.index[ds.index <= d][-1]
        print(f"\n  AT {idx.date()}:")
        print(f"    DGS1 level         : {dgs1[idx]:.0f} bp")
        print(f"    DFF  level         : {dff[idx]:.0f} bp")
        print(f"    spread_1yr         : {sp[idx]:+.0f} bp   (hiking still priced)")
        print(f"    DGS1  trailing 21td: {d21_y[idx]:+.0f} bp    63td: {d63_y[idx]:+.0f} bp")
        print(f"    spread trailing 21td: {d21_s[idx]:+.0f} bp    63td: {d63_s[idx]:+.0f} bp")

    print("\n" + "-" * 78)
    print("Compare the trailing 63td DGS1 move and spread level across the three:")
    print("  a REAL rollover = yields falling hard (big negative 63td DGS1) AND spread already")
    print("  near/through zero. If 2005 shows only a shallow yield move with spread still clearly")
    print("  positive, the exit was denominator-driven, not a genuine rollover.")


if __name__ == "__main__":
    main()
