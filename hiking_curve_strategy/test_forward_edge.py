"""
Forward-implied vs. realized yield test.

Question: does the signal predict yield moves BEYOND what the forward curve
already priced in at entry, or does it only catch moves the market already
expected (in which case ZT futures P&L — carry-netted — should be flat even
though DGS2/SHY cash P&L looks profitable, since cash carry bleed is what
makes an "already priced in" move look free money on paper)?

Construction of the forward-implied yield:

  Using the unbiased expectations hypothesis, today's spot curve (DGS1, DGS2)
  implies a forward 1yr yield, 1yr from today:

      (1 + y2)^2 = (1 + y1) * (1 + f)
      f = (1 + y2)^2 / (1 + y1) - 1

  f is the market's expectation, AT ENTRY, of what the 1yr yield will be one
  year later. This is the cleanest forward constructible from the DGS1/DGS2
  stack already used elsewhere in this project (no OIS curve available).

  At each signal entry date we compute f. At each signal exit date (or +1yr
  from entry, whichever this cycle's holding period approximates) we read the
  REALIZED DGS1 yield. The "surprise" is:

      surprise_bp = realized_y1_at_horizon - forward_implied_f_at_entry

  A payer strategy only earns "free" P&L (beyond what carry already prices in)
  to the extent surprise_bp > 0 — i.e. yields ended up higher than the curve
  already expected at entry. If surprise_bp ~ 0 across cycles, the signal is
  only catching regime-correct-but-already-priced moves, and ZT P&L (carry
  netted into the basis) should be much flatter than DGS2 total_ret suggests.

Run: python test_forward_edge.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from signal_logic import detect_signal, signal_to_cycles
from utils.fred_utils import fetch_fred_dataframe


def implied_forward_1y1y(y1: pd.Series, y2: pd.Series) -> pd.Series:
    """
    Forward 1yr yield, 1yr from today, implied by spot DGS1/DGS2 (decimals).

        (1+y2)^2 = (1+y1)(1+f)  =>  f = (1+y2)^2/(1+y1) - 1
    """
    return (1 + y2) ** 2 / (1 + y1) - 1


def forward_vs_realized(cycles: list[dict], y1: pd.Series, y2: pd.Series) -> pd.DataFrame:
    """
    For each cycle: compare the forward-implied 1yr yield (1yr out from entry)
    against the REALIZED 1yr yield at entry + 1yr (or at exit, if the cycle is
    shorter than 1yr — clipped to available data).

    Also reports the naive "realized move" (what a DGS2-total_ret-style backtest
    effectively rewards) for direct comparison.
    """
    fwd = implied_forward_1y1y(y1, y2)
    idx = y1.index

    rows = []
    for c in cycles:
        entry = c["first_hike"]
        exit_ = c["last_hike"]

        # nearest available trading day at/after entry
        pos_entry = idx.searchsorted(entry)
        if pos_entry >= len(idx):
            continue
        entry_date = idx[pos_entry]

        target_1y = entry_date + pd.DateOffset(years=1)
        pos_1y = idx.searchsorted(target_1y)
        # clip to exit date or end of data, whichever binds first
        pos_exit = idx.searchsorted(exit_)
        horizon_pos = min(pos_1y, pos_exit, len(idx) - 1)
        if horizon_pos <= pos_entry:
            continue
        horizon_date = idx[horizon_pos]

        y1_entry_spot   = y1.loc[entry_date]
        y2_entry_spot   = y2.loc[entry_date]
        forward_implied = fwd.loc[entry_date]           # market's expectation of y1, 1yr out
        y1_realized     = y1.loc[horizon_date]           # what y1 actually was at that horizon

        surprise_bp        = (y1_realized - forward_implied) * 1e4
        naive_realized_move_bp = (y1_realized - y1_entry_spot) * 1e4

        rows.append({
            "cycle":              c["label"],
            "entry":              str(entry_date.date()),
            "horizon":            str(horizon_date.date()),
            "days_to_horizon":    horizon_pos - pos_entry,
            "y1_spot_entry_%":    round(y1_entry_spot * 100, 2),
            "y2_spot_entry_%":    round(y2_entry_spot * 100, 2),
            "fwd_implied_y1_%":   round(forward_implied * 100, 2),
            "y1_realized_%":      round(y1_realized * 100, 2),
            "naive_move_bp":      round(naive_realized_move_bp, 1),
            "surprise_bp":        round(surprise_bp, 1),
        })

    return pd.DataFrame(rows).set_index("cycle")


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")
    if not api_key:
        print("Set FRED_API_KEY in .env and re-run.")
        return

    print("Detecting signal (FRED spreads, from 1976)...")
    signal = detect_signal(api_key, start="1976-01-01")
    cycles = signal_to_cycles(signal)
    if not cycles:
        print("No signal episodes detected.")
        return

    print(f"  {len(cycles)} episode(s) detected.\n")

    print("Fetching DGS1/DGS2 spot curve...")
    curve = fetch_fred_dataframe(api_key, {"y1": "DGS1", "y2": "DGS2"}, start="1976-01-01")
    y1 = curve["y1"] / 100
    y2 = curve["y2"] / 100

    result = forward_vs_realized(cycles, y1, y2)

    print(f"\n{'='*100}")
    print("  Forward-implied vs. realized 1yr yield, per signal episode")
    print(f"{'='*100}")
    print(result.to_string())

    if not result.empty:
        avg_surprise = result["surprise_bp"].mean()
        avg_naive    = result["naive_move_bp"].mean()
        print(f"\n  Mean naive realized move:      {avg_naive:+.1f} bp  (what DGS2 total_ret-style backtest rewards)")
        print(f"  Mean surprise vs. forward:     {avg_surprise:+.1f} bp  (what ZT futures P&L can actually capture)")
        print()
        if avg_surprise <= 0:
            print("  --> Forward curve was, on average, AT LEAST AS GOOD a predictor as the signal's entry timing.")
            print("      Expect ZT futures P&L to be flat-to-negative even where DGS2 'total_ret' looks profitable.")
        elif avg_surprise < avg_naive * 0.5:
            print("  --> Signal captures some surprise, but more than half the naive move was already priced in by the forward curve.")
            print("      Expect ZT futures P&L to be materially weaker than DGS2 'total_ret' suggests.")
        else:
            print("  --> Signal captures a surprise component close to the full realized move.")
            print("      ZT futures P&L should track DGS2 'total_ret' reasonably well — edge looks real, not just carry mislabeled as signal.")


if __name__ == "__main__":
    main()
