"""
Signal-driven backtest for the hiking-cycle payer — INSTRUMENT A: a HELD, AGING 2Y note.

CONTRAST WITH signal_trade_dgs.py (instrument B, constant-maturity):
  B re-strikes the 2Y point EVERY DAY: duration pinned at ~2Y, coupon = today's 2y,
  carry = (2y_today - ff)/252. No aging, no locked coupon.

  A (this file) SHORTS ONE SPECIFIC NOTE at each signal entry and HOLDS it, letting it
  AGE down the curve until the signal exits. This is the construction David's carry
  intuition points at: the coupon is LOCKED at entry, so as the Fed hikes and funding
  climbs past that locked coupon, the SHORT earns positive carry (funding_earned -
  coupon_paid grows). The cost is that the note DRIFTS off the 2Y point the signal
  targets: 6 months in it is a 1.5Y note (1.5Y duration/yield), so the price/duration
  edge is earned on a shrinking, drifting exposure. This file measures whether A's
  better carry outweighs that drift + the roll drag a short accumulates sliding DOWN an
  upward curve.

SAME as B (deliberately, so the comparison is clean):
  - Signal: detect_signal / signal_to_cycles (identical entry/exit DATES).
  - P&L decomposition: price + carry_fund + carry_roll, same three buckets, decimal
    return space, short = negate. The MECHANICS of each bucket differ (aging note vs
    constant maturity) but the accounting is the same so the two are comparable.

HELD-NOTE P&L MECHANICS (per episode, short position):
  At entry t0 we short a par 2Y note. Its coupon is fixed at the entry 2y yield y0.
  On each subsequent day t with age tau = (t - t0) in years, remaining maturity m = 2 - tau:
    - y_m(t) = the note's OWN yield today = curve interpolated at remaining maturity m
               (across DGS3MO/6MO/1/2). As the note ages m shrinks, so we read a
               shorter, lower point of an upward curve.
    - D_m    = modified duration at remaining maturity m (shrinks toward 0 as m->0).
    - price_ret (long) = -D_m * d(y_m)         : MTM of the note's own yield move
    - carry_fund(long) = (y0 - ff_t)/252       : LOCKED coupon y0 received, funding paid.
                         KEY DIFFERENCE vs B: y0 is frozen at entry, NOT re-struck.
    - carry_roll(long) = -D_m * (y_m - y_m_prevday_at_older_age)... implemented as the
                         pull-to-curve as m shrinks: +D_m * (dy/dm) * (dm) where the note
                         rolls DOWN the curve. We compute it directly from the change in
                         the interpolated yield attributable to AGING (holding the curve
                         fixed), separated from the market move above.
  short position = negate all three.

Data: DGS3MO, DGS6MO, DGS1, DGS2, DFF (FRED). Signal inputs unchanged (1982+).

Run:  cd hiking_curve_strategy && PYTHONPATH=. python3 strategies/signal_trade_held_note.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths  # noqa: F401 — registers core/, benchmark/, utils/ on sys.path
from data          import fetch_dgs2_full_pnl, fetch_carryless_dgs2_returns, _modified_duration
from backtest      import annualised_stats, calc_strat_ret, cycle_pnl
from plot          import equity_curve
from benchmark     import POST2003_FED_HIKE_CYCLES
from signal_logic  import detect_signal, signal_to_cycles
from utils.fred_utils import fetch_fred_dataframe

DATA_START = "1982-10-01"   # signal-input floor (DFEDTAR 1982-09-27); same as B
TRADE_DAYS = 252

# curve tenors (years) we interpolate the aging note's yield across, and their FRED ids.
# 0.0 anchored to fed funds (overnight) so a note aging toward maturity pulls to the
# front of the curve smoothly rather than falling off the shortest CMT point.
CURVE_TENORS = [0.25, 0.5, 1.0, 2.0]
CURVE_IDS    = {0.25: "DGS3MO", 0.5: "DGS6MO", 1.0: "DGS1", 2.0: "DGS2"}


def _load_curve(api_key: str) -> pd.DataFrame:
    """Daily yield curve (decimal) across CURVE_TENORS plus fed funds at tenor 0."""
    cols = {f"y{t}": CURVE_IDS[t] for t in CURVE_TENORS}
    cols["ff"] = "DFF"
    raw = fetch_fred_dataframe(api_key, cols, DATA_START, fill_method="ffill")
    return raw.dropna() / 100.0


def _interp_yield(curve_row: pd.Series, m: float) -> float:
    """Interpolate the yield at remaining maturity m (years) from one day's curve row.

    Piecewise-linear across [0 (fed funds), 0.25, 0.5, 1, 2]. m is clamped to [0, 2];
    below 0.25 we interpolate between fed funds (tenor 0) and the 3mo point."""
    xs = [0.0] + CURVE_TENORS
    ys = [curve_row["ff"]] + [curve_row[f"y{t}"] for t in CURVE_TENORS]
    m = min(max(m, 0.0), CURVE_TENORS[-1])
    return float(np.interp(m, xs, ys))


def held_note_pnl(api_key: str, cycles: list[dict], roll_months: int | None = None) -> pd.DataFrame:
    """Per-episode held-note (instrument A) P&L, returned as ONE daily decimal-return
    series aligned to the signal calendar (0 on non-position days), plus its three
    components, so it plugs into the same reporting as B.

    roll_months: re-strike the note back to a fresh 2Y every this-many months. None =
    never roll (A-pure: one note aged over the whole episode → decays to a bill). A finite
    value keeps the note's remaining maturity in [2 - roll_months/12, 2], so duration
    stays near the 2Y point the signal targets while STILL locking the coupon WITHIN each
    roll window (so the carry-handoff can still operate, just reset periodically).

    For each episode we walk the note day by day, re-anchoring (relocking coupon, resetting
    age to 2Y) at each roll. Because A's return depends on the current anchor (locked coupon
    + age since last roll), it MUST be built per-episode, not as a global series like B."""
    curve = _load_curve(api_key)
    idx = curve.index

    price = pd.Series(0.0, index=idx)
    cfund = pd.Series(0.0, index=idx)
    croll = pd.Series(0.0, index=idx)
    signal = pd.Series(0, index=idx)
    roll_days = None if roll_months is None else roll_months * 30  # calendar-day roll trigger

    for c in cycles:
        t0, t1 = c["first_hike"], c["last_hike"]
        window = idx[(idx >= t0) & (idx <= t1)]
        if len(window) < 2:
            continue

        anchor_date = window[0]                          # entry date of the CURRENT note
        y0 = _interp_yield(curve.loc[anchor_date], 2.0)  # LOCKED coupon of the current note

        prev_ym = None
        prev_date = None
        for d in window:
            # roll: if the current note has been held roll_days, re-strike to a fresh 2Y
            if roll_days is not None and (d - anchor_date).days >= roll_days:
                anchor_date = d
                y0 = _interp_yield(curve.loc[d], 2.0)
                prev_ym = None                            # fresh note: no prior-day yield to diff yet

            tau = (d - anchor_date).days / 365.25         # age of CURRENT note in years
            m   = 2.0 - tau                                # remaining maturity
            row = curve.loc[d]
            ym  = _interp_yield(row, m)                    # note's own yield today at remaining m
            Dm  = float(_modified_duration(pd.Series([max(ym, 1e-4)]), N=max(m, 1e-4), k=2).iloc[0])

            signal[d] = -1                                 # short (payer)
            # locked-coupon carry: long earns (y0 - ff); short negates below
            cfund[d] = (y0 - row["ff"]) / TRADE_DAYS

            if prev_ym is not None:
                # split daily yield change into market move + aging(roll) slide:
                # yesterday's curve re-read at TODAY's shorter maturity => pure aging.
                ym_roll = _interp_yield(curve.loc[prev_date], m)
                d_roll   = ym_roll - prev_ym
                d_total  = ym - prev_ym
                d_market = d_total - d_roll
                price[d] = -Dm * d_market
                croll[d] = -Dm * d_roll
            prev_ym, prev_date = ym, d

    # SHORT position: negate all three components
    price, cfund, croll = -price, -cfund, -croll
    total = price + cfund + croll
    return pd.DataFrame({
        "signal":     signal,
        "price_ret":  price,
        "carry_fund": cfund,
        "carry_roll": croll,
        "strat_ret":  total,
        "cum_equity": (1 + total.fillna(0)).cumprod(),
    })


def _summary(name: str, df: pd.DataFrame, cycles: list[dict]) -> None:
    """Print annualised stats + per-cycle/pooled P&L for a held-note df."""
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    # annualised_stats wants a buy-hold 'bond_ret' column: the LONG-note return is the
    # negative of our short strat_ret (signal=-1), so long = -strat_ret on active days.
    stats = annualised_stats(df.assign(bond_ret=-df["strat_ret"]))
    print(stats.to_string())
    print("\nPer-cycle compounded payer P&L (held note):")
    for c in cycles:
        mask = (df.index >= c["first_hike"]) & (df.index <= c["last_hike"]) & (df["signal"] == -1)
        r = df.loc[mask, "strat_ret"].dropna()
        print(f"  {c['label']:<40}  {((1 + r).prod() - 1) * 100:+.2f}%")
    all_mask = df["signal"] == -1
    pooled = (1 + df.loc[all_mask, "strat_ret"].dropna()).prod() - 1
    print(f"  {'pooled (all cycles)':<40}  {pooled * 100:+.2f}%")
    # component attribution (pooled), so we can SEE where A differs from B
    for comp in ["price_ret", "carry_fund", "carry_roll"]:
        s = df.loc[all_mask, comp].dropna()
        print(f"    └ {comp:<12} pooled contribution  {((1 + s).prod() - 1) * 100:+.2f}%")


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    print("Detecting signal (FRED spreads, from 1982)...")
    signal = detect_signal(api_key, start=DATA_START)
    cycles = signal_to_cycles(signal)
    print(f"  {len(cycles)} episode(s)")

    # ---- A: held aging note ----
    print("\nBuilding HELD-NOTE (instrument A) P&L (aging note, locked coupon)...")
    df_A = held_note_pnl(api_key, cycles)
    _summary("A — HELD AGING NOTE (locked coupon, drifts off 2Y)", df_A, cycles)

    # ---- B: constant maturity, for side-by-side ----
    print("\nBuilding CONSTANT-MATURITY (instrument B) P&L for comparison...")
    pnl_full = fetch_dgs2_full_pnl(api_key, start=DATA_START)
    ret_B = pnl_full["total_ret"].rename("ret")
    df_B = calc_strat_ret(ret_B, cycles)
    per_B, pooled_B = cycle_pnl(ret_B, cycles)
    print(f"\n{'='*55}\n  B — CONSTANT MATURITY (re-struck daily) [reference]\n{'='*55}")
    print("Per-cycle compounded payer P&L:")
    for c in cycles:
        print(f"  {c['label']:<40}  {per_B.get(c['label'], float('nan'))*100:+.2f}%")
    print(f"  {'pooled (all cycles)':<40}  {pooled_B*100:+.2f}%")

    print("\nGenerating A-vs-B equity comparison...")
    _plot_A_vs_B(df_A, df_B, cycles)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass


def _plot_A_vs_B(df_A: pd.DataFrame, df_B: pd.DataFrame, cycles: list[dict]) -> None:
    """Equity curves of A (held note) vs B (constant maturity) on the SAME axes, so the
    instrument difference is visible. Both are payer strategies on the identical signal;
    the only difference is the P&L construction."""
    # rebuild each cumulative equity restricted to active days (flat off-signal)
    eqA = (1 + df_A["strat_ret"].fillna(0)).cumprod()
    eqB = (1 + df_B["strat_ret"].fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(eqA.index, eqA, color="#c0392b", lw=1.6, label="A — held aging note (locked coupon)")
    ax.plot(eqB.index, eqB, color="#3b6ea5", lw=1.6, label="B — constant maturity (re-struck daily)")
    for c in cycles:
        ax.axvspan(c["first_hike"], c["last_hike"], color="0.5", alpha=0.10)
    ax.set_ylabel("Growth of $1 (rebased at plot start)")
    ax.set_title("Payer strategy — Instrument A (held note) vs B (constant maturity)\n"
                 "same signal; difference is purely the P&L construction")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()


if __name__ == "__main__":
    main()
