"""
OVERFITTING TEST — RATIO_EXIT_THRESHOLD (the maturity-ratio exit level, live = 0.10).

The ratio exit fires when  smoothed_spread_1yr / cum_bp_hiked_since_cycle_start  falls
below RATIO_EXIT_THRESHOLD: "hiking still priced" has shrunk to a small fraction of
"hiking already delivered", so the cycle is mostly behind us and we leave.

RESULT: PLATEAU at 0.10 — the local step in/out of 0.10 is an ordinary-sized
wiggle in pooled P&L, not an outlier. The raw sweep table shows WHY it's still not a
totally free choice: there is a flat CORRIDOR ~[0.06, 0.13] holding 8 clean cycles, but
BELOW ~0.06 the exit fires too weakly and the 2017-18 leg is lost (7 cycles), and ABOVE
~0.13 it false-fires in mid-cycle pauses, fragmenting cycles (9->15) and pushing the
2015-18 leg negative. So the CORRIDOR EDGES are load-bearing; 0.10 is a round value
comfortably inside. See the RATIO_EXIT_THRESHOLD comment in signal_logic.py.

WHY WE DON'T TRY TO "DERIVE" 0.10 FURTHER
-----------------------------------------
The ratio exit is a PRIMARY driver, so unlike the near-inert ROC gate you might expect a
value you could pin precisely. You can't — and this is not a gap in the tooling. The cost
of exiting too LATE is only definable relative to the last hike, which is UNKNOWABLE in
real time; any cost-crossing / "distance-to-ideal-exit" analysis must anchor "ideal" to
the oracle last-hike dates, which launders hindsight into the number and overfits n<=6
cycles. So there is no honest hindsight-free derivation of the exact value. The correct
conclusion is the plateau + corridor evidence above, documented as a disclosed
discretionary choice (the REENTRY_BLOCK_BP template). The real firm-up is MORE cycles
(pre-1990 FRED history), not a cleverer statistic on these six.

HOW IT WORKS
------------
Monkeypatch signal_logic.RATIO_EXIT_THRESHOLD and call the REAL detect_signal each time,
so we test true production logic. detect_signal reads the constant as a module global
inside its loop, so patching the attribute before each call is exact.

Run:  cd hiking_curve_strategy && PYTHONPATH=. python3 overfitting_tests/sweep_ratio_exit_threshold.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# make the package modules importable whether run from repo root or here
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))            # this dir, for overfit_test_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths  # noqa: F401 — registers core/, benchmark/, utils/ on sys.path

import signal_logic
from signal_logic import detect_signal, signal_to_cycles
from data import fetch_carryless_dgs2_returns
from backtest import calc_strat_ret, cycle_pnl
from overfit_test_utils import classify_plateau, local_jumps


# swept values of RATIO_EXIT_THRESHOLD (dimensionless: smoothed spread / cum bp hiked).
# 0.01 steps from just above zero to 0.30, bracketing the live 0.10 with plenty of room
# on both sides to see where any plateau ends. Rounded to avoid float display noise.
SWEEP_VALUES = [round(0.02 + 0.01 * i, 2) for i in range(29)]   # 0.02 .. 0.30
LIVE_VALUE = 0.10             # the value currently in signal_logic.py
DATA_START = "1982-10-01"     # detect_signal's own default; earliest date all signal inputs exist

# --- plateau/cliff verdict knobs (see overfit_test_utils.py) ----------------------
# Outcome units are pooled P&L in percentage points, same as the ROC-gate sweep, so the
# same absolute-triviality floor applies: a jump < this in pooled P&L is economically
# nothing and is never a cliff (also keeps the robust-z well-behaved on flat stretches).
FLAT_FLOOR_PP = 0.10
# Examine jumps within +/- this many sweep steps of the live value, so the robust stats
# reflect the neighbourhood the strategy actually uses rather than a distant regime (e.g.
# very high thresholds where the ratio exit fires almost immediately every cycle).
VERDICT_WINDOW = 8


def _run_one(value, api_key, ret_price):
    """Patch RATIO_EXIT_THRESHOLD -> value, run the real detector, return a result dict."""
    original = signal_logic.RATIO_EXIT_THRESHOLD
    try:
        signal_logic.RATIO_EXIT_THRESHOLD = value
        signal = detect_signal(api_key, start=DATA_START)
        cycles = signal_to_cycles(signal)
    finally:
        signal_logic.RATIO_EXIT_THRESHOLD = original   # always restore, even on error

    per_cycle, pooled = cycle_pnl(ret_price, cycles)
    exit_dates = tuple(str(c["last_hike"].date()) for c in cycles)
    return {
        "value":      value,
        "n_cycles":   len(cycles),
        "exit_dates": exit_dates,
        "pooled_pnl": pooled,
        "per_cycle":  per_cycle,
        "cycles":     cycles,
    }


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

    print("Fetching DGS2 price-only return series (FRED, from 1982)...")
    # canonical carryless source (DGS2-only calendar); this sweep uses price-only
    # returns only, so it never needs the full pnl decomposition.
    ret_price = fetch_carryless_dgs2_returns(api_key, start=DATA_START)

    print(f"Sweeping RATIO_EXIT_THRESHOLD over {SWEEP_VALUES[0]}..{SWEEP_VALUES[-1]} "
          f"(live value = {LIVE_VALUE})...\n")
    
    # This is a list of dictionaries
    results = [_run_one(v, api_key, ret_price) for v in SWEEP_VALUES]

    # ---- 1) sweep table: cycle count, pooled P&L, exit-date fingerprint ----
    print("=" * 96)
    print("SWEEP TABLE  —  RATIO_EXIT_THRESHOLD vs. strategy behavior (DGS2 price-only)")
    print("=" * 96)
    print(f"{'THRESHOLD':>10} {'n_cyc':>6} {'pooled P&L %':>13}   exit dates")
    print("-" * 96)
    for r in results:
        marker = "  <-- LIVE" if r["value"] == LIVE_VALUE else ""
        exits = ", ".join(r["exit_dates"])
        print(f"{r['value']:>10.2f} {r['n_cycles']:>6} {r['pooled_pnl']*100:>12.2f}%   {exits}{marker}")

    # ---- 2) per-cycle P&L across the sweep (does any single cycle swing?) ----
    print("\n" + "=" * 96)
    print("PER-CYCLE compounded payer P&L (%) across the sweep  (columns keyed by ENTRY date")
    print("so a cycle whose exit slides across a year boundary stays in ONE column)")
    print("=" * 96)
    # key columns on entry date (stable) rather than the year-embedding label, so an exit
    # sliding across a calendar year does not split one cycle into two ghost columns.
    all_entries = []
    for r in results:
        for c in r["cycles"]:
            e = str(c["first_hike"].date())
            if e not in all_entries:
                all_entries.append(e)
    # per-run map: entry-date -> that cycle's P&L
    def _entry_pnl_map(r):
        m = {}
        for c in r["cycles"]:
            e = str(c["first_hike"].date())
            m[e] = r["per_cycle"].get(c["label"], float("nan"))
        return m

    header = f"{'THRESHOLD':>10} " + " ".join(f"{e[2:]:>11}" for e in all_entries)  # drop '19'/'20'
    print(header)
    print("-" * len(header))
    for r in results:
        m = _entry_pnl_map(r)
        cells = " ".join(f"{m.get(e, float('nan'))*100:>10.2f}%" for e in all_entries)
        marker = "  <-- LIVE" if r["value"] == LIVE_VALUE else ""
        print(f"{r['value']:>10.2f} {cells}{marker}")

    # ---- 3) robust-z plateau / cliff verdict (see overfit_test_utils.py) ----
    live_idx = SWEEP_VALUES.index(LIVE_VALUE)
    pooled_pct = [r["pooled_pnl"] * 100 for r in results]   # outcome series: pooled P&L in pp
    verdict = classify_plateau(
        pooled_pct, live_idx,
        flat_floor=FLAT_FLOOR_PP,
        window=VERDICT_WINDOW,
        pooled_pnl=pooled_pct[live_idx],   # enables the absolute material backstop
    )

    pnl_spread_full = max(pooled_pct) - min(pooled_pct)
    jumps_full = local_jumps(pooled_pct)

    print("\n" + "=" * 96)
    print("VERDICT  —  robust-z (median + MAD) on local jumps in pooled P&L")
    print("=" * 96)
    print(f"Live value RATIO_EXIT_THRESHOLD = {LIVE_VALUE}   "
          f"(sweep index {live_idx} of {len(SWEEP_VALUES)})")
    wlo, whi = verdict["win_lo_idx"], verdict["win_hi_idx"]
    print(f"Examined window: THRESHOLD in [{SWEEP_VALUES[wlo]}, {SWEEP_VALUES[whi]}] "
          f"(+/-{VERDICT_WINDOW} steps around live)")
    print()
    print(f"  jumps adjacent to live     : {[round(j, 3) for j in verdict['adj_jumps']]} pp")
    print(f"  their robust-z scores      : {[round(z, 2) for z in verdict['adj_z']]}   "
          f"(cutoff {verdict['z_cutoff']})")
    print(f"  median jump (in window)    : {verdict['median_jump']:.3f} pp")
    print(f"  max jump    (in window)    : {verdict['max_jump']:.3f} pp   "
          f"(flat-floor {FLAT_FLOOR_PP} pp)")
    print(f"  robust sigma (1.4826*MAD)  : {verdict['robust_sigma']:.4f} pp")
    print(f"  fraction of window jumps flagged as cliffs : {verdict['cliff_frac']:.0%}")
    print()
    print(f"  full-sweep pooled P&L spread : {pnl_spread_full:.2f} pp "
          f"over {SWEEP_VALUES[0]}..{SWEEP_VALUES[-1]}")
    print(f"  full-sweep largest jump      : {jumps_full.max():.3f} pp")
    print()

    v = verdict["verdict"]
    print(f"  => {v}   ({verdict['reason']})")
    if v == "PLATEAU":
        print(f"     The step in/out of {LIVE_VALUE} is an ordinary-sized wiggle, not an")
        print(f"     outlier — the exact value is not load-bearing on this data, so 'round")
        print(f"     value in a wide flat band' IS the justification. Document like REENTRY_BLOCK_BP.")
    elif v == "FLAG_FOR_REVIEW":
        print(f"     Pooled P&L lurches by an outlier-sized amount within one step of {LIVE_VALUE},")
        print(f"     so the exact value IS load-bearing and cannot ride on a plateau alibi. Note the")
        print(f"     value still cannot be DERIVED hindsight-free (late-exit cost needs the unknowable")
        print(f"     last-hike). The honest response is NOT to tune 0.10 to a P&L peak (overfits n<=6)")
        print(f"     but to read the CORRIDOR EDGES off the sweep table (where cycle count / a cycle's")
        print(f"     sign changes) and document 0.10 as a disclosed choice inside them, pending more cycles.")
    else:  # NO_STABLE_REGION
        print(f"     A large fraction of steps are cliff-sized: the ratio exit lurches almost")
        print(f"     everywhere, so there is no stable region to sit in. That itself is the finding —")
        print(f"     on this data the exit rule is fragile to its threshold; report that honestly and")
        print(f"     firm it up with MORE cycles rather than picking any single value.")
    print()
    print("NOTE: a plateau here means robustness to THIS parameter on THIS data (from 1982),")
    print("NOT out-of-sample validation of the exit rule itself.")


if __name__ == "__main__":
    main()
