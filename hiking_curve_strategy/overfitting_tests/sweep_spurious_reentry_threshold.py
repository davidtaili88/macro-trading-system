"""
OVERFITTING TEST — REENTRY_BLOCK_BP (the spurious-re-entry block threshold, live = 350bp).

Once the strategy has exited, it only re-arms the entry latch within the SAME hiking-cycle
span if the cycle is still YOUNG: cum bp hiked since cycle start < REENTRY_BLOCK_BP. Past
this level the cycle is winding down, so re-entering risks a shock whipsaw with little edge
left (e.g. re-arming Mar-2023 straight into the SVB spread collapse: the spurious 2022-23
second episode). It is a RISK-ASYMMETRY gate — it only blocks RE-arming, never the first
entry — so it can't cap a live winner.

350bp is a DISCRETIONARY value, but UNLIKE the ratio-exit threshold it CAN be given clean
quantitative backing, because it sits on a hindsight-free separation structure:
  - legitimate re-entries (2004-06 / 2015-18 second legs) re-armed at <= ~225bp
  - the one spurious re-entry (2023 SVB) re-armed at ~425bp
  - 350 sits in the (225, 425) gap
"cum bp hiked since cycle start" is a running quantity the strategy already knows in real
time — no oracle, no last-hike distance — so this is a real gap, not laundered hindsight.

WHAT THIS TEST CHECKS (and its ceiling)
---------------------------------------
The comment ASSERTS "any value in the ~200bp gap behaves identically on our data". This
test VERIFIES that: sweep REENTRY_BLOCK_BP across 150..500bp and check that pooled P&L and
cycle count are FLAT across the (225, 425) gap and only change at the edges (below ~225 you
start blocking legitimate second legs; above ~425 you stop blocking the 2023 whipsaw).

Ceiling: this rests on n=3 re-entry events (really n=1 on the spurious side), so we do NOT
compute a sigma/MAD noise floor — n=1 can't support one and that would be fake precision.
The honest output is tier-2 "gap-verified discretionary": a round value in a gap the sweep
CONFIRMS is flat, not a derived number. Firm-up = more cycles, not a cleverer statistic.

HOW IT WORKS
------------
Monkeypatch signal_logic.REENTRY_BLOCK_BP and call the REAL detect_signal each time, so we
test true production logic. detect_signal reads the constant as a module global inside its
loop, so patching the attribute before each call is exact.

Run:  cd hiking_curve_strategy && PYTHONPATH=. python3 overfitting_tests/sweep_spurious_reentry_threshold.py
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


# swept values of REENTRY_BLOCK_BP, in bp (cum bp hiked since cycle start past which we
# don't re-arm). 25bp steps across 150..500, bracketing the (225, 425) gap with room on
# both sides so both edges (block-legitimate-legs below, stop-blocking-2023 above) show.
SWEEP_VALUES = list(range(150, 501, 25))   # 150, 175, ... 500
LIVE_VALUE = 350              # the value currently in signal_logic.py
DATA_START = "1982-10-01"     # detect_signal's own default; earliest date all signal inputs exist

# --- plateau/cliff verdict knobs (see overfit_test_utils.py) ----------------------
# Outcome units are pooled P&L in percentage points, same as the other sweeps, so the same
# absolute-triviality floor applies: a jump < this in pooled P&L is economically nothing
# and is never a cliff (also keeps the robust-z well-behaved on flat stretches).
FLAT_FLOOR_PP = 0.10
# Examine jumps within +/- this many sweep steps of the live value. With 25bp steps, +/-8
# spans 200bp each side (150..500 from live 350) — enough to cover the whole (225,425) gap
# and both edges.
VERDICT_WINDOW = 8


def _run_one(value, api_key, ret_price):
    """Patch REENTRY_BLOCK_BP -> value, run the real detector, return a result dict."""
    original = signal_logic.REENTRY_BLOCK_BP
    try:
        signal_logic.REENTRY_BLOCK_BP = value
        signal = detect_signal(api_key, start=DATA_START)
        cycles = signal_to_cycles(signal)
    finally:
        signal_logic.REENTRY_BLOCK_BP = original   # always restore, even on error

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

    print(f"Sweeping REENTRY_BLOCK_BP over {SWEEP_VALUES[0]}..{SWEEP_VALUES[-1]}bp "
          f"(live value = {LIVE_VALUE})...\n")
    results = [_run_one(v, api_key, ret_price) for v in SWEEP_VALUES]

    # ---- 1) sweep table: cycle count, pooled P&L, exit-date fingerprint ----
    print("=" * 96)
    print("SWEEP TABLE  —  REENTRY_BLOCK_BP vs. strategy behavior (DGS2 price-only)")
    print("=" * 96)
    print(f"{'BLOCK_BP':>10} {'n_cyc':>6} {'pooled P&L %':>13}   exit dates")
    print("-" * 96)
    for r in results:
        marker = "  <-- LIVE" if r["value"] == LIVE_VALUE else ""
        exits = ", ".join(r["exit_dates"])
        print(f"{r['value']:>10} {r['n_cycles']:>6} {r['pooled_pnl']*100:>12.2f}%   {exits}{marker}")

    # ---- 2) per-cycle P&L across the sweep (columns keyed by ENTRY date, stable) ----
    print("\n" + "=" * 96)
    print("PER-CYCLE compounded payer P&L (%) across the sweep  (columns keyed by ENTRY date")
    print("so a cycle whose exit slides across a year boundary stays in ONE column)")
    print("=" * 96)
    all_entries = []
    for r in results:
        for c in r["cycles"]:
            e = str(c["first_hike"].date())
            if e not in all_entries:
                all_entries.append(e)

    def _entry_pnl_map(r):
        m = {}
        for c in r["cycles"]:
            e = str(c["first_hike"].date())
            m[e] = r["per_cycle"].get(c["label"], float("nan"))
        return m

    header = f"{'BLOCK_BP':>10} " + " ".join(f"{e[2:]:>11}" for e in all_entries)  # drop century
    print(header)
    print("-" * len(header))
    for r in results:
        m = _entry_pnl_map(r)
        cells = " ".join(f"{m.get(e, float('nan'))*100:>10.2f}%" for e in all_entries)
        marker = "  <-- LIVE" if r["value"] == LIVE_VALUE else ""
        print(f"{r['value']:>10} {cells}{marker}")

    # ---- 3) robust-z plateau / cliff verdict (see overfit_test_utils.py) ----
    live_idx = SWEEP_VALUES.index(LIVE_VALUE)
    pooled_pct = [r["pooled_pnl"] * 100 for r in results]   # outcome series: pooled P&L in pp
    verdict = classify_plateau(
        pooled_pct, live_idx,
        flat_floor=FLAT_FLOOR_PP,
        window=VERDICT_WINDOW,
    )

    pnl_spread_full = max(pooled_pct) - min(pooled_pct)
    jumps_full = local_jumps(pooled_pct)

    print("\n" + "=" * 96)
    print("VERDICT  —  robust-z (median + MAD) on local jumps in pooled P&L")
    print("=" * 96)
    print(f"Live value REENTRY_BLOCK_BP = {LIVE_VALUE}bp   "
          f"(sweep index {live_idx} of {len(SWEEP_VALUES)})")
    wlo, whi = verdict["win_lo_idx"], verdict["win_hi_idx"]
    print(f"Examined window: BLOCK_BP in [{SWEEP_VALUES[wlo]}, {SWEEP_VALUES[whi]}]bp "
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
          f"over {SWEEP_VALUES[0]}..{SWEEP_VALUES[-1]}bp")
    print(f"  full-sweep largest jump      : {jumps_full.max():.3f} pp")
    print()

    v = verdict["verdict"]
    print(f"  => {v}   ({verdict['reason']})")
    if v == "PLATEAU":
        print(f"     The step in/out of {LIVE_VALUE}bp is an ordinary-sized wiggle, not an outlier.")
        print(f"     This VERIFIES the comment's claim: the (225, 425) gap is genuinely flat, so 350")
        print(f"     is a GAP-VERIFIED discretionary choice — a round value in a confirmed-flat gap,")
        print(f"     not a needle threaded to P&L. Check the table: behavior should change only at")
        print(f"     the EDGES (below ~225 blocks legitimate second legs; above ~425 stops blocking")
        print(f"     the 2023 SVB whipsaw), and be flat in between.")
    elif v == "CLIFF":
        print(f"     Pooled P&L lurches within one step of {LIVE_VALUE}bp — the (225,425) gap is NOT")
        print(f"     as flat as the comment claims, so 350 is doing more than 'round value in a gap'.")
        print(f"     Read the edges off the table and reconsider; do NOT tune to a P&L peak (n=3).")
    else:  # NO_STABLE_REGION
        print(f"     Behavior lurches at most steps: no flat gap to sit in. The (225,425) separation")
        print(f"     is not robust on this data — report that honestly and firm up with more cycles.")
    print()
    print("NOTE: rests on n=3 re-entry events (n=1 spurious). A plateau here VERIFIES the gap is")
    print("flat on THIS data; it is not a derived value and not out-of-sample validation.")


if __name__ == "__main__":
    main()
