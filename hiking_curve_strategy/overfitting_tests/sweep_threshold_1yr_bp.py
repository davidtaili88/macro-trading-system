"""
OVERFITTING TEST — THRESHOLD_1YR_BP (the 1-year entry spread bar, live = 50bp).

Entry requires the smoothed 1yr spread (DGS1 - DFF) to exceed THRESHOLD_1YR_BP: the
market must be broadly pricing a hiking cycle (~2 hikes over the coming year) before
we arm the payer. Paired with THRESHOLD_3MO_BP (near-term hike priced) — BOTH must
hold to enter. This sweep isolates the 1yr bar, holding 3mo at its live value.

50bp is a round entry value. This test asks the SAME question the ratio-exit and
ROC-gate sweeps ask: is 50 sitting on a FLAT PLATEAU (the exact number is a don't-care
-> "round value in a wide band" is itself the justification) or on a CLIFF (pooled P&L
lurches as you nudge the bar one step -> the value is load-bearing)?

WHAT MOVES AS THE BAR MOVES
---------------------------
Unlike the ratio EXIT (which trims the tail of a held cycle), THRESHOLD_1YR_BP is an
ENTRY gate, so its two failure modes are:
  - TOO LOW: the latch arms on ambiguous pricing before the cycle is real, adding
    early false-start episodes (cycles that never delivered a hike, later flushed by
    the false-promise exit — but still dragging entry P&L). Watch n_cycles rise and
    early episodes appear.
  - TOO HIGH: the bar out-runs how much the market ever prices 1yr out on the thinner
    cycles, so genuine cycles are entered LATE (worse entry level) or missed entirely.
    Watch n_cycles fall and a cycle's P&L collapse toward zero as it drops out.
The exit-date fingerprint and per-cycle table below make both visible; the plateau
verdict scores whether 50 sits between them on flat ground.

HOW IT WORKS
------------
Monkeypatch signal_logic.THRESHOLD_1YR_BP and call the REAL detect_signal each time, so
we test true production logic. detect_signal reads the constant as a module global inside
its entry-latch condition, so patching the attribute before each call is exact. 3mo bar is
left at its live value throughout (single-parameter sweep).

Run:  cd hiking_curve_strategy && PYTHONPATH=. python3 overfitting_tests/sweep_threshold_1yr_bp.py
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
from backtest import cycle_pnl
from overfit_test_utils import classify_plateau, local_jumps


# swept values of THRESHOLD_1YR_BP (basis points on the smoothed DGS1-DFF entry spread).
# 5bp steps from 10 to 100bp, bracketing the live 50bp with room on both sides to see
# where any plateau ends: below ~30bp the bar starts admitting ambiguous pre-cycle
# pricing, above ~70bp it starts out-running the thinner cycles' 1yr pricing.
SWEEP_VALUES = list(range(10, 101, 5))   # 10, 15, ..., 100
LIVE_VALUE = 50               # the value currently in signal_logic.py
DATA_START = "1982-10-01"     # earliest date all signal inputs exist (same as the other entry/exit sweeps)

# --- plateau/cliff verdict knobs (see overfit_test_utils.py) ----------------------
# Outcome units are pooled P&L in percentage points, same as the ratio-exit sweep, so
# the same absolute-triviality floor applies: a jump < this in pooled P&L is economically
# nothing and is never a cliff (also keeps the robust-z well-behaved on flat stretches).
FLAT_FLOOR_PP = 0.10
# Examine jumps within +/- this many sweep steps of the live value, so the robust stats
# reflect the neighbourhood the strategy actually uses rather than a distant regime (very
# low bars that arm on noise, or very high bars that never arm on the thin cycles).
VERDICT_WINDOW = 8


def _run_one(value, api_key, ret_price):
    """Patch THRESHOLD_1YR_BP -> value, run the real detector, return a result dict."""
    original = signal_logic.THRESHOLD_1YR_BP
    try:
        signal_logic.THRESHOLD_1YR_BP = value
        signal = detect_signal(api_key, start=DATA_START)
        cycles = signal_to_cycles(signal)
    finally:
        signal_logic.THRESHOLD_1YR_BP = original   # always restore, even on error

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

    print(f"Sweeping THRESHOLD_1YR_BP over {SWEEP_VALUES[0]}..{SWEEP_VALUES[-1]}bp "
          f"(live value = {LIVE_VALUE}bp, 3mo bar held at live)...\n")
    results = [_run_one(v, api_key, ret_price) for v in SWEEP_VALUES]

    # ---- 1) sweep table: cycle count, pooled P&L, exit-date fingerprint ----
    print("=" * 96)
    print("SWEEP TABLE  —  THRESHOLD_1YR_BP vs. strategy behavior (DGS2 price-only)")
    print("=" * 96)
    print(f"{'1YR_BP':>10} {'n_cyc':>6} {'pooled P&L %':>13}   exit dates")
    print("-" * 96)
    for r in results:
        marker = "  <-- LIVE" if r["value"] == LIVE_VALUE else ""
        exits = ", ".join(r["exit_dates"])
        print(f"{r['value']:>10d} {r['n_cycles']:>6} {r['pooled_pnl']*100:>12.2f}%   {exits}{marker}")

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

    header = f"{'1YR_BP':>10} " + " ".join(f"{e[2:]:>11}" for e in all_entries)  # drop '19'/'20'
    print(header)
    print("-" * len(header))
    for r in results:
        m = _entry_pnl_map(r)
        cells = " ".join(f"{m.get(e, float('nan'))*100:>10.2f}%" for e in all_entries)
        marker = "  <-- LIVE" if r["value"] == LIVE_VALUE else ""
        print(f"{r['value']:>10d} {cells}{marker}")

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
    print(f"Live value THRESHOLD_1YR_BP = {LIVE_VALUE}bp   "
          f"(sweep index {live_idx} of {len(SWEEP_VALUES)})")
    wlo, whi = verdict["win_lo_idx"], verdict["win_hi_idx"]
    print(f"Examined window: 1YR_BP in [{SWEEP_VALUES[wlo]}, {SWEEP_VALUES[whi]}]bp "
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
        print(f"     The step in/out of {LIVE_VALUE}bp is an ordinary-sized wiggle, not an")
        print(f"     outlier — the exact bar is not load-bearing on this data, so 'round value")
        print(f"     in a wide flat band' IS the justification. Document like REENTRY_BLOCK_BP.")
    elif v == "CLIFF":
        print(f"     Pooled P&L lurches by an outlier-sized amount within one step of {LIVE_VALUE}bp,")
        print(f"     so the exact entry bar IS load-bearing and cannot ride on a plateau alibi.")
        print(f"     Read the CORRIDOR EDGES off the sweep table (where n_cycles or a cycle's sign")
        print(f"     changes: too-low admits pre-cycle false starts, too-high misses thin cycles)")
        print(f"     and document {LIVE_VALUE}bp as a disclosed choice inside them, pending more cycles.")
    else:  # NO_STABLE_REGION
        print(f"     A large fraction of steps are cliff-sized: entry P&L lurches almost everywhere,")
        print(f"     so there is no stable region to sit in. That itself is the finding — on this")
        print(f"     data the entry is fragile to its 1yr bar; report that honestly and firm it up")
        print(f"     with MORE cycles rather than picking any single value.")
    print()
    print("NOTE: a plateau here means robustness to THIS parameter on THIS data (from 1982),")
    print("NOT out-of-sample validation of the entry rule itself. The 3mo bar was held fixed;")
    print("the two entry bars can interact, so this is a single-axis slice, not the joint surface.")


if __name__ == "__main__":
    main()
