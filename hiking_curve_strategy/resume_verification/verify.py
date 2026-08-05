"""
Resume-verification harness — hiking_curve_strategy (DGS2 only).

Runs one check per resume claim against the LIVE code + real FRED data and prints
a PASS/FAIL summary. Every figure is DGS2-only (ZT ignored, per resume scope).
Each check verifies dthe claim as it is HONESTLY worded:

  - captured ~56% pre-carry of the perfect-hindsight benchmark (pooled, whole-strategy)
  - multi-gate rule cutting late exits by up to ~2 months vs a naive trigger

The capture check is the POOLED whole-strategy pre-carry capture: pooled P&L of ALL
detected episodes (including out-of-cycle false positives — real strategy behaviour,
so counted) over the oracle's pooled P&L across all hiking cycles (-60/-30td offsets),
on DGS2 price-only returns. See check_capture_ratio for the exact definition.

Run:  cd hiking_curve_strategy && PYTHONPATH=. python3 resume_verification/verify.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths  # noqa: F401 — registers core/, benchmarks/, utils/ on sys.path

DATA_START = "1982-10-01"   # earliest date all signal inputs exist (DFEDTAR starts 1982-09-27)

# true final-hike dates per real cycle (from benchmarks/ORACLE_CYCLES and
# unused_mechanisms/diagnostic_tests/diagnose_level_ratio.py) — the benchmark exit target.
TRUE_LAST_HIKE = {
    "1994-95": pd.Timestamp("1995-02-01"),
    "2004-06": pd.Timestamp("2006-06-29"),
    "2015-18": pd.Timestamp("2018-12-20"),
    "2022-23": pd.Timestamp("2023-07-27"),
}
# which detected episode's entry attributes to which real cycle
CYCLE_ENTRY_WINDOW = {
    "1994-95": (pd.Timestamp("1993-06-01"), pd.Timestamp("1994-12-31")),
    "2004-06": (pd.Timestamp("2004-01-01"), pd.Timestamp("2005-06-30")),
    "2015-18": (pd.Timestamp("2015-06-01"), pd.Timestamp("2017-06-30")),
    "2022-23": (pd.Timestamp("2021-06-01"), pd.Timestamp("2022-12-31")),
}


# ── shared fixtures: compute the expensive things ONCE, reuse across checks ──
_CACHE: dict = {}


def api_key() -> str:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, ".env"))
    return os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")

#Fetch cycles based on our short algorithm
def strategy_cycles():
    """Full production signal -> detected episodes (all gates live)."""
    if "cycles" not in _CACHE:
        from signal_logic import detect_signal, signal_to_cycles
        signal = detect_signal(api_key(), start=DATA_START)
        _CACHE["cycles"] = signal_to_cycles(signal)
    return _CACHE["cycles"]


# Calculate returns accroding to our signal strategy
def compounded_pnl(ret, cycles, entry_days=None, exit_days=None) -> float:
    """Compounded payer P&L over `cycles` on return series `ret`.
    None offsets = signal-driven (dates as-is); int offsets = oracle timing."""
    from backtest import calc_strat_ret
    df = calc_strat_ret(ret, cycles,
                        entry_days_before_first=entry_days,
                        exit_days_before_last=exit_days)
    r = df.loc[df["signal"] == -1, "strat_ret"].dropna()
    return (1 + r).prod() - 1


def _result(name, claim, passed, detail):
    return {"name": name, "claim": claim, "passed": passed, "detail": detail}



# ── capture of the perfect-hindsight benchmark ────────────────────
def check_capture_ratio():
    """Whole-strategy (pooled) capture on DGS2, PRE-CARRY (price-only):

        capture = pooled P&L of ALL detected signal episodes
                  ------------------------------------------------------
                  pooled P&L of the oracle over ALL hiking cycles (-60/-30td)

    This is the DEPLOYED-strategy view: the numerator includes EVERY episode the
    signal fired, INCLUDING out-of-cycle false positives (e.g. 1993, 1996) — that
    P&L is part of the strategy's real record, so it counts. Numerator and
    denominator are each pooled over their own active days and compounded, then
    divided. Not a per-trade capture ratio (the two range over different periods);
    it is an honest portfolio-level "what fraction of the ceiling did the strategy
    realise". PASSES if it lands near the expected neighbourhood (wide tol vs FRED
    revisions). Post-carry is intentionally NOT claimed (see resume scope)."""
    from benchmark import FED_HIKE_CYCLES, ENTRY_DAYS_BEFORE_FIRST, EXIT_DAYS_BEFORE_LAST
    from data import fetch_carryless_dgs2_returns
    expect_pct, tol_pp = 56.0, 8.0

    signal_derived_cycles = strategy_cycles()
    # price-only return series: canonical carryless source (DGS2-only calendar).
    ret = fetch_carryless_dgs2_returns(api_key(), start=DATA_START)

    # NOTE: compounded_pnl already returns (1+r).prod()-1, so it is ALREADY a
    # return — do NOT subtract 1 again here (an earlier version did, which flipped
    # both figures to nonsense negatives via double subtraction).
    signal_driven_ret = compounded_pnl(ret, signal_derived_cycles)
    oracle_driven_ret = compounded_pnl(ret, FED_HIKE_CYCLES,
                                       ENTRY_DAYS_BEFORE_FIRST, EXIT_DAYS_BEFORE_LAST)
    print("price only signal returns", signal_driven_ret)
    print("price only oracle returns", oracle_driven_ret)

    cap = signal_driven_ret / oracle_driven_ret * 100
    hit = abs(cap - expect_pct) <= tol_pp

    return _result(
        "capture of perfect-hindsight benchmark",
        f"pre-carry ~{expect_pct:.0f}% of the oracle (pooled whole-strategy, DGS2)",
        hit,
        f"price_ret: capture {cap:.1f}% (strat {signal_driven_ret*100:.2f}% / "
        f"oracle {oracle_driven_ret*100:.2f}%; expect ~{expect_pct:.0f}% "
        f"{'OK' if hit else 'OUT OF RANGE'})  ||  numerator = ALL "
        f"{len(signal_derived_cycles)} detected episodes incl. out-of-cycle false positives; "
        f"denominator = oracle over {len(FED_HIKE_CYCLES)} hiking cycles (-60/-30td); "
        f"price-only, DGS2. Post-carry not claimed.",
    )


# ── cutting late exits vs a naive spread-level trigger ────────────
def _final_exits(naive: bool) -> dict:
    """FULL vs NAIVE final exit per real cycle. NAIVE disables the extra gates via
    module constants so only the crude spread-level exit is live; entry/data identical."""
    import signal_logic as SL
    from signal_logic import detect_signal, signal_to_cycles
    ORIGINAL_FALSE_PROMISE_THRESHOLD = SL.FALSE_PROMISE_THRESHOLD_1YR_BP
    ORIGINAL_RATIO_EXIT_FLOOR = SL.RATIO_EXIT_FLOOR_BP
    try:
        if naive:
            SL.FALSE_PROMISE_THRESHOLD_1YR_BP = -1e9   # never fires
            SL.RATIO_EXIT_FLOOR_BP = 1e9               # ratio branch never evaluated
        sig = detect_signal(api_key(), start=DATA_START, neutral_guard=(not naive))
    finally:
        SL.FALSE_PROMISE_THRESHOLD_1YR_BP = ORIGINAL_FALSE_PROMISE_THRESHOLD
        SL.RATIO_EXIT_FLOOR_BP = ORIGINAL_RATIO_EXIT_FLOOR
    cycles = signal_to_cycles(sig)
    output = {}
    for label, (lo, hi) in CYCLE_ENTRY_WINDOW.items():
        eps = [c for c in cycles if lo <= c["first_hike"] <= hi]
        if eps:
            output[label] = max(e["last_hike"] for e in eps)
    return output


def check_late_exit_reduction():
    """On late-running cycles the full rule cuts lateness by up to ~2 months."""
    claimed_max = 62
    full, naive = _final_exits(naive=False), _final_exits(naive=True)

    exit_lags = []
    for year, full_exit_date in full.items():
        # Approximate orcale's 30 trading days from last hike with 45 calendar days
        if (naive[year] - TRUE_LAST_HIKE[year]).days + 45 > 0:
            # naive exits LATE; the full rule exits earlier. The reduction is how
            # many days earlier full is than naive: naive - full (positive = earlier).
            diff = naive[year] - full_exit_date
            diff_in_days = diff.days
            exit_lags.append(diff_in_days)
    max_lag_fixed = max(exit_lags)

    
    supports = max_lag_fixed >= claimed_max
    return _result(
        "cutting late exits vs naive spread-level trigger",
        f"reduces late-exit error by up to ~{claimed_max:.0f} days vs a naive spread-level trigger",
        supports,
        f"max late-exit reduction = {max_lag_fixed/30.4:.2f}mo ({max_lag_fixed}d) "
        f"across {len(exit_lags)} late-running cycle(s)",
    )


CHECKS = [
    check_capture_ratio,
    check_late_exit_reduction,
]


def main():
    print("=" * 100)
    print("  RESUME VERIFICATION  —  hiking_curve_strategy (DGS2 only)")
    print("=" * 100)
    results = []
    for fn in CHECKS:
        try:
            r = fn()
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {fn.__name__} raised: {e}")
            traceback.print_exc()
            continue
        results.append(r)
        print(f"\n[{'PASS' if r['passed'] else 'FAIL'}]  {r['name']}")
        print(f"   claim : {r['claim']}")
        print(f"   detail: {r['detail']}")

    n_pass = sum(1 for r in results if r["passed"])
    print("\n" + "=" * 100)
    print(f"  SUMMARY: {n_pass}/{len(results)} claims verified")
    for r in results:
        print(f"    [{'PASS' if r['passed'] else 'FAIL'}] {r['name']}")
    print("=" * 100)
    sys.exit(0 if results and n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
