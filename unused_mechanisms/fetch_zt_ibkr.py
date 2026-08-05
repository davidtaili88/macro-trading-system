"""
Pull INDIVIDUAL expired ZT contract histories from Interactive Brokers, then build a
carry-PRESERVING manually-rolled 2yr-futures return series.

WHY: Yahoo's ZT=F is back-adjusted upstream — roll gaps (and thus carry) are removed, so
the Yahoo backtest is price-only (see memory: zt-futures-carry-data-limitation). IBKR keeps
expired-contract history, so we can pull each quarterly contract's OWN price path and roll
between them the correct way, keeping carry in.

THE ROLL, DONE RIGHT (this is the whole point):
  - On each day you hold ONE contract; your daily return is THAT contract's own pct_change.
    Within a contract's life the price converges toward spot as expiry nears — that
    convergence IS the carry, and it is fully present in the contract's own path.
  - At a roll date you switch contracts. You earn the OLD contract's return up to the roll,
    then the NEW contract's return after. You NEVER difference across contracts, so the
    price-level gap between them never enters P&L (no phantom jump) — but carry stays, because
    it lived inside each contract's convergence, not in the gap.
  - Charge the explicit roll transaction cost (bid/ask) on the roll day.

SETUP REQUIRED (this script connects to a LOCAL TWS / IB Gateway — it is not a web call):
  1. pip install ib_insync
  2. Run TWS or IB Gateway, logged in, with API connections enabled
     (TWS: Config > API > Settings > "Enable ActiveX and Socket Clients"; note the port —
      7497 paper TWS, 7496 live TWS, 4002 paper Gateway, 4001 live Gateway).
  3. CME futures historical market-data permission on the account (may need a subscription).
     This script will tell you clearly if that permission is missing.

Run:  python3 unused_mechanisms/fetch_zt_ibkr.py    (run from the project root)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local dir, for ibkr_utils (archived here)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hiking_curve_strategy"))
import _paths  # noqa: F401 — registers core/, benchmark/, utils/ on sys.path
from data import get_zt_roll_dates
from ibkr_utils import ibkr_session, fetch_futures_bars

# ---- connection knobs (edit to match your TWS/Gateway) ----------------------------
IB_HOST = "127.0.0.1"
IB_PORT = 7497          # 7497 paper TWS | 7496 live TWS | 4002 paper GW | 4001 live GW
IB_CLIENT_ID = 17       # any unused id

# ZT quarterly delivery months (Mar/Jun/Sep/Dec) and how far back to try.
DELIVERY_MONTHS = [3, 6, 9, 12]
START_YEAR = 2014       # IBKR expired-history depth varies; start modest and extend if it works
END_YEAR   = 2023
OUT_CSV = os.path.join(os.path.dirname(__file__), "zt_manual_roll_returns.csv")


def _contract_codes(start_year, end_year):
    """List of (year, month) quarterly contracts to request, oldest first."""
    out = []
    for y in range(start_year, end_year + 1):
        for m in DELIVERY_MONTHS:
            out.append((y, m))
    return out


def main():
    print(f"Connecting to IB at {IB_HOST}:{IB_PORT} (clientId={IB_CLIENT_ID})...")
    contracts = _contract_codes(START_YEAR, END_YEAR)
    series = {}
    perm_error_seen = False

    try:
        with ibkr_session(host=IB_HOST, port=IB_PORT, client_id=IB_CLIENT_ID) as ib:
            print("Connected. Requesting individual ZT contract histories...\n")
            for (y, m) in contracts:
                ym = f"{y}{m:02d}"
                s, status = fetch_futures_bars(ib, symbol="ZT", contract_month=ym,
                                               exchange="CBOT", duration="9 M")
                if status == "ok":
                    series[(y, m)] = s
                    print(f"  {ym}: {len(s)} bars  {s.index[0].date()}..{s.index[-1].date()}  "
                          f"price [{s.min():.2f}, {s.max():.2f}]")
                elif status == "permission":
                    perm_error_seen = True
                    print(f"  {ym}: MARKET-DATA PERMISSION error")
                elif status == "no_contract":
                    print(f"  {ym}: no contract found (may predate IBKR history)")
                elif status == "no_data":
                    print(f"  {ym}: no bars (permission or history-depth limit)")
                else:
                    print(f"  {ym}: error fetching bars")
    except RuntimeError as e:
        # connection / missing-library failure from the session context manager
        print(f"\n{e}")
        return

    if not series:
        print("\nNo contract data retrieved.")
        if perm_error_seen:
            print("=> Looks like a CME market-data PERMISSION issue. In Client Portal / TWS:")
            print("   Account > Market Data Subscriptions > add the CME real-time/historical")
            print("   bundle (e.g. 'CME Real-Time' or the delayed/historical entitlement), then")
            print("   re-run. Historical futures bars need that entitlement.")
        else:
            print("=> No permission error, but no data — likely IBKR expired-history depth does")
            print("   not reach these years. Try raising START_YEAR toward the present and re-run.")
        return

    # --- build the carry-preserving manual roll ----------------------------------
    print(f"\nRetrieved {len(series)} contracts. Building carry-preserving manual roll...")
    roll_ret = _manual_roll_returns(series)
    if roll_ret is None or roll_ret.empty:
        print("Could not assemble a continuous rolled return series from the contracts pulled")
        print("(need consecutive quarterly contracts that overlap around each roll date).")
        return

    roll_ret.to_frame("ret").to_csv(OUT_CSV)
    print(f"\nWrote {len(roll_ret)} daily rolled returns to {OUT_CSV}")
    print(f"  span {roll_ret.index[0].date()}..{roll_ret.index[-1].date()}")
    print("Next: feed this series into the faithfulness diff vs DGS2-total to see the REAL")
    print("carry-inclusive ZT P&L (compare_zt_vs_dgs2.py, swapping ret_zt for this series).")


def _manual_roll_returns(series: dict) -> pd.Series:
    """Stitch per-contract daily returns into ONE continuous series, switching active
    contract at each CME roll date. Return within a contract; never across contracts.

    series: {(year, month): close_series}. Uses get_zt_roll_dates to decide which
    contract is active on each date (the nearest not-yet-rolled quarterly).
    """
    # order contracts chronologically by delivery
    ordered = sorted(series.keys())
    if not ordered:
        return None

    # union index across all contracts
    all_days = sorted(set().union(*[set(series[k].index) for k in ordered]))
    all_days = pd.DatetimeIndex(all_days)

    span_start = str(all_days[0].date())
    span_end   = str(all_days[-1].date())
    rolls = get_zt_roll_dates(span_start, span_end)

    # for each day, pick the active contract = first quarterly whose roll date is in the
    # future (i.e. we haven't rolled out of it yet) AND which has a price that day.
    def _active_contract(day):
        for (y, m) in ordered:
            # this contract's own roll date (5 bdays before FND of its delivery month)
            r = get_zt_roll_dates(f"{y}-01-01", f"{y}-12-31")
            # roll date matching this delivery month = the one in the month before m
            fnd_month = {3: 2, 6: 5, 9: 8, 12: 11}[m]
            cand = [d for d in r if d.month == fnd_month and d.year == y]
            if not cand:
                continue
            roll_day = cand[0]
            if day <= roll_day and day in series[(y, m)].index:
                return (y, m)
        return None

    rets = []
    prev_contract = None
    prev_price = None
    for day in all_days:
        c = _active_contract(day)
        if c is None:
            rets.append((day, float("nan"))); prev_contract = None; prev_price = None
            continue
        price = float(series[c].loc[day])
        if prev_contract == c and prev_price is not None:
            rets.append((day, price / prev_price - 1.0))       # return WITHIN the same contract
        else:
            rets.append((day, float("nan")))                    # roll day or first obs: no cross-contract return
        prev_contract, prev_price = c, price

    s = pd.Series({d: r for d, r in rets}, name="ret").dropna()
    return s


if __name__ == "__main__":
    main()
