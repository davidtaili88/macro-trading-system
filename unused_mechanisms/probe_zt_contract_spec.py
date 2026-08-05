"""
Probe: ask IBKR what ZT contract specs it actually accepts.

Error 200 ("no security definition") on our ZT requests means the CONTRACT SPEC is wrong,
NOT that data/permission is missing (the connection succeeded). Rather than guess the right
exchange string / month format, we ask IBKR directly: request contract details for a bare
Future("ZT") and print every contract it returns — exchange, expiry, localSymbol, conId.
Those are the exact fields fetch_zt_ibkr.py must use.

Run (TWS/Gateway must be up):
  python3 unused_mechanisms/probe_zt_contract_spec.py    (run from the project root)
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local dir, for ibkr_utils (archived here)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hiking_curve_strategy"))
import _paths  # noqa: F401
from ibkr_utils import ibkr_session

IB_PORT = 7497   # match your TWS/Gateway


def main():
    try:
        from ib_insync import Future
    except ImportError:
        print("pip install ib_insync"); return

    with ibkr_session(port=IB_PORT) as ib:
        # try a few exchange spellings; whichever returns contracts is the right one
        for exch in ("ECBOT", "CBOT"):
            print(f"\n=== reqContractDetails for Future('ZT', exchange='{exch}') ===")
            try:
                dets = ib.reqContractDetails(Future(symbol="ZT", exchange=exch, currency="USD"))
            except Exception as e:
                print(f"  error: {type(e).__name__}: {e}")
                continue
            if not dets:
                print("  (no contracts returned)")
                continue
            print(f"  {len(dets)} contracts returned. First 12:")
            print(f"    {'expiry':>10}  {'exchange':>10}  {'localSym':>12}  {'conId':>10}  {'tradingClass':>12}")
            for d in dets[:12]:
                c = d.contract
                print(f"    {c.lastTradeDateOrContractMonth:>10}  {c.exchange:>10}  "
                      f"{c.localSymbol:>12}  {c.conId:>10}  {c.tradingClass:>12}")
            # also show the full range of expiries available
            expiries = sorted({d.contract.lastTradeDateOrContractMonth for d in dets})
            print(f"  expiries available: {expiries[0]} .. {expiries[-1]}  ({len(expiries)} total)")
            # once one exchange works, that's our answer
            return


if __name__ == "__main__":
    main()
