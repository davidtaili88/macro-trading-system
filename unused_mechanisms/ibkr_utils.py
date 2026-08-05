"""
Generic Interactive Brokers data-loading utilities.
Reusable across strategies — no strategy-specific logic here (no roll conventions, no ZT
knowledge). Just: connect to a local TWS / IB Gateway, fetch historical bars for a
fully-specified contract, and turn IBKR's cryptic errors into plain English.

Requires a LOCAL TWS or IB Gateway running with the API enabled — this is a socket
connection to that desktop app, not a web call. See fetch_zt_ibkr.py for setup steps.
"""

from contextlib import contextmanager

import pandas as pd


# default socket ports by app/mode (edit the call, not this table)
DEFAULT_PORTS = {
    "paper_tws": 7497,
    "live_tws":  7496,
    "paper_gw":  4002,
    "live_gw":   4001,
}


@contextmanager
def ibkr_session(host: str = "127.0.0.1", port: int = 7497, client_id: int = 17,
                 timeout: float = 15.0):
    """Context manager yielding a connected ib_insync IB handle; disconnects on exit.

    Raises RuntimeError with an actionable message if ib_insync is missing or the
    connection fails (wrong port, API not enabled, TWS/Gateway not running).

    Usage:
        with ibkr_session(port=7497) as ib:
            s = fetch_futures_bars(ib, "ZT", "202403")
    """
    try:
        from ib_insync import IB
    except ImportError as e:
        raise RuntimeError(
            "ib_insync is not installed. Run:  pip install ib_insync\n"
            "(and make sure TWS or IB Gateway is running with the API enabled)"
        ) from e

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=timeout)
    except Exception as e:
        raise RuntimeError(
            f"IBKR connection failed on {host}:{port} ({type(e).__name__}: {e}).\n"
            "Checklist:\n"
            "  - Is TWS or IB Gateway running and logged in?\n"
            "  - API enabled? (TWS: Global Config > API > Settings > Enable Socket Clients)\n"
            "  - Port match? paper TWS=7497, live TWS=7496, paper GW=4002, live GW=4001.\n"
            f"    (this attempt used {port})"
        ) from e
    try:
        yield ib
    finally:
        ib.disconnect()


def classify_ibkr_error(msg: str) -> str:
    """Turn an IBKR error string into a short human category, so callers can branch
    (permission vs depth vs other) without parsing raw messages everywhere."""
    m = msg.lower()
    if "market data" in m or "permission" in m or "not subscribed" in m or "162" in m:
        return "permission"
    if "no security definition" in m or "200" in m:
        return "no_contract"
    if "historical data" in m and ("no data" in m or "query returned no data" in m):
        return "no_data"
    return "other"


def fetch_futures_bars(
    ib,
    symbol: str,
    contract_month: str,
    exchange: str = "CBOT",
    currency: str = "USD",
    duration: str = "9 M",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    end_datetime: str = "",
    use_rth: bool = True,
) -> tuple[pd.Series | None, str]:
    """Fetch daily close bars for ONE fully-specified futures contract.

    symbol/contract_month: e.g. "ZT" / "202403" (the delivery YYYYMM).
    Returns (series, status) where status is "ok" | "permission" | "no_contract" |
    "no_data" | "other". series is a close-price pd.Series named contract_month, or None.

    No roll logic, no multi-contract stitching — that is strategy-level and lives in the
    caller. This function only knows how to pull one contract's bars.
    """
    try:
        from ib_insync import Future, util
    except ImportError:
        return None, "other"

    fut = Future(symbol=symbol, lastTradeDateOrContractMonth=contract_month,
                 exchange=exchange, currency=currency)
    try:
        details = ib.reqContractDetails(fut)
    except Exception as e:
        return None, classify_ibkr_error(str(e))
    if not details:
        return None, "no_contract"

    con = details[0].contract
    try:
        bars = ib.reqHistoricalData(
            con, endDateTime=end_datetime, durationStr=duration,
            barSizeSetting=bar_size, whatToShow=what_to_show,
            useRTH=use_rth, formatDate=1,
        )
    except Exception as e:
        return None, classify_ibkr_error(str(e))
    if not bars:
        return None, "no_data"

    df = util.df(bars)
    s = pd.Series(df["close"].values, index=pd.to_datetime(df["date"]),
                  name=contract_month).dropna()
    return s, "ok"
