"""
Generic FRED data-loading utilities.
Reusable across strategies — no strategy-specific logic here.
"""

import pandas as pd
from fredapi import Fred


def _fetch_fred_series(fred: Fred, series_id: str, start: str) -> pd.Series:
    s = fred.get_series(series_id, observation_start=start)
    s.index = pd.to_datetime(s.index)
    return s.dropna()


def fetch_fred_dataframe(
    fred_api_key: str,
    series: dict[str, str],
    start: str = "1990-01-01",
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    Fetch any set of FRED series into a single aligned DataFrame.

    Args:
        fred_api_key: your FRED API key
        series: mapping of {column_name: fred_series_id}, e.g.
                {"dff": "DFF", "dgs1": "DGS1"}
        start: observation start date (YYYY-MM-DD)
        fill_method: how to fill gaps across series — "ffill", "bfill", or None

    Returns:
        DataFrame with one column per series, DatetimeIndex, NaN rows dropped.
    """
    fred = Fred(api_key=fred_api_key)
    raw = {name: _fetch_fred_series(fred, sid, start).rename(name)
           for name, sid in series.items()}
    df = pd.concat(raw.values(), axis=1)
    if fill_method:
        df = df.ffill() if fill_method == "ffill" else df.bfill()
    return df.dropna()


# ── real-time neutral rate (Laubach-Williams) ────────────────────────────────
# The LW natural rate of interest r*, in its REAL-TIME (one-sided) vintages: for each
# quarter Q the estimate uses only data available AT Q (no hindsight). This is what the
# neutral-guard exit needs — a distance-to-neutral that could actually have been known in
# real time. The "current estimates" file would leak future data; we deliberately use the
# real-time file. Coverage starts 2005q1 (the earliest vintage the NY Fed publishes).
_LW_REALTIME_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/economists/"
    "williams/data/Laubach_Williams_real_time_estimates.xlsx"
)


def fetch_rstar_realtime() -> pd.Series:
    """Laubach-Williams real-time ONE-SIDED r* (percent), indexed by each vintage's as-of
    quarter-end date. No hindsight: sheet 'YYYYqN' holds the estimate known at that quarter.
    Returns a quarterly Series; forward-fill to a daily index at the call site."""
    import io
    import ssl
    import urllib.request

    req = urllib.request.Request(_LW_REALTIME_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context()).read()
    xl = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")

    rows = []
    for sh in [s for s in xl.sheet_names if s and s[0] == "2" and "q" in s.lower()]:
        df = xl.parse(sh, header=5)              # row 5 holds the column headers
        df = df[df["Date"].notna()].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"].notna()]
        # the FIRST 'rstar' column is the One-Sided (real-time) block; the second is Two-Sided
        onesided = [c for c in df.columns if str(c).strip() == "rstar"][0]
        last = df.sort_values("Date").iloc[-1]   # final data point = the vintage's own quarter
        rows.append({"asof": last["Date"], "rstar": float(last[onesided])})
    return pd.DataFrame(rows).set_index("asof").sort_index()["rstar"]


def fetch_nominal_neutral(fred_api_key: str, index: pd.DatetimeIndex) -> pd.Series:
    """Nominal neutral fed funds = real-time r* + 10y expected inflation (FRED EXPINF10YR),
    forward-filled onto `index` (percent). NaN before r* coverage (pre-2005q1) — callers must
    treat NaN as 'guard inactive', so pre-2005 cycles fall back to the unguarded exit."""
    rstar = fetch_rstar_realtime().reindex(index, method="ffill")
    fred = Fred(api_key=fred_api_key)
    infl = _fetch_fred_series(fred, "EXPINF10YR", "1990-01-01").reindex(index, method="ffill")
    return (rstar + infl).rename("nominal_neutral")
