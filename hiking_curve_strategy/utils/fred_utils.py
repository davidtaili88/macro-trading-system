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
