"""
loaders.py — Research Agent
============================
Loads 13 FRED macro series from the parquet cache in data/storage/.

Exports:
  - FRED_SERIES
  - pull_fred_data()

Data is fetched once by data/fetch/fred.py and stored as
data/storage/fred_macro.parquet. This loader reads from parquet — no live
FRED API calls on each run. If the parquet is missing, it triggers the
fetch script automatically.
"""

import pandas as pd

from data import STORAGE_DIR
from data.fetch.fred import FRED_SERIES, fetch_fred_macro

MACRO_PATH = STORAGE_DIR / "fred_macro.parquet"


def pull_fred_data(fred_api_key: str | None = None,
                   start: str = "1995-01-01",
                   end:   str = "2025-12-31") -> pd.DataFrame:
    """
    Load the 13 FRED macro series from data/storage/fred_macro.parquet.

    If the parquet file doesn't exist, fetches from FRED using fred_api_key
    and writes the parquet cache before returning. Subsequent calls read
    from the cache without making any network calls.

    Transformations already applied in the parquet:
        - GDPC1: quarterly → forward-fill → QoQ annualised % change
        - CPIAUCSL: YoY % change
        - INDPRO: YoY % change
        - All others: raw monthly level

    Note:
        T5YIE begins ~2003, T10YIE begins ~2004. After dropna(), the
        usable dataset starts ~2003-02.

    Returns:
        pd.DataFrame — monthly, indexed by date, columns = FRED_SERIES keys
    """
    if not MACRO_PATH.exists():
        if fred_api_key is None:
            raise FileNotFoundError(
                "fred_macro.parquet not found. Run:\n"
                "  from data.fetch.fred import fetch_fred_macro\n"
                "  fetch_fred_macro(fred_api_key='YOUR_KEY')"
            )
        fetch_fred_macro(fred_api_key=fred_api_key, start=start, end=end)

    macro_df = pd.read_parquet(MACRO_PATH)

    print(f"FRED macro data: {macro_df.shape}")
    print(f"Date range: {macro_df.index.min().date()} → {macro_df.index.max().date()}")
    return macro_df
