"""
loaders.py — Research Agent data loaders.

Reads from the shared parquet cache and never makes a live API call when the
cache exists:

  data/storage/fred_macro.parquet          13 FRED macro series (monthly)
  data/storage/crsp_market_index.parquet   CRSP value-weighted returns

On a cold cache, load_macro_data() delegates to data.fetch.fred.fetch_fred_macro
(the shared fetcher) and, failing that, fetches inline via fredapi. The CRSP
loader converts the bundled crsp_market_index.csv to parquet on first run.

The 13 macro series carry the same stationarity transforms applied by both the
notebook and data/fetch/fred.py (cpi/indpro YoY %, gdp QoQ annualised %), so the
feature pipeline can treat the loaded frame as model-ready.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
STORAGE_DIR = PROJECT_ROOT / "data" / "storage"

MACRO_PARQUET = STORAGE_DIR / "fred_macro.parquet"
CRSP_PARQUET = STORAGE_DIR / "crsp_market_index.parquet"
CRSP_CSV = _THIS_DIR / "crsp_market_index.csv"

FRED_SERIES = {
    # Original 6
    "yield_curve":   "T10Y2Y",
    "term_spread":   "T10Y3M",
    "fed_funds":     "FEDFUNDS",
    "unemployment":  "UNRATE",
    "cpi":           "CPIAUCSL",
    "credit_spread": "BAA10Y",
    # Expanded set — June 2026
    "indpro":        "INDPRO",
    "vix":           "VIXCLS",
    "gdp":           "GDPC1",
    "t5yie":         "T5YIE",
    "t10yie":        "T10YIE",
    "dgs5":          "DGS5",
    "dgs30":         "DGS30",
}


def _fetch_macro_inline(fred_api_key: str, start: str, end: str) -> pd.DataFrame:
    """Fetch and transform the 13 macro series directly (cold-cache fallback)."""
    from fredapi import Fred

    fred = Fred(api_key=fred_api_key)
    data = {}
    for name, ticker in FRED_SERIES.items():
        data[name] = fred.get_series(ticker, observation_start=start, observation_end=end)
        print(f"  Fetched {ticker}")

    df = pd.DataFrame(data)
    df.index.name = "date"
    df = df.resample("MS").last()

    df["gdp"] = df["gdp"].ffill()
    df["gdp"] = df["gdp"].pct_change(3, fill_method=None) * 400
    df["cpi"] = df["cpi"].pct_change(12, fill_method=None) * 100
    df["indpro"] = df["indpro"].pct_change(12, fill_method=None) * 100

    df.dropna(inplace=True)
    return df


def load_macro_data(
    fred_api_key: str | None = None,
    start: str = "1995-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Return the 13-series monthly macro frame.

    Cache-first: reads data/storage/fred_macro.parquet if it exists. On a cold
    cache, tries the shared fetcher, then an inline fredapi fetch. Raises
    FileNotFoundError if the cache is missing and no API key is available.
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    if MACRO_PARQUET.exists():
        macro_df = pd.read_parquet(MACRO_PARQUET)
        print(f"Loaded FRED macro data from parquet cache: {MACRO_PARQUET}")
        return macro_df

    if not fred_api_key:
        raise FileNotFoundError(
            f"Parquet cache missing at {MACRO_PARQUET} and no FRED API key supplied. "
            "Provide fred_api_key or pre-populate the cache via "
            "data.fetch.fred.fetch_fred_macro(fred_api_key=...)."
        )

    # Prefer the shared fetcher so the cache layout stays consistent.
    try:
        from data.fetch.fred import fetch_fred_macro
        fetch_fred_macro(fred_api_key=fred_api_key, start=start, end=end)
        if MACRO_PARQUET.exists():
            return pd.read_parquet(MACRO_PARQUET)
    except Exception as e:
        print(f"Shared fetcher unavailable ({e}); fetching inline.")

    macro_df = _fetch_macro_inline(fred_api_key, start, end)
    macro_df.to_parquet(MACRO_PARQUET)
    print(f"FRED data fetched and cached → {MACRO_PARQUET}")
    return macro_df


def load_crsp() -> pd.DataFrame | None:
    """
    Return CRSP value-weighted market returns indexed by month-start, or None
    if neither the parquet cache nor the bundled CSV is available (validation
    is optional and must never break the pipeline).
    """
    if CRSP_PARQUET.exists():
        crsp = pd.read_parquet(CRSP_PARQUET)
        print(f"Loaded CRSP data from parquet cache: {CRSP_PARQUET}")
        return crsp

    if CRSP_CSV.exists():
        crsp = pd.read_csv(CRSP_CSV, parse_dates=["date"]).set_index("date")
        crsp.index = crsp.index.to_period("M").to_timestamp("s")
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        crsp.to_parquet(CRSP_PARQUET)
        print(f"CRSP CSV converted and cached → {CRSP_PARQUET}")
        return crsp

    print("CRSP data not found — skipping return validation.")
    return None
