"""
fetch/fred.py — FRED data fetcher
===================================
Pulls two datasets from FRED and writes them as parquet:

  fred_macro.parquet   13 macro series, monthly, for the Research Agent
  fred_dgs10.parquet   10Y Treasury yield series, daily, for the Profile Agent

Both files are cached. Re-run with force=True to refresh.

Usage:
    from data.fetch.fred import fetch_fred_macro, fetch_fred_dgs10
    fetch_fred_macro(fred_api_key="...")
    fetch_fred_dgs10(fred_api_key="...")
"""

from pathlib import Path

import pandas as pd
from fredapi import Fred

from data import STORAGE_DIR

FRED_SERIES = {
    "yield_curve":   "T10Y2Y",
    "term_spread":   "T10Y3M",
    "fed_funds":     "FEDFUNDS",
    "unemployment":  "UNRATE",
    "cpi":           "CPIAUCSL",
    "credit_spread": "BAA10Y",
    "indpro":        "INDPRO",
    "vix":           "VIXCLS",
    "gdp":           "GDPC1",
    "t5yie":         "T5YIE",
    "t10yie":        "T10YIE",
    "dgs5":          "DGS5",
    "dgs30":         "DGS30",
}

MACRO_PATH  = STORAGE_DIR / "fred_macro.parquet"
DGS10_PATH  = STORAGE_DIR / "fred_dgs10.parquet"


def fetch_fred_macro(
    fred_api_key: str,
    start: str = "1995-01-01",
    end:   str = "2025-12-31",
    force: bool = False,
) -> Path:
    """
    Pull all 13 FRED macro series → data/storage/fred_macro.parquet.

    Applies the same stationarity transforms as the Research Agent:
      - gdp:    quarterly QoQ annualised % change
      - cpi:    YoY % change
      - indpro: YoY % change
      - others: raw monthly level

    Returns the path to the parquet file.
    """
    if MACRO_PATH.exists() and not force:
        print(f"[fred] fred_macro.parquet already exists. Use force=True to refresh.")
        return MACRO_PATH

    print("[fred] Pulling 13 FRED macro series...")
    fred = Fred(api_key=fred_api_key)

    data = {}
    for name, ticker in FRED_SERIES.items():
        data[name] = fred.get_series(ticker, observation_start=start, observation_end=end)
        print(f"  ✓ {ticker} ({name})")

    df = pd.DataFrame(data)
    df.index.name = "date"
    df = df.resample("MS").last()

    df["gdp"]    = df["gdp"].ffill()
    df["gdp"]    = df["gdp"].pct_change(3, fill_method=None) * 400
    df["cpi"]    = df["cpi"].pct_change(12, fill_method=None) * 100
    df["indpro"] = df["indpro"].pct_change(12, fill_method=None) * 100

    df.dropna(inplace=True)

    df.to_parquet(MACRO_PATH)
    mb = MACRO_PATH.stat().st_size / (1024 ** 2)
    print(f"[fred] Saved fred_macro.parquet  {df.shape[0]} rows × {df.shape[1]} cols  ({mb:.3f} MB)")
    return MACRO_PATH


def fetch_fred_dgs10(
    fred_api_key: str,
    start: str = "2000-01-01",
    end:   str = "2025-12-31",
    force: bool = False,
) -> Path:
    """
    Pull daily DGS10 series → data/storage/fred_dgs10.parquet.
    Used by the Profile Agent for the HC annuity discount rate.

    Returns the path to the parquet file.
    """
    if DGS10_PATH.exists() and not force:
        print(f"[fred] fred_dgs10.parquet already exists. Use force=True to refresh.")
        return DGS10_PATH

    print("[fred] Pulling DGS10 (10Y Treasury yield)...")
    fred = Fred(api_key=fred_api_key)
    series = fred.get_series("DGS10", observation_start=start, observation_end=end)

    df = series.to_frame(name="dgs10")
    df.index.name = "date"
    df.dropna(inplace=True)

    df.to_parquet(DGS10_PATH)
    mb = DGS10_PATH.stat().st_size / (1024 ** 2)
    print(f"[fred] Saved fred_dgs10.parquet  {len(df)} rows  ({mb:.3f} MB)")
    return DGS10_PATH


def latest_dgs10(fred_api_key: str | None = None, fallback: float = 0.044) -> float:
    """
    Return the most recent DGS10 value as a decimal.

    Reads from cached parquet if available. Falls back to live FRED call if
    cache is missing and fred_api_key is provided. Returns fallback (4.4%)
    if both fail.
    """
    if DGS10_PATH.exists():
        df = pd.read_parquet(DGS10_PATH)
        rate = df["dgs10"].dropna().iloc[-1] / 100.0
        return float(rate)

    if fred_api_key:
        try:
            fred = Fred(api_key=fred_api_key)
            val = fred.get_series("DGS10").dropna().iloc[-1]
            return float(val) / 100.0
        except Exception as e:
            print(f"[fred] FRED API call failed: {e}. Using fallback {fallback*100:.1f}%")

    print(f"[fred] DGS10 not cached and no API key. Using fallback {fallback*100:.1f}%")
    return fallback
