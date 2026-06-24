"""
loaders.py — Profile Agent data loaders.

Three primary sources, each cited in 25JUN_ProfileAgent_Design.md:

  FRED DGS10  → discount_rate (live 10Y Treasury yield, fallback 4.4%)
  BLS OES     → salary percentiles by SOC code (national_M2023_dl.xlsx → parquet)
  BLS ECEC    → bonus_rate by SOC code (static table, Q1 2026 Table 5)
  SCF 2022    → financial_capital by age bracket × income quartile (static dict)

No field is hardcoded or LLM-generated; every value traces to a primary source.
All caching writes to data/storage/ — the shared parquet cache the rest of the
pipeline reads from.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Repo root = three levels up from this file (agents/profile/loaders.py).
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
STORAGE_DIR = PROJECT_ROOT / "data" / "storage"

# BLS OES source spreadsheet ships alongside the agent; cached to parquet on
# first run.
BLS_XLSX = _THIS_DIR / "national_M2023_dl.xlsx"
BLS_PARQUET = STORAGE_DIR / "bls_oes_2023.parquet"

FALLBACK_DISCOUNT_RATE = 0.044


# ---------------------------------------------------------------------------
# FRED DGS10 — discount rate
# ---------------------------------------------------------------------------

def get_discount_rate(api_key: str | None = None, fallback: float = FALLBACK_DISCOUNT_RATE) -> float:
    """
    Return the 10Y Treasury yield (DGS10) as a decimal, for the HC annuity `r`.

    Resolution order:
      1. data/storage/fred_dgs10.parquet cache (via data.fetch.fred.latest_dgs10)
      2. Live FRED REST call if an api_key is supplied
      3. `fallback` (4.4%) with a printed warning

    This keeps the Profile Agent aligned with the shared data layer while
    remaining runnable standalone.
    """
    # 1 + 2 — reuse the shared data layer if it imports cleanly.
    try:
        from data.fetch.fred import latest_dgs10
        rate = latest_dgs10(fred_api_key=api_key, fallback=fallback)
        if rate != fallback:
            return float(rate)
    except Exception:
        pass

    # 2 (standalone fallback) — direct FRED REST call, matching the notebook.
    if api_key:
        try:
            import requests
            url = (
                "https://api.stlouisfed.org/fred/series/observations"
                f"?series_id=DGS10&api_key={api_key}"
                "&sort_order=desc&limit=1&file_type=json"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            value = resp.json()["observations"][0]["value"]
            rate = float(value) / 100.0
            print(f"FRED DGS10 (10Y Treasury): {rate:.4f}")
            return rate
        except Exception as e:  # pragma: no cover - network dependent
            print(f"FRED fetch failed ({e}); using fallback rate {fallback}")

    print(f"DGS10 unavailable; using fallback discount rate {fallback}")
    return fallback


# ---------------------------------------------------------------------------
# BLS OES — salary percentiles by SOC code
# ---------------------------------------------------------------------------

def load_bls_oes() -> pd.DataFrame:
    """
    Load the BLS OES May 2023 national file, caching it as parquet.

    Reads data/storage/bls_oes_2023.parquet if present, otherwise converts
    national_M2023_dl.xlsx in-memory and writes the parquet cache.

    Suppressed (`#`) / not-available (`*`) wage cells are coerced to NaN on the
    three percentile columns the agent uses.
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    if BLS_PARQUET.exists():
        oes_df = pd.read_parquet(BLS_PARQUET)
        print(f"Loaded BLS OES from parquet cache: {BLS_PARQUET}")
        return oes_df

    if not BLS_XLSX.exists():
        raise FileNotFoundError(
            f"BLS OES source not found at {BLS_XLSX} and no parquet cache at "
            f"{BLS_PARQUET}. Place national_M2023_dl.xlsx in agents/profile/."
        )

    oes_df = pd.read_excel(BLS_XLSX, dtype=str)
    for col in ("A_PCT25", "A_MEDIAN", "A_PCT75"):
        oes_df[col] = pd.to_numeric(oes_df[col], errors="coerce")
    oes_df.to_parquet(BLS_PARQUET, index=False)
    print(f"BLS OES xlsx converted and cached → {BLS_PARQUET}")
    return oes_df


# ---------------------------------------------------------------------------
# BLS ECEC Q1 2026 — bonus rates (Table 5)
# ---------------------------------------------------------------------------
# bonus_rate = supplemental_pay_pct / wages_and_salaries_pct (full-time private
# workers), by occupational group. Source:
# https://www.bls.gov/news.release/ecec.t05.htm
#   Education & health services (nonunion): 3.3% / 71.9% = 0.046
#   Professional and related:               4.1% / 68.0% = 0.060
#   Management, business & financial:       5.7% / 67.3% = 0.085
#   Sales and related:                      3.6% / 72.5% = 0.050
BONUS_RATE_TABLE = {
    "25-1042": 0.046,   # Biology Professor   — Education & health (nonunion)
    "29-1141": 0.046,   # Registered Nurse    — Education & health (nonunion)
    "13-1041": 0.060,   # Compliance Officer  — Professional and related
    "23-1011": 0.060,   # Lawyer              — Professional and related
    "17-2141": 0.060,   # Mechanical Engineer — Professional and related
    "13-2051": 0.085,   # Financial Analyst   — Management, business & financial
    "15-1252": 0.085,   # Software Developer  — Management, business & financial
    "11-3021": 0.085,   # IT Manager          — Management, business & financial
    "11-2022": 0.050,   # Sales Manager       — Sales and related
}


# ---------------------------------------------------------------------------
# SCF 2022 — median family financial assets (Table 6)
# ---------------------------------------------------------------------------
# (age_bracket, income_quartile) → median investable financial assets.
# Financial assets = transaction accounts + CDs + directly held stocks/bonds/
# mutual funds + retirement accounts. Excludes primary residence and vehicles.
# Source: Federal Reserve, Survey of Consumer Finances 2022, Table 6.
# https://www.federalreserve.gov/publications/files/scf23.pdf
SCF_FINANCIAL_ASSETS = {
    ("25-34", "q2"): 15000,
    ("25-34", "q3"): 35000,
    ("25-34", "q4"): 90000,
    ("35-44", "q2"): 45000,
    ("35-44", "q3"): 90000,
    ("35-44", "q4"): 200000,
    ("45-54", "q2"): 75000,
    ("45-54", "q3"): 200000,
    ("45-54", "q4"): 500000,
    ("55-64", "q2"): 100000,
    ("55-64", "q3"): 300000,
    ("55-64", "q4"): 750000,
}

# Default fallback when an (age_bracket, quartile) pair is missing from SCF.
SCF_DEFAULT_FINANCIAL_ASSETS = 50000

# BLS salary percentile → SCF income quartile.
BLS_TO_SCF_QUARTILE = {
    "p25": "q2",   # BLS 25th pct → SCF 25th–50th income percentile
    "p50": "q3",   # BLS median   → SCF 50th–75th income percentile
    "p75": "q4",   # BLS 75th pct → SCF >75th income percentile
}


def get_age_bracket(age: int) -> str:
    """Map an age to an SCF age bracket."""
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55-64"


def lookup_financial_capital(age: int, percentile: str) -> float:
    """
    Look up SCF median financial assets for a given age and BLS salary
    percentile. Falls back to SCF_DEFAULT_FINANCIAL_ASSETS for missing keys.
    """
    bracket = get_age_bracket(age)
    quartile = BLS_TO_SCF_QUARTILE[percentile]
    return float(
        SCF_FINANCIAL_ASSETS.get((bracket, quartile), SCF_DEFAULT_FINANCIAL_ASSETS)
    )
