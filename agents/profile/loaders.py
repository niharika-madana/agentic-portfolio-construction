"""
loaders.py — Profile Agent
===========================
Data loading utilities:
  - FRED DGS10 discount rate (reads from data/storage/fred_dgs10.parquet)
  - BLS OES national wage data (reads from data/storage/bls_oes.parquet)
  - SCF 2022 static financial asset table (kept as in-memory dict)

Both FRED and BLS data are fetched by data/fetch/ scripts and cached as
parquet. These loaders read from parquet — they never download raw data
themselves. If the parquet file is missing they trigger the fetch script
automatically on first run.
"""

import pandas as pd

from data import STORAGE_DIR
from data.fetch.fred import fetch_fred_dgs10, latest_dgs10

# ---------------------------------------------------------------------------
# FRED DGS10
# ---------------------------------------------------------------------------

FALLBACK_DISCOUNT_RATE = 0.044


def get_discount_rate_from_fred(api_key: str | None = None,
                                fallback: float = FALLBACK_DISCOUNT_RATE) -> float:
    """
    Return the most recent 10Y Treasury yield (DGS10) as a decimal.

    Read order:
      1. data/storage/fred_dgs10.parquet  (cached, fast, no network)
      2. Live FRED API call               (if parquet missing and api_key provided)
      3. fallback (4.4%)                  (if both above fail)

    Source:
        Board of Governors of the Federal Reserve System (US),
        Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity
        [DGS10], FRED, Federal Reserve Bank of St. Louis.
    """
    dgs10_path = STORAGE_DIR / "fred_dgs10.parquet"

    if not dgs10_path.exists() and api_key:
        fetch_fred_dgs10(fred_api_key=api_key)

    rate = latest_dgs10(fred_api_key=api_key, fallback=fallback)
    print(f"DGS10 (10Y Treasury): {rate:.4f} ({rate*100:.2f}%)")
    return rate


# ---------------------------------------------------------------------------
# BLS OES — national occupational wage data
# ---------------------------------------------------------------------------

def load_bls_oes() -> pd.DataFrame:
    """
    Load BLS Occupational Employment and Wage Statistics (OES) May 2023
    from data/storage/bls_oes.parquet. Downloads and converts from the BLS
    zip/xlsx source on first call.

    Source:
        U.S. Bureau of Labor Statistics, Occupational Employment and Wage
        Statistics, May 2023 National Occupational Employment and Wage
        Estimates. https://www.bls.gov/oes/current/oes_nat.htm

    Returns a DataFrame indexed by OCC_CODE (SOC format, e.g. "25-1042")
    with columns:
        OCC_TITLE  — occupation label
        A_PCT25    — 25th percentile annual wage
        A_MEDIAN   — median (50th percentile) annual wage
        A_PCT75    — 75th percentile annual wage
    """
    bls_path = STORAGE_DIR / "bls_oes.parquet"

    if not bls_path.exists():
        from data.fetch.bls import fetch_bls_oes
        fetch_bls_oes()

    df = pd.read_parquet(bls_path)

    wage_cols = [c for c in ["A_PCT25", "A_MEDIAN", "A_PCT75"] if c in df.columns]
    available = df[wage_cols].notna().sum()
    print(
        f"BLS OES loaded: {len(df)} occupations | "
        f"A_MEDIAN available: {available.get('A_MEDIAN', '?')}"
    )
    keep = ["OCC_TITLE"] + wage_cols if "OCC_TITLE" in df.columns else wage_cols
    return df[keep]


# ---------------------------------------------------------------------------
# SCF 2022 — median financial assets by age bracket × income quartile
# ---------------------------------------------------------------------------

# Source: Federal Reserve, Survey of Consumer Finances 2022, Table 6
#   "Median and Mean Family Financial Assets by Selected Characteristics"
#   https://www.federalreserve.gov/publications/files/scf23.pdf
#
# "Financial assets" here = transaction accounts + CDs + directly held stocks/
# bonds/mutual funds + retirement accounts (IRA, 401k, DC plans) + other
# financial. Excludes primary residence equity, vehicles, and business equity.
# This maps to what we call financial_capital — investable liquid wealth.
#
# Income quartile boundaries (2022 SCF):
#   q1: < 25th pctile  (~< $30k)
#   q2: 25th–50th      (~$30k–$60k)
#   q3: 50th–75th      (~$60k–$120k)
#   q4: > 75th         (> $120k)
#
# Age bracket → "25-34", "35-44", "45-54", "55-64"

SCF_FINANCIAL_ASSETS: dict[tuple[str, str], int] = {
    ("25-34", "q1"):     1_800,
    ("25-34", "q2"):     8_000,
    ("25-34", "q3"):    25_000,
    ("25-34", "q4"):   120_000,
    ("35-44", "q1"):     3_000,
    ("35-44", "q2"):    25_000,
    ("35-44", "q3"):    90_000,
    ("35-44", "q4"):   350_000,
    ("45-54", "q1"):     6_000,
    ("45-54", "q2"):    60_000,
    ("45-54", "q3"):   200_000,
    ("45-54", "q4"):   700_000,
    ("55-64", "q1"):    12_000,
    ("55-64", "q2"):   100_000,
    ("55-64", "q3"):   320_000,
    ("55-64", "q4"): 1_100_000,
}


def _age_bracket(age: int) -> str:
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55-64"


def _income_quartile(bls_percentile_label: str) -> str:
    """Map BLS salary percentile label to SCF income quartile."""
    return {"p25": "q2", "p50": "q3", "p75": "q4"}[bls_percentile_label]


def get_scf_financial_capital(age: int, bls_percentile_label: str) -> int:
    """
    Look up median financial assets from SCF 2022 given a client's age
    and the salary percentile they represent in the BLS distribution.

    BLS salary percentile → SCF income quartile mapping:
        p25 salary → q2 (25th–50th income percentile)
        p50 salary → q3 (50th–75th income percentile)
        p75 salary → q4 (>75th income percentile)
    """
    bracket  = _age_bracket(age)
    quartile = _income_quartile(bls_percentile_label)
    return SCF_FINANCIAL_ASSETS[(bracket, quartile)]
