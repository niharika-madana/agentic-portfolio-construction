"""
data/fetch/wrds.py — WRDS / CRSP / Compustat data loaders with parquet caching.

All agents pull historical data through this module instead of querying WRDS
on every run. The first call to any fetch_* function opens a WRDS connection,
runs the query, and writes a parquet file to data/storage/. Subsequent calls
read from the parquet cache (fast, offline-capable).

Usage (first run — requires WRDS credentials):
    from data.fetch.wrds import get_connection, fetch_crsp_monthly, fetch_crsp_daily
    conn = get_connection()
    fetch_crsp_monthly(DEFAULT_TICKERS, conn=conn)
    fetch_crsp_daily(DEFAULT_TICKERS, conn=conn)

Usage (subsequent runs — reads from cache):
    from data.fetch.wrds import load_crsp_monthly, load_crsp_daily
    monthly, permno_map = load_crsp_monthly(tickers)
    daily,   permno_map = load_crsp_daily(tickers)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data import STORAGE_DIR

try:
    import wrds as _wrds_module
    _WRDS_AVAILABLE = True
except ImportError:
    _WRDS_AVAILABLE = False
    _wrds_module = None


# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

CRSP_MONTHLY_PATH    = STORAGE_DIR / "crsp_monthly.parquet"
CRSP_DAILY_PATH      = STORAGE_DIR / "crsp_daily.parquet"
PERMNO_MAP_PATH      = STORAGE_DIR / "permno_map.json"
MKT_CAP_PATH         = STORAGE_DIR / "mkt_cap_weights.json"
FF_FACTORS_PATH      = STORAGE_DIR / "ff_risk_factors.parquet"

DEFAULT_START = "2000-01-01"
DEFAULT_END   = "2025-12-31"

# ---------------------------------------------------------------------------
# GICS sector codes → names
# ---------------------------------------------------------------------------

GICS_SECTOR_NAMES: dict[str, str] = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}

# Static ETF sector map — Compustat does not classify ETFs by GICS sector.
# Used as the canonical sector source for the 29-ETF universe.
ETF_SECTORS: dict[str, str] = {
    "SPY":  "Broad Market",
    "IWM":  "Broad Market",
    "EFA":  "Broad Market",
    "EEM":  "Broad Market",
    "AGG":  "Fixed Income",
    "TLT":  "Fixed Income",
    "IEF":  "Fixed Income",
    "SHY":  "Fixed Income",
    "HYG":  "Fixed Income",
    "LQD":  "Fixed Income",
    "TIP":  "Fixed Income",
    "GLD":  "Real Assets",
    "VNQ":  "Real Estate",
    "DJP":  "Real Assets",
    "XLK":  "Information Technology",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLE":  "Energy",
    "XLI":  "Industrials",
    "XLC":  "Communication Services",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "BIL":  "Cash",
}


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection():
    """
    Open a WRDS connection. Prompts for Fordham WRDS credentials on first use;
    stores them in ~/.pgpass for subsequent calls.
    """
    if not _WRDS_AVAILABLE:
        raise ImportError("wrds package not installed. Run: pip install wrds")
    return _wrds_module.Connection()


# ---------------------------------------------------------------------------
# Direct WRDS query functions (no caching)
# ---------------------------------------------------------------------------

def query_permno_map(
    tickers: list[str],
    as_of_date: str,
    conn,
) -> dict[str, int]:
    """
    Map ticker symbols to CRSP PERMNOs.
    Uses DISTINCT ON to pick the most recent name record per ticker,
    avoiding date-filter edge cases for ETFs with gap or NULL nameendt.
    """
    ticker_list = ", ".join(f"'{t.upper()}'" for t in tickers)
    query = f"""
        SELECT DISTINCT ON (ticker) ticker, permno
        FROM crsp.dsenames
        WHERE ticker IN ({ticker_list})
        ORDER BY ticker, namedt DESC
    """
    df = conn.raw_sql(query)
    return dict(zip(df["ticker"], df["permno"].astype(int)))


def query_crsp_monthly(
    permnos: list[int],
    start_date: str,
    end_date: str,
    conn,
) -> pd.DataFrame:
    """
    Monthly returns and market cap for given PERMNOs.
    Columns: permno, date, ret, mktcap.
    """
    permno_list = ", ".join(str(p) for p in permnos)
    query = f"""
        SELECT permno, date, ret, ABS(prc) * shrout AS mktcap
        FROM crsp.msf
        WHERE permno IN ({permno_list})
          AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY permno, date
    """
    df = conn.raw_sql(query, date_cols=["date"])
    df["ret"]    = pd.to_numeric(df["ret"],    errors="coerce")
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    return df


def query_crsp_daily(
    permnos: list[int],
    start_date: str,
    end_date: str,
    conn,
) -> pd.DataFrame:
    """
    Daily returns, prices, and volume for given PERMNOs.
    Columns: permno, date, ret, prc, vol.
    """
    permno_list = ", ".join(str(p) for p in permnos)
    query = f"""
        SELECT permno, date, ret, ABS(prc) AS prc, vol
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY permno, date
    """
    df = conn.raw_sql(query, date_cols=["date"])
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    return df


def query_fama_french_factors(
    start_date: str,
    end_date: str,
    conn,
) -> pd.DataFrame:
    """
    Monthly FF 3-factor + momentum + risk-free from WRDS.
    Columns: mktrf, smb, hml, umd, rf (decimal). Index: date.
    """
    query = f"""
        SELECT date, mktrf, smb, hml, umd, rf
        FROM ff.factors_monthly
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    df = conn.raw_sql(query, date_cols=["date"])
    for col in ["mktrf", "smb", "hml", "umd", "rf"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("date")


def query_market_cap_weights(
    permnos: list[int],
    as_of_date: str,
    conn,
) -> dict[int, float]:
    """
    Market cap weights (price × shares) at most recent month-end on/before as_of_date.
    Returns {permno: weight}, weights sum to 1.0.
    """
    permno_list = ", ".join(str(p) for p in permnos)
    query = f"""
        SELECT permno, ABS(prc) * shrout AS mktcap
        FROM crsp.msf
        WHERE permno IN ({permno_list})
          AND date = (
              SELECT MAX(date) FROM crsp.msf
              WHERE date <= '{as_of_date}'
          )
    """
    df = conn.raw_sql(query).dropna()
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    total = df["mktcap"].sum()
    return {int(row.permno): row.mktcap / total for row in df.itertuples()}


# ---------------------------------------------------------------------------
# Cached fetch functions — query WRDS once, write parquet, re-read on next call
# ---------------------------------------------------------------------------

def fetch_crsp_monthly(
    tickers: list[str],
    conn,
    start: str = DEFAULT_START,
    end:   str = DEFAULT_END,
    as_of_date: str = "2025-06-01",
    force: bool = False,
) -> Path:
    """
    Fetch CRSP monthly returns for tickers → data/storage/crsp_monthly.parquet.
    Also saves permno_map.json and mkt_cap_weights.json.

    Returns path to the parquet file.
    """
    if CRSP_MONTHLY_PATH.exists() and PERMNO_MAP_PATH.exists() and not force:
        print("[wrds] crsp_monthly.parquet already cached. Use force=True to refresh.")
        return CRSP_MONTHLY_PATH

    print(f"[wrds] Fetching permno map for {len(tickers)} tickers from CRSP ...")
    permno_map = query_permno_map(tickers, as_of_date, conn)

    missing = [t for t in tickers if t not in permno_map]
    if missing:
        print(f"[wrds] WARNING — no PERMNO found for: {missing}")

    permnos = list(permno_map.values())

    print(f"[wrds] Fetching CRSP monthly returns ({start} → {end}) ...")
    df = query_crsp_monthly(permnos, start, end, conn)
    df.to_parquet(CRSP_MONTHLY_PATH)
    mb = CRSP_MONTHLY_PATH.stat().st_size / (1024 ** 2)
    print(f"[wrds] Saved crsp_monthly.parquet  {df.shape[0]:,} rows  ({mb:.2f} MB)")

    PERMNO_MAP_PATH.write_text(json.dumps(permno_map))
    print(f"[wrds] Saved permno_map.json  ({len(permno_map)} tickers)")

    print(f"[wrds] Fetching market cap weights as of {as_of_date} ...")
    inv_map  = {v: k for k, v in permno_map.items()}
    perm_weights = query_market_cap_weights(permnos, as_of_date, conn)
    ticker_weights = {inv_map[p]: w for p, w in perm_weights.items() if p in inv_map}
    MKT_CAP_PATH.write_text(json.dumps(ticker_weights))
    print(f"[wrds] Saved mkt_cap_weights.json  ({len(ticker_weights)} tickers)")

    return CRSP_MONTHLY_PATH


def fetch_crsp_daily(
    tickers: list[str],
    conn,
    start: str = DEFAULT_START,
    end:   str = DEFAULT_END,
    as_of_date: str = "2025-06-01",
    force: bool = False,
) -> Path:
    """
    Fetch CRSP daily returns for tickers → data/storage/crsp_daily.parquet.
    Requires permno_map.json (run fetch_crsp_monthly first).

    Returns path to the parquet file.
    """
    if CRSP_DAILY_PATH.exists() and not force:
        print("[wrds] crsp_daily.parquet already cached. Use force=True to refresh.")
        return CRSP_DAILY_PATH

    if not PERMNO_MAP_PATH.exists():
        print("[wrds] permno_map.json missing — running fetch_crsp_monthly first ...")
        fetch_crsp_monthly(tickers, conn, start, end, as_of_date)

    permno_map = json.loads(PERMNO_MAP_PATH.read_text())
    permnos    = [permno_map[t] for t in tickers if t in permno_map]

    print(f"[wrds] Fetching CRSP daily returns ({start} → {end}) — this may take a minute ...")
    df = query_crsp_daily(permnos, start, end, conn)
    df.to_parquet(CRSP_DAILY_PATH)
    mb = CRSP_DAILY_PATH.stat().st_size / (1024 ** 2)
    print(f"[wrds] Saved crsp_daily.parquet  {df.shape[0]:,} rows  ({mb:.2f} MB)")
    return CRSP_DAILY_PATH


def fetch_ff_factors(
    conn,
    start: str = DEFAULT_START,
    end:   str = DEFAULT_END,
    force: bool = False,
) -> Path:
    """
    Fetch Fama-French 3-factor + momentum from WRDS ff.factors_monthly
    → data/storage/ff_risk_factors.parquet.
    """
    if FF_FACTORS_PATH.exists() and not force:
        print("[wrds] ff_risk_factors.parquet already cached. Use force=True to refresh.")
        return FF_FACTORS_PATH

    print(f"[wrds] Fetching FF risk factors from WRDS ({start} → {end}) ...")
    df = query_fama_french_factors(start, end, conn)
    df.to_parquet(FF_FACTORS_PATH)
    mb = FF_FACTORS_PATH.stat().st_size / (1024 ** 2)
    print(f"[wrds] Saved ff_risk_factors.parquet  {df.shape[0]} rows  ({mb:.3f} MB)")
    return FF_FACTORS_PATH


# ---------------------------------------------------------------------------
# Load functions — read from parquet cache (raise if not yet fetched)
# ---------------------------------------------------------------------------

def load_crsp_monthly(
    tickers: list[str],
    start: str = DEFAULT_START,
    end:   str = DEFAULT_END,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Load CRSP monthly returns from parquet cache.
    Returns (crsp_monthly_df, permno_map) filtered to requested tickers.
    Raises FileNotFoundError if cache not populated yet.
    """
    if not CRSP_MONTHLY_PATH.exists() or not PERMNO_MAP_PATH.exists():
        raise FileNotFoundError(
            "CRSP monthly cache not found. Run:\n"
            "  from data.fetch.wrds import get_connection, fetch_crsp_monthly\n"
            "  conn = get_connection()\n"
            "  fetch_crsp_monthly(tickers, conn=conn)"
        )
    permno_map_full = json.loads(PERMNO_MAP_PATH.read_text())
    permno_map      = {t: permno_map_full[t] for t in tickers if t in permno_map_full}
    permnos         = list(permno_map.values())

    df   = pd.read_parquet(CRSP_MONTHLY_PATH)
    df   = df[df["permno"].isin(permnos)]
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True), permno_map


def load_crsp_daily(
    tickers: list[str],
    start: str = DEFAULT_START,
    end:   str = DEFAULT_END,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Load CRSP daily returns from parquet cache.
    Returns (crsp_daily_df, permno_map) filtered to requested tickers.
    """
    if not CRSP_DAILY_PATH.exists() or not PERMNO_MAP_PATH.exists():
        raise FileNotFoundError(
            "CRSP daily cache not found. Run:\n"
            "  from data.fetch.wrds import get_connection, fetch_crsp_daily\n"
            "  conn = get_connection()\n"
            "  fetch_crsp_daily(tickers, conn=conn)"
        )
    permno_map_full = json.loads(PERMNO_MAP_PATH.read_text())
    permno_map      = {t: permno_map_full[t] for t in tickers if t in permno_map_full}
    permnos         = list(permno_map.values())

    df   = pd.read_parquet(CRSP_DAILY_PATH)
    df   = df[df["permno"].isin(permnos)]
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True), permno_map


def load_ff_factors(
    start: str = DEFAULT_START,
    end:   str = DEFAULT_END,
) -> pd.DataFrame:
    """
    Load FF risk factors from parquet cache.
    Falls back to Ken French download if WRDS cache not available.
    """
    if FF_FACTORS_PATH.exists():
        df   = pd.read_parquet(FF_FACTORS_PATH)
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask]

    # Fallback: Ken French data library
    from data.fetch.factors import load_ff_risk_factors
    return load_ff_risk_factors(start=start, end=end)


def load_market_cap_weights(tickers: list[str]) -> dict[str, float]:
    """
    Load ticker-level market cap weights from JSON cache.
    Returns equal weights if cache not available.
    """
    if MKT_CAP_PATH.exists():
        all_weights = json.loads(MKT_CAP_PATH.read_text())
        subset      = {t: all_weights[t] for t in tickers if t in all_weights}
        if subset:
            total  = sum(subset.values())
            return {t: w / total for t, w in subset.items()}

    # Fallback: equal weights
    print("[wrds] mkt_cap_weights.json not found — using equal weights")
    n = len(tickers)
    return {t: 1.0 / n for t in tickers}
