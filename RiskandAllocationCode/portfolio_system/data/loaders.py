from __future__ import annotations

import wrds
import pandas as pd


# ── Connection ─────────────────────────────────────────────────────────────────

def get_connection() -> wrds.Connection:
    """Open a WRDS connection. Prompts for Fordham credentials on first use."""
    return wrds.Connection()


# ── Ticker → PERMNO ────────────────────────────────────────────────────────────

def load_permno_map(
    tickers: list[str],
    as_of_date: str,
    conn: wrds.Connection,
) -> pd.DataFrame:
    """
    Map ticker symbols to CRSP PERMNOs as of as_of_date.
    Returns DataFrame with columns [ticker, permno].
    CRSP uses PERMNO as its primary key; all subsequent CRSP calls take PERMNOs.
    """
    ticker_list = ", ".join(f"'{t.upper()}'" for t in tickers)
    query = f"""
        SELECT DISTINCT ticker, permno
        FROM crsp.dsenames
        WHERE ticker IN ({ticker_list})
          AND namedt  <= '{as_of_date}'
          AND (nameendt >= '{as_of_date}' OR nameendt IS NULL)
    """
    return conn.raw_sql(query)


def tickers_to_permnos(
    tickers: list[str],
    as_of_date: str,
    conn: wrds.Connection,
) -> dict[str, int]:
    """Convenience wrapper: returns {ticker: permno}."""
    df = load_permno_map(tickers, as_of_date, conn)
    return dict(zip(df["ticker"], df["permno"].astype(int)))


# ── CRSP Daily Returns ─────────────────────────────────────────────────────────

def load_crsp_daily(
    permnos: list[int],
    start_date: str,
    end_date: str,
    conn: wrds.Connection,
) -> pd.DataFrame:
    """
    Daily returns and prices for the given PERMNOs.
    Columns: permno, date, ret, prc, vol.
    Used by core/risk.py for historical VaR/CVaR simulation.
    prc is stored as negative for bid-ask midpoints in CRSP — ABS() normalizes it.
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


# ── CRSP Monthly Returns ───────────────────────────────────────────────────────

def load_crsp_monthly(
    permnos: list[int],
    start_date: str,
    end_date: str,
    conn: wrds.Connection,
) -> pd.DataFrame:
    """
    Monthly returns and market cap for the given PERMNOs.
    Columns: permno, date, ret, mktcap.
    Used by core/allocation.py to build the covariance matrix.
    mktcap = ABS(prc) * shrout (shares outstanding in thousands).
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
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    return df


# ── CRSP Treasury / Risk-Free Rate ────────────────────────────────────────────

def load_risk_free_rate(
    start_date: str,
    end_date: str,
    conn: wrds.Connection,
) -> pd.Series:
    """
    Monthly risk-free rate from the Fama-French factors table on WRDS.
    Index: date, values: rf in decimal (e.g. 0.004 = 0.4% per month).
    Used as the discount rate for human capital (BMS) and BL equilibrium returns.
    """
    query = f"""
        SELECT date, rf
        FROM ff.factors_monthly
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    df = conn.raw_sql(query, date_cols=["date"])
    return df.set_index("date")["rf"].astype(float)


# ── Compustat GICS Sector Mapping ─────────────────────────────────────────────

_GICS_SECTOR_NAMES: dict[str, str] = {
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


def load_gics_sectors(
    tickers: list[str],
    conn: wrds.Connection,
) -> dict[str, str]:
    """
    Returns {ticker: sector_name} using Compustat GICS classifications.
    Used by core/constraints.py and core/risk.py for sector concentration checks.
    """
    ticker_list = ", ".join(f"'{t.upper()}'" for t in tickers)
    query = f"""
        SELECT tic, gsector
        FROM comp.company
        WHERE tic IN ({ticker_list})
    """
    df = conn.raw_sql(query).dropna(subset=["gsector"])
    df["gsector"] = df["gsector"].astype(str).str.strip()
    return {
        row.tic: _GICS_SECTOR_NAMES.get(row.gsector, f"Sector {row.gsector}")
        for row in df.itertuples()
    }


# ── Fama-French Factors ───────────────────────────────────────────────────────

def load_fama_french_factors(
    start_date: str,
    end_date: str,
    conn: wrds.Connection,
) -> pd.DataFrame:
    """
    Monthly Fama-French three-factor + momentum returns from WRDS.
    Columns: mktrf, smb, hml, umd, rf (all decimal, index = date).
    Used by core/allocation.py for view tilts and factor exposure regression.
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


# ── Market-Cap Weights ────────────────────────────────────────────────────────

def load_market_cap_weights(
    permnos: list[int],
    as_of_date: str,
    conn: wrds.Connection,
) -> dict[int, float]:
    """
    Returns {permno: weight} summing to 1.0, derived from market cap
    (ABS(prc) * shrout) at the most recent month-end on or before as_of_date.
    Used by core/allocation.py as the CAPM equilibrium starting point for BL.
    """
    permno_list = ", ".join(str(p) for p in permnos)
    query = f"""
        SELECT permno, ABS(prc) * shrout AS mktcap
        FROM crsp.msf
        WHERE permno IN ({permno_list})
          AND date = (
              SELECT MAX(date)
              FROM crsp.msf
              WHERE date <= '{as_of_date}'
          )
    """
    df = conn.raw_sql(query).dropna()
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    total = df["mktcap"].sum()
    return {int(row.permno): row.mktcap / total for row in df.itertuples()}
