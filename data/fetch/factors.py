"""
fetch/factors.py — Fama-French factor data fetcher
=====================================================
Downloads Fama-French 12 Industry Portfolio monthly returns from Ken French's
data library and saves to data/storage/ff12_monthly.parquet.

This replaces the live download in agents/research/loaders.py. The Research
Agent reads from the parquet cache instead of hitting the French data library
on every run.

Usage:
    from data.fetch.factors import fetch_ff12
    fetch_ff12()
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from data import STORAGE_DIR

FF12_URL     = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_CSV.zip"
FF12_PATH    = STORAGE_DIR / "ff12_monthly.parquet"

FF3_URL      = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
MOM_URL      = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
FF_RISK_PATH = STORAGE_DIR / "ff_risk_factors.parquet"


def fetch_ff12(force: bool = False) -> Path:
    """
    Download Fama-French 12 Industry Portfolios (value-weighted monthly returns)
    → data/storage/ff12_monthly.parquet.

    Returns are in percent (as published). Rows: monthly DatetimeIndex.
    Columns: the 12 industry names (NoDur, Durbl, Manuf, Enrgy, HiTec,
             Telcm, Shops, Hlth, Utils, Other, plus 2 more).

    Returns:
        Path to ff12_monthly.parquet.
    """
    if FF12_PATH.exists() and not force:
        print("[ff12] ff12_monthly.parquet already exists. Use force=True to refresh.")
        return FF12_PATH

    print(f"[ff12] Downloading FF12 from Ken French data library ...")
    response = requests.get(FF12_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv"))
        with zf.open(csv_name) as f:
            raw = f.read().decode("latin-1")

    # Ken French CSV files have multiple sections separated by blank lines.
    # The first section (before the first blank line) is the monthly VW returns.
    lines = raw.splitlines()

    header_row = None
    data_rows  = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if data_rows:
                break    # end of first data block
            continue

        if header_row is None:
            if stripped.startswith("NoDur") or "NoDur" in stripped:
                header_row = stripped.split(",")
                header_row = [h.strip() for h in header_row]
            continue

        parts = stripped.split(",")
        if len(parts) < 2:
            continue
        try:
            int(parts[0].strip())   # date column is YYYYMM int
        except ValueError:
            continue
        data_rows.append(parts)

    if not data_rows:
        # Fallback: more lenient parsing — scan for YYYYMM patterns
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 13:
                try:
                    date_int = int(parts[0].strip())
                    if 190001 <= date_int <= 203012:
                        data_rows.append(parts)
                except ValueError:
                    continue

    df = pd.DataFrame(data_rows)
    df = df.apply(lambda col: pd.to_numeric(col.str.strip(), errors="coerce"))
    df.dropna(how="all", inplace=True)

    # First column is YYYYMM date
    date_col = df.iloc[:, 0].astype(int)
    df = df.iloc[:, 1:]

    if header_row and len(header_row) == df.shape[1] + 1:
        df.columns = header_row[1:]
    elif header_row and len(header_row) == df.shape[1]:
        df.columns = header_row
    else:
        df.columns = [f"ind_{i+1}" for i in range(df.shape[1])]

    # Convert YYYYMM → DatetimeIndex (month-start)
    df.index = pd.to_datetime(date_col.astype(str), format="%Y%m") + pd.offsets.MonthBegin(0)
    df.index.name = "date"

    # Replace -99.99 sentinel with NaN (Ken French convention for missing data)
    df.replace(-99.99, float("nan"), inplace=True)

    df.to_parquet(FF12_PATH)
    mb = FF12_PATH.stat().st_size / (1024 ** 2)
    print(f"[ff12] Saved ff12_monthly.parquet  {df.shape[0]} rows × {df.shape[1]} cols  ({mb:.3f} MB)")
    return FF12_PATH


def _parse_french_csv_section(raw: str, expected_cols: list[str]) -> pd.DataFrame:
    """
    Parse the first data section from a Ken French CSV file.
    Returns a DataFrame with a DatetimeIndex (monthly) and float columns.
    """
    lines = raw.splitlines()
    data_rows: list[list[str]] = []
    in_data = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_data and data_rows:
                break
            continue
        parts = [p.strip() for p in stripped.split(",")]
        try:
            date_int = int(parts[0])
            if 190001 <= date_int <= 209912:
                in_data = True
                data_rows.append(parts)
        except (ValueError, IndexError):
            continue

    if not data_rows:
        raise ValueError("No data rows found in Ken French CSV")

    df = pd.DataFrame(data_rows)
    df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    df.dropna(how="all", inplace=True)
    date_col = df.iloc[:, 0].astype(int)
    df = df.iloc[:, 1: len(expected_cols) + 1]
    df.columns = expected_cols
    df.index   = pd.to_datetime(date_col.astype(str), format="%Y%m") + pd.offsets.MonthBegin(0)
    df.index.name = "date"
    df.replace(-99.99, float("nan"), inplace=True)
    df = df / 100.0    # French publishes in percent
    return df


def fetch_ff_risk_factors(force: bool = False) -> Path:
    """
    Download Fama-French 3-factor + momentum monthly risk factors from Ken French's
    data library → data/storage/ff_risk_factors.parquet.

    Columns (decimal, monthly): mktrf, smb, hml, umd, rf
    Index: DatetimeIndex (month-start)

    Used by agents/shared/core/allocation.py to build FF factor views.
    """
    if FF_RISK_PATH.exists() and not force:
        print("[ff_risk] ff_risk_factors.parquet already exists. Use force=True to refresh.")
        return FF_RISK_PATH

    print("[ff_risk] Downloading Fama-French 3-factor data from Ken French ...")
    r3 = requests.get(FF3_URL, timeout=60)
    r3.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r3.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw3 = zf.read(csv_name).decode("latin-1")

    df3 = _parse_french_csv_section(raw3, ["mktrf", "smb", "hml", "rf"])

    print("[ff_risk] Downloading Fama-French momentum factor from Ken French ...")
    rm = requests.get(MOM_URL, timeout=60)
    rm.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(rm.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        rawm = zf.read(csv_name).decode("latin-1")

    dfm = _parse_french_csv_section(rawm, ["umd"])

    df = df3.join(dfm[["umd"]], how="inner")
    df = df[["mktrf", "smb", "hml", "umd", "rf"]]

    df.to_parquet(FF_RISK_PATH)
    mb = FF_RISK_PATH.stat().st_size / (1024 ** 2)
    print(f"[ff_risk] Saved ff_risk_factors.parquet  {df.shape[0]} rows × {df.shape[1]} cols  ({mb:.3f} MB)")
    return FF_RISK_PATH


def load_ff_risk_factors(
    start: str = "2000-01-01",
    end:   str = "2025-12-31",
) -> "pd.DataFrame":
    """
    Load FF risk factors from parquet. Auto-fetches if missing.
    Returns DataFrame with columns [mktrf, smb, hml, umd, rf] in decimal (monthly).
    """
    import pandas as pd
    if not FF_RISK_PATH.exists():
        fetch_ff_risk_factors()
    df = pd.read_parquet(FF_RISK_PATH)
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask]
