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
