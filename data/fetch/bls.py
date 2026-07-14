"""
fetch/bls.py — BLS OES data fetcher
======================================
Downloads the BLS Occupational Employment and Wage Statistics national
flat file (May 2023), converts it from xlsx to parquet, and saves it to
data/storage/bls_oes.parquet.

Why xlsx at the source: BLS publishes their OES flat file as an Excel
workbook inside a zip archive. We download and unpack it exactly once,
convert to parquet immediately, and delete the xlsx. All subsequent reads
use the parquet file — nothing else in the pipeline ever touches xlsx.

Usage:
    from data.fetch.bls import fetch_bls_oes
    fetch_bls_oes()
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from data import STORAGE_DIR

BLS_OES_URL  = "https://www.bls.gov/oes/special.requests/oesm23nat.zip"
BLS_OUT_PATH = STORAGE_DIR / "bls_oes.parquet"

# Columns we actually use — drop the rest to keep the file small
KEEP_COLS = [
    "OCC_CODE", "OCC_TITLE", "OCC_GROUP",
    "TOT_EMP",
    "A_PCT10", "A_PCT25", "A_MEDIAN", "A_PCT75", "A_PCT90",
    "H_PCT10", "H_PCT25", "H_MEAN",   "H_PCT75", "H_PCT90",
]


def fetch_bls_oes(force: bool = False) -> Path:
    """
    Download BLS OES May 2023 national file → data/storage/bls_oes.parquet.

    The xlsx is downloaded into memory, parsed, and never written to disk.
    Only the parquet output is persisted.

    Args:
        force: Re-download even if parquet already exists.

    Returns:
        Path to bls_oes.parquet.
    """
    if BLS_OUT_PATH.exists() and not force:
        print("[bls] bls_oes.parquet already exists. Use force=True to refresh.")
        return BLS_OUT_PATH

    print(f"[bls] Downloading BLS OES May 2023 from {BLS_OES_URL} ...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer":  "https://www.bls.gov/oes/tables.htm",
        "Accept":   "application/zip,application/octet-stream,*/*",
    }
    response = requests.get(BLS_OES_URL, headers=headers, timeout=120)
    response.raise_for_status()
    print(f"[bls] Download complete ({len(response.content) / (1024**2):.1f} MB zip)")

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # Find the national xlsx — file name varies slightly across releases
        xlsx_names = [n for n in zf.namelist() if n.endswith(".xlsx") and "nat" in n.lower()]
        if not xlsx_names:
            xlsx_names = [n for n in zf.namelist() if n.endswith(".xlsx")]
        if not xlsx_names:
            raise FileNotFoundError(
                f"No xlsx found in BLS zip. Contents: {zf.namelist()}"
            )
        xlsx_name = xlsx_names[0]
        print(f"[bls] Parsing {xlsx_name} ...")

        with zf.open(xlsx_name) as f:
            df = pd.read_excel(
                io.BytesIO(f.read()),
                sheet_name=0,
                dtype=str,     # read everything as str; clean below
            )

    # Keep only columns we care about (tolerate missing ones gracefully)
    present = [c for c in KEEP_COLS if c in df.columns]
    df = df[present].copy()

    # Convert wage columns to numeric; BLS uses '#' for suppressed, '*' for N/A
    wage_cols = [c for c in present if c not in ("OCC_CODE", "OCC_TITLE", "OCC_GROUP")]
    for col in wage_cols:
        df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")

    df.set_index("OCC_CODE", inplace=True)

    df.to_parquet(BLS_OUT_PATH)
    mb = BLS_OUT_PATH.stat().st_size / (1024 ** 2)
    print(f"[bls] Saved bls_oes.parquet  {len(df)} occupations  ({mb:.3f} MB)")
    return BLS_OUT_PATH
