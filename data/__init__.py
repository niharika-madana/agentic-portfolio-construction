"""
data/ — Centralised data layer for the 5-agent advisory pipeline.

All raw data is fetched once, converted to parquet, and stored in
data/storage/. Agents read from parquet — they never download data
themselves.

Layout:
    data/fetch/      fetch scripts (one per data source)
    data/storage/    parquet files (gitignored)
    data/registry.py manifest: sizes, freshness, budget tracking

Exports:
    PROJECT_ROOT — absolute path to repo root
    STORAGE_DIR  — absolute path to data/storage/
    PRICES_DIR   — absolute path to data/storage/prices/
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
STORAGE_DIR  = Path(__file__).parent / "storage"
PRICES_DIR   = STORAGE_DIR / "prices"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
PRICES_DIR.mkdir(parents=True, exist_ok=True)
