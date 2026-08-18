"""
registry.py — Data storage manifest
=====================================
Lists all parquet files in data/storage/, their sizes, and last-updated
timestamps. Tracks usage against the 500MB project data budget.

Usage:
    from data.registry import show_registry, budget_status
    show_registry()
    budget_status()
"""

from pathlib import Path

from data import PRICES_DIR, STORAGE_DIR

BUDGET_MB = 500.0

# Expected datasets — used to flag what's missing
EXPECTED_DATASETS = {
    "fred_macro.parquet":  "Research Agent — 13 FRED macro series, monthly",
    "fred_dgs10.parquet":  "Profile Agent  — live 10Y Treasury yield (DGS10)",
    "bls_oes.parquet":     "Profile Agent  — BLS OES May 2023 national wages by SOC",
    "ff12_monthly.parquet":"Research Agent — Fama-French 12 Industry Portfolios (monthly)",
}


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def _last_modified(path: Path) -> str:
    import datetime
    ts = path.stat().st_mtime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def list_datasets() -> list[dict]:
    """Return metadata for every parquet file in storage."""
    rows = []

    # Top-level parquet files
    for p in sorted(STORAGE_DIR.glob("*.parquet")):
        rows.append({
            "name":          p.name,
            "path":          str(p.relative(STORAGE_DIR.parent)),
            "size_mb":       round(_file_size_mb(p), 3),
            "last_modified": _last_modified(p),
        })

    # Prices sub-folder
    for p in sorted(PRICES_DIR.glob("*.parquet")):
        rows.append({
            "name":          f"prices/{p.name}",
            "path":          str(p.relative(STORAGE_DIR.parent)),
            "size_mb":       round(_file_size_mb(p), 3),
            "last_modified": _last_modified(p),
        })

    return rows


def total_size_mb() -> float:
    return sum(r["size_mb"] for r in list_datasets())


def show_registry():
    """Print a formatted table of all stored datasets."""
    datasets = list_datasets()
    used_mb  = sum(r["size_mb"] for r in datasets)

    print(f"\n{'='*70}")
    print(f"  DATA REGISTRY — {len(datasets)} datasets | "
          f"{used_mb:.1f} MB used / {BUDGET_MB:.0f} MB budget "
          f"({used_mb/BUDGET_MB*100:.1f}%)")
    print(f"{'='*70}")

    if not datasets:
        print("  (empty — run data/fetch scripts to populate)")
    else:
        print(f"  {'Dataset':<35} {'Size MB':>8}  {'Last Updated'}")
        print(f"  {'-'*35} {'-'*8}  {'-'*16}")
        for r in datasets:
            print(f"  {r['name']:<35} {r['size_mb']:>8.3f}  {r['last_modified']}")

    # Flag missing expected datasets
    stored_names = {r["name"] for r in datasets}
    missing = {k: v for k, v in EXPECTED_DATASETS.items() if k not in stored_names}
    if missing:
        print(f"\n  MISSING ({len(missing)}):")
        for name, desc in missing.items():
            print(f"    ✗ {name:<35}  {desc}")

    print(f"{'='*70}\n")


def budget_status() -> dict:
    """Return budget usage summary dict."""
    used   = total_size_mb()
    remain = BUDGET_MB - used
    return {
        "budget_mb":    BUDGET_MB,
        "used_mb":      round(used, 2),
        "remaining_mb": round(remain, 2),
        "pct_used":     round(used / BUDGET_MB * 100, 1),
    }
