"""
fetch/prices.py — Equity and ETF price fetcher
================================================
Downloads daily adjusted OHLCV data via yfinance for the full asset-class
universe needed for regime-based portfolio backtesting.

One parquet file per ticker: data/storage/prices/{TICKER}.parquet
Columns: open, high, low, close, adj_close, volume  (Date as index)

Estimated storage: ~1-3 MB per ticker (~30-60 MB for the full universe).

Usage:
    from data.fetch.prices import fetch_prices, fetch_all
    fetch_all()                          # download full universe
    fetch_prices(["SPY", "AGG"])         # download subset
    fetch_prices(["AAPL"], force=True)   # force refresh
"""

from pathlib import Path

import pandas as pd

from data import PRICES_DIR

# ---------------------------------------------------------------------------
# Default ticker universe — covers all asset classes needed for backtesting
# ---------------------------------------------------------------------------

BROAD_EQUITY = [
    "SPY",   # S&P 500 (large-cap US)
    "IWM",   # Russell 2000 (small-cap US)
    "EFA",   # MSCI EAFE (developed international)
    "EEM",   # MSCI Emerging Markets
]

FIXED_INCOME = [
    "AGG",   # US Aggregate Bond (broad)
    "TLT",   # 20+ Year US Treasury
    "IEF",   # 7-10 Year US Treasury
    "SHY",   # 1-3 Year US Treasury
    "HYG",   # USD High Yield Corporate
    "LQD",   # USD Investment Grade Corporate
    "TIP",   # TIPS (inflation-protected)
]

REAL_ASSETS = [
    "GLD",   # Gold
    "VNQ",   # US REITs
    "DJP",   # Bloomberg Commodity Index
]

SECTOR_ETFS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLV",   # Health Care
    "XLE",   # Energy
    "XLI",   # Industrials
    "XLC",   # Communication Services
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLRE",  # Real Estate (sector)
]

CASH_PROXY = [
    "BIL",   # 1-3 Month T-Bill (cash equivalent)
]

DEFAULT_TICKERS = BROAD_EQUITY + FIXED_INCOME + REAL_ASSETS + SECTOR_ETFS + CASH_PROXY

# Regimes that need coverage — earliest start date drives the data pull
# Early Recovery starts 2003-01; pull from 2000 to give rolling-window room
DEFAULT_START = "2000-01-01"
DEFAULT_END   = "2025-12-31"


def _ticker_path(ticker: str) -> Path:
    return PRICES_DIR / f"{ticker}.parquet"


def fetch_prices(
    tickers: list[str] | None = None,
    start:   str = DEFAULT_START,
    end:     str = DEFAULT_END,
    force:   bool = False,
) -> dict[str, Path]:
    """
    Download daily OHLCV price data for each ticker via yfinance.

    Each ticker is saved to a separate parquet file. Existing files are
    skipped unless force=True. Tickers that fail to download (bad symbol,
    no data for the period) are logged and skipped — they do not crash
    the batch.

    Args:
        tickers: List of ticker symbols. Defaults to DEFAULT_TICKERS.
        start:   Start date string "YYYY-MM-DD".
        end:     End date string "YYYY-MM-DD".
        force:   Re-download even if parquet already exists.

    Returns:
        Dict mapping ticker → parquet path (only for tickers that succeeded).
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install it with: pip install yfinance")

    if tickers is None:
        tickers = DEFAULT_TICKERS

    saved    = {}
    skipped  = []
    failed   = []

    for ticker in tickers:
        out_path = _ticker_path(ticker)

        if out_path.exists() and not force:
            skipped.append(ticker)
            saved[ticker] = out_path
            continue

        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )

            if raw.empty:
                print(f"  [prices] {ticker}: no data returned — skipping")
                failed.append(ticker)
                continue

            # Flatten MultiIndex columns yfinance sometimes returns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = raw.rename(columns={
                "Open":      "open",
                "High":      "high",
                "Low":       "low",
                "Close":     "close",
                "Adj Close": "adj_close",
                "Volume":    "volume",
            })
            df.index.name = "date"

            # Keep only canonical columns that exist
            keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"]
                    if c in df.columns]
            df = df[keep]

            df.to_parquet(out_path)
            mb = out_path.stat().st_size / (1024 ** 2)
            print(f"  [prices] {ticker:<6}  {len(df):>5} rows  {mb:.2f} MB")
            saved[ticker] = out_path

        except Exception as e:
            print(f"  [prices] {ticker}: ERROR — {e}")
            failed.append(ticker)

    print(f"\n[prices] Done — {len(saved)} saved, {len(skipped)} already cached, "
          f"{len(failed)} failed")
    if failed:
        print(f"[prices] Failed tickers: {failed}")

    return saved


def fetch_all(force: bool = False) -> dict[str, Path]:
    """Download the full DEFAULT_TICKERS universe."""
    print(f"[prices] Fetching {len(DEFAULT_TICKERS)} tickers "
          f"({DEFAULT_START} → {DEFAULT_END}) ...")
    return fetch_prices(DEFAULT_TICKERS, force=force)


def load_prices(tickers: list[str]) -> pd.DataFrame:
    """
    Read adj_close prices from parquet for a list of tickers.

    Returns a wide DataFrame: Date index × ticker columns.
    Missing tickers (not yet fetched) are skipped with a warning.
    """
    frames = {}
    for ticker in tickers:
        path = _ticker_path(ticker)
        if not path.exists():
            print(f"[prices] {ticker} not in storage — run fetch_prices(['{ticker}'])")
            continue
        df = pd.read_parquet(path, columns=["adj_close"])
        frames[ticker] = df["adj_close"]

    if not frames:
        raise FileNotFoundError(
            "No price data found. Run data.fetch.prices.fetch_all() first."
        )
    return pd.DataFrame(frames)


def load_returns(tickers: list[str], freq: str = "D") -> pd.DataFrame:
    """
    Load adjusted prices and return a DataFrame of simple daily (or monthly)
    returns. freq='D' for daily, 'MS' for monthly.

    Used by the Allocation and Risk agents for regime-window backtesting.
    """
    prices = load_prices(tickers)
    if freq != "D":
        prices = prices.resample(freq).last()
    return prices.pct_change().dropna(how="all")
