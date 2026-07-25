"""
ticker_betas.py — per-ticker market betas for the ETF universe. No weights.

Action item (July 24 minutes, James, due Jul 30): "Implement beta-calculation
routine (no weights) and produce ticker-beta table for review."

Weights are deliberately not computed here. This module estimates and reports
exposures so the team can set beta caps and overlap constraints from measured
numbers rather than assumed ones; the optimiser stays where it is.

Two regressions per ticker, both on monthly CRSP total returns:

    CAPM      r_i - rf = alpha + beta_mkt * (r_m - rf) + e     (Fama-French mktrf)
    vs SPY    r_i - rf = alpha + beta_spy * (r_spy - rf) + e

`beta_spy` and its R-squared are the direct evidence for the review's §3 point
that the book is diversified by ticker count and not by underlying exposure. A
sector ETF with beta_spy near 1 and R-squared near 0.8 is not adding a new risk
factor to a book that already holds SPY — it is levering the one already there.

Standard errors are reported alongside every beta. A beta cap set from a point
estimate with a wide standard error is a cap set on noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
STORAGE_DIR  = PROJECT_ROOT / "data" / "storage"
OUTPUTS_DIR  = PROJECT_ROOT / "data" / "outputs"

_CRSP_MONTHLY = STORAGE_DIR / "crsp_monthly.parquet"
_FF_FACTORS   = STORAGE_DIR / "ff_risk_factors.parquet"
_PERMNO_MAP   = STORAGE_DIR / "permno_map.json"

TICKER_BETA_JSON = OUTPUTS_DIR / "ticker_betas.json"
TICKER_BETA_CSV  = OUTPUTS_DIR / "ticker_betas.csv"

MONTHS_PER_YEAR = 12

MIN_MONTHS = 36
"""
Minimum overlapping months before a beta is reported. Below this the estimate is
returned with `sufficient_history = False` rather than silently published —
XLC and XLRE are young funds and their betas rest on far less data than SPY's.
"""

BENCHMARK = "SPY"
"""Ticker used as the overlap benchmark for the second regression."""


@dataclass
class TickerBeta:
    """Estimated exposures for one ticker. All returns monthly, betas unitless."""

    ticker:             str
    n_months:           int
    sufficient_history: bool
    start:              str
    end:                str

    # CAPM against the Fama-French market factor
    beta_mkt:     float
    beta_mkt_se:  float
    alpha_annual: float
    r2_mkt:       float

    # Regression against SPY — the overlap diagnostic
    beta_spy:     float
    beta_spy_se:  float
    r2_spy:       float
    corr_spy:     float

    vol_annual:          float
    residual_vol_annual: float  # idiosyncratic vol left after removing SPY


def _load_returns() -> pd.DataFrame:
    """Return a month × ticker frame of CRSP total returns."""
    for path in (_CRSP_MONTHLY, _FF_FACTORS, _PERMNO_MAP):
        if not path.exists():
            raise FileNotFoundError(f"Required cache missing: {path}")

    permno_map = json.loads(_PERMNO_MAP.read_text())
    crsp = pd.read_parquet(_CRSP_MONTHLY)

    permno_to_ticker = {v: k for k, v in permno_map.items()}
    crsp = crsp[crsp["permno"].isin(permno_to_ticker)].copy()
    crsp["ticker"] = crsp["permno"].map(permno_to_ticker)
    crsp["date"] = pd.to_datetime(crsp["date"]).dt.to_period("M").dt.to_timestamp()

    return crsp.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last")


def _load_factors() -> pd.DataFrame:
    """Return the Fama-French factor frame indexed by month start."""
    ff = pd.read_parquet(_FF_FACTORS)
    ff.index = pd.to_datetime(ff.index).to_period("M").to_timestamp()
    return ff


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """
    Univariate OLS of y on x with intercept.

    Returns (slope, slope_std_error, intercept, r_squared). Computed in closed
    form rather than via a fitting library — this is the deterministic-math side
    of the pipeline and there is no reason to add a dependency for it.
    """
    n = len(y)
    x_mean, y_mean = x.mean(), y.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    if n < 3 or sxx <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    slope     = float(((x - x_mean) * (y - y_mean)).sum() / sxx)
    intercept = float(y_mean - slope * x_mean)

    resid   = y - (intercept + slope * x)
    dof     = n - 2
    sigma2  = float((resid ** 2).sum() / dof) if dof > 0 else float("nan")
    slope_se = float(np.sqrt(sigma2 / sxx)) if sigma2 == sigma2 else float("nan")

    ss_tot = float(((y - y_mean) ** 2).sum())
    r2     = float(1.0 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")
    return slope, slope_se, intercept, r2


def compute_ticker_betas(
    tickers: list[str] | None = None,
    benchmark: str = BENCHMARK,
) -> dict[str, TickerBeta]:
    """
    Estimate CAPM and vs-benchmark betas for every ticker with CRSP coverage.

    Parameters
    ----------
    tickers : list[str] | None
        Defaults to every ticker present in permno_map.json.
    benchmark : str
        Overlap benchmark for the second regression. Defaults to SPY.
    """
    returns = _load_returns()
    factors = _load_factors()

    if tickers is None:
        tickers = sorted(returns.columns)

    if benchmark not in returns.columns:
        raise ValueError(f"Benchmark {benchmark} has no CRSP returns")

    results: dict[str, TickerBeta] = {}

    for ticker in tickers:
        if ticker not in returns.columns:
            continue

        panel = pd.DataFrame({
            "ret":       returns[ticker],
            "bench":     returns[benchmark],
            "mktrf":     factors["mktrf"],
            "rf":        factors["rf"],
        }).dropna()

        if panel.empty:
            continue

        excess       = (panel["ret"] - panel["rf"]).to_numpy()
        bench_excess = (panel["bench"] - panel["rf"]).to_numpy()
        mktrf        = panel["mktrf"].to_numpy()

        beta_mkt, beta_mkt_se, alpha_m, r2_mkt = _ols(excess, mktrf)
        beta_spy, beta_spy_se, _, r2_spy       = _ols(excess, bench_excess)

        resid = excess - np.nanmean(excess) if len(excess) < 3 else (
            excess - (excess.mean() + beta_spy * (bench_excess - bench_excess.mean()))
        )

        corr = float(np.corrcoef(excess, bench_excess)[0, 1]) if len(excess) > 2 else float("nan")

        results[ticker] = TickerBeta(
            ticker             = ticker,
            n_months           = len(panel),
            sufficient_history = len(panel) >= MIN_MONTHS,
            start              = str(panel.index[0].date()),
            end                = str(panel.index[-1].date()),
            beta_mkt           = round(beta_mkt, 4),
            beta_mkt_se        = round(beta_mkt_se, 4),
            alpha_annual       = round(alpha_m * MONTHS_PER_YEAR, 4),
            r2_mkt             = round(r2_mkt, 4),
            beta_spy           = round(beta_spy, 4),
            beta_spy_se        = round(beta_spy_se, 4),
            r2_spy             = round(r2_spy, 4),
            corr_spy           = round(corr, 4),
            vol_annual         = round(float(excess.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)), 4),
            residual_vol_annual= round(float(resid.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)), 4),
        )

    return results


def beta_table(betas: dict[str, TickerBeta]) -> pd.DataFrame:
    """Render the beta estimates as a frame sorted by market beta, descending."""
    if not betas:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(b) for b in betas.values()])
    return df.sort_values("beta_mkt", ascending=False).reset_index(drop=True)


def overlap_report(betas: dict[str, TickerBeta], threshold: float = 0.70) -> pd.DataFrame:
    """
    Tickers whose variance is mostly explained by the benchmark.

    `r2_spy >= threshold` means holding this fund alongside SPY adds little
    independent risk. This is the measured form of the review's §3 objection.
    """
    df = beta_table(betas)
    if df.empty:
        return df
    overlapping = df[df["r2_spy"] >= threshold]
    return overlapping[["ticker", "beta_mkt", "beta_spy", "r2_spy",
                        "vol_annual", "residual_vol_annual", "n_months"]]


def run(save: bool = True) -> dict[str, TickerBeta]:
    """Compute the table, print it for review, and persist JSON + CSV."""
    betas = compute_ticker_betas()
    df    = beta_table(betas)

    print("\n=== Ticker beta table (monthly CRSP, excess returns) ===")
    print(df[["ticker", "n_months", "beta_mkt", "beta_mkt_se", "r2_mkt",
              "beta_spy", "r2_spy", "vol_annual", "residual_vol_annual",
              "alpha_annual"]].to_string(index=False))

    thin = df[~df["sufficient_history"]]
    if not thin.empty:
        print(f"\nThin history (< {MIN_MONTHS} months) — treat these betas with care:")
        print(thin[["ticker", "n_months", "start"]].to_string(index=False))

    overlap = overlap_report(betas)
    print(f"\n=== Overlap with {BENCHMARK} (R-squared >= 0.70) ===")
    if overlap.empty:
        print("  none")
    else:
        print(overlap.to_string(index=False))
        print(f"\n  {len(overlap)} of {len(df)} holdings are mostly explained by {BENCHMARK}.")

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        TICKER_BETA_JSON.write_text(
            json.dumps({k: asdict(v) for k, v in betas.items()}, indent=2)
        )
        df.to_csv(TICKER_BETA_CSV, index=False)
        print(f"\nSaved → {TICKER_BETA_JSON}")
        print(f"Saved → {TICKER_BETA_CSV}")

    return betas


if __name__ == "__main__":  # pragma: no cover
    run()
