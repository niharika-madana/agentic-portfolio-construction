"""
Real-data pipeline runner.
Uses yfinance for prices/returns/volume/market-cap and Kenneth French's
data library for monthly FF 3-factor + momentum factors.
No WRDS credentials required.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_system.schemas import HumanCapitalInput, RiskProfile, UserProfile
from portfolio_system.pipeline import run_pipeline


# ── Universe ───────────────────────────────────────────────────────────────────

TICKERS = [
    # ── US Equities ──────────────────────────────────────────────────────────
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM",  "BAC",  "GS",
    "JNJ",  "UNH",
    "XOM",  "CVX",
    "CAT",  "GE",
    "TSLA",
    # ── US Bond ETFs ─────────────────────────────────────────────────────────
    "TLT",   # iShares 20+ Year Treasury    — rallies during equity crashes
    "IEF",   # iShares 7-10 Year Treasury   — intermediate duration
    "LQD",   # iShares IG Corporate Bonds   — credit spread exposure
    "TIP",   # iShares TIPS                 — inflation hedge
    # ── International Equity ETFs ────────────────────────────────────────────
    "VEA",   # Vanguard Developed Markets   — Europe/Japan diversification
    "VWO",   # Vanguard Emerging Markets    — EM growth
    # ── Alternatives ─────────────────────────────────────────────────────────
    "GLD",   # SPDR Gold                    — crisis hedge, low equity corr.
    "VNQ",   # Vanguard Real Estate         — income + inflation hedge
    "IWM",   # iShares Russell 2000         — small-cap size premium
]

SECTORS = {
    # US Equities — GICS sectors
    "AAPL":  "Information Technology",
    "MSFT":  "Information Technology",
    "GOOGL": "Information Technology",
    "AMZN":  "Consumer Discretionary",
    "META":  "Communication Services",
    "JPM":   "Financials",
    "BAC":   "Financials",
    "GS":    "Financials",
    "JNJ":   "Health Care",
    "UNH":   "Health Care",
    "XOM":   "Energy",
    "CVX":   "Energy",
    "CAT":   "Industrials",
    "GE":    "Industrials",
    "TSLA":  "Consumer Discretionary",
    # Bond ETFs — split into two sub-sectors so optimizer can hold up to 40% bonds
    "TLT":   "US Treasuries",
    "IEF":   "US Treasuries",
    "TIP":   "US Treasuries",
    "LQD":   "US Corporate Bonds",
    # International equity — separate sector from US equities
    "VEA":   "International Developed",
    "VWO":   "International Emerging",
    # Alternatives
    "GLD":   "Commodities",
    "VNQ":   "Real Estate",
    "IWM":   "US Small Cap",
}

# Fake PERMNOs — the pipeline uses these only as DataFrame join keys
PERMNO_MAP = {t: 10000 + i for i, t in enumerate(TICKERS)}

START_DATE = "2005-01-01"
END_DATE   = "2025-06-01"


# ── Fama-French factors ────────────────────────────────────────────────────────

def _fetch_ff_zip(url: str, value_cols: list[str]) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=20) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
        raw = zf.read(zf.namelist()[0]).decode("utf-8")

    rows = []
    for line in raw.split("\n"):
        parts = line.strip().split(",")
        if len(parts) >= len(value_cols) + 1 and parts[0].strip().isdigit() and len(parts[0].strip()) == 6:
            try:
                rows.append([parts[0].strip()] + [float(x) / 100 for x in parts[1:len(value_cols) + 1]])
            except ValueError:
                continue

    df = pd.DataFrame(rows, columns=["date"] + value_cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.set_index("date")


def load_ff_factors(start: str, end: str) -> pd.DataFrame:
    print("  Downloading Fama-French 3-factor data...")
    ff3 = _fetch_ff_zip(
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
        ["mktrf", "smb", "hml", "rf"],
    )
    print("  Downloading Fama-French momentum data...")
    mom = _fetch_ff_zip(
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
        ["umd"],
    )
    ff = ff3.join(mom, how="inner")[["mktrf", "smb", "hml", "umd", "rf"]]
    return ff.loc[start:end]


# ── Price / return data ────────────────────────────────────────────────────────

def load_price_data(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (daily_prices, monthly_prices) DataFrames indexed by date,
    columns = tickers. Adjusted close prices from yfinance.
    Tickers with no data in the period are dropped and a warning is printed.
    """
    print(f"  Downloading price data for {len(tickers)} tickers ({start} → {end})...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"]

    # Drop tickers missing more than 20% of their expected daily observations
    threshold = 0.80 * len(close)
    good = [t for t in tickers if close[t].notna().sum() >= threshold]
    dropped = set(tickers) - set(good)
    if dropped:
        print(f"  WARNING: dropped tickers with insufficient history: {sorted(dropped)}")

    close = close[good]
    daily_prices   = close
    monthly_prices = close.resample("ME").last()
    return daily_prices, monthly_prices


def build_crsp_daily(
    daily_prices: pd.DataFrame,
    volume: pd.DataFrame,
    permno_map: dict[str, int],
) -> pd.DataFrame:
    tickers = daily_prices.columns.tolist()
    rets    = daily_prices.pct_change()

    rows = []
    for ticker in tickers:
        permno = permno_map[ticker]
        df = pd.DataFrame({
            "permno": permno,
            "date":   daily_prices.index,
            "ret":    rets[ticker].values,
            "prc":    daily_prices[ticker].values,
            "vol":    volume[ticker].values if ticker in volume.columns else np.nan,
        })
        rows.append(df.dropna(subset=["ret"]))

    return pd.concat(rows, ignore_index=True)


def build_crsp_monthly(
    monthly_prices: pd.DataFrame,
    mktcap: dict[str, float],
    permno_map: dict[str, int],
) -> pd.DataFrame:
    tickers = monthly_prices.columns.tolist()
    rets    = monthly_prices.pct_change()

    rows = []
    for ticker in tickers:
        permno = permno_map[ticker]
        df = pd.DataFrame({
            "permno": permno,
            "date":   monthly_prices.index,
            "ret":    rets[ticker].values,
            "mktcap": mktcap.get(ticker, 1.0),
        })
        rows.append(df.dropna(subset=["ret"]))

    return pd.concat(rows, ignore_index=True)


def get_market_caps(tickers: list[str]) -> dict[str, float]:
    print("  Fetching current market caps / AUM...")
    caps = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            # Stocks → market cap; ETFs → total net assets (AUM)
            val = tk.fast_info.market_cap
            if not val:
                val = tk.info.get("totalAssets") or 1.0
            caps[t] = float(val)
        except Exception:
            caps[t] = 1.0
    return caps


def market_cap_weights(caps: dict[str, float], tickers: list[str]) -> dict[str, float]:
    total = sum(caps.get(t, 0.0) for t in tickers)
    return {t: caps.get(t, 0.0) / total for t in tickers}


# ── Main ───────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  MULTI-AGENT PORTFOLIO CONSTRUCTION SYSTEM")
print("  Real data run — yfinance + Kenneth French factors")
print("=" * 60)

print("\n[1/4] Loading Fama-French factors...")
ff_factors = load_ff_factors(START_DATE, END_DATE)
risk_free_rate = float(ff_factors["rf"].mean() * 12)
print(f"      {len(ff_factors)} monthly observations  |  "
      f"mean annual rf: {risk_free_rate:.2%}")

print("\n[2/4] Loading price data...")
raw_all = yf.download(TICKERS, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
close  = raw_all["Close"]
volume = raw_all["Volume"]

threshold = 0.80 * len(close)
active_tickers = [t for t in TICKERS if close[t].notna().sum() >= threshold]
dropped = set(TICKERS) - set(active_tickers)
if dropped:
    print(f"  WARNING: dropped tickers with insufficient history: {sorted(dropped)}")

close  = close[active_tickers]
volume = volume[active_tickers]

daily_prices   = close
monthly_prices = close.resample("ME").last()
print(f"      {len(daily_prices)} daily rows  |  "
      f"{len(monthly_prices)} monthly rows  |  "
      f"{len(active_tickers)} tickers")

print("\n[3/4] Fetching market caps and building DataFrames...")
mktcaps = get_market_caps(active_tickers)
mkt_weights = market_cap_weights(mktcaps, active_tickers)

crsp_daily   = build_crsp_daily(daily_prices, volume, PERMNO_MAP)
crsp_monthly = build_crsp_monthly(monthly_prices, mktcaps, PERMNO_MAP)

active_sectors = {t: SECTORS[t] for t in active_tickers}
active_permno  = {t: PERMNO_MAP[t] for t in active_tickers}

print(f"      crsp_daily rows:   {len(crsp_daily):,}")
print(f"      crsp_monthly rows: {len(crsp_monthly):,}")

print("\n[4/4] Running pipeline...")

user_profile = UserProfile(
    financial_wealth=2_000_000,
    human_capital=HumanCapitalInput(
        present_value=250_000,
        employer_ticker="JPM",
        employer_sector="Financials",
        income_volatility=0.12,
        income_beta=0.6,
        years_to_retirement=15,
        discount_rate=0.04,
    ),
    risk_profile=RiskProfile.MODERATE,
)

print(f"\n  Client: {user_profile.risk_profile.value.title()} risk profile")
print(f"  Financial wealth:   ${user_profile.financial_wealth:>10,.0f}")
print(f"  Human capital (PV): ${user_profile.human_capital.present_value:>10,.0f}")
print(f"  Employer:            {user_profile.human_capital.employer_ticker} "
      f"({user_profile.human_capital.employer_sector})")
print(f"  Years to retirement: {user_profile.human_capital.years_to_retirement}")

final = run_pipeline(
    user_profile        = user_profile,
    universe_tickers    = active_tickers,
    market_cap_weights  = mkt_weights,
    sectors             = active_sectors,
    crsp_monthly        = crsp_monthly,
    crsp_daily          = crsp_daily,
    ff_factors          = ff_factors,
    risk_free_rate      = risk_free_rate,
    permno_map          = active_permno,
)


# ── Output ─────────────────────────────────────────────────────────────────────

alloc = final.allocation_output
rm    = final.risk_metrics

print("\n\n" + "=" * 60)
print("  FINAL RESULT")
print("=" * 60)

print(f"\n  DECISION: {final.decision.value.upper()}")

print("\n  -- ALLOCATION --------------------------------------------")
print(f"  Risky sleeve:  {alloc.risky_weight:.1%} of financial wealth")
print(f"  Safe/cash:     {alloc.safe_weight:.1%} of financial wealth")
print(f"  Expected return (risky): {alloc.portfolio_statistics.expected_return:.2%}")
print(f"  Volatility:              {alloc.portfolio_statistics.volatility:.2%}")
print(f"  Sharpe ratio:            {alloc.portfolio_statistics.sharpe_ratio:.2f}")

print("\n  Holdings (within risky sleeve):")
for w in sorted(alloc.weights, key=lambda w: w.total_weight, reverse=True):
    bar = "#" * int(w.total_weight * 200)
    print(f"    {w.ticker:<6} {w.total_weight:>6.2%}  {bar}")

fe = alloc.portfolio_statistics.factor_exposures
print(f"\n  Factor exposures — beta:{fe.market_beta:.2f}  SMB:{fe.smb:.2f}  "
      f"HML:{fe.hml:.2f}  Mom:{fe.mom:.2f}")

print("\n  -- RISK METRICS ------------------------------------------")
vc = rm.var_cvar
print(f"  Volatility (annualized): {rm.volatility:.2%}")
print(f"  VaR  95% / 99% (daily): {vc.var_95:.2%} / {vc.var_99:.2%}")
print(f"  CVaR 95% / 99% (daily): {vc.cvar_95:.2%} / {vc.cvar_99:.2%}")
print(f"  Max drawdown:            {rm.max_drawdown:.2%}  (cap: {rm.effective_drawdown_cap:.1%}  regime: {rm.market_regime.value.upper()})")
print(f"  Liquidity score:         {rm.liquidity_score:.2f}")

hca = rm.hc_adjusted
print(f"\n  Human capital adjusted:")
print(f"    Total wealth (fin + HC):    ${hca.total_wealth:>12,.0f}")
print(f"    HC fraction:                 {hca.hc_fraction:.1%}")
print(f"    Effective equity exposure:   {hca.effective_equity_exposure:.1%}")
print(f"    Employer concentration:      {hca.employer_concentration:.1%}")

print("\n  Stress tests:")
for s in rm.stress_results:
    flag = " !!" if s.threshold_breached else ""
    print(f"    {s.scenario:<30} loss: {s.portfolio_loss:.1%}  [{s.severity.value.upper()}]{flag}")

print("\n  -- ALLOCATION RATIONALE (LLM) ----------------------------")
print()
for line in alloc.rationale.split("\n"):
    print(f"  {line}")

print("\n  -- RISK REASONING TRACE (LLM) ----------------------------")
print()
for line in final.reasoning_trace.split("\n"):
    print(f"  {line}")

if final.constraints_violated:
    print("\n  Constraints violated:")
    for c in final.constraints_violated:
        print(f"    {c.constraint_type.value}: {c.target}  "
              f"({c.current_value:.1%} vs limit {c.limit:.1%})")

print("\n" + "=" * 60)
