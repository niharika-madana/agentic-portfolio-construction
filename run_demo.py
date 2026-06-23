"""
Demo runner --- full pipeline with synthetic data (no WRDS required).
Produces real portfolio output: weights, risk metrics, VaR, stress tests, LLM rationale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_system.schemas import (
    HumanCapitalInput, RiskProfile, UserProfile,
)
from portfolio_system.pipeline import run_pipeline


# -- Synthetic market data -------------------------------------------------------

np.random.seed(42)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM",  "BAC",  "GS",
    "JNJ",  "UNH",
    "XOM",  "CVX",
    "CAT",  "GE",
    "TSLA",
]

SECTORS = {
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "GOOGL":"Information Technology", "AMZN": "Consumer Discretionary",
    "META": "Communication Services",
    "JPM":  "Financials",             "BAC":  "Financials",
    "GS":   "Financials",
    "JNJ":  "Health Care",            "UNH":  "Health Care",
    "XOM":  "Energy",                 "CVX":  "Energy",
    "CAT":  "Industrials",            "GE":   "Industrials",
    "TSLA": "Consumer Discretionary",
}

N = len(TICKERS)
# Fake PERMNOs (integers) --- real WRDS data would have actual PERMNOs
PERMNO_MAP = {t: 10000 + i for i, t in enumerate(TICKERS)}
INV_PERMNO = {v: k for k, v in PERMNO_MAP.items()}

# Monthly returns: 2005-2024 so all stress periods (2008, 2020) are covered
dates_monthly = pd.date_range("2004-12-31", periods=240, freq="ME")  # 20 years
mean_monthly  = np.linspace(0.005, 0.012, N)
cov_base      = np.eye(N) * 0.002 + np.full((N, N), 0.0005)
monthly_rets  = np.random.multivariate_normal(mean_monthly, cov_base, size=len(dates_monthly))

crsp_monthly_rows = []
for t_idx, ticker in enumerate(TICKERS):
    permno = PERMNO_MAP[ticker]
    for d_idx, date in enumerate(dates_monthly):
        crsp_monthly_rows.append({
            "permno": permno,
            "date":   date,
            "ret":    monthly_rets[d_idx, t_idx],
            "mktcap": 1e9 * (1 + t_idx),
        })
crsp_monthly = pd.DataFrame(crsp_monthly_rows)

# Daily returns: 2005-2024 covers all historical stress windows
dates_daily = pd.date_range("2005-01-03", periods=5040, freq="B")
mean_daily  = mean_monthly.mean() / 21
cov_daily   = cov_base / 21
daily_rets  = np.random.multivariate_normal(
    np.full(N, mean_daily), cov_daily, size=len(dates_daily)
)

crsp_daily_rows = []
for t_idx, ticker in enumerate(TICKERS):
    permno = PERMNO_MAP[ticker]
    for d_idx, date in enumerate(dates_daily):
        crsp_daily_rows.append({
            "permno": permno,
            "date":   date,
            "ret":    daily_rets[d_idx, t_idx],
            "prc":    100.0 + t_idx * 10,
            "vol":    1e6 * (t_idx + 1),
        })
crsp_daily = pd.DataFrame(crsp_daily_rows)

# Fama-French factors matching monthly date range
ff_monthly_arr = np.random.randn(len(dates_monthly), 5) * np.array([0.04, 0.02, 0.02, 0.02, 0.003])
ff_factors = pd.DataFrame(
    ff_monthly_arr,
    index=dates_monthly,
    columns=["mktrf", "smb", "hml", "umd", "rf"],
)
ff_factors["rf"] = np.abs(ff_factors["rf"])  # rf must be positive

risk_free_rate = float(ff_factors["rf"].mean() * 12)

# Equal market-cap weights (by ticker, not PERMNO --- pipeline maps internally)
mkt_weights = {t: 1.0 / N for t in TICKERS}


# -- User profile --------------------------------------------------------------

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


# -- Run pipeline --------------------------------------------------------------

print("\n" + "=" * 60)
print("  MULTI-AGENT PORTFOLIO CONSTRUCTION SYSTEM")
print("  Demo run --- synthetic data, real math + LLM rationale")
print("=" * 60)
print(f"\n  Client: {user_profile.risk_profile.value.title()} risk profile")
print(f"  Financial wealth:   ${user_profile.financial_wealth:>10,.0f}")
print(f"  Human capital (PV): ${user_profile.human_capital.present_value:>10,.0f}")
print(f"  Employer:            {user_profile.human_capital.employer_ticker} ({user_profile.human_capital.employer_sector})")
print(f"  Years to retirement: {user_profile.human_capital.years_to_retirement}")

final = run_pipeline(
    user_profile        = user_profile,
    universe_tickers    = TICKERS,
    market_cap_weights  = mkt_weights,
    sectors             = SECTORS,
    crsp_monthly        = crsp_monthly,
    crsp_daily          = crsp_daily,
    ff_factors          = ff_factors,
    risk_free_rate      = risk_free_rate,
    permno_map          = PERMNO_MAP,
)


# -- Print full output -----------------------------------------------------------

alloc  = final.allocation_output
rm     = final.risk_metrics

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
sorted_weights = sorted(alloc.weights, key=lambda w: w.total_weight, reverse=True)
for w in sorted_weights:
    bar = "#" * int(w.total_weight * 200)
    print(f"    {w.ticker:<6} {w.total_weight:>6.2%}  {bar}")

fe = alloc.portfolio_statistics.factor_exposures
print(f"\n  Factor exposures --- beta:{fe.market_beta:.2f}  SMB:{fe.smb:.2f}  HML:{fe.hml:.2f}  Mom:{fe.mom:.2f}")

print("\n  -- RISK METRICS ------------------------------------------")
vc = rm.var_cvar
print(f"  Volatility (annualized): {rm.volatility:.2%}")
print(f"  VaR  95% / 99% (daily): {vc.var_95:.2%} / {vc.var_99:.2%}")
print(f"  CVaR 95% / 99% (daily): {vc.cvar_95:.2%} / {vc.cvar_99:.2%}")
print(f"  Max drawdown:            {rm.max_drawdown:.2%}")
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

