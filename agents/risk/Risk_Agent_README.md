# Risk Agent — Design Document
**AI Financial Advisor Pipeline | Agent 4 of 5**
*Fordham MSQF Capstone 2026*
*Last updated: 2026-06-23*

---

## Implementation Status

🔲 **Stub** — signatures and contracts are fully specified in `orchestrator.py` and `contracts.py`. Implementation pending.

Entry point to implement: `agents/risk/risk_agent.py → assess_risk()`

---

## Purpose

Evaluate whether the Allocation Agent's proposed portfolio is safe for the specific client across all five historical macro regimes. Applies deterministic threshold checks grounded in portfolio theory and regulatory standards. Returns a decision that either clears the portfolio for Compliance review or sends it back to the Allocation Agent for revision.

**The Risk Agent does not modify weights.** It only evaluates. Any portfolio changes are made by the Allocation Agent in response to FLAG violations.

---

## Design Principle

**No LLM decision-making.** All risk metrics are computed deterministically in code. The decision logic is a rule engine applying published thresholds. No model judges a portfolio as "good" or "bad" — the thresholds do.

---

## Inputs

```python
def assess_risk(
    allocation: AllocationAgentOutput,
    profile:    ProfileAgentOutput,
    macro:      MacroRegimeSnapshot,
) -> RiskAgentOutput:
```

**Key fields consumed from ProfileAgentOutput:**
- `income_equity_correlation` (ρ) — HC-adjusted sector limit formula
- `human_capital_type` — determines which sector method applies
- `RSU_concentration` — employer concentration check
- `risk_tolerance_level` — drawdown cap selection
- `income_volatility_sigma` (σ) — macro stress scenario scaling

**Key fields consumed from MacroRegimeSnapshot:**
- `regime_label` — which historical drawdown anchor to apply
- `regime_volatility` — elevates VaR threshold in high-volatility regimes
- `vix` — tail-risk scenario scaling

---

## Checks

### Regime Stress Tests

For each of the 5 academic regimes, compute the portfolio's simulated drawdown using historical regime-window returns from `data/storage/prices/`:

```
portfolio_drawdown = Σ (weight_i × regime_return_i)
drawdown_floor     = benchmark_drawdown × drawdown_multiplier(risk_tolerance)
passed             = portfolio_drawdown ≥ drawdown_floor
```

**Historical benchmark drawdowns per regime (S&P 500 reference):**

| Regime | Anchor Period | Benchmark Drawdown |
|---|---|---|
| Early Recovery | 2003-06 → 2004-06 | −15% (mild) |
| Late-Cycle Expansion | 2005-01 → 2007-06 | −22% |
| Financial Crisis & ZLB | 2008-09 → 2010-12 | −54% |
| Moderate Expansion | 2015-01 → 2019-06 | −18% |
| Inflation Shock | 2022-01 → 2023-06 | −25% |

**Drawdown floor multiplier by risk tolerance:**

| Profile | Multiplier | Floor (Financial Crisis) |
|---|---|---|
| Conservative | 0.70 | −37.8% |
| Moderate | 0.85 | −45.9% |
| Aggressive | 1.00 | −54.0% |

### Position Concentration Limits

For each ticker in the proposed portfolio, derive a position limit using marginal risk contribution:

```
marginal_risk_i = (Σ_j × w)_i / portfolio_volatility
derived_limit   = min(base_single_name_limit, 1 / (1 + marginal_risk_i × scaling_factor))
passed          = actual_weight ≤ derived_limit
```

**Base limits (from contracts.py / Uniform Prudent Investor Act):**
- Max single-name: 10%
- Max employer (employer stock + RSU): 15%

### HC-Correlation Adjusted Sector Limits

For equity-like human capital clients, sector limits are tightened to account for the career-embedded sector exposure:

```
adjusted_sector_limit = base_sector_limit × (1 − ρ)
```

| Client | ρ | Base Limit | Adjusted Limit |
|---|---|---|---|
| Biology Professor | 0.10 | 25% | 22.5% |
| Financial Analyst | 0.40 | 25% | 15.0% |
| Software Developer | 0.75 | 25% | 6.25% |

**Method label:** `"hc_correlation_adjusted"` for equity-like clients, `"standard"` for others.

This method label is validated by Compliance Check 1.3 — if `human_capital_type == "equity-like"` and `sector_method != "hc_correlation_adjusted"`, Compliance raises a HIGH violation.

### Volatility Check

Compute `portfolio_volatility_annual` from the covariance matrix of returns in `data/storage/prices/`:

```python
returns = load_returns(list(weights.keys()))  # from data.fetch.prices
cov     = returns.cov() * 252
w       = np.array([weights[t] for t in returns.columns])
vol     = float(np.sqrt(w @ cov @ w))
```

This field is passed to Compliance for Check 2.5 (volatility suitability bands).

---

## Decision Logic

| Condition | Decision |
|---|---|
| All regime tests PASS and all position/sector limits PASS | `PASS` → forward to Compliance |
| Any regime test or limit FAIL | `FLAG` → return violations to Allocation Agent |
| Structural issue (e.g. weights don't sum to 1.0, unknown tickers) | `REJECT` → halt for human review |

**Note:** Contracts.py uses `PASS / FLAG / REJECT` (not `APPROVE`). The Orchestrator checks `risk.risk_decision == "FLAG"` to trigger Loop A revisions.

---

## Output Schema (from contracts.py)

```python
class RiskAgentOutput(BaseModel):
    risk_decision:              RiskDecision          # PASS / FLAG / REJECT
    regime_evaluation:          dict[str, RegimeEvaluation]
    position_limits:            dict[str, PositionLimit]
    sector_limits:              dict[str, SectorLimit]
    violations:                 list[str]
    derivation:                 RiskDerivation
    warnings:                   list[str]
    portfolio_volatility_annual: float | None         # None if covariance unavailable
```

### Supporting models

```python
class RegimeEvaluation(BaseModel):
    benchmark_drawdown: float   # historical S&P drawdown
    drawdown_floor:     float   # risk-tolerance-adjusted floor
    portfolio_drawdown: float   # simulated portfolio drawdown
    passed:             bool

class PositionLimit(BaseModel):
    derived_limit:  float   # computed from marginal risk contribution
    actual_weight:  float
    passed:         bool
    method:         str     # "marginal_risk_contribution"

class SectorLimit(BaseModel):
    base_limit:          float
    correlation_with_hc: float
    adjusted_limit:      float
    actual_sector_weight: float
    passed:              bool

class RiskDerivation(BaseModel):
    drawdown_method:       str   # "benchmark_relative_per_regime"
    concentration_method:  str   # "marginal_risk_contribution"
    sector_method:         str   # "hc_correlation_adjusted" or "standard"
    hc_type:               str   # from ProfileAgentOutput
    rsu_concentration:     float
    data_source:           str   # "data/storage/prices/"
```

---

## Example Output (Software Developer, FLAG)

```json
{
    "risk_decision": "FLAG",
    "regime_evaluation": {
        "Early Recovery":         { "benchmark_drawdown": -0.15, "drawdown_floor": -0.15, "portfolio_drawdown": -0.11, "passed": true },
        "Late-Cycle Expansion":   { "benchmark_drawdown": -0.22, "drawdown_floor": -0.22, "portfolio_drawdown": -0.18, "passed": true },
        "Financial Crisis & ZLB": { "benchmark_drawdown": -0.54, "drawdown_floor": -0.54, "portfolio_drawdown": -0.48, "passed": true },
        "Moderate Expansion":     { "benchmark_drawdown": -0.18, "drawdown_floor": -0.18, "portfolio_drawdown": -0.14, "passed": true },
        "Inflation Shock":        { "benchmark_drawdown": -0.25, "drawdown_floor": -0.25, "portfolio_drawdown": -0.31, "passed": false }
    },
    "position_limits": {
        "AGG": { "derived_limit": 0.30, "actual_weight": 0.35, "passed": false, "method": "marginal_risk_contribution" },
        "TLT": { "derived_limit": 0.20, "actual_weight": 0.20, "passed": true, "method": "marginal_risk_contribution" }
    },
    "sector_limits": {
        "Technology": { "base_limit": 0.25, "correlation_with_hc": 0.75, "adjusted_limit": 0.0625, "actual_sector_weight": 0.00, "passed": true }
    },
    "violations": [
        "Inflation Shock: portfolio_drawdown -31.0% exceeds floor -25.0%",
        "AGG: actual_weight 35.0% exceeds derived_limit 30.0%"
    ],
    "derivation": {
        "drawdown_method":      "benchmark_relative_per_regime",
        "concentration_method": "marginal_risk_contribution",
        "sector_method":        "hc_correlation_adjusted",
        "hc_type":              "equity-like",
        "rsu_concentration":    0.35,
        "data_source":          "data/storage/prices/"
    },
    "warnings": [],
    "portfolio_volatility_annual": 0.089
}
```

---

## Data Sources

**`data/storage/prices/{TICKER}.parquet`** — daily adjusted prices, 2000–2025, for all tickers in the Allocation Agent's universe. Used to:
- Compute historical drawdowns per regime anchor window
- Compute covariance matrix for `portfolio_volatility_annual`
- Compute marginal risk contributions for position limits

Fetched by `data/fetch/prices.py` (yfinance). No WRDS or CRSP dependency.

**`data/storage/ff12_monthly.parquet`** — Fama-French 12 Industry Portfolios. Used for factor exposure decomposition (supplementary; not required for PASS/FLAG decision).

---

## Implementation Plan

**Suggested module structure:**

```
agents/risk/
├── risk_agent.py        ← assess_risk() — entry point
├── stress.py            ← regime drawdown simulation from parquet data
├── concentration.py     ← position limits via marginal risk contribution
├── sector_limits.py     ← HC-correlation adjusted sector check
├── vol.py               ← portfolio_volatility_annual from covariance
└── test_risk.py
```

**Implementation order:**
1. `vol.py` — simplest; just covariance math, no regime logic
2. `concentration.py` — position limit derivation
3. `sector_limits.py` — HC-adjusted formula, need ρ from profile
4. `stress.py` — regime drawdown; need parquet price data loaded correctly
5. `risk_agent.py` — wire everything into `RiskAgentOutput`
6. `test_risk.py` — adversarial tests: single-stock portfolio → REJECT; good diversified portfolio → PASS

---

## Key References

- Bodie, Merton & Samuelson (1992). *Labor Supply Flexibility and Portfolio Choice.* Journal of Economic Dynamics and Control.
- Davis & Willen (2000). *Using Financial Assets to Hedge Labor Income Risks.* SSRN.
- Ibbotson, Milevsky, Chen & Zhu (2007). *Lifetime Financial Advice.* CFA Institute Research Foundation.
- Basel III/IV — VaR/CVaR at 95%/99% confidence levels.
- Uniform Prudent Investor Act — single-name and employer concentration limits.
