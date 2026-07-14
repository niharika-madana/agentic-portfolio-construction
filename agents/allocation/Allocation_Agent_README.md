# Allocation Agent — Design Document
**AI Financial Advisor Pipeline | Agent 3 of 5**
*Fordham MSQF Capstone 2026*
*Last updated: 2026-06-23*

---

## Implementation Status

🔲 **Stub** — signatures and contracts are fully specified in `orchestrator.py` and `contracts.py`. Implementation pending.

Entry point to implement: `agents/allocation/allocation_agent.py → build_allocation()`

---

## Purpose

Construct a target portfolio that respects the client's residual equity budget after accounting for implicit Human Capital equity exposure. The portfolio must pass Risk Agent stress tests and survive Compliance fiduciary review.

**The single number that differentiates all portfolios:**

```
portfolio_equity_target = effective_risk_budget − implicit_equity_exposure
```

This comes from `ProfileAgentOutput` and is non-negotiable. A biology professor (implicit exposure 4.2%) gets a high equity target (~91%). A software developer (implicit exposure 86.2%) gets a negative equity target (~−24.5%) — their portfolio should underweight equities to hedge their career risk.

---

## Design Principle

**No LLM decision-making.** The optimizer produces all weights deterministically. The LLM's only role is to render the optimizer's output into readable rationale text that cites client-specific numbers. No model chooses weights, tilts, or factor exposures.

---

## Inputs

```python
def build_allocation(
    profile:                     ProfileAgentOutput,
    macro:                       MacroRegimeSnapshot,
    revision:                    int       = 0,
    prior_risk_flags:            list[str] = None,
    prior_compliance_violations: list[str] = None,
) -> AllocationAgentOutput:
```

**Key fields consumed from ProfileAgentOutput:**
- `portfolio_equity_target` (derived in orchestrator: `effective_risk_budget − implicit_equity_exposure`)
- `income_equity_correlation` (ρ) — feeds HC-adjusted sector limits
- `human_capital_type` — drives constraint stringency
- `RSU_concentration` — employer concentration floor
- `risk_tolerance_level` — secondary constraint
- `investment_horizon_years` — duration tilt
- `liquidity_needs` — cash floor
- `current_holdings` — baseline to rebalance from

**Key fields consumed from MacroRegimeSnapshot:**
- `regime_label` — anchors return expectations by regime
- `regime_confidence` — if < 0.60, tilt conservatively
- `regime_shift_date` — fresh transitions warrant more aggressive rebalancing

---

## Methodology

### Black-Litterman Optimisation

**Equilibrium returns** derived from CAPM market-cap weights (Sharpe, 1964), confidence parameter τ = 0.025 (Walters, 2013).

**Regime views** — the Research Agent's `regime_label` shifts the prior: for each of the 5 academic regimes, a return expectation vector is applied as a P-Q view matrix. Return expectations per regime are estimated from historical parquet data in `data/storage/prices/`.

**Human capital incorporation** — following Bodie, Merton & Samuelson (1992) and Campbell & Viceira (2002):

```
w_financial = α × (1 + HC / FC)
```

Where α is the unconstrained optimal risky share. For equity-like HC clients, `HC` is treated as an undiversifiable risky asset with high ρ to the equity market — the portfolio shifts away from equity accordingly.

### Regime-Conditional Return Expectations

Historical return estimates by regime from `data/storage/prices/{TICKER}.parquet`:

| Regime | Expected Equity Return | Expected Bond Return |
|---|---|---|
| Early Recovery | High positive | Low positive |
| Late-Cycle Expansion | Moderate positive | Low/negative |
| Financial Crisis & ZLB | Negative / high vol | Positive (flight to quality) |
| Moderate Expansion | Moderate positive | Low positive |
| Inflation Shock | Negative | Negative (rate duration risk) |

Low-confidence regimes (< 0.60) are handled by blending toward the equal-weight prior.

---

## Constraint Set

All constraints are enforced by the optimizer, not by post-hoc filtering.

| Constraint | Limit | Source |
|---|---|---|
| Max single-name position | 10% | Russell Investments, CFA Institute |
| Max sector exposure | 20% (base) | Industry practice |
| Max total economic sector (portfolio + HC) | 25% | Ibbotson et al. (2007) |
| Max employer concentration (portfolio + HC) | 15% | Uniform Prudent Investor Act |
| HC-adjusted sector limit | `base_limit × (1 − ρ)` | Davis & Willen (2000) |
| Cash floor | Low: 5% / Medium: 10% / High: 15% | From `liquidity_needs` |
| Portfolio equity weight | Anchored to `portfolio_equity_target` ± 5% | Profile Agent HC formula |

**Revision handling:** On Risk Agent FLAG, prior violations are passed as strings to `prior_risk_flags`. The optimizer must tighten the offending position or sector limit and re-solve.

---

## Instrument Universe

From `data/storage/prices/` (fetched by `data/fetch/prices.py`):

| Category | Tickers |
|---|---|
| Broad equity | SPY, IWM, EFA, EEM |
| Fixed income | AGG, TLT, IEF, SHY, HYG, LQD, TIP |
| Real assets | GLD, VNQ, DJP |
| Sectors | XLK, XLF, XLV, XLE, XLI, XLC, XLY, XLP, XLU, XLRE |
| Cash proxy | BIL |

---

## Output Schema (from contracts.py)

```python
class AllocationAgentOutput(BaseModel):
    proposed_portfolio:          dict[str, float]  # ticker → weight, sum = 1.0 ±0.01
    allocation_rationale:        dict[str, str]    # ticker → rationale (≥2 client terms)
    revision:                    int               # 0 = first run
    prior_risk_flags:            list[str]         # Risk violations carried in
    prior_compliance_violations: list[str]         # Compliance violations carried in
```

**Model validator enforced in contracts.py:**
- ≥2 positions
- No negative weights
- Weights sum to 1.0 ±0.01
- Every ticker in `proposed_portfolio` has an entry in `allocation_rationale`

### Rationale Requirements

Each rationale string must cite at least 2 of:
`{client_id, human_capital_type, income_equity_beta, implicit_equity_exposure, portfolio_equity_target, regime_label, risk_tolerance_level, RSU_concentration}`

Compliance Check 2.2 verifies this — generic boilerplate triggers a MEDIUM violation.

---

## Example Output

For Biology Professor (bond-like, portfolio_equity_target ≈ 0.916, Moderate Expansion regime):

```json
{
    "proposed_portfolio": {
        "SPY":  0.45,
        "EFA":  0.20,
        "IWM":  0.15,
        "AGG":  0.10,
        "GLD":  0.05,
        "BIL":  0.05
    },
    "allocation_rationale": {
        "SPY": "Core US equity allocation. Professor's bond-like income (β=0.05) implies minimal implicit equity exposure (4.2%), leaving a portfolio equity target of 91.6%. Broad S&P exposure anchors equity budget in Moderate Expansion regime.",
        "EFA": "International developed equity. Professor's stable academic income has near-zero correlation with international markets (ρ=0.10), allowing full diversification benefit.",
        "AGG": "Modest bond allocation. Portfolio equity target of 91.6% leaves limited room for bonds; 10% provides liquidity buffer consistent with low liquidity_needs.",
        ...
    },
    "revision": 0,
    "prior_risk_flags": [],
    "prior_compliance_violations": []
}
```

For Software Developer (equity-like, portfolio_equity_target ≈ −0.245, Late-Cycle Expansion):

```json
{
    "proposed_portfolio": {
        "AGG":  0.35,
        "TLT":  0.20,
        "GLD":  0.15,
        "IEF":  0.15,
        "BIL":  0.10,
        "VNQ":  0.05
    },
    "allocation_rationale": {
        "AGG": "Developer's equity-like income (β=0.90, ρ=0.75) produces an implicit equity exposure of 86.2% against a risk budget of 61.7%, yielding a negative portfolio equity target of −24.5%. Bond allocation offsets career-embedded equity risk.",
        ...
    },
    "revision": 0,
    "prior_risk_flags": [],
    "prior_compliance_violations": []
}
```

---

## Implementation Plan

**Suggested module structure:**

```
agents/allocation/
├── allocation_agent.py      ← build_allocation() — entry point
├── optimizer.py             ← Black-Litterman solver (scipy.optimize or cvxpy)
├── views.py                 ← Regime-conditional return views from parquet data
├── constraints.py           ← Constraint set builder (equity target, sector, RSU)
├── rationale.py             ← Deterministic rationale renderer (no LLM inference)
└── test_allocation.py
```

---

## Key References

- Sharpe (1964). *Capital Asset Prices.* Journal of Finance.
- Merton (1971). *Optimum Consumption and Portfolio Rules.* Journal of Economic Theory.
- Bodie, Merton & Samuelson (1992). *Labor Supply Flexibility and Portfolio Choice.* Journal of Economic Dynamics and Control.
- Campbell & Viceira (2002). *Strategic Asset Allocation.* Oxford University Press.
- Walters (2013). *The Black-Litterman Model in Detail.* SSRN.
- Ibbotson, Milevsky, Chen & Zhu (2007). *Lifetime Financial Advice.* CFA Institute Research Foundation.
