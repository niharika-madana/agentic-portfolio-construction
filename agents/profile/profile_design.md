# Profile Agent — Design Document
**AI Financial Advisor Pipeline | Agent 1 of 5**
*Fordham MSQF Capstone 2026*
*Last updated: 2026-07-02 (post-refactor sync — module structure consolidated into `profile_agent.py` + `profile_model.py` [loaders.py/hc_beta_table.py/personas.py/human_capital.py/beta.py removed]; income_stability proof-of-categorization added; output paths moved to `data/outputs/`. Prior: 2026-06-30 — schema alignment enforced; `include_percentile_variants` exposed; unit tests added; pandas pinned to 2.x)*

---

## Objective

Produce a validated `ProfileAgentOutput` for each client that captures **total wealth** — financial assets plus the present value of future earnings (human capital) — and quantifies how much of that total wealth is already implicitly exposed to equity market risk through the client's career. This implicit exposure is the number that drives differentiated portfolio construction downstream.

The agent is fully data-driven. Every numeric field traces to a primary source (FRED, BLS OES, SCF). No field is hardcoded or LLM-generated.

---

## Pipeline Overview

```
FRED DGS10              → discount_rate (live 10Y Treasury yield)
BLS OES May 2023        → salary percentiles by SOC code
                           cached as data/storage/bls_oes_2023.parquet
BLS ECEC Q1 2026        → bonus_rate by SOC code (BONUS_RATE_TABLE)
SCF 2022 (static table) → financial_capital by age × income quartile
                               ↓
        build_bls_personas()  → raw persona dicts
          effective_salary = annual_salary × (1 + bonus_rate)
                               ↓
        For each persona:
          hc_type = HUMAN_CAPITAL_TYPE[income_stability]
          β, ρ    = lookup_hc_beta(hc_type)             ← calibrated table
          profile = build_profile(persona)               ← single call
          output  = to_profile_agent_output(profile)     ← Pydantic validation
                               ↓
        list[ProfileAgentOutput]  →  orchestrator.run_pipeline()
        also saved as data/storage/profiles_all.parquet
```

Single-pass architecture. Beta is known before `build_profile()` is called — there is no two-pass design.

---

## Module Structure

```
agents/profile/
├── profile_agent.py    run_profile_agent() — entry point; data loading (FRED DGS10
│                       via _get_discount_rate, BLS OES via _load_bls_oes)
├── profile_model.py    Domain model — merged from hc_beta_table.py + human_capital.py
│                       + personas.py. Holds: HC_BETA_TABLE / INCOME_VOLATILITY_SIGMA /
│                       HUMAN_CAPITAL_TYPE (β/ρ/σ tables), lookup_hc_beta(),
│                       compute_human_capital()/build_profile(), TARGET_OCCUPATIONS +
│                       build_bls_personas(), BONUS_RATE_TABLE, SCF_FINANCIAL_ASSETS,
│                       get_age_bracket(), lookup_financial_capital(),
│                       and INCOME_STABILITY_BASIS (proof-of-categorization + integrity check)
└── __init__.py
```

Unit tests live at the top-level `tests/test_profile.py` and `tests/test_human_capital.py` (no live API calls).

> **Refactor note (post-June 30):** `loaders.py`, `hc_beta_table.py`, `personas.py`, `human_capital.py`, and the deprecated `beta.py` have all been **removed**. Their data-loading moved to `profile_agent.py`; everything else was consolidated inline into `profile_model.py`. Any `from loaders import ...` / `from hc_beta_table import ...` / `from beta import ...` must be updated to import from `profile_model` (or `profile_agent` for loaders).

---

## Core Formulas

### Human Capital Valuation

Present value of the full expected earnings stream (base + bonus), discounted at the 10Y Treasury yield:

```
Effective Annual Earnings = Annual Salary × (1 + bonus_rate)
HC = Effective Annual Earnings × [1 − (1 + r)^(−n)] / r
```

- `bonus_rate` = industry-specific bonus rate from BLS ECEC Q1 2026 (see `BONUS_RATE_TABLE`)
- `r` = FRED DGS10 (live). Fallback: 4.4% if API call fails.
- `n` = years to retirement (= 65 − age)

Ignoring bonuses would systematically understate HC for every occupation — particularly management and financial roles where supplemental pay is a meaningful share of total compensation.

> **Source:** Board of Governors of the Federal Reserve System (US), Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity [DGS10], FRED, Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/DGS10

### Effective Risk Budget

Scales total wealth by how much of it is at genuine risk. Stable income (low σ) behaves like a bond already held — so the portfolio can carry more equity. Variable income (high σ) behaves like an equity position — so the portfolio needs to hedge it.

```
Effective Risk Budget = (FC + HC × (1 − σ)) / Total Wealth
```

### Implicit Equity Exposure

The fraction of total wealth that is already exposed to equity market risk through the client's career — before a single stock is purchased.

```
Implicit Equity Exposure = HC_share × β

where  HC_share = HC / Total Wealth
       β        = income_equity_beta (from HC_BETA_TABLE)
```

> ⚠ The old design doc (June 22) incorrectly stated `Implicit Equity Exposure = (HC × σ) / Total Wealth`. That formula uses income volatility (σ) where it should use income equity beta (β). The correct formula is enforced by the `_check_implicit_equity_exposure` model validator in `contracts.py`.

### Portfolio Equity Target

The residual equity capacity after subtracting career-embedded equity risk. This is the constraint the Allocation Agent builds around — not a suggestion.

```
Portfolio Equity Target = Effective Risk Budget − Implicit Equity Exposure
```

A negative value means the client's career already provides more equity exposure than their total risk budget allows. The Allocation Agent should underweight equity relative to a naive risk-tolerance-only approach.

---

## Income Risk Parameters

| Stability | σ | HC Type | Applies To | β | ρ |
|---|---|---|---|---|---|
| High | 0.05 | bond-like | Academia, government, military, nurses | 0.05 | 0.10 |
| Medium | 0.20 | mixed | Legal, engineering, healthcare admin, analysts | 0.35 | 0.40 |
| Low | 0.40 | equity-like | Tech (RSU), finance (bonus), sales (commission) | 0.90 | 0.75 |

`σ` (income_volatility_sigma) — total annualised earnings uncertainty. Used in the effective risk budget formula.

`β` (income_equity_beta) — systematic sensitivity of income to equity market returns. Used to compute implicit equity exposure.

`ρ` (income_equity_correlation) — correlation of income changes with equity market returns. Used by the Risk Agent to compute HC-correlation adjusted sector limits: `adjusted_limit = base_limit × (1 − ρ)`. A software developer with ρ = 0.75 faces a much tighter technology sector limit than a biology professor with ρ = 0.10.

HC type boundaries (enforced by `contracts.py` `_check_hc_type_consistent_with_beta` validator):
- `bond-like`: β ≤ 0.30
- `mixed`: 0.30 < β ≤ 0.80
- `equity-like`: β > 0.80

### Income-Stability Categorization — Proof

Each occupation's `income_stability` label (which sets σ and the HC type) is **not** a bare hand-assignment. It is derived from three observable, citable labor-economics signals, and corroborated by the pipeline's own comp-structure fields (`bonus_rate`, `rsu_eligible`, `has_pension`):

1. **Variable-compensation share** — fraction of pay that is not fixed base salary. Proxied by the BLS ECEC supplemental-pay share (`BONUS_RATE_TABLE` source) plus equity comp (`rsu_eligible`) and commission. More variable pay → income co-moves with markets → higher σ and β.
2. **Cyclical employment risk** — occupational unemployment level/cyclicality (BLS CPS). Education, healthcare, and government run structurally low and near-acyclical; professional/technical roles are moderate; technology and sales are the most layoff-cyclical.
3. **Institutional job protection** — tenure, occupational licensure, civil-service status, and DB pension coverage (`has_pension`).

The per-occupation justification is recorded in `INCOME_STABILITY_BASIS` (`profile_model.py`), and an **import-time integrity check** asserts every `TARGET_OCCUPATIONS` label matches its documented basis — so a label and its proof cannot silently diverge.

> Sources: BLS ECEC Q1 2026, Table 5 (supplemental-pay share by occupational group); BLS CPS occupational unemployment; Davis & Willen (2000), occupational income betas.

---

## Data Sources

### FRED DGS10 — Discount Rate
Live 10Y Treasury yield pulled via FRED API at runtime. Falls back to 4.4% on failure. This is the `r` in the HC annuity formula.

> Source: FRED series `DGS10`, Federal Reserve Bank of St. Louis.

### BLS OES — Salary Data

**Bureau of Labor Statistics, Occupational Employment and Wage Statistics, May 2023 National Estimates.**

Downloaded as a flat file from BLS (`national_M2023_dl.xlsx`) and cached locally as `data/storage/bls_oes_2023.parquet` on first run. Subsequent runs load from parquet. Columns used:

| Column | Description |
|---|---|
| `OCC_CODE` | SOC code (e.g. `25-1042`) |
| `A_PCT25` | 25th percentile annual wage |
| `A_MEDIAN` | Median (50th percentile) annual wage |
| `A_PCT75` | 75th percentile annual wage |

Values of `#` (suppressed) or `*` (not available) are treated as NaN and those variants are skipped.

> Source: U.S. Bureau of Labor Statistics. Occupational Employment and Wage Statistics, May 2023. https://www.bls.gov/oes/2023/may/oes_nat.htm

### BLS ECEC — Bonus Rates

**Bureau of Labor Statistics, Employer Costs for Employee Compensation (ECEC), Q1 2026, Table 5.**

Bonus rates are derived from supplemental pay as a percentage of total compensation by occupational group (full-time private industry workers), converted to a fraction of base salary:

```
bonus_rate = supplemental_pay_pct / wages_and_salaries_pct
```

| SOC | Occupation | BLS Occupational Group | Supp. Pay % of TC | Wages % of TC | bonus_rate |
|---|---|---|---|---|---|
| 25-1042 | Biology Professor | Education & health services (nonunion) | 3.3% | 71.9% | 0.046 |
| 29-1141 | Registered Nurse | Education & health services (nonunion) | 3.3% | 71.9% | 0.046 |
| 13-1041 | Compliance Officer | Professional and related | 4.1% | 68.0% | 0.060 |
| 23-1011 | Lawyer | Professional and related | 4.1% | 68.0% | 0.060 |
| 17-2141 | Mechanical Engineer | Professional and related | 4.1% | 68.0% | 0.060 |
| 13-2051 | Financial Analyst | Management, business & financial | 5.7% | 67.3% | 0.085 |
| 15-1252 | Software Developer | Management, business & financial | 5.7% | 67.3% | 0.085 |
| 11-3021 | IT Manager | Management, business & financial | 5.7% | 67.3% | 0.085 |
| 11-2022 | Sales Manager | Sales and related | 3.6% | 72.5% | 0.050 |

> Source: U.S. Bureau of Labor Statistics, Employer Costs for Employee Compensation, Q1 2026. Table 5: Private industry workers by bargaining and work status — full-time workers by occupational group. Last modified June 12, 2026. https://www.bls.gov/news.release/ecec.t05.htm

### SCF 2022 — Financial Capital

**Federal Reserve, Survey of Consumer Finances 2022, Table 6: "Median Family Financial Assets."**

Published every 3 years; no live API. Implemented as a static dict in `loaders.SCF_FINANCIAL_ASSETS`:

```
(age_bracket, income_quartile) → median investable financial assets
```

"Financial assets" in SCF = transaction accounts + CDs + directly held stocks/bonds/mutual funds + retirement accounts (IRA, 401k, DC plans). This is what we call `financial_capital` — liquid investable wealth, excluding primary residence and vehicles.

**Income quartile mapping (BLS salary percentile → SCF income quartile):**
- BLS p25 salary → SCF q2 (25th–50th income percentile)
- BLS p50 salary → SCF q3 (50th–75th income percentile)
- BLS p75 salary → SCF q4 (>75th income percentile)

**Age bracket mapping:**
- Age < 35 → "25-34"
- 35 ≤ age < 45 → "35-44"
- 45 ≤ age < 55 → "45-54"
- 55 ≤ age < 65 → "55-64"

> Source: Federal Reserve, Survey of Consumer Finances 2022. https://www.federalreserve.gov/publications/files/scf23.pdf

---

## Beta and Correlation — Calibrated Table

Beta and correlation are looked up from the `HC_BETA_TABLE` in `profile_model.py` (via `lookup_hc_beta()`) — the **only** authorised source. This replaces the OLS regression in the prior design, which regressed synthetic Gaussian noise (`σ × N(0,1)`) against Fama-French sector returns and produced β ≈ 0 for every client regardless of career type.

```python
HC_BETA_TABLE = {
    "bond-like":   {"beta": 0.05, "correlation": 0.10},
    "mixed":       {"beta": 0.35, "correlation": 0.40},
    "equity-like": {"beta": 0.90, "correlation": 0.75},
}
```

**Academic basis:**

Ibbotson, Milevsky, Chen, Zhu (2007), *"Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance,"* CFA Institute Research Foundation. Establishes the framework linking HC type to portfolio equity offsets. Calibrated β values by occupation class align with the paper's central results.

Davis & Willen (2000), *"Using Financial Assets to Hedge Labor Income Risks: Estimating the Benefits,"* SSRN. Estimated income betas from PSID wage data by education/occupation; find range −0.1 to +0.5 for most occupations, rising toward +0.9 for finance and technology roles with equity compensation.

**Consistency with contracts.py:** The three calibrated β values (0.05, 0.35, 0.90) fall squarely within the threshold ranges validated by `_check_hc_type_consistent_with_beta`. No edge cases.

**Why not OLS?** OLS requires observed income data correlated with market returns. BLS OES provides cross-sectional wage levels, not a time series of income changes by individual. Without panel data (e.g., PSID, NLSY79), a regression would require synthetic income shocks — which is what the prior design did, producing meaningless results. The calibrated table is more honest about this limitation and cites real empirical estimates.

---

## Persona Construction — BLS + SCF

`build_bls_personas()` generates personas per target occupation. The `include_percentile_variants` flag controls how many:

| Mode | `include_percentile_variants` | Personas Generated | Use Case |
|---|---|---|---|
| Production (package default) | `False` | 9 (p50 only, one per SOC) | Orchestrator default |
| Extended (notebook default) | `True` | 27 (p25 + p50 + p75 per SOC) | Sensitivity analysis, full audit |

The same flag is exposed on the entry point `run_profile_agent(include_percentile_variants=False)` so the orchestrator can reproduce either set deterministically.

### Target Occupations (9)

| SOC | Label | Career Type | Income Stability | HC Type | Has Pension | RSU Eligible |
|---|---|---|---|---|---|---|
| 25-1042 | Biology Professor | Academia | High | bond-like | ✓ | — |
| 29-1141 | Registered Nurse | Healthcare | High | bond-like | — | — |
| 13-1041 | Compliance Officer | Government | High | bond-like | ✓ | — |
| 23-1011 | Lawyer | Legal | Medium | mixed | — | — |
| 17-2141 | Mechanical Engineer | Engineering | Medium | mixed | — | — |
| 13-2051 | Financial Analyst | Finance | Medium | mixed | — | — |
| 15-1252 | Software Developer | Technology | Low | equity-like | — | ✓ |
| 11-3021 | IT Manager | Technology | Low | equity-like | — | ✓ |
| 11-2022 | Sales Manager | Sales | Low | equity-like | — | — |

---

## Qualitative Field Derivation Rules

Qualitative fields not available from BLS or SCF are derived from documented rules. No field is made up.

### Risk Tolerance

| HC Type | Age | risk_tolerance_level |
|---|---|---|
| bond-like | < 50 | moderate |
| bond-like | ≥ 50 | conservative |
| mixed | any | moderate |
| equity-like | < 45 | aggressive |
| equity-like | ≥ 45 | moderate |

**Rationale:** Bond-like income acts as a portfolio stabiliser, increasing risk capacity. Equity-like income already carries market risk, so the portfolio should not amplify it with an aggressive stance.

### Liquidity Needs

| Income Stability | liquidity_needs |
|---|---|
| High (regular salary) | low |
| Medium (bonus-driven) | medium |
| Low (RSU/commission) | medium |

**Rationale:** Stable monthly salary makes the client their own liquidity buffer. Variable income (bonuses, vesting events) creates unpredictable cash flow timing.

### Investment Objective

| Years to Retirement | investment_objective |
|---|---|
| ≥ 10 years | growth |
| < 10 years | income |

### Current Holdings

For RSU holders:
```
employer_RSU  = rsu_concentration
US_equity     = (1 − rsu_concentration) × 0.55
bonds         = (1 − rsu_concentration) × 0.25
cash          = (1 − rsu_concentration) × 0.20
```

For non-RSU holders, derived from SCF Table 7 "Family Holdings of Financial Assets" by age and risk tolerance:

| Risk Tolerance | Age | Equity | Intl Equity | Bonds | Cash |
|---|---|---|---|---|---|
| aggressive | < 45 | 0.65 | 0.20 | 0.10 | 0.05 |
| aggressive | ≥ 45 | 0.55 | 0.20 | 0.20 | 0.05 |
| moderate | < 45 | 0.50 | 0.15 | 0.25 | 0.10 |
| moderate | ≥ 45 | 0.40 | 0.15 | 0.35 | 0.10 |
| conservative | any | 0.25 | 0.10 | 0.50 | 0.15 |

All holdings sum to exactly 1.0 by construction.

### RSU Concentration

Only applies to RSU-eligible occupations (software developers, IT managers):

| BLS Percentile | RSU Concentration |
|---|---|
| p25 | 0.20 |
| p50 | 0.35 |
| p75 | 0.55 |

Higher salary percentile → larger equity grant as a fraction of total compensation.

---

## Three Illustrative Personas (BLS p50, r = 4.4%)

| | Biology Professor | Financial Analyst | Software Developer |
|---|---|---|---|
| **SOC** | 25-1042 | 13-2051 | 15-1252 |
| **BLS Median Salary** | ~$83,920 | ~$99,010 | ~$132,270 |
| **Bonus Rate (BLS ECEC Q1 2026)** | 4.6% | 8.5% | 8.5% |
| **Effective Annual Earnings** | ~$87,782 | ~$107,426 | ~$143,513 |
| **Age** | 47 | 40 | 38 |
| **Financial Capital (SCF)** | $200,000 | $90,000 | $90,000 |
| **Human Capital** | ~$1,075,965 | ~$1,601,000 | ~$2,241,835 |
| **Total Wealth** | ~$1,275,965 | ~$1,691,000 | ~$2,331,835 |
| **HC Share** | 0.843 | 0.947 | 0.961 |
| **σ** | 0.05 | 0.20 | 0.40 |
| **HC Type** | bond-like | mixed | equity-like |
| **β** | 0.05 | 0.35 | 0.90 |
| **ρ** | 0.10 | 0.40 | 0.75 |
| **Implicit Equity Exposure** | **0.042** | **0.332** | **0.865** |
| **Effective Risk Budget** | 0.958 | 0.811 | 0.616 |
| **Portfolio Equity Target** | **+0.916** | **+0.479** | **−0.249** |
| **Risk Tolerance** | moderate | moderate | aggressive |

**Key insight:** The software developer's portfolio equity target is negative. Their career already provides 86.5% equity exposure through RSU vesting, layoff correlation, and bonus sensitivity to tech market conditions. Their effective risk budget is only 61.6%. Even before looking at their portfolio, they are over-exposed to equity. The Allocation Agent should underweight equities relative to what risk tolerance alone would suggest and use bonds and alternatives to hedge the career risk.

The biology professor's bond-like income — a stable, government-indifferent salary — contributes almost no equity exposure (4.2%). They can hold 91.6% equity in their portfolio without exceeding their risk budget. Note that including the 4.6% bonus rate increases their HC by ~$46,000 relative to a base-salary-only calculation.

---

## Output Schema

`ProfileAgentOutput` is defined in `contracts.py` and validated by four `model_validator` functions. Field names and types here are the authoritative schema — any rename must be mirrored in `agents/risk/contracts.py` (RiskAgentInput) and `agents/research/contracts.py` (ProfileContextForResearch):

1. `_check_holdings_sum` — `current_holdings` weights must sum to 1.0 ± 0.01
2. `_check_total_wealth_consistency` — `total_wealth == financial_capital + human_capital_valuation` within 0.5%
3. `_check_implicit_equity_exposure` — enforces `hc_share × β` within 0.01 tolerance
4. `_check_hc_type_consistent_with_beta` — β thresholds must match `human_capital_type` label

Example output for Biology Professor (p50):

```json
{
  "client_id": "bls_25-1042_p50",
  "career_type": "Academia",
  "age": 47,
  "financial_capital": 200000,
  "human_capital_valuation": 1075965,
  "total_wealth": 1275965,
  "human_capital_pct_of_total": 84.3,
  "income_volatility_sigma": 0.05,
  "income_equity_beta": 0.05,
  "income_equity_correlation": 0.10,
  "implicit_equity_exposure": 0.042,
  "human_capital_type": "bond-like",
  "income_stability": "High",
  "effective_risk_budget": 0.958,
  "portfolio_equity_target": 0.916,
  "industry_exposure_sector": "Education",
  "RSU_concentration": 0.0,
  "has_pension": true,
  "bonus_rate": 0.046,
  "current_holdings": {
    "US_equity": 0.50,
    "intl_equity": 0.15,
    "bonds": 0.25,
    "cash": 0.10
  },
  "investment_horizon_years": 18,
  "risk_tolerance_level": "moderate",
  "liquidity_needs": "low",
  "investment_objective": "growth"
}
```

---

## Unit Tests

Deterministic unit tests in `test_profile.py` (no live API calls — uses fixed discount rate 4.4%). Beta/correlation are not passed in; `build_profile()` looks them up from `hc_beta_table` by HC type, so the fixtures mirror `build_bls_personas()` output. The four load-bearing cases:

| Test | Persona | What It Checks |
|---|---|---|
| 1 | Software Developer p50 (salary $132,270) | HC annuity formula, β=0.90, effective_salary includes bonus_rate=0.085, IEE = HC_share × β, validates through the contract |
| 2 | Biology Professor p50 (salary $83,920) | bond-like type, β=0.05, pension=True, professor `portfolio_equity_target` > developer `portfolio_equity_target` |
| 3 | Bad total_wealth (10% off) | `_check_total_wealth_consistency` validator fires |
| 4 | β=0.10 with equity-like label | `_check_hc_type_consistent_with_beta` validator fires |

Supporting tests cover the annuity formula directly, the effective-risk-budget formula, that the bonus raises HC above a base-salary-only PV, the calibrated `lookup_hc_beta` return type, the holdings-sum validator, and the adapter round-trip — 11 tests, all passing.

HC values confirmed against the annuity formula (r = 4.4%): Software Developer (effective $143,513, n=27) HC = $2,241,835; Biology Professor (effective $87,780, n=18) HC = $1,075,965.

---

## Downstream Agent Field Routing

**Allocation Agent:**
- `implicit_equity_exposure` + `effective_risk_budget` → `portfolio_equity_target = effective_risk_budget − implicit_equity_exposure`
- `income_equity_beta` → avoids doubling sector risk already embedded in income
- `current_holdings` → baseline to rebalance from
- `investment_objective` → anchors return target

**Risk Agent:**
- `income_volatility_sigma` → parameterises earnings stress scenarios
- `income_equity_beta` → scales income shock in drawdown simulations
- `income_equity_correlation` → HC-correlation adjusted sector limits: `adjusted_limit = base_limit × (1 − ρ)`
- `RSU_concentration` → single-stock concentration risk flag
- `liquidity_needs` → minimum cash floor in stress scenarios
- `has_pension` → supplementary income floor in retirement stress scenarios

**Compliance Agent:**
- `human_capital_type` → classifies income stream, drives portfolio constraint framing
- `industry_exposure_sector` → sector concentration check against regulatory limits
- `risk_tolerance_level`, `investment_horizon_years`, `age` → FINRA Rule 2111 suitability
- `RSU_concentration` → ICA §5b1 concentration breach trigger
- `bonus_rate` → included in total compensation audit trail

---

## Failure Modes

### BLS SOC Code Not Found
A SOC code in `TARGET_OCCUPATIONS` is missing from the downloaded OES flat file (e.g. new occupation not yet surveyed, or file format change).

**Handling:** `build_bls_personas()` prints a warning and skips the occupation. Pipeline does not crash. Run `load_bls_oes()` interactively and check available SOC codes if this occurs.

### BLS Wage Suppressed
BLS suppresses wages (`#`) for small sample sizes or for the 90th percentile of very high-paying roles.

**Handling:** `pd.to_numeric(..., errors='coerce')` converts `#` and `*` to `NaN`. That salary variant is skipped with a print message. Median (p50) is almost never suppressed for national-level data.

### Pydantic ValidationError
`to_profile_agent_output()` raises `pydantic.ValidationError` if any formula constraint is violated — wrong `implicit_equity_exposure`, holdings not summing to 1.0, β/hc_type mismatch, etc.

**Handling:** Error propagates immediately — the profile does not reach the orchestrator. Fix the upstream data or derivation rule that produced the bad value.

### FRED API Failure
FRED is unreachable or the API key is invalid.

**Handling:** Clean fallback to 4.4% with a printed warning. The pipeline continues. Log the fallback rate in the output JSON so it's visible in the audit trail.

### Holdings Not Summing to 1.0
`_derive_current_holdings()` constructs all allocations arithmetically from a base of 1.0, so this cannot occur from the BLS pipeline. Only possible if holdings are set manually. Caught by `_check_holdings_sum` Pydantic validator before the profile exits the agent.

---

## Implementation Notes

- **Entry point:** `agents/profile/profile_agent.py` → `run_profile_agent()`
- **Beta source:** `HC_BETA_TABLE` in `agents/profile/profile_model.py` (via `lookup_hc_beta()`) — the only authorised source. The former `hc_beta_table.py` and deprecated `beta.py` have been removed.
- **Environment:** pandas pinned to `>=2.0,<3.0` across all agents (pandas 3.x caused parquet load errors — team decision June 25 2026)
- **BLS OES source file:** `national_M2023_dl.xlsx` — converted and cached as `data/storage/bls_oes_2023.parquet` on first run
- **Output (JSON):** `data/outputs/profiles_all.json`
- **Output (Parquet):** `data/storage/profiles_all.parquet` — consumed by Allocation, Risk, Compliance agents
- **Runtime environment:** Google Colab or local Python 3.11+
- **API keys needed:** `FRED_API` only. No Anthropic or OpenAI keys required.
- **Dependencies:** `requests`, `"pandas>=2.0,<3.0"`, `openpyxl`, `numpy`, `pydantic`
- **Removed dependencies:** `anthropic`, `openai`, `scipy`, `ruptures`, `sklearn` (no longer used by profile agent)

### Default run (9 personas, p50 only)
```python
from agents.profile.profile_agent import run_profile_agent
profiles = run_profile_agent(fred_api_key="YOUR_KEY")
# include_percentile_variants defaults to False → 9 personas
```

### Extended run (27 personas, p25 + p50 + p75)
```python
profiles = run_profile_agent(
    fred_api_key="YOUR_KEY",
    include_percentile_variants=True,
)
# → 27 personas (9 SOC codes × 3 percentiles)
```
