# Profile Agent — Design Document
**AI Financial Advisor Pipeline | Agent 1 of 5**
*Fordham MSQF Capstone 2026*
*Last updated: 2026-07-20 (July 14 meeting action item — `implied_market_volatility` diagnostic added (σ_market = ρ × σ_income / β); calibration-inconsistency finding documented; corrected two doc errors: `income_stability` enum case in the example JSON, and the BLS OES parquet filename; added `tier_derivation.py` — a reproducible, executable check that recomputes `income_stability` tiers from encoded signals and asserts the result matches the declared label. Prior: 2026-07-02 — post-refactor sync, module structure consolidated into `profile_agent.py` + `profile_model.py` [loaders.py/hc_beta_table.py/personas.py/human_capital.py/beta.py removed]; income_stability proof-of-categorization added; output paths moved to `data/outputs/`. Prior: 2026-06-30 — schema alignment enforced; `include_percentile_variants` exposed; unit tests added; pandas pinned to 2.x)*

---

## Objective

Produce a validated `ProfileAgentOutput` for each client that captures **total wealth** — financial assets plus the present value of future earnings (human capital) — and quantifies how much of that total wealth is already implicitly exposed to equity market risk through the client's career. This implicit exposure is the number that drives differentiated portfolio construction downstream.

The agent is fully data-driven. Every numeric field traces to a primary source (FRED, BLS OES, SCF). No field is hardcoded or LLM-generated.

---

## Pipeline Overview

```
FRED DGS10              → discount_rate (live 10Y Treasury yield)
BLS OES May 2023        → salary percentiles by SOC code
                           cached as data/storage/bls_oes.parquet
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
│                       implied_market_volatility() [diagnostic],
│                       compute_human_capital()/build_profile(), TARGET_OCCUPATIONS +
│                       build_bls_personas(), BONUS_RATE_TABLE, SCF_FINANCIAL_ASSETS,
│                       get_age_bracket(), lookup_financial_capital(),
│                       and INCOME_STABILITY_BASIS (proof-of-categorization + integrity check)
├── tier_derivation.py  Reproducible derivation of the income_stability tiers — recomputes
│                       each tier from BONUS_RATE_TABLE + RSU_BY_PERCENTILE + has_pension and
│                       asserts it matches the label in TARGET_OCCUPATIONS. See
│                       [Tier Derivation — Reproducibility Check](#tier-derivation--reproducibility-check).
│                       Run via `python -m agents.profile.tier_derivation`.
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

### Implied Market Volatility — Diagnostic Only

*Added per the July 14 meeting action item.* The single-factor identity that relates the three income-risk parameters is:

```
β = ρ × σ_income / σ_market     ⇒     σ_market = ρ × σ_income / β
```

`implied_market_volatility()` in `profile_model.py` inverts the identity and emits the result per profile. **Nothing downstream consumes it** — it exists to make the calibration's internal consistency visible and auditable. Returns `None` when β ≤ 0, where the identity is undefined.

Because σ, β and ρ are calibrated independently (see [Beta and Correlation](#beta-and-correlation--calibrated-table)), this does **not** resolve to a single market volatility — see the table in Income Risk Parameters below. That spread is the finding, not a bug.

---

## Income Risk Parameters

| Stability | σ | HC Type | Applies To | β | ρ | Implied σ_market |
|---|---|---|---|---|---|---|
| High | 0.05 | bond-like | Academia, government, military, nurses | 0.05 | 0.10 | 0.1000 |
| Medium | 0.20 | mixed | Legal, engineering, healthcare admin, analysts | 0.35 | 0.40 | 0.2286 |
| Low | 0.40 | equity-like | Tech (RSU), finance (bonus), sales (commission) | 0.90 | 0.75 | 0.3333 |

`σ` (income_volatility_sigma) — total annualised earnings uncertainty. Used in the effective risk budget formula.

`β` (income_equity_beta) — systematic sensitivity of income to equity market returns. Used to compute implicit equity exposure.

`ρ` (income_equity_correlation) — correlation of income changes with equity market returns. Used by the Risk Agent to compute HC-correlation adjusted sector limits: `adjusted_limit = base_limit × (1 − ρ)`. A software developer with ρ = 0.75 faces a much tighter technology sector limit than a biology professor with ρ = 0.10.

`Implied σ_market` (implied_market_volatility) — **diagnostic only, consumed by nothing.** `ρ × σ / β`, i.e. the market volatility each row would imply under a strict single-factor model. See the calibration note below.

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

### Tier Derivation — Reproducibility Check

`INCOME_STABILITY_BASIS` documents *why* each occupation carries its tier, in prose. `tier_derivation.py` makes that reasoning executable: it recomputes every tier from the encoded signals and asserts the result matches the declared label. Run `python -m agents.profile.tier_derivation` to print the derivation table.

**Scope — 2 of the 3 cited signals are encoded.** The prose basis cites three signals (variable-compensation share, cyclical employment risk, institutional job protection). Only the first, plus the `has_pension` slice of the third, are actually encoded anywhere in this repo:

1. **Variable-compensation share** — `BONUS_RATE_TABLE` (BLS ECEC Q1 2026, Table 5) plus equity comp via `rsu_eligible`. **Encoded.**
2. **Cyclical employment risk** — per-occupation BLS CPS unemployment. **Not encoded.** No CPS fetcher exists in this repo; the only unemployment series present is FRED's aggregate `UNRATE`, consumed by the Research Agent, which is not occupation-specific.
3. **Institutional job protection** — partially encoded, as `has_pension`. Licensure, tenure, and civil-service status are not separate fields.

So the derivation is a scoring rule over signals (1) and the `has_pension` slice of (3) — not the full three-signal basis the prose describes.

**Scoring rule.** `score_occupation()` computes a market-exposure score in units of `BONUS_RATE_TABLE`'s supplemental-pay-to-wages ratio:

```
score = BONUS_RATE_TABLE[soc]
      + RSU_PREMIUM        if rsu_eligible   (0.35 — reuses RSU_BY_PERCENTILE["p50"])
      − PENSION_DISCOUNT    if has_pension    (0.02)
```

Tier cutoffs sit at the gaps between BLS ECEC occupational-group ratios: `HIGH_MAX = 0.050`, `MEDIUM_MAX = 0.100`.

`PENSION_DISCOUNT = 0.02` is the rule's one free parameter — **calibrated, not measured**: it's the smallest round value that pulls a pensioned worker in the ECEC "Professional and related" group (0.060) below `HIGH_MAX`, without pulling an unpensioned one below it. It's what distinguishes Compliance Officer (High) from Lawyer and Mechanical Engineer (Medium) — all three share `bonus_rate = 0.060`. It stands in for the civil-service/tenure protection that signal (3) describes but does not fully encode.

**Result: 8 of 9 tiers reproduced.** The ninth, Sales Manager (11-2022), is a documented override rather than a tuned threshold:

| SOC | Occupation | Rule computes | Declared tier | Why |
|---|---|---|---|---|
| 11-2022 | Sales Manager | Medium (score 0.050, below Financial Analyst's 0.085 and Lawyer's 0.060) | Low | Commission is not a field in this repo. The ECEC "Sales and related" ratio (0.050) captures only supplemental pay, not commission — so the rule under-scores it. This is an artifact of the missing field, not a claim that sales income is stable: commission revenue tracks consumer-discretionary demand, giving this occupation equity-like earnings variance. Encoding a commission share would remove the need for this override. |

Every override must state what the rule computes, what the true tier is, and which unencoded signal justifies the gap — this keeps `OVERRIDES` from becoming a dumping ground for rows the rule simply fails to explain.

**Sensitivity note.** Registered Nurse (score 0.046, no pension) sits only 0.004 below `HIGH_MAX`: its High tier is reproduced by the rule here, but it's the row most sensitive to the threshold, and its prose basis in `INCOME_STABILITY_BASIS` leans on licensure and counter-cyclical healthcare demand — both unencoded signals. If `HIGH_MAX` is ever retuned, this is the row to check first.

---

## Data Sources

### FRED DGS10 — Discount Rate
Live 10Y Treasury yield pulled via FRED API at runtime. Falls back to 4.4% on failure. This is the `r` in the HC annuity formula.

> Source: FRED series `DGS10`, Federal Reserve Bank of St. Louis.

### BLS OES — Salary Data

**Bureau of Labor Statistics, Occupational Employment and Wage Statistics, May 2023 National Estimates.**

Downloaded as a flat file from BLS (`national_M2023_dl.xlsx`) and cached locally as `data/storage/bls_oes.parquet` on first run. Subsequent runs load from parquet. Columns used:

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

Published every 3 years; no live API. Implemented as a static dict in `profile_model.SCF_FINANCIAL_ASSETS`:

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

### Known Calibration Inconsistency

*Raised in the July 14 review; documented rather than silently corrected.*

σ, β and ρ are each calibrated **independently**, from a different source, for a different downstream consumer:

| Parameter | Calibration source | Consumed by |
|---|---|---|
| σ | BLS ECEC supplemental-pay share + occupational cyclicality | `effective_risk_budget` |
| β | Davis & Willen (2000) occupational income betas | `implicit_equity_exposure` |
| ρ | Ibbotson et al. (2007) HC-type framework | Risk Agent HC-adjusted sector limits |

They are **not** jointly estimated. Inverting the single-factor identity `σ_market = ρ × σ / β` therefore yields a different market volatility per tier rather than one common value:

| HC Type | Calculation | Implied σ_market |
|---|---|---|
| bond-like | 0.10 × 0.05 / 0.05 | **10.00%** |
| mixed | 0.40 × 0.20 / 0.35 | **22.86%** |
| equity-like | 0.75 × 0.40 / 0.90 | **33.33%** |

A true single-factor model requires all three to equal one σ_market. Realised US equity volatility sits around 16–20%, so the mixed tier is roughly plausible while bond-like is far too low and equity-like far too high. The bond-like row is the largest outlier — β = 0.05 is small enough that the ratio is highly sensitive to it, making β the most likely culprit if the team recalibrates.

**Interpretation.** This confirms what the table already claims to be: a pragmatic calibration anchored to published estimates, not a strict econometric model. Presenting it as single-factor-consistent would be a misrepresentation.

**Why it is left uncorrected.** Retuning β or ρ to force a common σ_market would change `implicit_equity_exposure` for every persona, which propagates directly into `portfolio_equity_target` and therefore into every downstream allocation and risk check. That is a team decision with blast radius well outside the Profile Agent, not a side effect of adding a diagnostic. Surfacing the number per-profile makes the discrepancy visible and auditable in the meantime.

> **Deliberately unvalidated.** No `model_validator` cross-checks `implied_market_volatility` against the identity — such a check would fail every profile the agent produces. The field is descriptive, not a constraint.

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
| **Implied σ_market** *(diagnostic)* | 0.1000 | 0.2286 | 0.3333 |
| **Implicit Equity Exposure** | **0.042** | **0.332** | **0.865** |
| **Effective Risk Budget** | 0.958 | 0.811 | 0.616 |
| **Portfolio Equity Target** | **+0.916** | **+0.479** | **−0.249** |
| **Risk Tolerance** | moderate | moderate | aggressive |

**Key insight:** The software developer's portfolio equity target is negative. Their career already provides 86.5% equity exposure through RSU vesting, layoff correlation, and bonus sensitivity to tech market conditions. Their effective risk budget is only 61.6%. Even before looking at their portfolio, they are over-exposed to equity. The Allocation Agent should underweight equities relative to what risk tolerance alone would suggest and use bonds and alternatives to hedge the career risk.

The biology professor's bond-like income — a stable, government-indifferent salary — contributes almost no equity exposure (4.2%). They can hold 91.6% equity in their portfolio without exceeding their risk budget. Note that including the 4.6% bonus rate increases their HC by ~$46,000 relative to a base-salary-only calculation.

---

## Output Schema

`ProfileAgentOutput` is defined in `contracts.py` and validated by four `model_validator` functions. Field names and types here are the authoritative schema. The root `contracts.py` is the single definition site — the per-agent `contracts.py` files this section used to reference (carrying `RiskAgentInput` / `ProfileContextForResearch`) no longer exist, and neither do those types; a rename now needs updating only in `contracts.py` and its consumers:

1. `_check_holdings_sum` — `current_holdings` weights must sum to 1.0 ± 0.01
2. `_check_total_wealth_consistency` — `total_wealth == financial_capital + human_capital_valuation` within 0.5%
3. `_check_implicit_equity_exposure` — enforces `hc_share × β` within 0.01 tolerance
4. `_check_hc_type_consistent_with_beta` — β thresholds must match `human_capital_type` label

Note the enum casing: `income_stability`, `human_capital_type`, `risk_tolerance_level`, `liquidity_needs` and `investment_objective` are all **lowercase** on the wire (`"high"`, not `"High"`). The `High`/`Medium`/`Low` keys used in `INCOME_VOLATILITY_SIGMA` and `TARGET_OCCUPATIONS` are internal only — `build_profile()` maps them via `_STABILITY_TO_CONTRACT` before validation.

Example output for Biology Professor (p50, r = 4.4%):

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
  "implied_market_volatility": 0.1,
  "implicit_equity_exposure": 0.042,
  "human_capital_type": "bond-like",
  "income_stability": "high",
  "effective_risk_budget": 0.958,
  "portfolio_equity_target": 0.916,
  "industry_exposure_sector": "Education",
  "RSU_concentration": 0.0,
  "has_pension": true,
  "bonus_rate": 0.046,
  "current_holdings": {
    "US_equity": 0.40,
    "intl_equity": 0.15,
    "bonds": 0.35,
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

Deterministic unit tests in `test_profile.py` (no live API calls — uses fixed discount rate 4.4%). Beta/correlation are not passed in; `build_profile()` looks them up from `HC_BETA_TABLE` (in `profile_model.py`) by HC type, so the fixtures mirror `build_bls_personas()` output. The four load-bearing cases:

| Test | Persona | What It Checks |
|---|---|---|
| 1 | Software Developer p50 (salary $132,270) | HC annuity formula, β=0.90, effective_salary includes bonus_rate=0.085, IEE = HC_share × β, validates through the contract |
| 2 | Biology Professor p50 (salary $83,920) | bond-like type, β=0.05, pension=True, professor `portfolio_equity_target` > developer `portfolio_equity_target` |
| 3 | Bad total_wealth (10% off) | `_check_total_wealth_consistency` validator fires |
| 4 | β=0.10 with equity-like label | `_check_hc_type_consistent_with_beta` validator fires |

Supporting tests cover the annuity formula directly, the effective-risk-budget formula, that the bonus raises HC above a base-salary-only PV, the calibrated `lookup_hc_beta` return type, the holdings-sum validator, and the adapter round-trip — 11 tests, all passing.

HC values confirmed against the annuity formula (r = 4.4%): Software Developer (effective $143,513, n=27) HC = $2,241,835; Biology Professor (effective $87,780, n=18) HC = $1,075,965.

`tier_derivation.py` additionally asserts, at minimum, that every entry in `OVERRIDES` is load-bearing (i.e. the rule's computed tier for that SOC code genuinely differs from the declared one) — so `OVERRIDES` can't silently accumulate rows the rule already explains correctly.

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

**Handling:** `build_bls_personas()` prints a warning and skips the occupation. Pipeline does not crash. Run `_load_bls_oes()` (in `profile_agent.py`) interactively and check available SOC codes if this occurs.

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
`derive_current_holdings()` constructs all allocations arithmetically from a base of 1.0, so this cannot occur from the BLS pipeline. Only possible if holdings are set manually. Caught by `_check_holdings_sum` Pydantic validator before the profile exits the agent.

---

## Beta Calibration — One Market (resolves 24 Jul review §7)

**Was:** σ, β and ρ were each calibrated independently, so inverting the single-factor identity `β = ρ × σ_income / σ_market` implied three different market volatilities — 10.0%, 22.9% and 33.3%. One model claiming three markets.

**Now:** σ_market is measured once and β is derived from it. `SIGMA_MARKET = 0.1571` is the annualised standard deviation of the Fama-French excess market return (`mktrf`), 312 monthly observations, 2000-01 to 2025-12 — the same factor and file `agents/research/ticker_betas.py` regresses against, so human-capital beta and asset beta are denominated in one market. `measure_market_volatility()` recomputes it from the parquet.

| HC type | ρ | σ_income | β (was) | β (now) |
|---|---|---|---|---|
| bond-like | 0.10 | 0.05 | 0.05 | **0.032** |
| mixed | 0.40 | 0.20 | 0.35 | **0.509** |
| equity-like | 0.75 | 0.40 | 0.90 | **1.910** |

β is now computed in the table comprehension, never hand-set, so the identity holds by construction and cannot drift again. `implied_market_volatility` changes meaning: it was a diagnostic that exposed the inconsistency, and is now a consistency check that must round-trip to ≈0.157 for every profile.

**Downstream impact.** No persona changes human-capital type or corner status; magnitudes move. Equity-like `portfolio_equity_target` goes from −0.25 to −1.22 (already at the zero bound, stays there); mixed from 0.48 to 0.33; bond-like from 0.92 to 0.93.

**Consequence worth flagging:** equity-like β now sits at 1.91 against the contract's 2.0 ceiling, so a +10% extraction error on σ_income or ρ leaves the admissible range entirely. Twelve sensitivity cells are unmeasurable for that reason. Either the ceiling or `σ_income = 0.40` needs revisiting.

---

## Intake Layer — Transcript to Typed Profile

The 'extract' half of the standing rule (LLMs extract, classify, narrate; deterministic code computes). Nothing here produces a portfolio number — it produces what the client *said*, typed and sourced, which `build_profile()` then computes on.

**Contracts** (`contracts.py`): `FactSource` (STATED / INFERRED / DEFAULT / UNKNOWN), `ExtractedField`, `ExtractedProfile`. Validators make the failure modes unconstructable rather than merely discouraged — a STATED field without an `evidence_quote` raises, and an UNKNOWN field carrying a value or lacking a `follow_up_question` raises.

**`transcript_generator.py`** — synthetic discovery calls rendered *from* ground-truth personas by deterministic templating. The answer key is the input that produced the transcript, not something recovered from it afterwards, so no extractor can score well by sharing a prior with the generator. Facts are planted at four salience levels (prominent / mentioned / parenthetical / omitted); omitted fields are the refusal-to-guess tests. One fact is planted as an advisor-supplied estimate the client merely assents to, recorded `is_stated=False` — the mentor's point about Priya's beta of 1.2.

**`intake.py`** — three extractors behind one protocol: `RuleBasedExtractor` (deterministic, offline, the floor the LLMs must beat), `NaiveExtractor` (one call, flat output — the control asked for directly), `StructuredExtractor` (field-by-field, provenance, explicit unknown, stated confidence). `call_model` is a thin swappable backend per the 14 Jul minutes; it raises rather than silently degrading when no key is set.

**`intake_eval.py`** — scores any extractor on the five properties from *Now and Forward* §4. Provenance is *verified* against the transcript, not trusted: a quote the model composed rather than copied fails.

### Harness result — the naive-vs-designed comparison

18 transcripts / 9 personas, extraction on `claude-opus-5`, 0 failures.

| metric | rule_based | naive | **structured** |
|---|---|---|---|
| overall accuracy | 98.6% | 86.1% | **98.6%** |
| recall: prominent | 91.7% | 91.7% | 91.7% |
| recall: mentioned | 100% | 100% | 100% |
| recall: parenthetical | 100% | 100% | 100% |
| provenance (has quote) | 100% | 0% | 100% |
| quote verified | 87.8% | 0% | **100%** |
| stated vs inferred | 100% | **12.2%** | 90.5% |
| refusal recall | 100% | 60.3% | 100% |
| hallucination rate | 0% | **39.7%** | **0%** |
| calibration error | 16.6% | 32.9% | **6.4%** |

**On fields that were actually discussed, all three are near-perfect.** The per-field breakdown shows the real story — naive fails on exactly three fields, and they are exactly the ones the transcripts never mention:

| field | naive | rule_based | structured |
|---|---|---|---|
| `investment_horizon_years` | 0.722 | 1.000 | 1.000 |
| `investment_objective` | 0.556 | 1.000 | 1.000 |
| `liquidity_needs` | 0.556 | 1.000 | 1.000 |
| `income_stability` | 0.944 | 0.833 | 0.833 |
| *(the other eight fields)* | 1.000 | 1.000 | 1.000 |

**The entire gap is refusal.** Asked about something never discussed, the single-call control invents a plausible value **39.7%** of the time; the structured extractor does it **0%** of the time, returning `unknown` plus a follow-up question. For a suitability file that is decisive — a fabricated liquidity need is indistinguishable from a real one once it reaches the optimiser.

Two secondary results worth keeping:

- **Naive cannot separate what the client said from what the advisor said** (stated-vs-inferred 12.2%). The transcripts plant an advisor-supplied beta the client merely assents to — the mentor's Priya beta-1.2 point — and naive records it as the client's own assertion.
- **Only the structured extractor is calibrated** (6.4% error against naive's 32.9%), so only its confidence can be used as a routing signal.

`income_stability` is the one field everybody finds hard, and naive is *best* at it (0.944 vs 0.833) — inferring a category from a qualitative cue is the one place an unconstrained single call has an edge.

**Adding the units spec halved naive's hallucination rate** (97.1% → 39.7%). Telling it that some values are derived rather than stated implicitly taught it when to answer "inferred" instead of inventing. Still 39.7% against 0%.

The rule-based floor now matches structured on overall accuracy (98.6%) but plateaus where regex must: 87.8% on quote verification, because its cue-window slices don't always land on a verbatim span, and **0 statements classified** — see below.

**Two harness bugs were found and fixed while producing this table**, both of which had penalised all three extractors equally and hidden the real differences:

1. Categorical fields (`income_stability` et al.) had no value space in the prompt, so extractors returned semantically correct prose — "stable base salary but unpredictable bonus" — that an exact-match scorer must reject. Fixed by `FIELD_VALUE_SPACE` in `intake.py`; the constraint is now stated in both prompts.
2. The answer key recorded the persona's unrounded `bonus_rate` (0.046) while the transcript said "around 5%". The key now records **what the transcript says**, since 4.6% is not recoverable from the conversation and grading against it penalises correct reading.

Reproduce with `python -m agents.profile.intake_eval`. Sweep cost/quality with `INTAKE_MODEL=claude-haiku-4-5 python -m agents.profile.intake_eval` — same harness, same scoring, different backend.

### The bridge — transcript actually drives the pipeline

`intake_bridge.py` converts an `ExtractedProfile` into a persona dict and runs it through the **same `build_profile()`** the BLS personas use. Without it the intake layer was a dead end: measured carefully, connected to nothing — the review's own "a stage computes something correct and the next stage never receives it", reproduced inside this agent.

All 9 synthetic conversations now produce validated `ProfileAgentOutput`s, and they reproduce the BLS-built profiles: identical β and equity target, with human capital differing only by the bonus rounding the transcript introduces (1,099,343 vs 1,095,155 for the biology professor — the client says "around 5%", the persona holds 4.6%).

**Which source wins — the answer to the oral question.** *"If the client's own words disagree with what your data says about them, which wins, and why?"*

| Kind of fact | Winner | Why |
|---|---|---|
| Situation — age, salary, balances, employer | **Client** | They are the authority on their own circumstances; no table knows their account balance |
| Model parameters — β, ρ, σ | **Measured calibration** | These are estimated, not observed. A transcript β is rarely the client's measurement |
| Never discussed, defaultable | Neither — **defaulted and listed** | A suitability file must separate what the client asserted from what the system chose |
| Never discussed, not defaultable | **Nothing is built** | `REQUIRED_FIELDS` cannot be guessed without inventing the client |

The parameter rule is the interesting one. In the reference intake the *advisor* proposed β = 1.2 and the client merely assented, hedging it might be low in a bad year. Soft assent to someone else's estimate is not evidence, and letting it overwrite a calibrated parameter launders an opinion into a number the optimiser treats as fact. So the calibration is used — **and the disagreement is recorded rather than discarded**:

```
CONFLICT: income_equity_beta: transcript says 0.80 (advisor-supplied, client
assented); calibration derives 0.032 from income_stability='High'. Calibration
used — the transcript figure is an estimate, not a measurement. Flag for
advisor review.
```

Refusal carries through the layer too. A transcript that never mentions salary or balances returns `profile=None` and the follow-up questions, rather than defaulting its way to a fabricated client — it would be strange for the extractor to honestly decline and the bridge to invent the same value one layer down.

**Entry point.** The transcript path is exposed on the agent itself, not just as an importable helper:

```python
run_profile_agent(transcripts={client_id: text, ...})           # conversations
run_profile_agent(transcripts=..., extractor=StructuredExtractor())  # LLM path
run_profile_agent()                                             # BLS, unchanged default
```

`build_profiles_from_transcripts()` returns `(profiles, results)` — the second element carries one `BridgeResult` per conversation **including the ones that produced no profile**, so callers can surface the follow-up questions. A conversation that did not build is a client to go back to, not an error to swallow.

### Sorting statements, not just extracting fields (*Now and Forward* §3)

Field extraction recovers typed values. It does not answer the question §3 poses: *"the things a client says are not all the same kind of object, and the system has to sort them before it can act."*

`StatementKind` in `contracts.py` encodes the five categories, and `ClientStatement` carries each one with its quote, its subject, and — the load-bearing part — **its destinations**:

| kind | what it is | typical destinations |
|---|---|---|
| `HARD_CONSTRAINT` | something the client will not hold | `universe_exclusion` |
| `SOFT_PREFERENCE` | a tilt they'd like reflected, not binding | `bl_view` |
| `RISK_FACT` | arrives sounding like a preference, is really an exposure | `concentration_limit` + `human_capital_beta` + `sector_underweight` |
| `SUITABILITY_FACT` | horizon, liquidity, what is in or out of scope | `suitability_record`, `scope_boundary` |
| `CHALLENGE` | a contradiction worth putting back to the client | `advisor_review` |

`RISK_FACT` is the category the mentor calls *"the one that makes the whole exercise worthwhile"*, and the reason `destinations` is a **list**. Priya's ARVX holding is not a preference for her employer's stock — it is a concentration, a human-capital beta, and a forced sector underweight, from one passage.

A validator enforces the thing that would otherwise quietly go wrong: **a statement with no destination raises.** Classifying a statement and then routing it nowhere is the same failure the review named at the top of Doc 1, one level up.

**Classification is where the language model earns its cost — and this is measured, not asserted.** `transcript_generator.py` plants five statements per transcript (one of each kind, with their required destinations) as a classification answer key, on the same principle as the field key: the ground truth is the input that produced the transcript, not something read back out of it.

76 planted statements, scored by `intake_eval.py`:

| metric | rule_based | naive | **structured** |
|---|---|---|---|
| statements returned | 0 | 0 | 272 |
| recall | 0% | 0% | **100%** |
| kind accuracy | n/a | n/a | **81.6%** |
| destinations complete | n/a | n/a | **100%** |

Recall is 100% for **every** kind — hard constraint, soft preference, risk fact, suitability fact, challenge — and so is kind accuracy and destination completeness.

**Kind accuracy started at 81.6%, and the diagnosis matters more than the number.** Adding a confusion matrix to the scorer showed all 7 errors were a single pair: `challenge → risk_fact`. The cause was the answer key, not the classifier. The planted challenge read *"Pretty cautious, I'd say. Though I suppose most of what I own is in the one stock"* — which **asserts a concentration**, and asserted it for all nine personas including the six holding no employer stock at all. That is a risk fact wearing a challenge's label, so a classifier filing it as `risk_fact` was reading it correctly.

Three fixes, in descending order of how much they mattered:

1. **The planted challenge is now a genuine contradiction** — *"Pretty cautious, I'd say. Although I would like this to roughly double over the next ten years"* — two client statements that cannot both hold, with no exposure claim and nothing depending on the persona's holdings.
2. **The prompt distinguishes the pair explicitly**: a risk fact states an exposure, a challenge states a contradiction between two things the client said, and a passage doing both should emit **two** statements rather than picking a winner.
3. **The scorer no longer takes the first quote match.** Since one passage can legitimately yield several statements, it now prefers a kind match among the candidates — otherwise a correct multi-statement answer scored wrong purely on ordering.

**Fixing it introduced a fourth bug, which is the useful part of the story.** The replacement challenge first read *"I would like this to roughly double over the next ten years"* — a clean contradiction with no exposure claim, and therefore correct on the axis being fixed. But it hands the extractor an **investment horizon** and a **growth objective**, both of which are planted as OMITTED refusal tests. Structured hallucination went from 0% to **52.9%** in a single run, every point of it an extractor correctly reading a horizon this generator had just put into the transcript.

So a planted statement has two constraints, not one: it must be the kind it claims to be, **and** it must not mention any field planted as omitted. The final wording — *"Pretty cautious, I'd say. Though honestly a 30% drop wouldn't bother me all that much"* — satisfies both.

`_assert_omissions_are_omitted()` now enforces the second constraint at generation time, raising if a transcript contains a phrase that would let an extractor legitimately recover a field its own answer key calls omitted. A first version tripped on the bare word "years" and fired on *"works out around 5% most years"*, so the cues are phrases — a guard that cries wolf gets switched off.

This is the **fourth** harness bug this work has surfaced, after the missing value space, the unrounded `bonus_rate` key, and the mislabelled challenge. The pattern is consistent enough to state plainly in the paper: **most of the apparent capability gaps in this evaluation turned out to be specification gaps in the evaluation.** A 52-point swing in a headline metric caused by a single reworded sentence is also the argument for the guard — without it, that swing is silent and reads as a model regression.

The floor scores **0%** — not because it does badly but because it does not implement classification. That is the point: on field extraction rule-based ties structured at 98.6%, so the language model buys nothing. On sorting prose into constraints, preferences, exposures and contradictions, the gap is total rather than incremental.

---

## The Reference Discovery Call

`agents/profile/reference_case.py` runs the mentor's own example, `intake_priya_raman.txt`, through the whole path. Until this existed, the intake layer had **only** ever been tested on transcripts this repo generated itself — uniform speaker turns, fixed phrasings, facts planted at chosen salience. The 98.6% was measured entirely on that corpus.

**The real call broke two things the synthetic corpus could not have caught.**

1. **It is advisor prose notes, not a dialogue.** There are no `CLIENT:` lines. The rule-based extractor read only those, so it recovered **0 of 12 fields — with no error**. `_client_voice` / `_advisor_voice` now handle both formats; in prose notes, attribution is decided per fact via advisor-judgement cues ("I'd put her at…", "when I floated…") rather than per line.

2. **`RSU_concentration` came back as `360000`** — the dollar figure, for a field the contract bounds to `[0, 1]`. The prompt never stated units, and the failure surfaced at the bridge, one layer from the cause. `FIELD_UNITS` now states them.

Current result on the real call:

| | fields recovered | statements classified |
|---|---|---|
| rule_based | **1 of 14** | 0 (cannot classify) |
| structured | **13 of 14** | 18 |

The one refusal is `has_pension`, which genuinely never comes up — correct behaviour, not a miss.

**What it produces.** Human capital $4,834,483 against financial capital $905,000 — **84.2% of her total wealth is her career**. Income beta 1.910 (equity-like), implicit equity exposure **1.608** against a risk budget of 0.66, giving a portfolio equity target of **−0.945**. Before she opens a brokerage account she is already over-exposed to equities, and the $360k of ARVX sits on top of that. Her holdings are extracted from the transcript, so the profile carries **ARVX at 39.8%** rather than a neutral default book.

Two statements are routed to `advisor_review` as challenges, both of which are the contradiction the mentor names: she describes herself as middle-of-the-road while holding a double-barrelled biotech bet, and she reports discomfort with the ARVX position while having taken no action on it.

And the source conflict fires on the real numbers — the transcript's advisor-supplied β of 1.20 against the calibration's 1.910, with the calibration used and the disagreement recorded.

---

## Input Sensitivity — How Far Does an Extraction Error Travel?

`sensitivity.py` answers *Now and Forward* §5. Perturb an intake-extractable field ±10/20%, rebuild the profile through the real formulas, re-run the real allocation agent, measure the move. 132 cells measured, 12 skipped as out-of-contract.

At ±20% input error, max |Δ equity share|:

| field | max Δ | bond-like | mixed | equity-like |
|---|---|---|---|---|
| income_volatility_sigma | **13.5pp** | 1.5pp | 13.5pp | 0 |
| income_equity_correlation | 9.7pp | 0.6pp | 9.7pp | 0 |
| annual_salary | 1.4pp | 0.2pp | 1.4pp | 0 |
| financial_capital | 1.3pp | 0.3pp | 1.3pp | 0 |

Against the mentor's own scale — 1% means extraction precision is not binding, 15% means the intake layer is the bottleneck — **σ_income sits at the bottleneck end for mixed personas and at the irrelevant end for everyone else.** The answer is persona-dependent, which is itself the result.

Two caveats that change the reading:

1. Turnover must be measured on the **assembled** book (`risky_weight × sleeve + safe_weight × safe`). Measured on `proposed_portfolio` alone it is 0.0 for every perturbation even when equity share moves 13 points, because that field is sleeve-relative — a direct symptom of review §4.1.
2. Once §4.1's unit conversion is fixed, **all 132 cells sit at a corner solution** and stop responding to intake error entirely. The moves above are what the *current* engine does. After the fix, a bound rather than the input sets these allocations.

---

## Implementation Notes

- **Entry point:** `agents/profile/profile_agent.py` → `run_profile_agent()`
- **Beta source:** `HC_BETA_TABLE` in `agents/profile/profile_model.py` (via `lookup_hc_beta()`) — the only authorised source. The former `hc_beta_table.py` and deprecated `beta.py` have been removed.
- **Environment:** pandas pinned to `>=2.0,<3.0` across all agents (pandas 3.x caused parquet load errors — team decision June 25 2026)
- **BLS OES source file:** `national_M2023_dl.xlsx` — converted and cached as `data/storage/bls_oes.parquet` on first run
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
