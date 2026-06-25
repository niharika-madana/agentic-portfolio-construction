# Agentic Portfolio Construction

A multi-agent AI system that constructs personalized investment portfolios grounded in Human Capital theory, macro regime detection, and regulatory compliance. Built for the Fordham MSQF Capstone 2026.

---

## Overview

The system runs five specialized agents in sequence to produce a compliance-cleared, client-specific portfolio recommendation. Each agent has a defined Pydantic contract in `contracts.py`; no agent invents a number — every quantitative output is produced by deterministic code and cited in the rationale text.

**Core idea:** A client's total wealth = Financial Capital + Human Capital (PV of future earnings). The portfolio's equity allocation must account for how much equity risk the client already carries implicitly through their career. A software developer with RSUs and a salary correlated to the S&P 500 needs a very different portfolio than a tenured biology professor.

---

## Agent Status

| Agent | Status | Entry Point |
|-------|--------|-------------|
| Profile Agent | ✅ Implemented | `agents/profile/profile_agent.py → run_profile_agent()` |
| Research Agent | ✅ Implemented | `agents/research/research_agent.py → run_research_agent()` |
| Allocation Agent | ✅ Implemented | `agents/allocation/agent.py → run_allocation_agent()` |
| Risk Agent | ✅ Implemented | `agents/risk/agent.py → run_risk_agent()` |
| Compliance Agent | ✅ Implemented | `agents/compliance/compliance_agent.py → run_compliance()` |
| Orchestrator | ✅ Implemented | `agents/orchestrator/orchestrator.py → run_pipeline() / run_all()` |

---

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  Profile Agent  │     │  Research Agent  │
│                 │     │                  │
│  BLS OES → HC   │     │  FRED → PELT     │
│  SCF → FC       │     │  KMeans → XGB    │
│  β table → exp. │     │  Regime label    │
└────────┬────────┘     └────────┬─────────┘
         │ ProfileAgentOutput    │ MacroRegimeSnapshot
         └──────────┬────────────┘
                    ▼
         ┌──────────────────────┐
         │   Allocation Agent   │◄──────────────────────────┐
         │                      │                           │
         │  Black-Litterman BL  │◄── Risk FLAG feedback     │
         │  equity_target drives│     violations, ≤3 rev.   │
         │  portfolio weights   │                           │
         └──────────┬───────────┘                           │
                    │ AllocationOutput / AllocationAgentOutput│
                    ▼                                        │
         ┌──────────────────────┐                           │
         │     Risk Agent       │──── FLAG ─────────────────┘
         │                      │
         │  VaR 95/99, CVaR     │
         │  Regime stress tests │
         │  HC-corr. adj. limits│
         └──────────┬───────────┘
                    │ RiskAgentOutput   APPROVE / FLAG / REJECT
                    ▼
         ┌──────────────────────┐     ┌──────────────────────────┐
         │  Orchestrator        │────►│   Compliance Agent       │
         │  (assemble input)    │     │                          │
         └──────────────────────┘     │  Job 1: Audit Risk Agent │
                    ▲                 │  Job 2: Fiduciary checks │
                    │ FAIL (≤2 rev.)  └──────────┬───────────────┘
                    └────────────────────────────┘
                                       │ PASS / PASS_WITH_WARNINGS
                                       ▼
                            ┌──────────────────────┐
                            │    AdvisorPackage    │
                            └──────────────────────┘
```

---

## Human Capital Framework

### Human Capital Valuation

```
HC = Annual Salary × [1 − (1 + r)^(−n)] / r
```

- `r` = FRED DGS10 (live 10Y Treasury yield; 4.4% fallback if no API key)
- `n` = years to retirement (= 65 − age)

### Income Risk Parameters

| Income Stability | σ | HC Type | β | ρ |
|---|---|---|---|---|
| High (academia, govt, nursing) | 0.05 | bond-like | 0.05 | 0.10 |
| Medium (legal, engineering, finance) | 0.20 | mixed | 0.35 | 0.40 |
| Low (tech RSU, sales, commissions) | 0.40 | equity-like | 0.90 | 0.75 |

β and ρ are from Ibbotson, Milevsky, Chen & Zhu (2007) and Davis & Willen (2000).

### Portfolio Equity Target

```
implicit_equity_exposure = HC_share × β
effective_risk_budget    = (FC + HC × (1 − σ)) / total_wealth
portfolio_equity_target  = effective_risk_budget − implicit_equity_exposure
```

This target is what makes each persona's allocation fundamentally different, and what the Allocation Agent builds around.

---

## Pipeline Walkthrough

### Step 1 — Profile Agent → `ProfileAgentOutput`

Generates one validated profile per BLS occupation at median salary (p50).

**Data sources:**
- BLS OES May 2023 national file → salary percentiles by SOC code → `data/storage/bls_oes.parquet`
- SCF 2022 Table 6 → median investable financial assets by age × income quartile (static dict)
- FRED DGS10 → annuity discount rate
- Calibrated β/ρ table (`agents/profile/hc_beta_table.py`)

**Nine target occupations:**

| SOC | Label | HC Type |
|---|---|---|
| 25-1042 | Biology Professor | bond-like |
| 29-1141 | Registered Nurse | bond-like |
| 13-1041 | Compliance Officer | bond-like |
| 23-1011 | Lawyer | mixed |
| 17-2141 | Mechanical Engineer | mixed |
| 13-2051 | Financial Analyst | mixed |
| 15-1252 | Software Developer | equity-like |
| 11-3021 | IT Manager | equity-like |
| 11-2022 | Sales Manager | equity-like |

**Illustrative outputs (BLS p50, r = 4.4%):**

| | Biology Professor | Financial Analyst | Software Developer |
|---|---|---|---|
| Salary | ~$81,840 | ~$99,890 | ~$130,160 |
| HC | ~$1.0M | ~$1.5M | ~$2.0M |
| β | 0.05 | 0.35 | 0.90 |
| Implicit equity exposure | ~4% | ~33% | ~86% |
| Portfolio equity target | **+92%** | **+48%** | **−24%** |

The software developer's career already delivers ~86% equity exposure — their portfolio target is negative, meaning they should underweight equities relative to a naive risk-tolerance approach.

---

### Step 2 — Research Agent → `MacroRegimeSnapshot`

Classifies the current macro regime using a four-stage deterministic pipeline on 13 FRED series:

1. **Feature engineering** — z-scores, log-transforms, first differences, YoY changes
2. **PELT change-point detection** (`ruptures`) — identifies structural breaks without date assumptions
3. **K-means + XGBoost** — clusters segments then maps to academic regime labels via anchor windows
4. **6-month rolling majority vote** — suppresses transient signal noise at boundaries

**Five academic regimes:**

| ID | Label | Anchor Period | Key Signal |
|----|-------|--------------|-----------|
| 0 | Early Recovery | 2003-06 → 2004-06 | Accommodative policy, spreads narrowing |
| 1 | Late-Cycle Expansion | 2005-01 → 2007-06 | Low VIX, rising rates, tight credit |
| 2 | Financial Crisis & ZLB | 2008-09 → 2010-12 | Credit stress, QE, near-zero rates |
| 3 | Moderate Expansion | 2015-01 → 2019-06 | Stable macro, low volatility |
| 4 | Inflation Shock | 2022-01 → 2023-06 | Above-target CPI, aggressive tightening |

---

### Step 3 — Allocation Agent → `AllocationAgentOutput`

Constructs a Black-Litterman portfolio anchored to the client's residual equity budget after subtracting implicit HC exposure.

**BL pipeline (`agents/shared/core/allocation.py`):**
1. Build annualised covariance from CRSP monthly excess returns (2000–2024)
2. CAPM equilibrium returns: `π = δ · Σ · w_mkt`
3. Fama-French factor views (per-asset expected alpha from FF4 regression)
4. BL posterior: `μ_BL, Σ_BL`
5. Mean-variance optimisation with sector + single-name + employer concentration constraints
6. BMS (1992) `w_fin` scaling → risky / safe weight split
7. Per-ticker weight decomposition: equilibrium baseline + view tilt + HC offset

**ETF universe (28 tickers, CRSP-covered):**

| Category | Tickers |
|---|---|
| Broad equity | SPY, IWM, EFA, EEM |
| Fixed income | AGG, TLT, IEF, SHY, HYG, LQD, TIP |
| Real assets | GLD, VNQ |
| Sector ETFs | XLK, XLF, XLV, XLE, XLI, XLC, XLY, XLP, XLU, XLRE |
| Cash proxy | BIL |

**Data source:** WRDS/CRSP monthly returns + Fama-French risk factors → `data/storage/`.

---

### Step 4 — Risk Agent → `RiskAgentOutput`

Stress-tests the allocation across all five academic regime windows. For equity-like HC clients, sector limits are HC-correlation adjusted:

```
adjusted_sector_limit = base_limit × (1 − ρ)
```

A software developer (ρ=0.75) faces a tech sector limit of 25% × (1−0.75) = **6.25%** — far tighter than the base 25% limit.

**Risk metrics computed:**
- Annualised volatility, VaR 95/99, CVaR 95/99
- Max drawdown vs 80/20 benchmark per regime window
- Per-ticker marginal risk contribution limits
- HC-adjusted sector and employer concentration limits

**Decision:** `APPROVE` → forward to Compliance. `FLAG` → return violated constraints to Allocation (max 3 revisions). `REJECT` → terminal (portfolio still delivered, failure recorded in metadata).

**Data source:** WRDS/CRSP daily returns → `data/storage/crsp_daily.parquet`.

---

### Step 5 — Compliance Agent → `ComplianceAgentOutput`

Two parallel jobs (52 deterministic checks):

**Job 1 — Constraint Set Verification (audits Risk Agent)**
- 1.1 Completeness — all 5 regimes present, every ticker has a limit, derivation fields populated
- 1.2 Consistency — actual weight vs limit vs pass/fail flag cross-check
- 1.3 Derivation Audit — equity-like HC → `hc_correlation_adjusted` method enforced

**Job 2 — Content & Fiduciary Checks**
- 2.1 Rationale completeness — every ticker has a rationale
- 2.2 Rationale specificity — cites ≥2 client-specific terms
- 2.3 HC acknowledgment — RSU >10% → RSU language; β>0.8 → HC beta language
- 2.4 Weight integrity — sum=1.0, no negatives, ≥2 positions
- 2.5 Volatility suitability — annualised vol within tolerance band for risk profile

**Status mapping:**

| Highest Severity | `compliance_status` | `clearance` |
|---|---|---|
| Any HIGH | FAIL | False |
| MEDIUM or LOW | PASS_WITH_WARNINGS | True |
| None | PASS | True |

---

### Step 6 — Dual Feedback Loops (Orchestrator)

**Loop A — Risk → Allocation (max 3 revisions)**

```python
while risk.decision == FLAG and revision < 3:
    allocation = run_allocation_agent(..., flag_constraints=risk.constraints_violated)
    risk = run_risk_agent(allocation, profile)
    revision += 1
```

**Loop B — Compliance → Allocation (max 2 revisions)**

```python
while not compliance.clearance and revision < 2:
    if "allocation_agent" in compliance.agent_feedback:
        allocation, _ = run_allocation_agent(...)
        compliance = run_compliance(assemble_compliance_input(...), risk_ao)
    revision += 1
```

---

## Data Layer

All external data is fetched once and cached as parquet in `data/storage/` (gitignored, ~500 MB budget). Agents read from parquet — they never make live API calls on each pipeline run.

| File | Source | Fetcher | Used by |
|---|---|---|---|
| `crsp_monthly.parquet` | WRDS/CRSP `msf` | `fetch_crsp_monthly()` | Allocation (BL covariance + market weights) |
| `crsp_daily.parquet` | WRDS/CRSP `dsf` | `fetch_crsp_daily()` | Risk (VaR, drawdown) |
| `ff_risk_factors.parquet` | WRDS/FF monthly | `fetch_ff_factors()` | Allocation (FF views) |
| `ff12_monthly.parquet` | Ken French library | `fetch_ff12()` | Research (regime validation) |
| `bls_oes.parquet` | BLS OES May 2023 | `fetch_bls_oes()` | Profile (salary distributions) |
| `fred_macro.parquet` | FRED API | `fetch_fred_macro()` | Research (13 macro series) |
| `permno_map.json` | WRDS | `fetch_crsp_monthly()` | Allocation + Risk (ticker → PERMNO) |
| `mkt_cap_weights.json` | WRDS | `fetch_crsp_monthly()` | Allocation (BL equilibrium prior) |

---

## File Structure

```
agentic-portfolio-construction-dev/
├── contracts.py                      ← All Pydantic inter-agent schemas (source of truth)
├── pipeline_demo.ipynb               ← Full end-to-end demo notebook
├── requirements.txt
│
├── data/
│   ├── __init__.py                   ← STORAGE_DIR path anchor
│   ├── fetch/
│   │   ├── fred.py                   ← FRED macro + DGS10
│   │   ├── bls.py                    ← BLS OES May 2023 → parquet
│   │   ├── wrds.py                   ← WRDS/CRSP + FF factors loaders
│   │   └── factors.py                ← Ken French FF12 + FF risk factor fallback
│   └── storage/                      ← Parquet cache (gitignored)
│
├── agents/
│   ├── shared/core/                  ← Quantitative core (no LLM)
│   │   ├── allocation.py             ← Black-Litterman optimizer
│   │   ├── risk.py                   ← VaR, drawdown, stress tests, HC metrics
│   │   ├── constraints.py            ← Limit constants + check functions
│   │   └── human_capital.py          ← BMS 1992 w_fin, Merton risky share
│   │
│   ├── profile/                      ← Agent 1
│   │   ├── profile_agent.py          ← run_profile_agent() → list[ProfileAgentOutput]
│   │   ├── hc_beta_table.py          ← Calibrated β/ρ table (Ibbotson 2007)
│   │   ├── human_capital.py          ← HC valuation + build_profile()
│   │   ├── personas.py               ← BLS-grounded persona builder
│   │   ├── loaders.py                ← BLS OES + FRED DGS10 loaders
│   │   └── test_profile.py
│   │
│   ├── research/                     ← Agent 2
│   │   ├── research_agent.py         ← run_research_agent() → MacroRegimeSnapshot
│   │   ├── loaders.py                ← FRED macro series loader
│   │   ├── features.py               ← 13-feature engineering pipeline
│   │   ├── detection.py              ← PELT change-point detection
│   │   ├── regime_model.py           ← KMeans + XGBoost + smoothing
│   │   ├── model_comparison.py       ← HMM/GMM benchmarking (paper validation)
│   │   └── adapters.py               ← MacroRegimeSnapshot builder
│   │
│   ├── allocation/                   ← Agent 3
│   │   ├── agent.py                  ← run_allocation_agent() → (AllocationOutput, AllocationAgentOutput)
│   │   └── adapters.py               ← ProfileAgentOutput → AllocationInput; ETF sector map
│   │
│   ├── risk/                         ← Agent 4
│   │   └── agent.py                  ← run_risk_agent() → (RiskOutput, RiskAgentOutput)
│   │
│   ├── compliance/                   ← Agent 5
│   │   ├── compliance_agent.py       ← run_compliance() → ComplianceAgentOutput
│   │   ├── constraint_checks.py      ← Job 1: audits Risk Agent output
│   │   ├── content_checks.py         ← Job 2: fiduciary checks
│   │   ├── report.py                 ← ComplianceAgentOutput assembler
│   │   └── test_compliance.py        ← 52 unit tests
│   │
│   └── orchestrator/
│       ├── orchestrator.py           ← run_pipeline() / run_all()
│       └── pipeline.py               ← Allocation ↔ Risk FLAG loop
│
└── tests/
    ├── test_allocation_core.py
    ├── test_constraints.py
    └── test_human_capital.py
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Populate the data cache (one-time, ~10 min, requires WRDS access)

```python
from data.fetch.wrds import get_connection, fetch_crsp_monthly, fetch_crsp_daily, fetch_ff_factors
from data.fetch.fred import fetch_fred_macro
from data.fetch.bls  import fetch_bls_oes
from data.fetch.factors import fetch_ff12
from agents.allocation.adapters import DEFAULT_TICKERS

conn = get_connection()                        # prompts for WRDS credentials
fetch_crsp_monthly(DEFAULT_TICKERS, conn=conn) # crsp_monthly.parquet + permno_map + mkt_cap_weights
fetch_crsp_daily(DEFAULT_TICKERS, conn=conn)   # crsp_daily.parquet  (~5 min)
fetch_ff_factors(conn=conn)                    # ff_risk_factors.parquet

fetch_fred_macro(fred_api_key="YOUR_FRED_KEY") # fred_macro.parquet
fetch_bls_oes()                                # bls_oes.parquet
fetch_ff12()                                   # ff12_monthly.parquet
```

### 3. Run the full pipeline

```python
from agents.orchestrator.orchestrator import run_all

packages = run_all(fred_api_key="YOUR_FRED_KEY")
# Returns list[AdvisorPackage] — one per BLS persona
```

Or use the demo notebook:

```bash
jupyter notebook pipeline_demo.ipynb
```

### 4. Run tests

```bash
pytest agents/compliance/test_compliance.py -v   # 52 compliance checks
pytest agents/profile/test_profile.py -v         # HC formula + contract validation
pytest tests/ -v                                 # allocation core + constraints
```

---

## Key References

**Human Capital Framework**
- Ibbotson, Milevsky, Chen & Zhu (2007). *Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance.* CFA Institute Research Foundation.
- Davis & Willen (2000). *Using Financial Assets to Hedge Labor Income Risks.* SSRN.
- Bodie, Merton & Samuelson (1992). *Labor Supply Flexibility and Portfolio Choice.* Journal of Economic Dynamics and Control.
- Campbell & Viceira (2002). *Strategic Asset Allocation.* Oxford University Press.

**Regime Detection**
- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica 57(2).
- Clarida, Galí & Gertler (1999). *The Science of Monetary Policy.* Journal of Economic Literature 37(4).

**Portfolio Optimisation**
- Black & Litterman (1992). *Global Portfolio Optimization.* Financial Analysts Journal.
- Walters (2013). *The Black-Litterman Model in Detail.* SSRN.
- Sharpe (1964). *Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk.* Journal of Finance.

**Regulatory**
- FINRA Rule 2111 — Suitability
- SEC Regulation Best Interest (Reg BI), 17 CFR 240.15l-1
- FINRA Rule 2090 — Know Your Customer
- Uniform Prudent Investor Act

**Data Sources**
- FRED (Federal Reserve Bank of St. Louis) — macro series and DGS10
- WRDS/CRSP — ETF monthly and daily returns, market cap weights
- BLS OES May 2023 — occupational wage statistics
- Federal Reserve SCF 2022 — household financial assets
- Ken French Data Library — Fama-French factors and 12 Industry Portfolios
