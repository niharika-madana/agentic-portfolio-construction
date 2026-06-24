# Agentic Portfolio Construction

A multi-agent AI system that constructs personalized investment portfolios grounded in Human Capital theory, macro regime detection, and regulatory compliance. Built for the Fordham MSQF Capstone 2026.

---

## Overview

The system runs five specialized agents in sequence to produce a compliance-cleared, client-specific portfolio recommendation. Each agent has a defined Pydantic contract in `contracts.py`; no agent invents a number — every quantitative output is produced by deterministic code and cited in the rationale text.

**Core idea:** A client's total wealth = Financial Capital + Human Capital (PV of future earnings). The portfolio's equity allocation must account for how much equity risk the client is already carrying implicitly through their career. A software developer with RSUs and a salary correlated to the S&P 500 needs a very different portfolio than a tenured biology professor.

---

## Agent Implementation Status

| Agent | Status | Entry Point |
|-------|--------|-------------|
| Profile Agent | ✅ Implemented | `agents/profile/profile_agent.py → run_profile_agent()` |
| Research Agent | ✅ Implemented | `agents/research/research_agent.py → run_research_agent()` |
| Compliance Agent | ✅ Implemented | `agents/compliance/compliance_agent.py → run_compliance()` |
| Allocation Agent | 🔲 Stub | `orchestrator.py` — raises `NotImplementedError` |
| Risk Agent | 🔲 Stub | `orchestrator.py` — raises `NotImplementedError` |
| Orchestrator | ⚠️ Partial | `orchestrator.py → run_pipeline()` — wired; awaits Allocation + Risk |

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
         │  BL optimisation     │◄── Risk feedback          │
         │  equity_target drives│     (violations, ≤3 rev.) │
         │  portfolio weights   │                           │
         └──────────┬───────────┘                           │
                    │ AllocationAgentOutput                 │
                    ▼                                       │
         ┌──────────────────────┐                          │
         │     Risk Agent       │──── FLAG ────────────────┘
         │                      │
         │  Regime stress tests │
         │  Position/sector lim.│
         │  HC-corr. adj. limits│
         └──────────┬───────────┘
                    │ RiskAgentOutput
                    ▼
         ┌──────────────────────┐     ┌──────────────────────────┐
         │  Orchestrator        │────►│   Compliance Agent       │
         │  (assemble input)    │     │                          │
         └──────────────────────┘     │  Job 1: Audit Risk Agent │
                    ▲                 │  Job 2: Fiduciary checks │
                    │                 └──────────┬───────────────┘
                    │ FAIL (≤2 rev.)             │
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

- `r` = FRED DGS10 (10Y Treasury yield, cached in `data/storage/fred_dgs10.parquet`)
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

Generates one validated profile per BLS occupation at median salary (or p25/p75 with `include_percentile_variants=True`).

**Data sources:**
- BLS OES May 2023 national file → salary percentiles by SOC code → `data/storage/bls_oes.parquet`
- SCF 2022 Table 6 → median investable financial assets by age × income quartile (static dict)
- FRED DGS10 → annuity discount rate → `data/storage/fred_dgs10.parquet`
- Calibrated β/ρ table (Ibbotson 2007, Davis & Willen 2000)

**Nine target occupations (default universe):**

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
| Implicit equity exposure | 4.2% | 33.0% | 86.2% |
| Portfolio equity target | **+91.6%** | **+48.1%** | **−24.5%** |

The software developer's career already delivers 86% equity exposure — their portfolio target is negative, meaning they should underweight equities relative to a naive risk-tolerance approach.

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

**Data source:** `data/storage/fred_macro.parquet` (fetched once by `data/fetch/fred.py`).

---

### Step 3 — Allocation Agent → `AllocationAgentOutput`

Constructs a Black-Litterman portfolio anchored to the client's residual equity budget after subtracting implicit HC exposure. Receives prior risk and compliance violations on revision runs.

**Key constraint:** `portfolio_equity_target = effective_risk_budget − implicit_equity_exposure` from ProfileAgentOutput.

**Data source:** `data/storage/prices/{TICKER}.parquet` via `data.fetch.prices.load_returns()`.

---

### Step 4 — Risk Agent → `RiskAgentOutput`

Stress-tests the allocation across all five academic regimes. For equity-like HC clients, sector limits are HC-correlation adjusted:

```
adjusted_sector_limit = base_limit × (1 − ρ)
```

A software developer (ρ=0.75) faces a tech sector limit of 25% × (1−0.75) = 6.25% — far tighter than the base 25% limit.

Decision: `PASS` → forward to Compliance. `FLAG` → return to Allocation with violations (max 3 revisions). `REJECT` → halt for human review.

**Data source:** `data/storage/prices/{TICKER}.parquet` for regime-window return distributions.

---

### Step 5 — Compliance Agent → `ComplianceAgentOutput`

Two parallel jobs:

**Job 1 — Constraint Set Verification (audits Risk Agent)**
- 1.1 Completeness — all 5 regimes present, every ticker has a limit, derivation fields populated
- 1.2 Consistency — actual vs limit vs pass/fail flag cross-check
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
while risk.risk_decision == FLAG and risk_revision < 3:
    allocation = run_allocation_agent(..., prior_risk_flags=risk.violations)
    risk = run_risk_agent(allocation, profile, macro)
    risk_revision += 1
```

**Loop B — Compliance → Allocation/Risk (max 2 revisions)**

```python
while compliance.status == FAIL and comp_revision < 2:
    if 'allocation_agent' in compliance.agent_feedback:
        allocation = run_allocation_agent(..., prior_compliance_violations=...)
        risk = run_risk_agent(allocation, profile, macro)  # must re-run
    elif 'risk_agent' in compliance.agent_feedback:
        risk = run_risk_agent(allocation, profile, macro)
    else:
        break  # profile/research issue — cannot fix by re-running downstream
    compliance = run_compliance(assemble_compliance_input(...), risk)
    comp_revision += 1
```

---

## Data Layer

All external data is fetched once and cached as parquet in `data/storage/`. Agents read from parquet — they never make live API calls on each run.

```
data/
├── fetch/
│   ├── fred.py        → fred_macro.parquet + fred_dgs10.parquet
│   ├── bls.py         → bls_oes.parquet       (xlsx → parquet in-memory)
│   ├── prices.py      → prices/{TICKER}.parquet  (29 ETFs via yfinance)
│   └── factors.py     → ff12_monthly.parquet
└── storage/           (gitignored; ~50-80 MB estimated)
    ├── fred_macro.parquet
    ├── fred_dgs10.parquet
    ├── bls_oes.parquet
    ├── ff12_monthly.parquet
    └── prices/
        ├── SPY.parquet
        ├── AGG.parquet
        └── ...
```

**Budget:** 500 MB total. Run `from data.registry import show_registry; show_registry()` to check current usage.

---

## File Structure

```
agentic-portfolio-construction-dev/
├── contracts.py                  ← All Pydantic inter-agent schemas (source of truth)
├── orchestrator.py               ← Pipeline runner + dual feedback loops + report renderer
├── demo.py                       ← Three-persona compliance demo
├── requirements.txt
│
├── data/                         ← Centralised data layer (new)
│   ├── __init__.py               ← STORAGE_DIR, PRICES_DIR path anchors
│   ├── registry.py               ← show_registry(), budget_status()
│   ├── fetch/
│   │   ├── fred.py               ← FRED macro + DGS10 → parquet
│   │   ├── bls.py                ← BLS OES May 2023 → parquet
│   │   ├── prices.py             ← yfinance ETF universe → parquet
│   │   └── factors.py            ← Fama-French 12 → parquet
│   └── storage/                  ← Parquet cache (gitignored)
│
└── agents/
    ├── profile/                  ← ✅ Implemented
    │   ├── profile_agent.py      ← run_profile_agent()
    │   ├── hc_beta_table.py      ← Calibrated β/ρ table
    │   ├── human_capital.py      ← HC valuation + build_profile()
    │   ├── personas.py           ← BLS-grounded persona builder
    │   ├── loaders.py            ← Reads from data/storage/ parquet
    │   └── test_profile.py
    │
    ├── research/                 ← ✅ Implemented
    │   ├── research_agent.py     ← run_research_agent()
    │   ├── loaders.py            ← Reads from data/storage/ parquet
    │   ├── features.py           ← 13-feature engineering pipeline
    │   ├── detection.py          ← PELT change-point detection
    │   ├── regime_model.py       ← KMeans + XGBoost + smoothing
    │   ├── model_comparison.py   ← HMM/GMM benchmarking
    │   ├── adapters.py           ← MacroRegimeSnapshot builder
    │   └── test_research.py
    │
    ├── compliance/               ← ✅ Implemented (52 unit tests)
    │   ├── compliance_agent.py   ← run_compliance()
    │   ├── constraint_checks.py  ← Job 1: audits Risk Agent output
    │   ├── content_checks.py     ← Job 2: fiduciary checks
    │   ├── report.py             ← ComplianceAgentOutput assembler
    │   └── test_compliance.py    ← 52 unit tests (pytest)
    │
    ├── allocation/               ← 🔲 Stub — design doc only
    │   └── Allocation_Agent_README.md
    │
    └── risk/                     ← 🔲 Stub — design doc only
        └── Risk_Agent_README.md
```

---

## Setup

```bash
pip install -r requirements.txt

# Step 1: Fetch all data (one-time setup, ~50-80 MB)
python - <<'EOF'
from data.fetch.fred    import fetch_fred_macro, fetch_fred_dgs10
from data.fetch.bls     import fetch_bls_oes
from data.fetch.prices  import fetch_all
from data.fetch.factors import fetch_ff12

fetch_fred_macro(fred_api_key="YOUR_FRED_KEY")
fetch_fred_dgs10(fred_api_key="YOUR_FRED_KEY")
fetch_bls_oes()
fetch_all()     # 29 ETFs, 2000-2025 — takes ~2 min
fetch_ff12()
EOF

# Step 2: Check data budget
python -c "from data.registry import show_registry; show_registry()"

# Step 3: Run profile agent (9 BLS personas)
python -c "
from agents.profile.profile_agent import run_profile_agent
profiles = run_profile_agent(fred_api_key='YOUR_FRED_KEY')
print(f'{len(profiles)} profiles generated')
"

# Step 4: Run research agent
python -c "
from agents.research.research_agent import run_research_agent
snapshot = run_research_agent(fred_api_key='YOUR_FRED_KEY')
print(snapshot.regime_label, snapshot.regime_confidence)
"

# Step 5: Run compliance tests (52 tests, no API keys needed)
pytest agents/compliance/test_compliance.py -v

# Step 6: Run three-persona compliance demo
python demo.py
```

---

## Key References

**Human Capital Framework**
- Ibbotson, Milevsky, Chen & Zhu (2007). *Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance.* CFA Institute Research Foundation.
- Davis & Willen (2000). *Using Financial Assets to Hedge Labor Income Risks.* SSRN.
- Campbell & Viceira (2002). *Strategic Asset Allocation.* Oxford University Press.
- Bodie, Merton & Samuelson (1992). *Labor Supply Flexibility and Portfolio Choice.* Journal of Economic Dynamics and Control.

**Regime Detection(needs to be revamped)**
- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica 57(2).
- Clarida, Galí & Gertler (1999). *The Science of Monetary Policy.* Journal of Economic Literature 37(4).

**Portfolio Optimisation(needs to be revamped)**
- Sharpe (1964). *Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk.* Journal of Finance.
- Walters (2013). *The Black-Litterman Model in Detail.* SSRN.

**Regulatory**
- FINRA Rule 2111 — Suitability
- SEC Regulation Best Interest (Reg BI), 17 CFR 240.15l-1
- FINRA Rule 2090 — Know Your Customer
- Uniform Prudent Investor Act

**Data Sources**
- FRED (Federal Reserve Bank of St. Louis) — macro series and DGS10
- BLS OES May 2023 — occupational wage statistics
- Federal Reserve SCF 2022 — household financial assets
- yfinance — ETF price history for backtesting
- Ken French Data Library — Fama-French 12 Industry Portfolios
