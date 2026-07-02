# Agentic Portfolio Construction

A five-agent AI system that constructs personalized investment portfolios grounded in Human Capital theory, macro regime detection, and regulatory compliance. Built for the Fordham MSQF Capstone 2026.

---

## Core Idea

A client's total wealth = Financial Capital + Human Capital (PV of future earnings). The portfolio's equity allocation must account for how much equity risk the client already carries implicitly through their career. A software developer with RSUs correlated to the S&P 500 needs a very different portfolio than a tenured biology professor.

```
implicit_equity_exposure = HC_share × β
effective_risk_budget    = (FC + HC × (1 − σ)) / total_wealth
portfolio_equity_target  = effective_risk_budget − implicit_equity_exposure
```

---

## Pipeline

```
Profile Agent ──┐
                ├──► Allocation Agent ◄──── FLAG (≤3 revisions)
Research Agent ─┘         │                       │
                           ▼                      │
                       Risk Agent ─── FLAG ───────┘
                           │
                           ▼
                    Compliance Agent ◄──── FAIL (≤2 revisions)
                           │
                           ▼
                      AdvisorPackage
```

| Agent | Entry Point | Output |
|-------|------------|--------|
| Profile | `agents/profile/profile_agent.py → run_profile_agent()` | `ProfileAgentOutput` |
| Research | `agents/research/research_agent.py → run_research_agent()` | `MacroRegimeSnapshot` |
| Allocation | `agents/allocation/agent.py → run_allocation_agent()` | `AllocationAgentOutput` |
| Risk | `agents/risk/agent.py → run_risk_agent()` | `RiskAgentOutput` |
| Compliance | `agents/compliance/compliance_agent.py → run_compliance()` | `ComplianceAgentOutput` |
| Orchestrator | `agents/orchestrator/orchestrator_agent.py → run_pipeline()` | `AdvisorPackage` |

All inter-agent schemas are defined in `contracts.py` (Pydantic v2).

---

## Human Capital Parameters

| Income Stability | σ | HC Type | β | ρ |
|---|---|---|---|---|
| High (academia, government, nursing) | 0.05 | bond-like | 0.05 | 0.10 |
| Medium (legal, engineering, finance) | 0.20 | mixed | 0.35 | 0.40 |
| Low (tech RSU, sales, commissions) | 0.40 | equity-like | 0.90 | 0.75 |

β and ρ from Ibbotson, Milevsky, Chen & Zhu (2007) and Davis & Willen (2000).

**Nine target occupations (BLS p50):**

| SOC | Label | HC Type | Portfolio Equity Target |
|---|---|---|---|
| 25-1042 | Biology Professor | bond-like | ~+92% |
| 29-1141 | Registered Nurse | bond-like | ~+88% |
| 13-1041 | Compliance Officer | bond-like | ~+85% |
| 23-1011 | Lawyer | mixed | ~+60% |
| 17-2141 | Mechanical Engineer | mixed | ~+55% |
| 13-2051 | Financial Analyst | mixed | ~+48% |
| 15-1252 | Software Developer | equity-like | ~−24% |
| 11-3021 | IT Manager | equity-like | ~−18% |
| 11-2022 | Sales Manager | equity-like | ~−10% |

---

## Setup

### 1. Install

```bash
git clone https://github.com/niharika-madana/agentic-portfolio-construction.git
cd agentic-portfolio-construction
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e .
```

### 2. Populate data cache (one-time, ~10 min, requires WRDS access)

```python
from data.fetch.wrds import get_connection, fetch_crsp_monthly, fetch_crsp_daily, fetch_ff_factors
from data.fetch.fred import fetch_fred_macro
from data.fetch.bls import fetch_bls_oes
from data.fetch.factors import fetch_ff12
from agents.allocation.adapters import DEFAULT_TICKERS

conn = get_connection()                         # prompts for WRDS credentials
fetch_crsp_monthly(DEFAULT_TICKERS, conn=conn)  # + permno_map + mkt_cap_weights
fetch_crsp_daily(DEFAULT_TICKERS, conn=conn)
fetch_ff_factors(conn=conn)
fetch_fred_macro(fred_api_key="YOUR_FRED_KEY")
fetch_bls_oes()
fetch_ff12()
```

Data is cached as parquet in `data/storage/` (data also pushed for reference).

### 3. Run the pipeline

```python
from agents.profile.profile_agent import run_profile_agent
from agents.research.research_agent import run_research_agent
from agents.orchestrator.orchestrator_agent import run_pipeline

profiles = run_profile_agent()
macro    = run_research_agent()
package  = run_pipeline(profiles[0], macro, fred_api_key="YOUR_FRED_KEY")
```

Or open `pipeline_demo.ipynb` for a guided walkthrough.

### 4. Run tests

```bash
pytest tests/ -v
```

---

## File Structure

```
contracts.py                        ← Pydantic v2 inter-agent schemas
pipeline_demo.ipynb                 ← End-to-end demo notebook
pyproject.toml                      ← Dependencies + pytest config
│
data/
├── fetch/                          ← One-time data fetchers (FRED, BLS, WRDS, FF)
├── storage/                        ← Parquet cache (gitignored)
└── outputs/                        ← JSON/CSV pipeline outputs
│
agents/
├── shared/core/
│   ├── allocation.py               ← Black-Litterman optimizer
│   ├── risk.py                     ← VaR, drawdown, stress tests
│   ├── constraints.py              ← Limit constants + helpers
│   └── human_capital.py            ← BMS 1992 w_fin, Merton risky share
│
├── profile/
│   ├── profile_agent.py            ← run_profile_agent()
│   └── profile_model.py            ← HC valuation, β table, persona builder
│
├── research/
│   ├── research_agent.py           ← run_research_agent()
│   ├── pipeline.py                 ← PELT → KMeans → XGBoost → smoothing
│   └── adapters.py                 ← MacroRegimeSnapshot builder
│
├── allocation/
│   ├── agent.py                    ← run_allocation_agent()
│   └── adapters.py                 ← ProfileAgentOutput → AllocationInput
│
├── risk/
│   └── agent.py                    ← run_risk_agent()
│
├── compliance/
│   ├── compliance_agent.py         ← run_compliance()
│   ├── constraint_checks.py        ← Job 1: audit Risk Agent (checks 1.1–1.3)
│   ├── content_checks.py           ← Job 2: fiduciary checks (checks 2.1–2.5)
│   └── report.py                   ← ComplianceAgentOutput assembler
│
└── orchestrator/
    └── orchestrator_agent.py       ← run_pipeline(), dual feedback loops
│
tests/
├── test_compliance.py
├── test_profile.py
├── test_research.py
├── test_allocation_core.py
├── test_human_capital.py
└── test_constraints.py
```

---

## References

**Human Capital**
- Ibbotson, Milevsky, Chen & Zhu (2007). *Lifetime Financial Advice.* CFA Institute Research Foundation.
- Davis & Willen (2000). *Using Financial Assets to Hedge Labor Income Risks.* SSRN.
- Bodie, Merton & Samuelson (1992). *Labor Supply Flexibility and Portfolio Choice.* JEDC.
- Campbell & Viceira (2002). *Strategic Asset Allocation.* Oxford University Press.

**Regime Detection**
- Hamilton (1989). *A New Approach to Nonstationary Time Series and the Business Cycle.* Econometrica.
- Clarida, Galí & Gertler (1999). *The Science of Monetary Policy.* JEL.

**Portfolio Optimisation**
- Black & Litterman (1992). *Global Portfolio Optimization.* Financial Analysts Journal.
- Sharpe (1964). *Capital Asset Prices.* Journal of Finance.

**Regulatory**
- FINRA Rule 2111 — Suitability
- SEC Regulation Best Interest (Reg BI), 17 CFR 240.15l-1
- FINRA Rule 2090 — Know Your Customer
- Uniform Prudent Investor Act
