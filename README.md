# Agentic Portfolio Construction

A five-agent system that builds personalised portfolios from Human Capital theory, macro
regime detection, and fiduciary compliance. Fordham MSQF Capstone 2026.

---

## Core idea

Total wealth = Financial Capital + Human Capital (PV of future earnings). A portfolio's equity budget must be set **net of the equity risk the client already carries through their career**. A software engineer paid in RSUs and a tenured biology professor have opposite problems, and the same portfolio cannot be right for both.

```
implicit_equity_exposure = hc_share × β
effective_risk_budget    = (FC + HC × (1 − σ)) / total_wealth
portfolio_equity_target  = effective_risk_budget − implicit_equity_exposure
```

`portfolio_equity_target` can go **negative**, the career alone already exceeds the total risk budget, and the correct portfolio holds no equity at all.

---

## Pipeline

```
Profile ──┐
          ├─► Allocation ◄──► Risk        Loop A  (FLAG,  ≤3 revisions)
Research ─┘        │
                   ▼
              Compliance                  Loop B  (FAIL,  ≤2 revisions)
                   │
                   ▼
             AdvisorPackage
```

| Agent | Entry point | Output |
|---|---|---|
| Profile | `agents/profile/profile_agent.py → run_profile_agent()` | `ProfileAgentOutput` |
| Research | `agents/research/research_agent.py → run_research_agent()` | `MacroRegimeSnapshot` |
| Allocation | `agents/allocation/agent.py → run_allocation_agent()` | `AllocationAgentOutput` |
| Risk | `agents/risk/agent.py → run_risk_agent()` | `RiskAgentOutput` |
| Compliance | `agents/compliance/compliance_agent.py → run_compliance()` | `ComplianceAgentOutput` |
| Orchestrator | `agents/orchestrator/orchestrator_agent.py → run_pipeline()` | `AdvisorPackage` |

`contracts.py` is the single Pydantic v2 source of truth for every inter-agent schema with 18 enums, 40 models, 14 validators. Validators encode business rules, not just types.

**Division of labour.** Language models extract, classify and narrate; deterministic code computes and validates. The Allocation Agent's LLM proposes the risky-sleeve composition and `agents/allocation/validator.py` re-checks it against the same hard bounds the optimizer enforces. `risky_weight` i.e., how much of financial wealth is risky at all, stays fully deterministic, because that number *is* the human-capital thesis.

---

## Human capital calibration

| Income stability | σ | ρ | HC type | β |
|---|---:|---:|---|---:|
| High — academia, government, nursing | 0.05 | 0.10 | bond-like | 0.0318 |
| Medium — legal, engineering, finance | 0.20 | 0.40 | mixed | 0.5092 |
| Low — tech RSU, sales, commission | 0.40 | 0.75 | equity-like | 1.9096 |

β is **derived, not looked up**: `β = ρ × σ_income / σ_market`, against a single measured `SIGMA_MARKET = 0.1571` (annualised Fama-French `mktrf`, 312 months, 2000–2025). One model, one market. `ProfileAgentOutput.implied_market_volatility` inverts the identity as an audit, it must round-trip to ≈0.157 for every profile, whatever the HC type.

Sources: Ibbotson, Milevsky, Chen & Zhu (2007); Davis & Willen (2000).

---

## Results — nine BLS personas

Measured end-to-end, one `run_pipeline` per persona (`data/outputs/persona_runs/`).

| SOC | Occupation | HC type | β | Equity target | Portfolio vol | Compliance |
|---|---|---|---:|---:|---:|---|
| 25-1042 | Biology Professor | bond-like | 0.03 | +0.931 | 7.51% | PASS |
| 29-1141 | Registered Nurse | bond-like | 0.03 | +0.923 | 7.52% | PASS |
| 13-1041 | Compliance Officer | bond-like | 0.03 | +0.924 | 7.51% | PASS_WITH_WARNINGS |
| 23-1011 | Lawyer | mixed | 0.51 | +0.354 | 6.20% | PASS |
| 17-2141 | Mechanical Engineer | mixed | 0.51 | +0.329 | 5.77% | PASS |
| 13-2051 | Financial Analyst | mixed | 0.51 | +0.327 | 5.73% | PASS |
| 15-1252 | Software Developer | equity-like | 1.91 | −1.223 | 0.00% | PASS_WITH_WARNINGS |
| 11-3021 | IT Manager | equity-like | 1.91 | −1.230 | 0.00% | PASS_WITH_WARNINGS |
| 11-2022 | Sales Manager | equity-like | 1.91 | −1.208 | 0.00% | PASS_WITH_WARNINGS |

**Portfolio vol falls monotonically as career equity risk rises**: 7.5% → 5.8% → 0.00%.
The three equity-like personas receive a fully defensive portfolio: their careers already consume the entire risk budget. All nine cleared compliance (9/9 `APPROVE`, 9/9 clearance).

---

## Asset universe

**33 single stocks** (three largest by market cap in each of the 11 GICS sectors) and **5 defensive funds**(`TLT`, `SHY`, `TIP`, `LQD`, `GLD`). Single names are used because an ETF universe cannot enforce its own concentration limits.

`AGG` and `BIL` are deliberately absent: `AGG` duplicates the treasury and credit funds beside it, and `BIL` (0.6% annualised vol) *is* the risk-free asset that `safe_weight = 1 − w_fin` already represents.

| Limit | Value |
|---|---:|
| Single name | 10% |
| Sector | 20% |
| Employer's sector | 10% |
| Employer's own stock | 0% |

Because single names carry real GICS sectors, the employer-sector cap binds (9.82% and 10.01% in the runs above).

> **Estimation caveat.** `build_returns_matrix` ends in `pivot.dropna()`, so the covariance sample is the *intersection* of every ticker's history. The current universe yields **151 months × 38 assets (T/N ≈ 4.0)**, bounded by META's 2012 listing. Adequate, not comfortable — dropping the three post-2008 listings would extend it to 2005.

---

## Compliance Agent

Three jobs, 13 check functions, 20 named checks. **Every decision is deterministic** and the
LLM never sets a pass/fail.

| Job | File | Covers |
|---|---|---|
| 1 | `constraint_checks.py` | Audits the **Risk Agent's** output: completeness, actual-vs-limit-vs-flag consistency, derivation-method audit |
| 2 | `content_checks.py` | Fiduciary content (Reg BI, FINRA 2111): rationale completeness, specificity, HC acknowledgment, weight integrity, volatility suitability, **rationale distinctness** |
| 3 | `job3_robo_adviser_checks.py` | Robo-adviser duties (SEC IM 2017-02): composition↔objective, client-mandate consistency, algorithm-limitation disclosure, ETF-provider conflict |

`report.py` applies the severity ladder — any `HIGH` → `FAIL` and no clearance; `MEDIUM`/`LOW` → `PASS_WITH_WARNINGS` — and groups violations by responsible agent into `agent_feedback`, which is the routing table Loop B consumes.

---

## Setup

```bash
git clone https://github.com/niharika-madana/agentic-portfolio-construction.git
cd agentic-portfolio-construction
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -e .
```

Create `.env` with `FRED_API_KEY` and `ANTHROPIC_API_KEY`.

**Populate the cache** (one-time, ~10 min, needs WRDS credentials):

```python
from data.fetch.wrds import get_connection, fetch_crsp_monthly, fetch_crsp_daily, fetch_ff_factors
from data.fetch.fred import fetch_fred_macro
from data.fetch.bls import fetch_bls_oes
from data.fetch.factors import fetch_ff12
from agents.allocation.adapters import DEFAULT_TICKERS

conn = get_connection()                                      # prompts for WRDS credentials
fetch_crsp_monthly(DEFAULT_TICKERS, conn=conn, force=True)   # + permno_map + mkt_cap_weights
fetch_crsp_daily(DEFAULT_TICKERS,  conn=conn, force=True)
fetch_ff_factors(conn=conn)
fetch_fred_macro(fred_api_key="...")
fetch_bls_oes()
fetch_ff12()
```

`force=True` is required when a cache already exists, the fetchers short-circuit otherwise,
which silently leaves an older universe in place.

**Run:**

```python
from agents.profile.profile_agent import run_profile_agent
from agents.research.research_agent import run_research_agent
from agents.orchestrator.orchestrator_agent import run_pipeline

profiles = run_profile_agent(save=False)
macro    = run_research_agent()
package  = run_pipeline(profiles[0], macro)
```

Or open `pipeline_demo.ipynb` — environment → data → Profile → Research → both loops → full
`AdvisorPackage`.

**Test:** `pytest tests/ agents/ -q`

---

## Layout

```
contracts.py                     Pydantic v2 inter-agent schemas — the single source of truth
pipeline_demo.ipynb              End-to-end walkthrough

data/
├── fetch/                       WRDS/CRSP, FRED, BLS, Fama-French fetchers
├── storage/                     Parquet cache
└── outputs/                     Regime snapshot, profiles, per-persona pipeline runs

agents/shared/core/
├── allocation.py                Black-Litterman + Merton/HC risky-weight solve
├── risk.py                      VaR, drawdown, stress tests
├── constraints.py               Limit constants + breach checks
└── human_capital.py             BMS 1992 w_fin, Merton risky share

agents/profile/                  profile_agent · profile_model · sector_guard
                                 intake · intake_bridge · intake_eval · transcript_generator
agents/research/                 research_agent · pipeline (PELT→KMeans→XGBoost)
                                 rebalance · regime_returns · regime_sleeves · adapters
agents/allocation/               agent · adapters · validator
agents/risk/                     agent
agents/compliance/               compliance_agent · constraint_checks · content_checks
                                 job3_robo_adviser_checks · report
agents/orchestrator/             orchestrator_agent — control plane, both feedback loops

tests/                           compliance · job3 · profile · research · allocation_core
                                 human_capital · constraints · intake_weld
```

---

## References

**Human capital** — Ibbotson, Milevsky, Chen & Zhu (2007), *Lifetime Financial Advice*;
Davis & Willen (2000); Bodie, Merton & Samuelson (1992); Campbell & Viceira (2002).

**Regime detection** — Hamilton (1989); Clarida, Galí & Gertler (1999); Killick, Fearnhead &
Eckley (2012), *PELT*.

**Optimisation** — Black & Litterman (1992); Merton (1971); Sharpe (1964).

**Regulatory** — SEC Regulation Best Interest, 17 CFR 240.15l-1; SEC IM Guidance Update
2017-02 (Robo-Advisers); Investment Advisers Act of 1940; FINRA Rules 2111 and 2090;
Uniform Prudent Investor Act.
