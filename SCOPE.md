# Project Scope Document
## Agentic Portfolio Construction
**Fordham University — Master of Science in Quantitative Finance**
**Capstone Project 2026**
*Last updated: 2026-06-24*

---

## 1. Problem Statement

Standard robo-advisors treat portfolio construction as a single-variable optimization problem — they optimize for risk tolerance alone. They ignore the single largest asset most working professionals own: their **Human Capital** — the present value of all future earnings.

A tenured biology professor and a software developer with RSUs may have identical financial portfolios and identical stated risk tolerances. But the professor's income is uncorrelated with markets (β ≈ 0.05), so her financial portfolio can hold nearly all equities. The software developer's income moves with the S&P 500 (β ≈ 0.90) — his career is already an undiversified equity position. Treating them identically violates basic fiduciary principles.

This project builds an agentic AI system that incorporates Human Capital into portfolio construction, subjects every recommendation to deterministic risk and compliance validation, and produces a regulatory-defensible, client-specific `AdvisorPackage`.

---

## 2. Objectives

1. Quantify each client's implicit equity exposure through Human Capital and use it as the primary portfolio constraint
2. Classify the current macro regime using a deterministic ML pipeline to anchor return expectations
3. Construct a personalized portfolio via Black-Litterman optimization that respects the residual equity budget
4. Stress-test the portfolio across five historical regimes with HC-correlation adjusted sector limits
5. Validate the recommendation against FINRA Rule 2111 (Suitability) and SEC Regulation Best Interest fiduciary standards
6. Use LLMs for text generation only — rationale, narrative, and report prose — while keeping all quantitative decisions in deterministic code

---

## 3. In Scope

### 3.1 Agent Pipeline

| Agent | Responsibility | Owner |
|---|---|---|
| **Profile Agent** | HC valuation (BLS OES + SCF + FRED), income β/ρ/σ, equity target | James Huang |
| **Research Agent** | Macro regime classification (PELT + KMeans + XGBoost), 13 FRED series | James Huang |
| **Allocation Agent** | Black-Litterman portfolio construction, HC-adjusted weights, LLM rationale | Aidan Altorelli |
| **Risk Agent** | Regime stress tests, position/sector limits, HC-correlation adjustment | Aidan Altorelli |
| **Compliance Agent** | Constraint set audit (Job 1), fiduciary content checks (Job 2) | Nihar |
| **Orchestrator** | Pipeline sequencing, dual feedback loops, AdvisorPackage assembly | Nihar |

### 3.2 Human Capital Framework
- HC valuation via annuity PV formula with FRED DGS10 as discount rate
- Calibrated β and ρ from Ibbotson et al. (2007) and Davis & Willen (2000)
- Three HC types: bond-like (β=0.05), mixed (β=0.35), equity-like (β=0.90)
- Portfolio equity target = effective risk budget − implicit equity exposure
- HC-correlation adjusted sector limits: `adjusted_limit = base_limit × (1 − ρ)`

### 3.3 Client Universe
- **9 BLS occupations** at p25/p50/p75 salary percentiles (up to 27 personas)
- Data-driven: salary from BLS OES May 2023, financial capital from SCF 2022

### 3.4 Macro Regime Detection
- 13 FRED macro series, monthly, 2003–2025
- 5 academic regimes: Early Recovery, Late-Cycle Expansion, Financial Crisis & ZLB, Moderate Expansion, Inflation Shock
- Pipeline: PELT change-point detection → K-means clustering → XGBoost anchor mapping → 6-month rolling majority vote
- HMM/GMM benchmarked and documented; XGBoost retained

### 3.5 Portfolio Construction
- Black-Litterman optimization with regime-conditional return views
- Instrument universe: 29 ETFs across broad equity, fixed income, real assets, sectors, cash (WRDS/CRSP)
- Constraint set: 10% single-name, 25% total economic sector, 15% employer, HC-adjusted sector limits

### 3.6 Risk Assessment
- Regime stress tests across all 5 academic regimes using historical ETF returns from CRSP parquet cache
- Marginal risk contribution position limits
- Portfolio volatility (annualized) for Compliance suitability check

### 3.7 Compliance Validation
**Job 1 — Constraint Set Verification (audits Risk Agent):**
- 1.1 Completeness: all 5 regimes present, all tickers have limits, derivation fields populated
- 1.2 Consistency: actual vs limit vs pass/fail flags cross-check
- 1.3 Derivation Audit: equity-like HC → HC-correlation-adjusted method enforced

**Job 2 — Content & Fiduciary Checks:**
- 2.1 Rationale completeness
- 2.2 Rationale specificity (≥2 client-specific terms)
- 2.3 HC acknowledgment (RSU >10% → RSU language; β>0.8 → HC beta language)
- 2.4 Weight integrity (sum=1.0, no negatives, ≥2 positions)
- 2.5 Volatility suitability (annualized vol within tolerance band per risk profile)

**Regulatory references:** FINRA Rule 2111, SEC Reg BI (17 CFR 240.15l-1), FINRA Rule 2090, Uniform Prudent Investor Act

### 3.8 Feedback Loops (Orchestrator)
- **Loop A — Risk → Allocation:** max 3 revisions; Risk violations passed as strings
- **Loop B — Compliance → Allocation/Risk:** max 2 revisions; agent_feedback routes re-runs

### 3.9 LLM Integration (3 touchpoints)
- **Allocation rationale:** LLM writes per-ticker rationale from optimizer decomposition; Compliance validates specificity
- **Research narrative:** LLM synthesizes top feature importances into 2-3 sentence regime explanation
- **Executive summary:** LLM writes plain-English client-facing paragraph in final report

### 3.10 Data Layer
- Centralized `data/storage/` parquet cache — all agents read from parquet, no live API calls on each run
- 500 MB total data budget
- Sources: FRED (fredapi), BLS OES (zip → xlsx → parquet), SCF 2022 (static), WRDS/CRSP (ETF prices + returns + market cap), WRDS ff.factors_monthly (FF risk factors), Ken French (FF12 industry portfolios for Research Agent)

### 3.11 Contracts & Validation
- `contracts.py` — single Pydantic v2 source of truth for all 5 agent output schemas
- 4 model validators on `ProfileAgentOutput` (holdings sum, wealth consistency, IEE formula, HC type consistency)
- No agent can produce a malformed output that passes to the next agent

### 3.12 Testing
- 52 unit tests for Compliance Agent (pytest, no API keys needed)
- Determinism tests: same input → identical output
- Adversarial tests: single-stock portfolio → REJECT; generic rationale → MEDIUM violation

---

## 4. Out of Scope

| Excluded | Reason |
|---|---|
| Live brokerage execution | Capstone is a recommendation engine, not a trading system |
| Real client data | All personas are BLS/SCF-grounded synthetic profiles |
| Options, derivatives, alternatives beyond listed ETFs | Instrument universe restricted to liquid ETFs for tractability |
| Real-time data feeds (WebSocket, Bloomberg) | FRED and WRDS via parquet cache is sufficient |
| Mobile/web UI | Output is a structured `AdvisorPackage` + Markdown report |
| Tax optimization | Out of scope for this version |
| Multi-period dynamic rebalancing | Single-period portfolio recommendation |
| CRSP individual stock data | ETF universe covers factor exposures without per-stock data |
| LLM choosing portfolio weights, risk thresholds, or compliance decisions | All quantitative decisions are deterministic; LLM handles text only |
| Model fine-tuning | Using Claude/GPT-4o API via prompting only |

---

## 5. Deliverables

| # | Deliverable | Format | Status |
|---|---|---|---|
| 1 | `contracts.py` — all Pydantic inter-agent schemas | Python | ✅ Complete |
| 2 | Profile Agent | Python module | ✅ Complete |
| 3 | Research Agent | Python module | ✅ Complete |
| 4 | Compliance Agent | Python module + 52 tests | ✅ Complete |
| 5 | Orchestrator | Python module | ✅ Complete |
| 6 | Allocation Agent | Python module | ✅ Complete |
| 7 | Risk Agent | Python module | ✅ Complete |
| 8 | Centralised data layer | `data/fetch/` + `data/storage/` | ✅ Complete |
| 9 | Design docs (all 5 agents + orchestrator) | Markdown | ✅ Complete |
| 10 | Three-persona demo | `demo.py` | ✅ Complete |
| 11 | Full pipeline end-to-end run | `orchestrator.run_all()` | ✅ Complete |
| 12 | Final report / paper | PDF | 🔲 Pending |

---

## 6. Technical Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      data/ layer                             │
│  FRED → parquet  │  BLS OES → parquet  │  yfinance → parquet │
└──────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      Profile Agent    Research Agent    (prices for
      [BLS + SCF +     [13 FRED series   Allocation +
       FRED DGS10]      KMeans + XGB]    Risk agents)
              │               │
              └───────────────┘
                      │
              ┌───────▼──────────────────────────────┐
              │        Orchestrator (control plane)  │
              │                                      │
              │  Allocation Agent ◄──── Risk Agent   │
              │       (BL optimizer)   (thresholds)  │
              │           ▲                          │
              │           └── Compliance Agent       │
              │                (2 jobs, 5 checks)    │
              └──────────────────────────────────────┘
                              │
                      AdvisorPackage
                    (Pydantic-validated)
```

**Language:** Python 3.11+
**Validation:** Pydantic v2
**ML:** XGBoost, scikit-learn, ruptures (PELT), hmmlearn
**Optimisation:** scipy / cvxpy (Black-Litterman)
**LLM:** Anthropic Claude (claude-sonnet-4-6) via API
**Data:** fredapi, wrds, requests (BLS), pandas, pyarrow
**Testing:** pytest

---

## 7. Data Sources

| Source | Data | Used By | Cached As |
|---|---|---|---|
| FRED (fredapi) | DGS10 (discount rate), 13 macro series | Profile, Research | `fred_dgs10.parquet`, `fred_macro.parquet` |
| BLS OES May 2023 | Salary percentiles by SOC code | Profile | `bls_oes.parquet` |
| SCF 2022 (Federal Reserve) | Median financial assets by age × income quartile | Profile | Static dict in code |
| WRDS / CRSP (`crsp.msf`, `crsp.dsf`) | Monthly + daily ETF returns, market cap, 2000–2025 | Allocation, Risk | `crsp_monthly.parquet`, `crsp_daily.parquet` |
| WRDS (`ff.factors_monthly`) | FF 3-factor + momentum + risk-free rate | Allocation | `ff_risk_factors.parquet` |
| Ken French Data Library | FF12 industry portfolios (monthly) | Research | `ff12_monthly.parquet` |

---

## 8. Constraints & Assumptions

| Constraint / Assumption | Detail |
|---|---|
| Retirement age | Fixed at 65 for all personas |
| HC discount rate | FRED DGS10; fallback 4.4% |
| β and ρ | Calibrated table (Ibbotson 2007, Davis & Willen 2000) — not estimated per-client |
| Salary → financial capital | BLS p25→SCF q2, p50→q3, p75→q4 |
| Regime history | Usable from 2003-02 (T5YIE / T10YIE data availability) |
| ETF history | Some sector ETFs begin 1998–2007; yfinance returns NaN for pre-inception dates |
| CRSP 2025 gap | CRSP returns unavailable for 2025; regime validation covers 1995–2024 only |
| No short selling | All portfolio weights ≥ 0 |
| Single-period | No dynamic rebalancing, no transaction costs |
| LLM determinism | LLM calls use temperature=0 for reproducibility; Compliance validates output, not the call |

---

## 9. Success Criteria

| Criterion | Threshold |
|---|---|
| All 52 compliance unit tests passing | 100% |
| Portfolio equity target differentiated across HC types | Professor target > Analyst target > Developer target |
| Regime detection recovers all 5 academic labels | 5 / 5 |
| Inflation Shock correctly produces negative expected return | Avg monthly return < 0 |
| Compliance correctly FAILs portfolios with RSU >10% and no RSU acknowledgment | 100% catch rate |
| Full pipeline (all 5 agents) runs end-to-end for all 9 personas | Zero unhandled exceptions |
| LLM rationale passes Compliance Check 2.2 specificity test | ≥2 client-specific terms per ticker |
| Data storage within budget | < 500 MB |

---

## 10. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| BLS OES SOC code not found in flat file | Low | Medium | Warning + skip; fallback to median salary for occupation group |
| FRED API failure at demo time | Medium | Low | Parquet cache eliminates live FRED dependency after first fetch |
| LLM rationale fails Compliance specificity check | Medium | Medium | Prompt includes explicit term list; Compliance violation triggers revision loop |
| Black-Litterman numerical instability (non-PD covariance) | Medium | High | Ledoit-Wolf shrinkage; fallback to equal-weight prior |
| Regime label bleed (XGBoost applies same label to distinct periods) | Medium | Medium | Anchor windows chosen from historically unambiguous period cores |
| ETF inception date mismatch for historical regime windows | Low | Low | NaN rows dropped from return matrix; documented in regime stress test |

---

## 11. Academic References

- Ibbotson, Milevsky, Chen & Zhu (2007). *Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance.* CFA Institute Research Foundation.
- Davis & Willen (2000). *Using Financial Assets to Hedge Labor Income Risks.* SSRN.
- Bodie, Merton & Samuelson (1992). *Labor Supply Flexibility and Portfolio Choice.* Journal of Economic Dynamics and Control.
- Campbell & Viceira (2002). *Strategic Asset Allocation.* Oxford University Press.
- Hamilton (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series.* Econometrica 57(2).
- Clarida, Galí & Gertler (1999). *The Science of Monetary Policy.* Journal of Economic Literature 37(4).
- Sharpe (1964). *Capital Asset Prices.* Journal of Finance.
- Walters (2013). *The Black-Litterman Model in Detail.* SSRN.
- FINRA Rule 2111 — Suitability
- SEC Regulation Best Interest (Reg BI), 17 CFR 240.15l-1
- Uniform Prudent Investor Act
