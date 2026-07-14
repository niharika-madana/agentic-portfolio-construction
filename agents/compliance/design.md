# Compliance Agent — Design Document
**AI Financial Advisor Pipeline | Agent 5 of 5**
*Fordham MSQF Capstone 2026*
*Last updated: 2026-06-23*

---

## Purpose

Final validation gate in the pipeline. Audits the Risk Agent's constraint set for completeness and consistency, and independently validates the recommendation's fiduciary quality. Does not compute portfolio metrics.

**Implementation status:** ✅ Fully implemented — 52 unit tests passing.

---

## Architectural Role

```
Profile → Research → Allocation → Risk → Orchestrator (assemble) → Compliance
                          ↑                                              │
                          └──────────── FAIL (up to 2 revisions) ───────┘
```

The Compliance Agent is the last line of defense before the `AdvisorPackage` is finalized. It can trigger re-runs of the Allocation Agent, the Risk Agent, or both — but cannot re-run itself or the upstream Profile/Research agents.

---

## Two Parallel Jobs

**Job 1 — Constraint Set Verification:** Is the Risk Agent's report complete, consistent, and correctly derived?

**Job 2 — Content & Fiduciary Checks:** Is the recommendation client-specific, complete, and fiduciary-compliant?

---

## Input

The Orchestrator assembles a `ComplianceInput` object from all upstream agents and passes it alongside the `RiskAgentOutput`.

### ComplianceInput (from contracts.py)

```json
{
    "client_profile": {
        "persona":           "bls_15-1252_p50",
        "age":               38,
        "risk_tolerance":    "aggressive",
        "time_horizon_years": 27,
        "liquidity_needs":   "medium",
        "investment_objective": "growth"
    },
    "human_capital": {
        "industry":                 "Technology",
        "income_stability":         "low",
        "income_equity_beta":       0.90,
        "income_equity_correlation": 0.75,
        "implicit_equity_exposure": 0.862,
        "rsu_concentration":        0.35,
        "human_capital_type":       "equity-like",
        "has_pension":              false
    },
    "macro_regime": {
        "current_regime":    "Late-Cycle Expansion",
        "regime_volatility": "medium",
        "regime_confidence": 0.847
    },
    "proposed_portfolio": {
        "SPY": 0.20, "AGG": 0.30, "GLD": 0.15,
        "IEF": 0.20, "BIL": 0.10, "VNQ": 0.05
    },
    "allocation_rationale": {
        "SPY": "...", "AGG": "...", "GLD": "...",
        "IEF": "...", "BIL": "...", "VNQ": "..."
    },
    "regime_change_flag": { "detected": false }
}
```

### RiskAgentOutput (from contracts.py)

```json
{
    "risk_decision": "FLAG",
    "regime_evaluation": {
        "Early Recovery":          { "benchmark_drawdown": -0.44, "drawdown_floor": -0.44, "portfolio_drawdown": -0.38, "passed": true },
        "Late-Cycle Expansion":    { "benchmark_drawdown": -0.22, "drawdown_floor": -0.22, "portfolio_drawdown": -0.19, "passed": true },
        "Financial Crisis & ZLB":  { "benchmark_drawdown": -0.51, "drawdown_floor": -0.51, "portfolio_drawdown": -0.62, "passed": false },
        "Moderate Expansion":      { "benchmark_drawdown": -0.18, "drawdown_floor": -0.18, "portfolio_drawdown": -0.12, "passed": true },
        "Inflation Shock":         { "benchmark_drawdown": -0.22, "drawdown_floor": -0.22, "portfolio_drawdown": -0.29, "passed": false }
    },
    "position_limits": {
        "SPY": { "derived_limit": 0.25, "actual_weight": 0.20, "passed": true,  "method": "marginal_risk_contribution" },
        "AGG": { "derived_limit": 0.30, "actual_weight": 0.30, "passed": true,  "method": "marginal_risk_contribution" }
    },
    "sector_limits": {
        "Technology": { "base_limit": 0.25, "correlation_with_hc": 0.75, "adjusted_limit": 0.0625, "actual_sector_weight": 0.20, "passed": true }
    },
    "violations": ["Financial Crisis & ZLB: portfolio drawdown -62% exceeds floor -51%"],
    "derivation": {
        "drawdown_method":       "benchmark_relative_per_regime",
        "concentration_method":  "marginal_risk_contribution",
        "sector_method":         "hc_correlation_adjusted",
        "hc_type":               "equity-like",
        "rsu_concentration":     0.35,
        "data_source":           "data/storage/prices/"
    },
    "warnings": [],
    "portfolio_volatility_annual": 0.142
}
```

---

## Checks

### Job 1: Constraint Set Verification (audits Risk Agent)

#### Check 1.1 — Completeness
- All 5 academic regime labels present in `regime_evaluation`
  (`Early Recovery`, `Late-Cycle Expansion`, `Financial Crisis & ZLB`, `Moderate Expansion`, `Inflation Shock`)
- Every ticker in `proposed_portfolio` has a corresponding entry in `position_limits`
- `derivation` section exists and all method fields are non-empty

#### Check 1.2 — Consistency
- Cross-check actual values vs limits vs `passed` flags
- If `actual_weight > derived_limit` but `passed: true` → flag as Risk Agent internal error (HIGH)
- If `actual_sector_weight > adjusted_limit` but `passed: true` → flag as Risk Agent internal error (HIGH)
- If `portfolio_drawdown < drawdown_floor` (more negative) but `passed: true` → flag

#### Check 1.3 — Derivation Audit
- If `human_capital_type == "equity-like"`, `sector_method` must be `hc_correlation_adjusted`
- If `rsu_concentration > 0.10`, employer sector `adjusted_limit` must be strictly less than `base_limit`
- If `regime_change_detected == true`, `regime_evaluation` must include the current regime label

---

### Job 2: Content & Fiduciary Checks (Compliance-owned)

#### Check 2.1 — Rationale Completeness (HIGH)
Every ticker in `proposed_portfolio` must have a matching, non-empty entry in `allocation_rationale`.

#### Check 2.2 — Rationale Specificity (MEDIUM)
Rationale text must reference at least 2 client-specific terms from:
`(industry, human_capital_type, risk_tolerance, investment_objective, persona, rsu_concentration, regime_label)`

Generic boilerplate without client context → MEDIUM violation.

#### Check 2.3 — HC Acknowledgment
- **2.3a (HIGH):** If `rsu_concentration > 0.10`, at least one rationale must reference RSU exposure or employer concentration
- **2.3b (MEDIUM):** If `income_equity_beta > 0.80`, at least one rationale must acknowledge the HC-equity linkage

#### Check 2.4 — Weight Integrity (HIGH)
- Weights sum to 1.0 ±0.01
- No negative weights
- Minimum 2 positions

#### Check 2.5 — Volatility Suitability
Checks `portfolio_volatility_annual` against tolerance band for `risk_tolerance`:

| Risk Tolerance | Expected Ann. Vol Range | Severity if Above Band |
|---|---|---|
| Conservative | < 0.10 | HIGH |
| Moderate | 0.10 – 0.18 | HIGH if > 0.18; MEDIUM if < 0.10 |
| Aggressive | 0.14 – 0.25 | HIGH if > 0.25 |

Skipped if `portfolio_volatility_annual` is `None` (Risk Agent did not compute it).

---

## Severity → Status Mapping

| Highest Severity Found | `compliance_status` | `clearance` |
|---|---|---|
| Any HIGH | `FAIL` | `false` |
| MEDIUM or LOW (no HIGH) | `PASS_WITH_WARNINGS` | `true` |
| None | `PASS` | `true` |

---

## Output

```json
{
    "compliance_status": "FAIL",
    "overall_severity": "HIGH",
    "violations": [
        {
            "check":              "1.1_completeness",
            "severity":           "HIGH",
            "description":        "regime_evaluation missing 'Moderate Expansion'",
            "rule_reference":     "contracts.py RiskAgentOutput — 5 academic regimes required",
            "responsible_agent":  "risk_agent",
            "action_required":    "Re-run Risk Agent with all 5 regime anchor windows"
        }
    ],
    "passed_checks": ["1.2_consistency", "2.1_rationale_completeness", "2.4_weight_integrity"],
    "clearance": false,
    "agent_feedback": {
        "risk_agent": [
            { "check": "1.1_completeness", "action_required": "Add Moderate Expansion to regime_evaluation" }
        ]
    },
    "recommendation": "Return risk_agent for revision — regime evaluation incomplete"
}
```

---

## Feedback Loop Behavior

The Orchestrator routes agent_feedback keys to decide what to re-run:

| `agent_feedback` keys | Orchestrator action |
|---|---|
| `allocation_agent` | Re-run Allocation → then re-run Risk → then re-run Compliance |
| `risk_agent` only | Re-run Risk only → then re-run Compliance |
| `profile_agent` or `research_agent` | Break — cannot fix by re-running downstream agents |

Maximum 2 compliance revisions. If FAIL persists after 2 revisions, `AdvisorPackage` is assembled with the outstanding failure recorded in `RunMetadata.pipeline_warnings`.

---

## Failure Modes

| Failure | Handling |
|---|---|
| Missing upstream fields | Input validation before any check; missing fields → immediate FAIL with identifying error |
| Risk Agent internal contradiction | Check 1.2 catches actual > limit marked as passed — HIGH violation assigned to risk_agent |
| Persona misclassification | Check 1.3 re-validates that HC-adjusted limits were applied for equity-like clients |
| Generic rationale | Check 2.2 flags boilerplate without client context — fiduciary concern under Reg BI |
| Regime name mismatch | Check 1.1 validates against the exact 5 academic labels from contracts.py |

---

## File Structure

```
agents/compliance/
├── design.md             ← this document
├── compliance_agent.py   ← run_compliance() — entry point
├── constraint_checks.py  ← Job 1: checks 1.1, 1.2, 1.3
├── content_checks.py     ← Job 2: checks 2.1–2.5
├── report.py             ← ComplianceAgentOutput assembler
└── test_compliance.py    ← 52 unit tests (pytest, no API keys needed)
```

---

## Validation Coverage (52 Tests)

**Unit tests (mock data):**
- Each of the 7 checks tested independently with crafted inputs
- Boundary conditions: RSU at exactly 10%, weights summing to 0.99, confidence at exactly 0.60
- All three HC types produce meaningfully different compliance outcomes

**Determinism tests:**
- Same input → identical output on repeated calls (no stochastic elements)

**Integration tests:**
- Full pipeline with injected agent outputs for all 9 BLS personas
- Deliberate Risk Agent inconsistencies injected to verify Check 1.2 catches them
- Feedback loop verified: FAIL → allocation fixes → re-run → PASS

---

## Key References

**Regulatory**
- FINRA Rule 2111 — Suitability
- SEC Regulation Best Interest (Reg BI), 17 CFR 240.15l-1
- FINRA Rule 2090 — Know Your Customer
- Uniform Prudent Investor Act — fiduciary standard

**Academic**
- Ibbotson, Milevsky, Chen & Zhu (2007) — total wealth framework, HC type thresholds
