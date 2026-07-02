# Orchestrator — Design Document
**AI Financial Advisor Pipeline | Control Plane**
*Fordham MSQF Capstone 2026*
*Last updated: 2026-06-23*

---

## Purpose

The Orchestrator is the control plane of the pipeline. It does not produce analysis — it wires the five agents together, manages the two feedback loops, and assembles the final `AdvisorPackage`. It is stateless: given the same `ProfileAgentOutput` and `MacroRegimeSnapshot`, it always produces the same sequence of agent calls.

**File:** `orchestrator.py` (project root)

**Implementation status:** ⚠️ Partial — fully wired. Awaits Allocation and Risk agent implementations. Compliance agent is live and tested.

---

## Responsibilities

1. Receive `ProfileAgentOutput` (Agent 1) and `MacroRegimeSnapshot` (Agent 2)
2. Compute `portfolio_equity_target` and pass to Allocation Agent
3. Run Allocation → Risk with feedback loop (max 3 revisions)
4. Assemble `ComplianceInput` from upstream outputs
5. Run Compliance with feedback loop (max 2 revisions)
6. Collect warnings, revision counts, and final decisions into `RunMetadata`
7. Return a complete `AdvisorPackage`

---

## Pipeline Steps

### Step 1 — Pre-flight Checks

Before running any agent:

- `macro.is_low_confidence` (confidence < 0.60) → log WARNING, append to `pipeline_warnings`
- After risk runs: `risk.portfolio_volatility_annual is None` → log WARNING (Compliance Check 2.5 will be skipped)

These do not stop the pipeline — they are recorded in `RunMetadata.pipeline_warnings`.

---

### Step 2 — Compute Portfolio Equity Target

```python
portfolio_equity_target = profile.effective_risk_budget - profile.implicit_equity_exposure
```

This value is not stored in `ProfileAgentOutput` — it is derived by the orchestrator and passed to the Allocation Agent. It is the single number that makes each client's portfolio fundamentally different.

| Client | effective_risk_budget | implicit_equity_exposure | portfolio_equity_target |
|---|---|---|---|
| Biology Professor | 0.958 | 0.042 | **+0.916** |
| Financial Analyst | 0.811 | 0.330 | **+0.481** |
| Software Developer | 0.617 | 0.862 | **−0.245** |

---

### Step 3 — Initial Allocation

```python
allocation = run_allocation_agent(profile, macro, revision=0)
```

The Allocation Agent receives `ProfileAgentOutput` in full. The key fields it builds around:
- `portfolio_equity_target` (computed above)
- `income_equity_correlation` → HC-adjusted sector limits
- `RSU_concentration` → employer concentration constraint
- `liquidity_needs` → cash floor
- `current_holdings` → baseline for rebalancing

---

### Step 4 — Risk → Allocation Feedback Loop

```python
risk = run_risk_agent(allocation, profile, macro)
risk_revision = 0

while risk.risk_decision == FLAG and risk_revision < MAX_RISK_REVISIONS:
    risk_revision += 1
    allocation = run_allocation_agent(
        profile, macro,
        revision         = risk_revision,
        prior_risk_flags = risk.violations,
    )
    risk = run_risk_agent(allocation, profile, macro)
```

**Constants:** `MAX_RISK_REVISIONS = 3`

**Termination:**
- `PASS` → exit loop
- `REJECT` → exit loop immediately (structural issue, no point revising)
- `risk_revision >= 3` → log WARNING, proceed to compliance with outstanding flags

On each revision, the Allocation Agent receives the full violation string list and must revise to address them. The Risk Agent re-evaluates the new allocation from scratch.

---

### Step 5 — Assemble ComplianceInput

`assemble_compliance_input(profile, macro, allocation)` maps upstream fields to the flat `ComplianceInput` structure defined in `contracts.py`.

**Key field mappings:**

| `ComplianceInput` field | Source |
|---|---|
| `client_profile.persona` | `profile.client_id` |
| `client_profile.risk_tolerance` | `profile.risk_tolerance_level.value` (lowercased) |
| `client_profile.time_horizon_years` | `profile.investment_horizon_years` |
| `human_capital.income_equity_beta` | `profile.income_equity_beta` — drives Check 2.3b |
| `human_capital.income_equity_correlation` | `profile.income_equity_correlation` — used in Check 1.3a sector audit |
| `human_capital.implicit_equity_exposure` | `profile.implicit_equity_exposure` — cited in violation text |
| `human_capital.rsu_concentration` | `profile.RSU_concentration` — drives Check 2.3a |
| `human_capital.human_capital_type` | `profile.human_capital_type.value` — drives Check 1.3a |
| `macro_regime.current_regime` | `macro.regime_label` |
| `macro_regime.regime_volatility` | `_discretise_regime_volatility(macro.regime_volatility)` |
| `macro_regime.regime_confidence` | `macro.regime_confidence` |
| `proposed_portfolio` | `allocation.proposed_portfolio` |
| `allocation_rationale` | `allocation.allocation_rationale` |
| `regime_change_flag.detected` | `macro.regime_change_detected` |

**Volatility discretisation** (`_discretise_regime_volatility`):

| Float range | Label |
|---|---|
| < 0.01 | `"low"` |
| 0.01 – 0.03 | `"medium"` |
| ≥ 0.03 | `"high"` |

---

### Step 6 — Compliance → Allocation/Risk Feedback Loop

```python
compliance = run_compliance(compliance_input, risk)
comp_revision = 0

while compliance.compliance_status == FAIL and comp_revision < MAX_COMPLIANCE_REVISIONS:
    comp_revision += 1
    feedback = compliance.agent_feedback

    if 'allocation_agent' in feedback:
        allocation = run_allocation_agent(
            profile, macro,
            revision                    = risk_revision + comp_revision,
            prior_risk_flags            = risk.violations,
            prior_compliance_violations = [f.action_required for f in feedback['allocation_agent']],
        )
        risk = run_risk_agent(allocation, profile, macro)  # must re-run — weights changed

    elif 'risk_agent' in feedback:
        risk = run_risk_agent(allocation, profile, macro)  # allocation unchanged

    else:
        break  # responsible agent is profile or research — cannot fix by re-running

    compliance_input = assemble_compliance_input(profile, macro, allocation)
    compliance = run_compliance(compliance_input, risk)
```

**Constants:** `MAX_COMPLIANCE_REVISIONS = 2`

**Decision logic:**

| `agent_feedback` keys | Action |
|---|---|
| `allocation_agent` | Re-run Allocation + Risk (weights must change to fix fiduciary issues) |
| `risk_agent` only | Re-run Risk only (allocation is fine; Risk Agent needs to recompute) |
| `profile_agent` or `research_agent` | Break — cannot fix by re-running downstream agents |

**Why Risk must re-run when Allocation changes:** Compliance Check 2.5 (vol suitability) and Check 1.2 (consistency) both depend on Risk Agent outputs derived from specific allocation weights. If weights change, risk metrics become stale.

---

### Step 7 — Assemble AdvisorPackage

```python
metadata = RunMetadata(
    risk_revisions          = risk_revision,
    compliance_revisions    = comp_revision,
    final_risk_decision     = risk.risk_decision,
    final_compliance_status = compliance.compliance_status,
    pipeline_warnings       = pipeline_warnings,
)

return AdvisorPackage(
    profile    = profile,
    macro      = macro,
    allocation = allocation,
    risk       = risk,
    compliance = compliance,
    metadata   = metadata,
)
```

---

## Revision Limits and Rationale

| Loop | Limit | Rationale |
|---|---|---|
| Risk → Allocation | 3 | Three attempts exposes fundamentally mis-specified portfolios without over-optimising to stress tests |
| Compliance → Allocation/Risk | 2 | Compliance failures should resolve in one revision; second is a safety net; persistent failure requires human review |

---

## Agent Stubs (Current State)

Until Allocation and Risk agents are implemented:

```python
# orchestrator.py — replace these raise statements:

def run_allocation_agent(
    profile:                     ProfileAgentOutput,
    macro:                       MacroRegimeSnapshot,
    revision:                    int       = 0,
    prior_risk_flags:            list[str] = None,
    prior_compliance_violations: list[str] = None,
) -> AllocationAgentOutput:
    raise NotImplementedError("Allocation Agent not yet implemented")
    # → replace with:
    # from agents.allocation.allocation_agent import build_allocation
    # return build_allocation(profile, macro, revision, prior_risk_flags, prior_compliance_violations)

def run_risk_agent(
    allocation: AllocationAgentOutput,
    profile:    ProfileAgentOutput,
    macro:      MacroRegimeSnapshot,
) -> RiskAgentOutput:
    raise NotImplementedError("Risk Agent not yet implemented")
    # → replace with:
    # from agents.risk.risk_agent import assess_risk
    # return assess_risk(allocation, profile, macro)
```

---

## Injectable Callables (Testing)

`run_pipeline()` accepts injectable callables for testing without real implementations:

```python
package = run_pipeline(
    profile,
    macro,
    _run_allocation_fn = mock_allocation,
    _run_risk_fn       = mock_risk,
    _run_compliance_fn = mock_compliance,
)
```

The Compliance Agent is imported lazily (deferred import inside `run_pipeline()`) to avoid import-time failures if the compliance module has a dependency issue.

---

## Report Renderer

`render_report(package: AdvisorPackage) → str` produces a Markdown report containing:

- Client profile + income risk decomposition table (σ, ρ, β, implicit exposure, equity target)
- Macro regime snapshot with flags (`is_low_confidence`, `regime_change_detected`)
- Portfolio allocation table (ticker | weight | rationale)
- Risk assessment with regime stress test results for all 5 academic regimes
- Compliance status, all violations, and recommendation
- Pipeline metadata (revision counts, warnings, final decisions)

**Batch runner:**

```python
packages = run_all(personas=profiles, macro=snapshot)
for pkg in packages:
    print(render_report(pkg))
```

---

## Error Handling Philosophy

The orchestrator does not swallow agent exceptions. Real exceptions propagate to the call site. `pipeline_warnings` is only for **expected partial failures**:
- Regime low confidence (< 0.60)
- Missing `portfolio_volatility_annual` (Risk Agent didn't compute it)
- Revision limit exhausted without convergence

The compliance agent is deterministic — same inputs always return the same output. The revision limit cap prevents infinite loops when neither agent changes their output.

---

## Key Constants

```python
MAX_RISK_REVISIONS       = 3   # orchestrator.py line ~47
MAX_COMPLIANCE_REVISIONS = 2   # orchestrator.py line ~48
```
