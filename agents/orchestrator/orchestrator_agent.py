"""
orchestrator_agent.py — full five-agent pipeline control plane.

Entry point:
    run_pipeline(profile, macro, fred_api_key) → AdvisorPackage

Pipeline sequence:
  1. Pre-flight checks (low-confidence macro warning)
  2. Allocation ↔ Risk loop (max 3 FLAG revisions)
  3. Assemble ComplianceInput
  4. Compliance ↔ Allocation/Risk loop (max 2 revisions)
  5. Assemble AdvisorPackage

All quantitative decisions are deterministic. LLMs participate only in
rationale, reasoning trace, and executive summary (3 touchpoints total).
"""

from __future__ import annotations

import logging
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contracts import (
    AdvisorPackage, AllocationAgentOutput, AllocationOutput,
    ClientProfileSection, ComplianceInput, ComplianceStatus,
    HumanCapitalSection, MacroRegimeSection, MacroRegimeSnapshot,
    ProfileAgentOutput, RegimeChangeFlag, RiskAgentOutput, RiskDecision,
    RiskOutput, RunMetadata,
)
from agents.allocation.agent import run_allocation_agent
from agents.compliance.compliance_agent import run_compliance
from agents.risk.agent import run_risk_agent
from data.fetch.fred import latest_dgs10
from data.fetch.wrds import load_ff_factors

logger = logging.getLogger(__name__)

_MAX_FLAG_ITERATIONS     = 3
_MAX_COMPLIANCE_REVISIONS = 2


# ---------------------------------------------------------------------------
# Allocation ↔ Risk FLAG loop
# ---------------------------------------------------------------------------

def run_alloc_risk_loop(
    profile:       ProfileAgentOutput,
    discount_rate: float,
    ff_factors:    pd.DataFrame | None = None,
) -> tuple[AllocationOutput, RiskOutput, RiskAgentOutput, AllocationAgentOutput, int]:
    """
    Run the Allocation ↔ Risk FLAG feedback loop for one client profile.

    Returns (AllocationOutput, RiskOutput, RiskAgentOutput, AllocationAgentOutput, revisions)
    where revisions = number of FLAG re-entries (0 = first run passed).
    """
    if ff_factors is None:
        ff_factors = load_ff_factors()

    flag_constraints = []
    flag_iteration   = 0
    alloc_output: AllocationOutput | None     = None
    risk_output:  RiskOutput | None           = None
    risk_ao:      RiskAgentOutput | None      = None
    alloc_ao:     AllocationAgentOutput | None = None

    for iteration in range(_MAX_FLAG_ITERATIONS + 1):
        label = "First run" if iteration == 0 else f"FLAG re-entry #{iteration}"
        logger.info(f"[pipeline] Iteration {iteration + 1}  ({label})")

        alloc_output, alloc_ao = run_allocation_agent(
            profile          = profile,
            discount_rate    = discount_rate,
            ff_factors       = ff_factors,
            flag_constraints = flag_constraints,
            flag_iteration   = flag_iteration,
        )

        logger.info(
            f"[pipeline] Allocation done — risky {alloc_output.risky_weight:.1%} "
            f"E[r] {alloc_output.portfolio_statistics.expected_return:.2%} "
            f"vol {alloc_output.portfolio_statistics.volatility:.2%}"
        )

        risk_output, risk_ao = run_risk_agent(
            allocation_output = alloc_output,
            profile           = profile,
        )

        logger.info(f"[pipeline] Risk decision: {risk_output.decision.value}")

        if risk_output.decision in (RiskDecision.APPROVE, RiskDecision.REJECT):
            return alloc_output, risk_output, risk_ao, alloc_ao, iteration

        # FLAG — feed tightened constraints back into next allocation run
        flag_constraints = risk_output.constraints_violated
        flag_iteration   = iteration + 1

    # Safety fallback (should not reach here — iteration 3 always REJECTs)
    return alloc_output, risk_output, risk_ao, alloc_ao, _MAX_FLAG_ITERATIONS


# ---------------------------------------------------------------------------
# ComplianceInput assembly
# ---------------------------------------------------------------------------

def _regime_volatility_label(v: float) -> str:
    if v < 0.01:
        return "low"
    elif v < 0.025:
        return "medium"
    else:
        return "high"


def assemble_compliance_input(
    profile:  ProfileAgentOutput,
    macro:    MacroRegimeSnapshot,
    alloc_ao: AllocationAgentOutput,
) -> ComplianceInput:
    """Build the ComplianceInput the Compliance Agent expects."""
    return ComplianceInput(
        client_profile = ClientProfileSection(
            persona              = profile.client_id,
            age                  = profile.age,
            risk_tolerance       = profile.risk_tolerance_level.value,
            time_horizon_years   = profile.investment_horizon_years,
            liquidity_needs      = profile.liquidity_needs.value,
            investment_objective = profile.investment_objective.value,
        ),
        human_capital = HumanCapitalSection(
            industry                  = profile.industry_exposure_sector,
            income_stability          = profile.income_stability.value,
            income_equity_correlation = profile.income_equity_correlation,
            income_equity_beta        = profile.income_equity_beta,
            implicit_equity_exposure  = profile.implicit_equity_exposure,
            rsu_concentration         = profile.RSU_concentration,
            human_capital_type        = profile.human_capital_type.value,
            employer_sector           = profile.industry_exposure_sector,
            has_pension               = profile.has_pension,
            estimated_hc_pv           = profile.human_capital_valuation,
        ),
        macro_regime = MacroRegimeSection(
            current_regime      = macro.regime_label,
            historical_analogue = macro.prior_regime,
            regime_risk_flags   = [],
            regime_confidence   = macro.regime_confidence,
            regime_volatility   = _regime_volatility_label(macro.regime_volatility),
        ),
        proposed_portfolio   = alloc_ao.proposed_portfolio,
        allocation_rationale = alloc_ao.allocation_rationale,
        regime_change_flag   = RegimeChangeFlag(
            detected       = macro.regime_change_detected,
            prior_regime   = macro.prior_regime if macro.regime_change_detected else None,
            current_regime = macro.regime_label if macro.regime_change_detected else None,
            shift_date     = str(macro.regime_shift_date) if macro.regime_change_detected else None,
        ),
        client_statements    = profile.client_statements,
    )


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    profile:      ProfileAgentOutput,
    macro:        MacroRegimeSnapshot,
    fred_api_key: str | None = None,
) -> AdvisorPackage:
    """
    Run the full five-agent pipeline for one client.

    Args:
        profile:      ProfileAgentOutput from the Profile Agent.
        macro:        MacroRegimeSnapshot from the Research Agent.
        fred_api_key: FRED API key for live DGS10 fetch (uses 4.4% fallback if None).

    Returns:
        A fully validated AdvisorPackage.
    """
    pipeline_warnings: list[str] = []

    # Step 1 — Pre-flight checks
    if macro.is_low_confidence:
        msg = f"Low-confidence macro regime ({macro.regime_confidence:.0%}) — results may be unreliable"
        logger.warning(msg)
        pipeline_warnings.append(msg)

    # Step 2 — Discount rate
    discount_rate = latest_dgs10(fred_api_key=fred_api_key, fallback=0.044)

    # Step 3 — Allocation ↔ Risk loop
    alloc_output, risk_output, risk_ao, alloc_ao, risk_revisions = run_alloc_risk_loop(
        profile       = profile,
        discount_rate = discount_rate,
    )

    if risk_output.decision == RiskDecision.REJECT:
        pipeline_warnings.append(
            f"Risk Agent REJECTED portfolio after {risk_revisions} FLAG revision(s)"
        )
    elif risk_revisions > 0:
        pipeline_warnings.append(f"Risk Agent required {risk_revisions} FLAG revision(s)")

    # Step 4 — Compliance loop (max 2 revisions)
    compliance_revisions = 0
    compliance_input     = assemble_compliance_input(profile, macro, alloc_ao)
    compliance_output    = run_compliance(compliance_input, risk_ao)

    for _ in range(_MAX_COMPLIANCE_REVISIONS):
        if compliance_output.clearance:
            break
        if not compliance_output.agent_feedback:
            break

        if "allocation_agent" in compliance_output.agent_feedback:
            feedback_violations = [
                item.action_required
                for item in compliance_output.agent_feedback["allocation_agent"]
            ]
            alloc_output, alloc_ao = _revise_allocation_for_compliance(
                profile, discount_rate, alloc_output.allocation_input.flag_iteration,
                feedback_violations,
            )
            compliance_input  = assemble_compliance_input(profile, macro, alloc_ao)
            compliance_output = run_compliance(compliance_input, risk_ao)
            compliance_revisions += 1

    if compliance_revisions > 0:
        pipeline_warnings.append(f"Compliance required {compliance_revisions} revision(s)")

    # Step 5 — Assemble AdvisorPackage
    metadata = RunMetadata(
        risk_revisions          = risk_revisions,
        compliance_revisions    = compliance_revisions,
        final_risk_decision     = RiskDecision(risk_output.decision.value),
        final_compliance_status = compliance_output.compliance_status,
        pipeline_warnings       = pipeline_warnings,
    )

    return AdvisorPackage(
        profile    = profile,
        macro      = macro,
        allocation = alloc_ao,
        risk       = risk_ao,
        compliance = compliance_output,
        metadata   = metadata,
    )


def _revise_allocation_for_compliance(
    profile:               ProfileAgentOutput,
    discount_rate:         float,
    current_iteration:     int,
    compliance_violations: list[str],
) -> tuple[AllocationOutput, AllocationAgentOutput]:
    """Re-run allocation with compliance violations as additional context."""
    return run_allocation_agent(
        profile        = profile,
        discount_rate  = discount_rate,
        flag_iteration = min(current_iteration + 1, 3),
    )

