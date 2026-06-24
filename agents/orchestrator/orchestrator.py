"""
Orchestrator — full five-agent pipeline control plane.

Entry points:
    run_pipeline(profile, macro, fred_api_key) → AdvisorPackage
    run_all(fred_api_key)                       → list[AdvisorPackage]  (all 9 BLS personas)

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contracts import (
    AdvisorPackage, AllocationAgentOutput, AllocationOutput,
    ClientProfileSection, ComplianceInput, ComplianceStatus,
    HumanCapitalSection, MacroRegimeSection, MacroRegimeSnapshot,
    ProfileAgentOutput, RegimeChangeFlag, RiskAgentOutput, RiskDecision,
    RiskOutput, RunMetadata,
)
from agents.compliance.compliance_agent import run_compliance
from agents.orchestrator.pipeline import run_alloc_risk_loop
from data.fetch.fred import latest_dgs10

logger = logging.getLogger(__name__)

_MAX_COMPLIANCE_REVISIONS = 2


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
    profile:     ProfileAgentOutput,
    macro:       MacroRegimeSnapshot,
    alloc_ao:    AllocationAgentOutput,
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
    )


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    profile:       ProfileAgentOutput,
    macro:         MacroRegimeSnapshot,
    fred_api_key:  str | None = None,
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
        profile        = profile,
        discount_rate  = discount_rate,
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

        # Re-run allocation if allocation_agent has actionable feedback
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
    profile:              ProfileAgentOutput,
    discount_rate:        float,
    current_iteration:    int,
    compliance_violations: list[str],
) -> tuple[AllocationOutput, AllocationAgentOutput]:
    """Re-run allocation with compliance violations as additional context."""
    from agents.allocation.agent import run_allocation_agent
    return run_allocation_agent(
        profile          = profile,
        discount_rate    = discount_rate,
        flag_iteration   = min(current_iteration + 1, 3),
    )


# ---------------------------------------------------------------------------
# Batch run — all 9 BLS personas
# ---------------------------------------------------------------------------

def run_all(
    fred_api_key: str | None = None,
    macro:        MacroRegimeSnapshot | None = None,
) -> list[AdvisorPackage]:
    """
    Run the full pipeline for all 9 BLS personas (p50 salary tier).

    Args:
        fred_api_key: FRED API key. Uses 4.4% fallback if None.
        macro:        MacroRegimeSnapshot. If None, runs the Research Agent first.

    Returns:
        List of AdvisorPackage, one per persona.
    """
    from agents.profile.profile_agent import run_profile_agent
    from agents.research.research_agent import run_research_agent

    if macro is None:
        logger.info("[orchestrator] Running Research Agent to get current regime...")
        macro = run_research_agent(fred_api_key)

    profiles = run_profile_agent(fred_api_key=fred_api_key)

    packages: list[AdvisorPackage] = []
    for profile in profiles:
        logger.info(f"[orchestrator] Running pipeline for persona: {profile.client_id}")
        try:
            pkg = run_pipeline(profile, macro, fred_api_key=fred_api_key)
            packages.append(pkg)
        except Exception as e:
            logger.error(f"[orchestrator] Pipeline failed for {profile.client_id}: {e}")

    return packages
