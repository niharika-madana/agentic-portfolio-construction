"""
Orchestrator — wires all five agents into a single pipeline.

Pipeline flow:
  1. Receive ProfileAgentOutput and MacroRegimeSnapshot from upstream agents
  2. Run AllocationAgent
  3. Run RiskAgent
     → If FLAG: re-run AllocationAgent with risk flags (up to MAX_RISK_REVISIONS)
  4. Assemble ComplianceInput
  5. Run ComplianceAgent
     → If FAIL with allocation_agent issues: re-run AllocationAgent + RiskAgent
     → If FAIL with risk_agent issues only:  re-run RiskAgent only
     (up to MAX_COMPLIANCE_REVISIONS)
  6. Return AdvisorPackage

Stub functions (run_allocation_agent, run_risk_agent) define the expected
signatures. Replace the NotImplementedError bodies once those agents are built.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from contracts import (
    AdvisorPackage,
    AgentFeedbackItem,
    AllocationAgentOutput,
    ClientProfileSection,
    ComplianceAgentOutput,
    ComplianceInput,
    ComplianceStatus,
    HumanCapitalSection,
    MacroRegimeSection,
    MacroRegimeSnapshot,
    ProfileAgentOutput,
    RegimeChangeFlag,
    ResponsibleAgent,
    RiskAgentOutput,
    RiskDecision,
    RunMetadata,
)

logger = logging.getLogger(__name__)

MAX_RISK_REVISIONS       = 3
MAX_COMPLIANCE_REVISIONS = 2

# ---------------------------------------------------------------------------
# Agent stubs — replace bodies once allocation and risk agents are built
# ---------------------------------------------------------------------------

def run_allocation_agent(
    profile:              ProfileAgentOutput,
    macro:                MacroRegimeSnapshot,
    revision:             int       = 0,
    prior_risk_flags:     list[str] = None,
    prior_compliance_violations: list[str] = None,
) -> AllocationAgentOutput:
    """
    Call the Allocation Agent.

    The agent uses profile.implicit_equity_exposure to compute the
    portfolio equity target:

        portfolio_risky_share = profile.effective_risk_budget
                                - profile.implicit_equity_exposure

    This single subtraction is what makes each persona's allocation
    meaningfully different — see contracts.py AllocationAgentOutput docstring.

    TODO: replace body with:
        from agents.allocation.allocation_agent import build_allocation
        return build_allocation(profile, macro, revision,
                                prior_risk_flags, prior_compliance_violations)
    """
    raise NotImplementedError(
        "Allocation Agent not yet implemented. "
        "Import and call build_allocation() from agents/allocation/allocation_agent.py"
    )


def run_risk_agent(
    allocation: AllocationAgentOutput,
    profile:    ProfileAgentOutput,
    macro:      MacroRegimeSnapshot,
) -> RiskAgentOutput:
    """
    Call the Risk Agent.

    The agent stress-tests the allocation across all five historical
    regimes and computes HC-correlation-adjusted sector limits using
    profile.income_equity_correlation.

    TODO: replace body with:
        from agents.risk.risk_agent import assess_risk
        return assess_risk(allocation, profile, macro)
    """
    raise NotImplementedError(
        "Risk Agent not yet implemented. "
        "Import and call assess_risk() from agents/risk/risk_agent.py"
    )


# ---------------------------------------------------------------------------
# Assembler — builds ComplianceInput from upstream agent outputs
# ---------------------------------------------------------------------------

def _discretise_regime_volatility(vol: float) -> str:
    """Convert Research Agent's float regime_volatility to the 'low'|'medium'|'high' label."""
    if vol < 0.01:
        return "low"
    elif vol < 0.03:
        return "medium"
    return "high"


def _derive_regime_risk_flags(macro: MacroRegimeSnapshot) -> list[str]:
    """Derive plain-English risk flags from the current macro snapshot."""
    flags = []
    if macro.yield_curve < 0:
        flags.append("inverted yield curve")
    if macro.credit_spread > 3.0:
        flags.append("elevated credit spreads")
    if macro.cpi > 4.0:
        flags.append("high inflation")
    if macro.unemployment > 6.0:
        flags.append("elevated unemployment")
    if macro.fed_funds > 4.5:
        flags.append("restrictive monetary policy")
    if macro.is_low_confidence:
        flags.append("low regime classification confidence")
    return flags


def assemble_compliance_input(
    profile:    ProfileAgentOutput,
    macro:      MacroRegimeSnapshot,
    allocation: AllocationAgentOutput,
) -> ComplianceInput:
    """
    Assemble the ComplianceInput from upstream agent outputs.

    Field mapping notes:
      ClientProfileSection.persona          ← profile.client_id
      ClientProfileSection.risk_tolerance   ← profile.risk_tolerance_level (lowercase)
      ClientProfileSection.time_horizon_years ← profile.investment_horizon_years
      HumanCapitalSection.industry          ← profile.industry_exposure_sector
      HumanCapitalSection.income_equity_beta ← profile.income_equity_beta (primary HC risk measure)
      HumanCapitalSection.income_equity_correlation ← profile.income_equity_correlation (for sector audit)
      HumanCapitalSection.implicit_equity_exposure  ← profile.implicit_equity_exposure
      HumanCapitalSection.employer_sector   ← profile.industry_exposure_sector
      HumanCapitalSection.estimated_hc_pv   ← profile.human_capital_valuation
      MacroRegimeSection.regime_volatility  ← discretised from macro.regime_volatility (float→string)
    """
    client_profile = ClientProfileSection(
        persona              = profile.client_id,
        age                  = profile.age,
        risk_tolerance       = profile.risk_tolerance_level.value,
        time_horizon_years   = profile.investment_horizon_years,
        liquidity_needs      = profile.liquidity_needs.value,
        investment_objective = profile.investment_objective.value,
    )

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
    )

    macro_regime = MacroRegimeSection(
        current_regime     = macro.regime_label,
        historical_analogue = None,  # populated by Research Agent if available
        regime_risk_flags  = _derive_regime_risk_flags(macro),
        regime_confidence  = macro.regime_confidence,
        regime_volatility  = _discretise_regime_volatility(macro.regime_volatility),
    )

    regime_change_flag = RegimeChangeFlag(
        detected       = macro.regime_change_detected,
        prior_regime   = macro.prior_regime if macro.regime_change_detected else None,
        current_regime = macro.regime_label if macro.regime_change_detected else None,
        shift_date     = macro.regime_shift_date.isoformat() if macro.regime_change_detected else None,
        rebalance_proposed = None,  # set by Allocation Agent if it proposes a rebalance
    )

    return ComplianceInput(
        client_profile       = client_profile,
        human_capital        = human_capital,
        macro_regime         = macro_regime,
        proposed_portfolio   = allocation.proposed_portfolio,
        allocation_rationale = allocation.allocation_rationale,
        regime_change_flag   = regime_change_flag,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    profile: ProfileAgentOutput,
    macro:   MacroRegimeSnapshot,
    *,
    _run_allocation_fn=None,
    _run_risk_fn=None,
    _run_compliance_fn=None,
) -> AdvisorPackage:
    """
    Run the full advisory pipeline for one client persona.

    Parameters
    ----------
    profile : ProfileAgentOutput
        Output from the Profile Agent (Agent 1).
    macro : MacroRegimeSnapshot
        Output from the Research Agent (Agent 2), most recent snapshot.
    _run_allocation_fn, _run_risk_fn, _run_compliance_fn : callable, optional
        Injectable replacements for testing — pass mock callables to avoid
        hitting the real agent implementations.

    Returns
    -------
    AdvisorPackage
        Complete pipeline output including all agent outputs and metadata.
    """
    pipeline_warnings: list[str] = []

    # ── Resolve agent callables (real or injected mock) ───────────────
    _allocation = _run_allocation_fn or run_allocation_agent
    _risk       = _run_risk_fn       or run_risk_agent

    if _run_compliance_fn is None:
        from agents.compliance.compliance_agent import run_compliance as _compliance_fn
    else:
        _compliance_fn = _run_compliance_fn

    # ── Warn if macro confidence is low ───────────────────────────────
    if macro.is_low_confidence:
        msg = (
            f"Regime '{macro.regime_label}' has low classification confidence "
            f"({macro.regime_confidence:.2f} < 0.60). Allocations are based on an "
            f"uncertain regime signal."
        )
        logger.warning(msg)
        pipeline_warnings.append(msg)

    # ── Warn if portfolio_volatility_annual is missing ────────────────
    # (checked after risk runs, but noted here for clarity)

    # ------------------------------------------------------------------
    # Step 2 — Initial allocation
    # ------------------------------------------------------------------
    logger.info("PIPELINE [%s] — initial allocation (revision 0)", profile.client_id)
    allocation = _allocation(profile, macro, revision=0)

    # ------------------------------------------------------------------
    # Step 3 — Risk → Allocation feedback loop
    # ------------------------------------------------------------------
    risk = _risk(allocation, profile, macro)
    risk_revision = 0

    while risk.risk_decision == RiskDecision.FLAG and risk_revision < MAX_RISK_REVISIONS:
        risk_revision += 1
        logger.info(
            "PIPELINE [%s] — risk FLAG on revision %d, re-running allocation "
            "(violations: %s)",
            profile.client_id, risk_revision, risk.violations,
        )
        allocation = _allocation(
            profile, macro,
            revision          = risk_revision,
            prior_risk_flags  = risk.violations,
        )
        risk = _risk(allocation, profile, macro)

    if risk.risk_decision == RiskDecision.FLAG:
        msg = (
            f"Portfolio still FLAG after {MAX_RISK_REVISIONS} risk revisions. "
            f"Proceeding to compliance with outstanding violations."
        )
        logger.warning("PIPELINE [%s] — %s", profile.client_id, msg)
        pipeline_warnings.append(msg)

    if risk.portfolio_volatility_annual is None:
        msg = "Risk Agent did not supply portfolio_volatility_annual; SUIT-VOL check (2.5) will be skipped."
        logger.warning("PIPELINE [%s] — %s", profile.client_id, msg)
        pipeline_warnings.append(msg)

    # ------------------------------------------------------------------
    # Step 4 — Assemble ComplianceInput
    # ------------------------------------------------------------------
    compliance_input = assemble_compliance_input(profile, macro, allocation)

    # ------------------------------------------------------------------
    # Step 5 — Compliance → Allocation/Risk feedback loop
    # ------------------------------------------------------------------
    compliance = _compliance_fn(compliance_input, risk)
    comp_revision = 0

    while (
        compliance.compliance_status == ComplianceStatus.FAIL
        and comp_revision < MAX_COMPLIANCE_REVISIONS
    ):
        comp_revision += 1
        feedback = compliance.agent_feedback

        needs_allocation_rerun = ResponsibleAgent.ALLOCATION.value in feedback
        needs_risk_rerun       = ResponsibleAgent.RISK.value in feedback

        if needs_allocation_rerun:
            comp_violations = _extract_compliance_violation_descriptions(
                feedback[ResponsibleAgent.ALLOCATION.value]
            )
            logger.info(
                "PIPELINE [%s] — compliance FAIL (revision %d), re-running allocation "
                "with compliance violations: %s",
                profile.client_id, comp_revision, comp_violations,
            )
            allocation = _allocation(
                profile, macro,
                revision                     = risk_revision + comp_revision,
                prior_risk_flags             = risk.violations,
                prior_compliance_violations  = comp_violations,
            )
            # Re-run risk on the revised allocation before re-checking compliance
            risk = _risk(allocation, profile, macro)

        elif needs_risk_rerun:
            logger.info(
                "PIPELINE [%s] — compliance FAIL (revision %d), re-running risk agent only "
                "(allocation unchanged)",
                profile.client_id, comp_revision,
            )
            risk = _risk(allocation, profile, macro)

        else:
            # Compliance failed on profile or research agent issues — cannot fix by re-running
            msg = (
                f"Compliance FAIL on revision {comp_revision} cannot be resolved by "
                f"re-running allocation or risk (responsible: "
                f"{list(feedback.keys())}). Breaking loop."
            )
            logger.warning("PIPELINE [%s] — %s", profile.client_id, msg)
            pipeline_warnings.append(msg)
            break

        compliance_input = assemble_compliance_input(profile, macro, allocation)
        compliance = _compliance_fn(compliance_input, risk)

    if compliance.compliance_status == ComplianceStatus.FAIL and comp_revision >= MAX_COMPLIANCE_REVISIONS:
        msg = (
            f"Portfolio still FAIL after {MAX_COMPLIANCE_REVISIONS} compliance revisions. "
            f"Returning package with FAIL status for manual review."
        )
        logger.warning("PIPELINE [%s] — %s", profile.client_id, msg)
        pipeline_warnings.append(msg)

    # ------------------------------------------------------------------
    # Step 6 — Assemble and return
    # ------------------------------------------------------------------
    metadata = RunMetadata(
        risk_revisions          = risk_revision,
        compliance_revisions    = comp_revision,
        final_risk_decision     = risk.risk_decision,
        final_compliance_status = compliance.compliance_status,
        pipeline_warnings       = pipeline_warnings,
    )

    logger.info(
        "PIPELINE [%s] — complete. Risk: %s (revisions: %d) | Compliance: %s (revisions: %d)",
        profile.client_id,
        risk.risk_decision.value, risk_revision,
        compliance.compliance_status.value, comp_revision,
    )

    return AdvisorPackage(
        profile    = profile,
        macro      = macro,
        allocation = allocation,
        risk       = risk,
        compliance = compliance,
        metadata   = metadata,
    )


# ---------------------------------------------------------------------------
# Report renderer
# ---------------------------------------------------------------------------

def render_report(package: AdvisorPackage) -> str:
    """Render a human-readable markdown report from an AdvisorPackage."""
    lines: list[str] = []
    p   = package.profile
    m   = package.macro
    a   = package.allocation
    r   = package.risk
    c   = package.compliance
    md  = package.metadata

    lines.append(f"# Advisory Report — {p.client_id}")
    lines.append("")

    # ── Profile ───────────────────────────────────────────────────────
    lines.append("## Client Profile")
    lines.append(f"- **Age**: {p.age} | **Career**: {p.career_type}")
    lines.append(f"- **HC type**: {p.human_capital_type.value} | **Income stability**: {p.income_stability.value}")
    lines.append(f"- **Financial capital**: ${p.financial_capital:,.0f}")
    lines.append(f"- **Human capital (PV)**: ${p.human_capital_valuation:,.0f} ({p.human_capital_pct_of_total:.1f}% of total wealth)")
    lines.append(f"- **Total wealth**: ${p.total_wealth:,.0f}")
    lines.append("")
    lines.append("### Income Risk Decomposition")
    lines.append(f"- **σ (income volatility)**: {p.income_volatility_sigma:.2f}")
    lines.append(f"- **ρ (income-equity correlation)**: {p.income_equity_correlation:.2f}")
    lines.append(f"- **β (income equity beta)**: {p.income_equity_beta:.2f}")
    lines.append(f"- **Implicit equity exposure** (HC share × β): {p.implicit_equity_exposure:.3f}")
    lines.append(f"- **Effective risk budget**: {p.effective_risk_budget:.3f}")
    lines.append(f"- **Portfolio equity target** (budget − implicit): {p.effective_risk_budget - p.implicit_equity_exposure:.3f}")
    lines.append(f"- **RSU concentration**: {p.RSU_concentration:.0%}")
    lines.append("")

    # ── Macro ─────────────────────────────────────────────────────────
    lines.append("## Macro Regime")
    lines.append(f"- **Current regime**: {m.regime_label} (confidence: {m.regime_confidence:.1%})")
    lines.append(f"- **Regime since**: {m.regime_shift_date} | **As of**: {m.as_of}")
    if m.regime_change_detected:
        lines.append(f"- **Regime change detected**: {m.prior_regime} → {m.regime_label}")
    lines.append(f"- **Yield curve (10Y-2Y)**: {m.yield_curve:+.2f}pp | **Credit spread**: {m.credit_spread:.2f}pp")
    lines.append(f"- **Fed funds**: {m.fed_funds:.2f}% | **Unemployment**: {m.unemployment:.1f}% | **CPI**: {m.cpi:.1f}%")
    if m.is_low_confidence:
        lines.append("- **WARNING**: Low regime confidence — allocations carry classification uncertainty")
    lines.append("")

    # ── Allocation ────────────────────────────────────────────────────
    lines.append("## Portfolio Allocation")
    lines.append(f"- **Revision**: {a.revision}")
    lines.append("")
    lines.append("| Ticker | Weight | Rationale |")
    lines.append("|--------|--------|-----------|")
    for ticker, weight in sorted(a.proposed_portfolio.items(), key=lambda x: -x[1]):
        rationale = a.allocation_rationale.get(ticker, "—")
        lines.append(f"| {ticker} | {weight:.2%} | {rationale} |")
    lines.append("")

    # ── Risk ──────────────────────────────────────────────────────────
    lines.append("## Risk Assessment")
    lines.append(f"- **Decision**: **{r.risk_decision.value}**")
    if r.portfolio_volatility_annual is not None:
        lines.append(f"- **Annualised volatility**: {r.portfolio_volatility_annual:.2%}")
    lines.append("")
    lines.append("### Regime Stress Tests")
    lines.append("| Regime | Portfolio DD | Benchmark DD | Passed |")
    lines.append("|--------|-------------|--------------|--------|")
    for regime, ev in r.regime_evaluation.items():
        lines.append(
            f"| {regime} | {ev.portfolio_drawdown:.2%} | {ev.benchmark_drawdown:.2%} "
            f"| {'✓' if ev.passed else '✗'} |"
        )
    lines.append("")
    if r.violations:
        lines.append("**Violations:**")
        for v in r.violations:
            lines.append(f"- {v}")
        lines.append("")

    # ── Compliance ────────────────────────────────────────────────────
    lines.append("## Compliance")
    lines.append(f"- **Status**: **{c.compliance_status.value}** | **Cleared**: {'YES' if c.clearance else 'NO'}")
    lines.append(f"- **Severity**: {c.overall_severity.value}")
    lines.append(f"- **Recommendation**: {c.recommendation}")
    lines.append("")
    if c.violations:
        lines.append("### Violations")
        lines.append("| Check | Severity | Description | Responsible Agent | Action |")
        lines.append("|-------|----------|-------------|-------------------|--------|")
        for v in c.violations:
            lines.append(
                f"| {v.check} | {v.severity.value} | {v.description} "
                f"| {v.responsible_agent} | {v.action_required} |"
            )
        lines.append("")
    if c.passed_checks:
        lines.append(f"**Passed checks**: {', '.join(c.passed_checks)}")
        lines.append("")

    # ── Metadata ──────────────────────────────────────────────────────
    lines.append("## Pipeline Metadata")
    lines.append(f"- Risk revisions: {md.risk_revisions} | Compliance revisions: {md.compliance_revisions}")
    if md.pipeline_warnings:
        lines.append("### Warnings")
        for w in md.pipeline_warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all(
    personas: list[ProfileAgentOutput],
    macro:    MacroRegimeSnapshot,
    **pipeline_kwargs,
) -> list[AdvisorPackage]:
    """
    Run the pipeline for a list of personas against the same macro snapshot.
    Returns one AdvisorPackage per persona.

    Example
    -------
    packages = run_all(
        personas = [prof_profile, tech_profile, fin_profile],
        macro    = current_macro_snapshot,
    )
    for pkg in packages:
        print(render_report(pkg))
    """
    packages = []
    for profile in personas:
        logger.info("run_all — starting pipeline for %s", profile.client_id)
        pkg = run_pipeline(profile, macro, **pipeline_kwargs)
        packages.append(pkg)
    return packages


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _extract_compliance_violation_descriptions(
    feedback_items: list[AgentFeedbackItem],
) -> list[str]:
    """Convert AgentFeedbackItem list to plain strings for the allocation agent."""
    return [f"{item.check}: {item.action_required}" for item in feedback_items]
