"""
orchestrator_agent.py — full five-agent pipeline control plane.

Entry point:
    run_pipeline(profile, macro, fred_api_key) → AdvisorPackage

Pipeline sequence:
  1. Pre-flight checks (low-confidence macro warning)
  2. Discount rate, then the regime tilt on the client's equity target
  3. Allocation ↔ Risk loop (max 3 FLAG revisions)
  4. Assemble ComplianceInput
  5. Compliance ↔ Allocation/Risk loop (max 2 revisions)
  6. Assemble AdvisorPackage

Step 2's tilt is the Research Agent's only route into the portfolio — see
apply_regime_tilt(). Everywhere else macro reaches (the pre-flight warning, the
compliance narrative, the final package) it is reported, not acted on.

All quantitative decisions are deterministic. LLMs participate only in
rationale, reasoning trace, and executive summary (3 touchpoints total).
"""

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contracts import (
    AdvisorPackage, AllocationAgentOutput, AllocationOutput,
    ClientProfileSection, ComplianceInput, ComplianceStatus,
    HumanCapitalSection, MacroRegimeSection, MacroRegimeSnapshot,
    ProfileAgentOutput, RebalanceDecision, RebalanceEvaluation,
    RegimeChangeFlag, RiskAgentOutput, RiskDecision,
    RiskOutput, RunMetadata,
)
from agents.allocation.agent import run_allocation_agent
from agents.compliance.compliance_agent import run_compliance
from agents.research.rebalance import evaluate_rebalance
from agents.research.regime_returns import compute_regime_stats, regime_tilt
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


# ---------------------------------------------------------------------------
# Regime tilt — the Research Agent's only route into the portfolio
# ---------------------------------------------------------------------------

_REGIME_FEATURES_PARQUET = (
    Path(__file__).resolve().parent.parent.parent / "data" / "storage" / "fred_macro_regimes.parquet"
)
"""
The smoothed regime sequence, written by research_agent._save_outputs().

MacroRegimeSnapshot carries a single month; compute_regime_stats() needs the full
monthly label path to measure each regime's realised volatility. Loading it from
the cache mirrors how this module already sources ff_factors, and keeps
run_pipeline's signature unchanged.
"""


def _load_regime_stats() -> dict:
    """
    Per-regime equity tilts from the cached smoothed regime sequence.

    Returns an empty dict when the cache is absent or malformed. That is not a
    silent failure: evaluate_rebalance() treats an empty stats dict as
    INSUFFICIENT_HISTORY rather than a zero tilt, so an unmeasured regime is
    reported instead of being quietly acted on with a fabricated number.
    """
    if not _REGIME_FEATURES_PARQUET.exists():
        return {}
    try:
        df = pd.read_parquet(_REGIME_FEATURES_PARQUET)
    except Exception as e:  # pragma: no cover — corrupt cache
        logger.warning(f"[pipeline] Could not read regime cache: {e}")
        return {}
    if "regime_label_smoothed" not in df.columns:
        return {}
    return compute_regime_stats(df["regime_label_smoothed"])


def apply_regime_tilt(
    profile:           ProfileAgentOutput,
    macro:             MacroRegimeSnapshot,
    pipeline_warnings: list[str],
) -> tuple[ProfileAgentOutput, RebalanceEvaluation | None]:
    """
    Route the macro regime into the portfolio via the client's equity target.

    Until this existed the Research Agent influenced nothing: run_allocation_agent
    takes no macro argument, and MacroRegimeSnapshot.for_allocation() and
    rebalance.evaluate_rebalance() both had zero callers. The regime reached the
    compliance narrative and the final package, and moved no weight.

    The tilt is applied here rather than inside the Allocation Agent because the
    Allocation Agent already reads portfolio_equity_target off the profile, and
    the orchestrator already owns input assembly — so macro reaches allocation
    through a field the optimizer respects, with no change to Aidan's signature
    and no quantitative decision made in this module. evaluate_rebalance() decides
    whether to move; regime_tilt() sizes the move; this function only carries the
    result.

    Only a JUSTIFIED decision applies the tilt. That is the point of
    RebalanceEvaluation: the smoothed 1995-2025 label path contains 18 runs with a
    median length of 5 months, so acting on regime_change_detected alone would
    trade the book on classifier noise. Every non-JUSTIFIED verdict is recorded in
    pipeline_warnings with its explanation, so a no-op is visible rather than
    silent.

    Returns (profile, evaluation). `profile` is a copy with the tilted target when
    the tilt applied, and the original object otherwise.
    """
    if macro.regime_change_evidence is None:
        msg = (
            "Regime tilt skipped — snapshot carries no regime_change_evidence "
            "(build_snapshot was called without PELT break_dates)"
        )
        logger.info(f"[pipeline] {msg}")
        pipeline_warnings.append(msg)
        return profile, None

    regime_stats = _load_regime_stats()
    evaluation   = evaluate_rebalance(macro, profile, regime_stats)

    if evaluation.decision != RebalanceDecision.JUSTIFIED:
        msg = f"Regime tilt not applied ({evaluation.decision.value}): {evaluation.explanation}"
        logger.info(f"[pipeline] {msg}")
        pipeline_warnings.append(msg)
        return profile, evaluation

    # The tilt is a RELATIVE adjustment (±TILT_CAP as a fraction of the target),
    # so it is applied multiplicatively in whatever units the field already holds.
    # RebalanceEvaluation.proposed_equity_target is deliberately NOT used here:
    # rebalance.to_financial_units() converts it to FINANCIAL-wealth units and
    # clips to [0, 1], while ProfileAgentOutput.portfolio_equity_target is in
    # TOTAL-WEALTH units (effective_risk_budget − implicit_equity_exposure).
    # Assigning one to the other would be a unit error that no validator catches.
    base = profile.portfolio_equity_target
    if base is None:
        base = profile.effective_risk_budget - profile.implicit_equity_exposure

    tilt   = regime_tilt(regime_stats, macro.regime_label)
    tilted = base * (1.0 + tilt)

    # model_copy does not re-run validators, which is safe here: none of
    # ProfileAgentOutput's four model_validators reference portfolio_equity_target.
    profile = profile.model_copy(update={"portfolio_equity_target": tilted})

    msg = (
        f"Regime tilt applied ({evaluation.prior_regime} → {evaluation.current_regime}): "
        f"equity target {base:+.4f} → {tilted:+.4f} ({tilt:+.1%} tilt)"
    )
    logger.info(f"[pipeline] {msg}")
    pipeline_warnings.append(msg)
    return profile, evaluation


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

    # Step 2b — Regime tilt. Must precede the allocation loop: it adjusts the
    # equity target the optimizer caps its risky weight at, so applying it after
    # would leave the allocation built on the untilted target.
    profile, _rebalance_evaluation = apply_regime_tilt(profile, macro, pipeline_warnings)

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

