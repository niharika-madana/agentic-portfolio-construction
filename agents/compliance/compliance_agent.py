"""
Compliance Agent — main entry point.

run_compliance(compliance_input, risk_output) → ComplianceAgentOutput

Execution order:
  1. Job 1 — Constraint set verification (constraint_checks.py)
     1.1  Completeness
     1.2  Consistency
     1.3  Derivation audit
  2. Job 2 — Content & fiduciary checks (content_checks.py)
     2.1  Rationale completeness
     2.2  Rationale specificity
     2.3  HC acknowledgment
     2.4  Weight integrity
     2.5  Volatility suitability (skipped if Risk Agent omits portfolio_volatility_annual)
     2.6  Rationale distinctness (rationales differentiated across positions)
  3. Job 3 — Robo-adviser checks (job3_robo_adviser_checks.py)
     3.1  Composition–objective consistency
     3.2  Client-mandate consistency (optional LLM reviewer; rule decides)
     3.3  Algorithm-limitation disclosure
     3.4  ETF-provider conflict screen
  4. Compile report (report.py)

Input validation is handled entirely by the Pydantic contracts in contracts.py.
By the time run_compliance() is called, both compliance_input and risk_output
are already validated Pydantic objects — no separate schema validation step needed.

The compliance agent's decisions are deterministic — same inputs always produce
the same output. Job 3 Check 3.2 may optionally consult an injected LLM reviewer
to surface mandate conflicts, but a deterministic rule assigns every severity, so
clearance never depends on a model call. With no reviewer injected (the default,
and the state under pytest) no LLM calls are made.
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contracts import (
    ComplianceAgentOutput,
    ComplianceInput,
    RiskAgentOutput,
)
from agents.compliance.constraint_checks import (
    check_completeness,
    check_consistency,
    check_derivation_audit,
)
from agents.compliance.content_checks import (
    check_hc_acknowledgment,
    check_rationale_completeness,
    check_rationale_distinctness,
    check_rationale_specificity,
    check_volatility_suitability,
    check_weight_integrity,
)
from agents.compliance.job3_robo_adviser_checks import (
    MandateReviewer,
    check_algorithm_limitation_disclosure,
    check_client_mandate,
    check_composition_objective,
    check_etf_provider_conflict,
)
from agents.compliance.report import compile_report

logger = logging.getLogger(__name__)


def run_compliance(
    compliance_input: ComplianceInput,
    risk_output:      RiskAgentOutput,
    mandate_reviewer: Optional[MandateReviewer] = None,
) -> ComplianceAgentOutput:
    """
    Run all compliance checks and return a ComplianceAgentOutput.

    Parameters
    ----------
    compliance_input : ComplianceInput
        Assembled by the orchestrator from ProfileAgentOutput +
        MacroRegimeSnapshot + AllocationAgentOutput.
    risk_output : RiskAgentOutput
        Output from the Risk Agent — audited by Job 1 checks.
    mandate_reviewer : Optional[MandateReviewer]
        Optional LLM hook for Job 3 Check 3.2 (client-mandate consistency). When
        provided, its findings augment the deterministic set-membership pass, but
        a deterministic rule still assigns every severity — the LLM never sets
        clearance (SCOPE §3.9). Left None by the orchestrator and under pytest,
        keeping the agent fully deterministic and key-free by default.

    Returns
    -------
    ComplianceAgentOutput
        compliance_status=FAIL              → orchestrator re-runs responsible agents
        compliance_status=PASS_WITH_WARNINGS → cleared but warnings noted in report
        compliance_status=PASS              → cleared, no action required
    """
    client_id = compliance_input.client_profile.persona
    logger.info("COMPLIANCE [%s] — starting checks", client_id)

    all_violations = []
    all_passed     = []

    # ------------------------------------------------------------------
    # Job 1 — Constraint set verification (audits Risk Agent output)
    # ------------------------------------------------------------------

    v, p = check_completeness(compliance_input, risk_output)
    all_violations.extend(v); all_passed.extend(p)

    v, p = check_consistency(compliance_input, risk_output)
    all_violations.extend(v); all_passed.extend(p)

    v, p = check_derivation_audit(compliance_input, risk_output)
    all_violations.extend(v); all_passed.extend(p)

    logger.info(
        "COMPLIANCE [%s] — Job 1 complete: %d violation(s), %d passed",
        client_id, len(all_violations), len(all_passed),
    )

    # ------------------------------------------------------------------
    # Job 2 — Content & fiduciary checks (Compliance-owned)
    # ------------------------------------------------------------------

    v, p = check_rationale_completeness(compliance_input)
    all_violations.extend(v); all_passed.extend(p)

    v, p = check_rationale_specificity(compliance_input)
    all_violations.extend(v); all_passed.extend(p)

    v, p = check_hc_acknowledgment(compliance_input)
    all_violations.extend(v); all_passed.extend(p)

    v, p = check_weight_integrity(compliance_input)
    all_violations.extend(v); all_passed.extend(p)

    v, p, skipped = check_volatility_suitability(compliance_input, risk_output)
    all_violations.extend(v); all_passed.extend(p)
    if skipped:
        logger.warning(
            "COMPLIANCE [%s] — Check 2.5 (SUIT-VOL) skipped: "
            "Risk Agent did not supply portfolio_volatility_annual",
            client_id,
        )

    v, p = check_rationale_distinctness(compliance_input)
    all_violations.extend(v); all_passed.extend(p)

    logger.info(
        "COMPLIANCE [%s] — Job 2 complete: %d total violation(s), %d passed checks",
        client_id, len(all_violations), len(all_passed),
    )

    # ------------------------------------------------------------------
    # Job 3 — Robo-adviser checks (SEC IM Guidance Update No. 2017-02)
    # ------------------------------------------------------------------

    v, p = check_composition_objective(compliance_input)
    all_violations.extend(v); all_passed.extend(p)

    v, p = check_client_mandate(compliance_input, mandate_reviewer)
    all_violations.extend(v); all_passed.extend(p)

    v, p = check_algorithm_limitation_disclosure(compliance_input)
    all_violations.extend(v); all_passed.extend(p)

    v, p = check_etf_provider_conflict(compliance_input)
    all_violations.extend(v); all_passed.extend(p)

    logger.info(
        "COMPLIANCE [%s] — Job 3 complete: %d total violation(s), %d passed checks",
        client_id, len(all_violations), len(all_passed),
    )

    # ------------------------------------------------------------------
    # Compile and return report
    # ------------------------------------------------------------------

    report = compile_report(all_violations, all_passed)

    logger.info(
        "COMPLIANCE [%s] — status: %s | clearance: %s | severity: %s",
        client_id,
        report.compliance_status.value,
        report.clearance,
        report.overall_severity.value,
    )

    return report
