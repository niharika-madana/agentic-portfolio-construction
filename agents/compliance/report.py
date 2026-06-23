"""
Compliance report compiler.

Takes the raw violation and passed-check lists from all checks and
assembles the final ComplianceAgentOutput with:
  - compliance_status  (PASS | PASS_WITH_WARNINGS | FAIL)
  - overall_severity   (HIGH | MEDIUM | LOW | NONE)
  - clearance          (False if any HIGH severity violation exists)
  - agent_feedback     (violations grouped by responsible agent)
  - recommendation     (plain-English next step)

Rule:
  - Any HIGH severity violation  → FAIL,               clearance=False
  - MEDIUM or LOW only           → PASS_WITH_WARNINGS,  clearance=True
  - No violations                → PASS,                clearance=True
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contracts import (
    AgentFeedbackItem,
    ComplianceAgentOutput,
    ComplianceStatus,
    ComplianceViolation,
    Severity,
)


def compile_report(
    violations:    list[ComplianceViolation],
    passed_checks: list[str],
) -> ComplianceAgentOutput:
    """
    Compile the final ComplianceAgentOutput from the collected violations
    and passed check names.

    Parameters
    ----------
    violations : list[ComplianceViolation]
        All violations raised across checks 1.1–2.5.
    passed_checks : list[str]
        Names of checks that produced no violations.

    Returns
    -------
    ComplianceAgentOutput
    """
    # ── Determine overall severity ────────────────────────────────────
    severities = {v.severity for v in violations}

    if Severity.HIGH in severities:
        overall_severity   = Severity.HIGH
        compliance_status  = ComplianceStatus.FAIL
        clearance          = False
    elif Severity.MEDIUM in severities:
        overall_severity   = Severity.MEDIUM
        compliance_status  = ComplianceStatus.PASS_WITH_WARNINGS
        clearance          = True
    elif Severity.LOW in severities:
        overall_severity   = Severity.LOW
        compliance_status  = ComplianceStatus.PASS_WITH_WARNINGS
        clearance          = True
    else:
        overall_severity   = Severity.NONE
        compliance_status  = ComplianceStatus.PASS
        clearance          = True

    # ── Group feedback by responsible agent ───────────────────────────
    agent_feedback: dict[str, list[AgentFeedbackItem]] = {}
    for v in violations:
        agent = v.responsible_agent
        if agent not in agent_feedback:
            agent_feedback[agent] = []
        agent_feedback[agent].append(AgentFeedbackItem(
            check           = v.check,
            action_required = v.action_required,
        ))

    # ── Build recommendation ──────────────────────────────────────────
    if compliance_status == ComplianceStatus.PASS:
        recommendation = "Portfolio cleared. All compliance checks passed."

    elif compliance_status == ComplianceStatus.PASS_WITH_WARNINGS:
        n = len(violations)
        agents = sorted(agent_feedback.keys())
        recommendation = (
            f"Portfolio passes with {n} warning(s). "
            f"Review before client presentation. "
            f"Warnings raised against: {', '.join(agents)}."
        )

    else:  # FAIL
        high_violations = [v for v in violations if v.severity == Severity.HIGH]
        agents_to_fix   = sorted(agent_feedback.keys())
        recommendation  = (
            f"Portfolio FAILED compliance ({len(high_violations)} HIGH severity violation(s)). "
            f"Return to {', '.join(agents_to_fix)} for revision before proceeding. "
            f"Key issues: "
            + "; ".join(v.description[:80] for v in high_violations[:3])
            + ("..." if len(high_violations) > 3 else ".")
        )

    return ComplianceAgentOutput(
        compliance_status = compliance_status,
        overall_severity  = overall_severity,
        violations        = violations,
        passed_checks     = list(dict.fromkeys(passed_checks)),  # deduplicate, preserve order
        clearance         = clearance,
        agent_feedback    = agent_feedback,
        recommendation    = recommendation,
    )
