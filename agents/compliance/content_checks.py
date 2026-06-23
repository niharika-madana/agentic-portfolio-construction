"""
Job 2 — Content & Fiduciary Checks (Compliance-owned).

Five checks:
  2.1  Rationale completeness — every ticker has a rationale entry
  2.2  Rationale specificity  — rationale cites client-specific context (not boilerplate)
  2.3  HC acknowledgment      — RSU/beta exposure referenced when material
  2.4  Weight integrity       — weights sum to 1.0, no negatives, min 2 positions
  2.5  Volatility suitability — portfolio vol within band for stated risk tolerance
       (skipped with warning if Risk Agent did not supply portfolio_volatility_annual)

These are Compliance's own independent checks — they do not re-audit the
Risk Agent (that is Job 1). They verify that the recommendation is complete,
client-specific, and fiduciary-compliant.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contracts import (
    ComplianceInput,
    ComplianceViolation,
    ResponsibleAgent,
    RiskAgentOutput,
    Severity,
)

# ---------------------------------------------------------------------------
# Client-context keyword vocabulary for Check 2.2
# ---------------------------------------------------------------------------
# A rationale must contain at least 2 of these terms to be considered
# client-specific. They cover HC type, sector, risk profile, and regime.
# Generic phrases like "diversification" or "stability" do not count.

_CLIENT_CONTEXT_TERMS: set[str] = {
    # Human capital labels
    "human capital", "hc", "bond-like", "equity-like", "mixed",
    # Income nature
    "beta", "income stability", "income volatility", "sigma",
    "implicit equity", "implicit exposure",
    # RSU / employer concentration
    "rsu", "employer", "concentration", "stock compensation",
    # Sector
    "technology", "tech", "education", "finance", "financial",
    "healthcare", "energy", "sector",
    # Risk profile
    "risk tolerance", "conservative", "moderate", "aggressive",
    "risk budget", "effective risk budget",
    # Time horizon & liquidity
    "horizon", "liquidity", "retirement",
    # Regime context
    "regime", "rate hike", "dot-com", "gfc", "covid", "ai boom",
    "yield curve", "credit spread", "inflation",
    # Portfolio role
    "hedge", "offset", "diversif",  # prefix match covers diversify / diversification
}

# Suitability volatility bands (annualised) per risk tolerance level
_SUITABILITY_VOL_BANDS: dict[str, tuple[float, float]] = {
    "conservative": (0.00, 0.12),
    "moderate":     (0.00, 0.20),
    "aggressive":   (0.00, 0.30),
}


# ---------------------------------------------------------------------------
# Check 2.1 — Rationale Completeness
# ---------------------------------------------------------------------------

def check_rationale_completeness(
    compliance_input: ComplianceInput,
) -> tuple[list[ComplianceViolation], list[str]]:
    """
    Every ticker in proposed_portfolio must have a non-empty rationale entry.

    Severity HIGH — missing rationale means the recommendation cannot be
    justified to the client, violating Reg BI's care obligation.
    """
    violations: list[ComplianceViolation] = []
    passed: list[str] = []
    missing = 0

    portfolio = compliance_input.proposed_portfolio
    rationale = compliance_input.allocation_rationale

    for ticker in portfolio:
        if ticker not in rationale:
            violations.append(ComplianceViolation(
                check             = "check_2_1_rationale_completeness",
                severity          = Severity.HIGH,
                description       = f"Missing rationale for portfolio position: {ticker}",
                rule_reference    = "SEC Reg BI — Care Obligation (17 CFR 240.15l-1(a)(2)(ii))",
                responsible_agent = ResponsibleAgent.ALLOCATION.value,
                action_required   = f"Add rationale for {ticker} citing at least one computed number",
            ))
            missing += 1
        elif not rationale[ticker].strip():
            violations.append(ComplianceViolation(
                check             = "check_2_1_rationale_completeness",
                severity          = Severity.HIGH,
                description       = f"Rationale for {ticker} is empty or whitespace-only",
                rule_reference    = "SEC Reg BI — Care Obligation",
                responsible_agent = ResponsibleAgent.ALLOCATION.value,
                action_required   = f"Populate rationale for {ticker}",
            ))
            missing += 1

    if missing == 0:
        passed.append("check_2_1_rationale_completeness")

    return violations, passed


# ---------------------------------------------------------------------------
# Check 2.2 — Rationale Specificity
# ---------------------------------------------------------------------------

def check_rationale_specificity(
    compliance_input: ComplianceInput,
) -> tuple[list[ComplianceViolation], list[str]]:
    """
    Each rationale must reference at least 2 client-specific terms from
    _CLIENT_CONTEXT_TERMS. Generic boilerplate without client context
    (e.g. "provides diversification and stability") does not constitute
    a fiduciary recommendation under Reg BI.

    Severity MEDIUM — the position may be suitable but the justification
    is not documented to a fiduciary standard.
    """
    violations: list[ComplianceViolation] = []
    passed: list[str] = []
    flagged = 0

    for ticker, text in compliance_input.allocation_rationale.items():
        text_lower = text.lower()
        matched_terms = [t for t in _CLIENT_CONTEXT_TERMS if t in text_lower]
        if len(matched_terms) < 2:
            violations.append(ComplianceViolation(
                check             = "check_2_2_rationale_specificity",
                severity          = Severity.MEDIUM,
                description       = (
                    f"{ticker} rationale references {len(matched_terms)} client-specific "
                    f"term(s) (minimum 2 required). "
                    f"Found: {matched_terms if matched_terms else 'none'}. "
                    f"Excerpt: '{text[:120]}...'"
                ),
                rule_reference    = (
                    "SEC Reg BI — Care Obligation; FINRA Rule 2111 Suitability — "
                    "recommendations must be based on client's investment profile"
                ),
                responsible_agent = ResponsibleAgent.ALLOCATION.value,
                action_required   = (
                    f"Rewrite {ticker} rationale to reference ≥2 of: HC type, income beta, "
                    f"sector, risk tolerance, RSU concentration, macro regime"
                ),
            ))
            flagged += 1

    if flagged == 0:
        passed.append("check_2_2_rationale_specificity")

    return violations, passed


# ---------------------------------------------------------------------------
# Check 2.3 — HC Acknowledgment
# ---------------------------------------------------------------------------

def check_hc_acknowledgment(
    compliance_input: ComplianceInput,
) -> tuple[list[ComplianceViolation], list[str]]:
    """
    When RSU concentration > 10% or income_equity_beta > 0.8 (equity-like HC),
    at least one position rationale must explicitly reference the client's
    income concentration or implicit equity exposure.

    Fiduciary concern: recommending a portfolio without acknowledging that the
    client is already heavily exposed to equity markets through their career
    violates the total-wealth framework and Reg BI's care obligation.

    Severity HIGH when RSU > 10% (material single-name concentration).
    Severity MEDIUM when equity-like HC type but no RSU (general market beta).
    """
    violations: list[ComplianceViolation] = []
    passed: list[str] = []
    hc = compliance_input.human_capital

    _RSU_TERMS  = {"rsu", "employer", "stock compensation", "concentration", "single-name"}
    _BETA_TERMS = {
        "human capital", "hc", "equity-like", "implicit equity",
        "implicit exposure", "income beta", "beta", "career risk",
    }

    full_rationale = " ".join(compliance_input.allocation_rationale.values()).lower()

    # ── 2.3a: RSU concentration > 10% ────────────────────────────────
    if hc.rsu_concentration > 0.10:
        if not any(term in full_rationale for term in _RSU_TERMS):
            violations.append(ComplianceViolation(
                check             = "check_2_3a_rsu_acknowledgment",
                severity          = Severity.HIGH,
                description       = (
                    f"RSU concentration {hc.rsu_concentration:.0%} > 10% (material single-name "
                    f"concentration) but no position rationale references RSU/employer exposure. "
                    f"The recommendation ignores a major component of the client's total wealth."
                ),
                rule_reference    = (
                    "Ibbotson et al. (2007) — total wealth framework; "
                    "SEC Reg BI — Care Obligation"
                ),
                responsible_agent = ResponsibleAgent.ALLOCATION.value,
                action_required   = (
                    "Add RSU/employer concentration context to at least one position rationale "
                    "(e.g. explain why a position hedges or avoids compounding RSU risk)"
                ),
            ))
        else:
            passed.append("check_2_3a_rsu_acknowledgment")
    else:
        passed.append("check_2_3a_rsu_acknowledgment")  # rule not triggered

    # ── 2.3b: Equity-like HC type (high income_equity_beta) ───────────
    if hc.income_equity_beta > 0.8:
        if not any(term in full_rationale for term in _BETA_TERMS):
            violations.append(ComplianceViolation(
                check             = "check_2_3b_implicit_equity_exposure_acknowledgment",
                severity          = Severity.MEDIUM,
                description       = (
                    f"Client has income_equity_beta={hc.income_equity_beta:.2f} (equity-like HC) "
                    f"with implicit_equity_exposure={hc.implicit_equity_exposure:.3f}, "
                    f"but no position rationale acknowledges this implicit market exposure. "
                    f"Portfolio construction should reference the HC offset."
                ),
                rule_reference    = (
                    "Ibbotson et al. (2007) — implicit_equity_exposure must be offset; "
                    "FINRA Rule 2111"
                ),
                responsible_agent = ResponsibleAgent.ALLOCATION.value,
                action_required   = (
                    "Reference human capital beta or implicit equity exposure in at least "
                    "one position rationale to show the portfolio accounts for career risk"
                ),
            ))
        else:
            passed.append("check_2_3b_implicit_equity_exposure_acknowledgment")
    else:
        passed.append("check_2_3b_implicit_equity_exposure_acknowledgment")  # rule not triggered

    return violations, passed


# ---------------------------------------------------------------------------
# Check 2.4 — Weight Integrity
# ---------------------------------------------------------------------------

def check_weight_integrity(
    compliance_input: ComplianceInput,
) -> tuple[list[ComplianceViolation], list[str]]:
    """
    Independent verification that portfolio weights are mathematically valid.
    These are also caught by the AllocationAgentOutput Pydantic validator,
    but Compliance re-checks independently as a pipeline audit gate.

    Severity HIGH — any weight integrity failure means the portfolio
    cannot be implemented as described.
    """
    violations: list[ComplianceViolation] = []
    passed: list[str] = []
    portfolio = compliance_input.proposed_portfolio
    issues = 0

    # ── 2.4a: Minimum 2 positions ─────────────────────────────────────
    if len(portfolio) < 2:
        violations.append(ComplianceViolation(
            check             = "check_2_4a_minimum_positions",
            severity          = Severity.HIGH,
            description       = f"Portfolio has {len(portfolio)} position(s); minimum 2 required",
            rule_reference    = "Internal — minimum diversification standard",
            responsible_agent = ResponsibleAgent.ALLOCATION.value,
            action_required   = "Add at least one more position to the portfolio",
        ))
        issues += 1

    # ── 2.4b: No negative weights ─────────────────────────────────────
    for ticker, w in portfolio.items():
        if w < 0:
            violations.append(ComplianceViolation(
                check             = "check_2_4b_no_negative_weights",
                severity          = Severity.HIGH,
                description       = f"{ticker} has negative weight {w:.4f}; short positions are not permitted",
                rule_reference    = "Internal — long-only portfolio constraint",
                responsible_agent = ResponsibleAgent.ALLOCATION.value,
                action_required   = f"Remove short position in {ticker}",
            ))
            issues += 1

    # ── 2.4c: Weights sum to 1.0 ─────────────────────────────────────
    valid_weights = [w for w in portfolio.values() if isinstance(w, (int, float))]
    if valid_weights:
        total = sum(valid_weights)
        if abs(total - 1.0) > 0.01:
            violations.append(ComplianceViolation(
                check             = "check_2_4c_weights_sum_to_one",
                severity          = Severity.HIGH,
                description       = (
                    f"Portfolio weights sum to {total:.4f}; must be 1.0 ±0.01. "
                    f"Portfolio cannot be fully implemented."
                ),
                rule_reference    = "Internal — portfolio weight integrity",
                responsible_agent = ResponsibleAgent.ALLOCATION.value,
                action_required   = f"Adjust weights to sum to 1.0 (current sum: {total:.4f})",
            ))
            issues += 1

    if issues == 0:
        passed.append("check_2_4_weight_integrity")

    return violations, passed


# ---------------------------------------------------------------------------
# Check 2.5 — Volatility Suitability
# ---------------------------------------------------------------------------

def check_volatility_suitability(
    compliance_input: ComplianceInput,
    risk_output: RiskAgentOutput,
) -> tuple[list[ComplianceViolation], list[str], bool]:
    """
    Verify that the portfolio's annualised volatility falls within the
    suitability band for the client's stated risk tolerance.

    This is Compliance's independent suitability check — it uses the
    portfolio_volatility_annual from the Risk Agent (computed from actual
    return history) as the authoritative number.

    Returns (violations, passed_checks, was_skipped).
    was_skipped=True means Risk Agent did not supply portfolio_volatility_annual;
    the orchestrator logs a pipeline warning in this case.

    Reference: FINRA Rule 2111 — suitability requires that recommendations
    are appropriate for the customer's financial situation and risk tolerance.
    """
    violations: list[ComplianceViolation] = []
    passed: list[str] = []

    ann_vol = risk_output.portfolio_volatility_annual
    if ann_vol is None:
        return violations, passed, True  # skipped

    risk_tolerance = compliance_input.client_profile.risk_tolerance.lower()
    low, high = _SUITABILITY_VOL_BANDS.get(risk_tolerance, (0.0, 0.30))

    if low <= ann_vol <= high:
        passed.append("check_2_5_volatility_suitability")
    else:
        direction = "above" if ann_vol > high else "below"
        violations.append(ComplianceViolation(
            check             = "check_2_5_volatility_suitability",
            severity          = Severity.HIGH if ann_vol > high else Severity.MEDIUM,
            description       = (
                f"Annualised portfolio volatility {ann_vol:.2%} is {direction} the suitability "
                f"band [{low:.0%}, {high:.0%}] for risk_tolerance='{risk_tolerance}'. "
                f"Portfolio risk is {'excessive' if ann_vol > high else 'unnecessarily low'} "
                f"for this client."
            ),
            rule_reference    = "FINRA Rule 2111 — Suitability; FINRA Rule 2090 — Know Your Customer",
            responsible_agent = ResponsibleAgent.ALLOCATION.value,
            action_required   = (
                f"Revise allocation to bring annualised volatility within "
                f"[{low:.0%}, {high:.0%}] for '{risk_tolerance}' client"
            ),
        ))

    return violations, passed, False
