"""
Job 1 — Constraint Set Verification (audits the Risk Agent's output).

Three checks:
  1.1  Completeness   — all required sections, tickers, and derivation fields present
  1.2  Consistency    — actual vs limit values agree with the pass/fail flags Risk set
  1.3  Derivation audit — correct method was applied for this client's HC type

None of these checks recompute risk numbers. They only verify that the Risk
Agent's own numbers are internally consistent and that the right methodology
was used given the client's profile.
"""

from __future__ import annotations

import sys
import os

# Allow imports from the project root (contracts.py lives there)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contracts import (
    ComplianceInput,
    ComplianceViolation,
    ResponsibleAgent,
    RiskAgentOutput,
    Severity,
)

_REQUIRED_REGIMES = {"dot_com", "gfc", "covid", "rate_hike", "ai_boom"}
_REQUIRED_DERIVATION_FIELDS = {
    "drawdown_method", "concentration_method", "sector_method",
    "hc_type", "rsu_concentration", "data_source",
}
_HC_CORRELATION_ADJUSTED_METHOD = "hc_correlation_adjusted"


# ---------------------------------------------------------------------------
# Check 1.1 — Completeness
# ---------------------------------------------------------------------------

def check_completeness(
    compliance_input: ComplianceInput,
    risk_output: RiskAgentOutput,
) -> tuple[list[ComplianceViolation], list[str]]:
    """
    Verify that the Risk Agent's output is structurally complete.

    Catches:
    - Missing regime evaluation entries (not all 5 regimes covered)
    - Portfolio tickers with no corresponding position limit entry
    - Missing derivation fields (cannot audit methodology without them)

    Returns (violations, passed_checks).
    """
    violations: list[ComplianceViolation] = []
    passed: list[str] = []

    # ── 1.1a: All five regimes present ───────────────────────────────
    present_regimes = set(risk_output.regime_evaluation.keys())
    missing_regimes = _REQUIRED_REGIMES - present_regimes
    if missing_regimes:
        violations.append(ComplianceViolation(
            check             = "check_1_1a_regime_completeness",
            severity          = Severity.HIGH,
            description       = (
                f"Risk Agent regime_evaluation is missing required regimes: "
                f"{sorted(missing_regimes)}. All five regimes must be evaluated."
            ),
            rule_reference    = "Project specification — 4-5 historical regime backtests required",
            responsible_agent = ResponsibleAgent.RISK.value,
            action_required   = f"Add regime evaluation entries for: {sorted(missing_regimes)}",
        ))
    else:
        passed.append("check_1_1a_regime_completeness")

    # ── 1.1b: Every portfolio ticker has a position limit ─────────────
    portfolio_tickers = set(compliance_input.proposed_portfolio.keys())
    limit_tickers     = set(risk_output.position_limits.keys())
    missing_limits    = portfolio_tickers - limit_tickers
    if missing_limits:
        violations.append(ComplianceViolation(
            check             = "check_1_1b_position_limit_completeness",
            severity          = Severity.HIGH,
            description       = (
                f"Portfolio contains tickers with no Risk Agent position limit: "
                f"{sorted(missing_limits)}. Every position must be evaluated."
            ),
            rule_reference    = "FINRA Rule 2111 — suitability requires evaluating all positions",
            responsible_agent = ResponsibleAgent.RISK.value,
            action_required   = f"Add position_limits entries for: {sorted(missing_limits)}",
        ))
    else:
        passed.append("check_1_1b_position_limit_completeness")

    # ── 1.1c: Derivation section is complete ─────────────────────────
    derivation_dict = risk_output.derivation.model_dump()
    missing_fields  = _REQUIRED_DERIVATION_FIELDS - set(derivation_dict.keys())
    if missing_fields:
        violations.append(ComplianceViolation(
            check             = "check_1_1c_derivation_completeness",
            severity          = Severity.HIGH,
            description       = (
                f"Risk Agent derivation section is missing fields: {sorted(missing_fields)}. "
                f"Cannot audit methodology without full derivation."
            ),
            rule_reference    = "SEC IM Guidance Update 2017-02 — algorithmic outputs must be explainable",
            responsible_agent = ResponsibleAgent.RISK.value,
            action_required   = f"Populate derivation fields: {sorted(missing_fields)}",
        ))
    else:
        passed.append("check_1_1c_derivation_completeness")

    return violations, passed


# ---------------------------------------------------------------------------
# Check 1.2 — Consistency
# ---------------------------------------------------------------------------

def check_consistency(
    compliance_input: ComplianceInput,
    risk_output: RiskAgentOutput,
) -> tuple[list[ComplianceViolation], list[str]]:
    """
    Cross-check actual values vs limits vs the pass/fail flags Risk Agent set.

    A position limit where actual_weight > derived_limit but passed=True is a
    Risk Agent error — the compliance agent catches it here so it does not
    silently propagate to the client recommendation.

    Catches:
    - Position limits: actual > limit but passed=True (false positive)
    - Position limits: actual ≤ limit but passed=False (false negative)
    - Sector limits: same logic
    - Regime evaluations: drawdown > floor but passed=True

    Returns (violations, passed_checks).
    """
    violations: list[ComplianceViolation] = []
    passed: list[str] = []
    pos_inconsistencies   = 0
    sec_inconsistencies   = 0
    regime_inconsistencies = 0

    # ── 1.2a: Position limit pass/fail flags ──────────────────────────
    for ticker, limit in risk_output.position_limits.items():
        actual_exceeds_limit = limit.actual_weight > limit.derived_limit

        if actual_exceeds_limit and limit.passed:
            violations.append(ComplianceViolation(
                check             = "check_1_2a_position_limit_consistency",
                severity          = Severity.HIGH,
                description       = (
                    f"{ticker}: actual_weight {limit.actual_weight:.2%} > "
                    f"derived_limit {limit.derived_limit:.2%} but Risk Agent "
                    f"marked passed=True. This is a Risk Agent error."
                ),
                rule_reference    = "Internal Risk Agent consistency — FINRA Rule 2111",
                responsible_agent = ResponsibleAgent.RISK.value,
                action_required   = f"Correct position_limits.{ticker}.passed to false",
            ))
            pos_inconsistencies += 1

        elif not actual_exceeds_limit and not limit.passed:
            violations.append(ComplianceViolation(
                check             = "check_1_2a_position_limit_consistency",
                severity          = Severity.MEDIUM,
                description       = (
                    f"{ticker}: actual_weight {limit.actual_weight:.2%} ≤ "
                    f"derived_limit {limit.derived_limit:.2%} but Risk Agent "
                    f"marked passed=False. Verify derived_limit calculation."
                ),
                rule_reference    = "Internal Risk Agent consistency",
                responsible_agent = ResponsibleAgent.RISK.value,
                action_required   = f"Verify and correct position_limits.{ticker}.passed or derived_limit",
            ))
            pos_inconsistencies += 1

    if pos_inconsistencies == 0:
        passed.append("check_1_2a_position_limit_consistency")

    # ── 1.2b: Sector limit pass/fail flags ────────────────────────────
    for sector, limit in risk_output.sector_limits.items():
        actual_exceeds_limit = limit.actual_sector_weight > limit.adjusted_limit

        if actual_exceeds_limit and limit.passed:
            violations.append(ComplianceViolation(
                check             = "check_1_2b_sector_limit_consistency",
                severity          = Severity.HIGH,
                description       = (
                    f"Sector '{sector}': actual_sector_weight {limit.actual_sector_weight:.2%} > "
                    f"adjusted_limit {limit.adjusted_limit:.2%} but Risk Agent "
                    f"marked passed=True. This is a Risk Agent error."
                ),
                rule_reference    = "Internal Risk Agent consistency — HC-adjusted sector limits",
                responsible_agent = ResponsibleAgent.RISK.value,
                action_required   = f"Correct sector_limits.{sector}.passed to false",
            ))
            sec_inconsistencies += 1

        elif not actual_exceeds_limit and not limit.passed:
            violations.append(ComplianceViolation(
                check             = "check_1_2b_sector_limit_consistency",
                severity          = Severity.MEDIUM,
                description       = (
                    f"Sector '{sector}': actual_sector_weight {limit.actual_sector_weight:.2%} ≤ "
                    f"adjusted_limit {limit.adjusted_limit:.2%} but Risk Agent "
                    f"marked passed=False. Verify adjusted_limit calculation."
                ),
                rule_reference    = "Internal Risk Agent consistency",
                responsible_agent = ResponsibleAgent.RISK.value,
                action_required   = f"Verify and correct sector_limits.{sector}.passed or adjusted_limit",
            ))
            sec_inconsistencies += 1

    if sec_inconsistencies == 0:
        passed.append("check_1_2b_sector_limit_consistency")

    # ── 1.2c: Regime evaluation pass/fail flags ───────────────────────
    for regime, ev in risk_output.regime_evaluation.items():
        dd_exceeds_floor = ev.portfolio_drawdown < ev.drawdown_floor  # both negative, so < means worse

        if dd_exceeds_floor and ev.passed:
            violations.append(ComplianceViolation(
                check             = "check_1_2c_regime_evaluation_consistency",
                severity          = Severity.HIGH,
                description       = (
                    f"Regime '{regime}': portfolio_drawdown {ev.portfolio_drawdown:.2%} "
                    f"< drawdown_floor {ev.drawdown_floor:.2%} (worse than floor) "
                    f"but Risk Agent marked passed=True."
                ),
                rule_reference    = "Internal Risk Agent consistency — benchmark-relative drawdown evaluation",
                responsible_agent = ResponsibleAgent.RISK.value,
                action_required   = f"Correct regime_evaluation.{regime}.passed to false",
            ))
            regime_inconsistencies += 1

        elif not dd_exceeds_floor and not ev.passed:
            violations.append(ComplianceViolation(
                check             = "check_1_2c_regime_evaluation_consistency",
                severity          = Severity.MEDIUM,
                description       = (
                    f"Regime '{regime}': portfolio_drawdown {ev.portfolio_drawdown:.2%} "
                    f"≥ drawdown_floor {ev.drawdown_floor:.2%} (within floor) "
                    f"but Risk Agent marked passed=False."
                ),
                rule_reference    = "Internal Risk Agent consistency",
                responsible_agent = ResponsibleAgent.RISK.value,
                action_required   = f"Verify regime_evaluation.{regime}.passed and drawdown_floor",
            ))
            regime_inconsistencies += 1

    if regime_inconsistencies == 0:
        passed.append("check_1_2c_regime_evaluation_consistency")

    return violations, passed


# ---------------------------------------------------------------------------
# Check 1.3 — Derivation Audit
# ---------------------------------------------------------------------------

def check_derivation_audit(
    compliance_input: ComplianceInput,
    risk_output: RiskAgentOutput,
) -> tuple[list[ComplianceViolation], list[str]]:
    """
    Verify that the Risk Agent applied the correct methodology given the
    client's human capital profile.

    Catches:
    - equity-like HC type but sector_method != hc_correlation_adjusted
    - RSU concentration > 10% but employer sector adjusted_limit not reduced
    - hc_type mismatch between profile and Risk Agent's own derivation record

    Returns (violations, passed_checks).
    """
    violations: list[ComplianceViolation] = []
    passed: list[str] = []
    hc = compliance_input.human_capital
    derivation = risk_output.derivation

    # ── 1.3a: equity-like HC must use hc_correlation_adjusted sector method
    if hc.human_capital_type == "equity-like":
        if derivation.sector_method != _HC_CORRELATION_ADJUSTED_METHOD:
            violations.append(ComplianceViolation(
                check             = "check_1_3a_sector_method_for_equity_like_hc",
                severity          = Severity.HIGH,
                description       = (
                    f"Client has equity-like human capital (β={hc.income_equity_beta:.2f}) "
                    f"but Risk Agent used sector_method='{derivation.sector_method}' "
                    f"instead of required '{_HC_CORRELATION_ADJUSTED_METHOD}'. "
                    f"Sector limits are materially understated without HC correlation adjustment."
                ),
                rule_reference    = "Ibbotson et al. (2007) — total wealth framework; FINRA Rule 2111",
                responsible_agent = ResponsibleAgent.RISK.value,
                action_required   = f"Re-run risk agent with sector_method='{_HC_CORRELATION_ADJUSTED_METHOD}'",
            ))
        else:
            passed.append("check_1_3a_sector_method_for_equity_like_hc")
    else:
        passed.append("check_1_3a_sector_method_for_equity_like_hc")  # rule not triggered

    # ── 1.3b: RSU > 10% → employer sector adjusted_limit must be < base_limit
    if hc.rsu_concentration > 0.10:
        employer_sector = hc.employer_sector
        if employer_sector in risk_output.sector_limits:
            sector = risk_output.sector_limits[employer_sector]
            if sector.adjusted_limit >= sector.base_limit:
                violations.append(ComplianceViolation(
                    check             = "check_1_3b_rsu_sector_limit_reduction",
                    severity          = Severity.HIGH,
                    description       = (
                        f"RSU concentration {hc.rsu_concentration:.0%} > 10%, "
                        f"but employer sector '{employer_sector}' adjusted_limit "
                        f"({sector.adjusted_limit:.2%}) is not below base_limit "
                        f"({sector.base_limit:.2%}). HC correlation adjustment not applied."
                    ),
                    rule_reference    = "Ibbotson et al. (2007) — RSU concentration compounds sector exposure",
                    responsible_agent = ResponsibleAgent.RISK.value,
                    action_required   = (
                        f"Reduce sector_limits.{employer_sector}.adjusted_limit below "
                        f"{sector.base_limit:.2%} given RSU concentration"
                    ),
                ))
            else:
                passed.append("check_1_3b_rsu_sector_limit_reduction")
        else:
            violations.append(ComplianceViolation(
                check             = "check_1_3b_rsu_sector_limit_reduction",
                severity          = Severity.MEDIUM,
                description       = (
                    f"RSU concentration {hc.rsu_concentration:.0%} > 10%, "
                    f"but employer sector '{employer_sector}' has no sector limit entry. "
                    f"Cannot verify HC concentration adjustment."
                ),
                rule_reference    = "Ibbotson et al. (2007)",
                responsible_agent = ResponsibleAgent.RISK.value,
                action_required   = f"Add sector_limits entry for '{employer_sector}'",
            ))
    else:
        passed.append("check_1_3b_rsu_sector_limit_reduction")  # rule not triggered

    # ── 1.3c: hc_type in derivation must match the profile ───────────
    if derivation.hc_type != hc.human_capital_type:
        violations.append(ComplianceViolation(
            check             = "check_1_3c_hc_type_consistency",
            severity          = Severity.HIGH,
            description       = (
                f"Risk Agent derivation records hc_type='{derivation.hc_type}' "
                f"but client profile states human_capital_type='{hc.human_capital_type}'. "
                f"Risk limits were calibrated for the wrong client archetype."
            ),
            rule_reference    = "FINRA Rule 2090 — Know Your Customer",
            responsible_agent = ResponsibleAgent.RISK.value,
            action_required   = f"Re-run risk agent with hc_type='{hc.human_capital_type}'",
        ))
    else:
        passed.append("check_1_3c_hc_type_consistency")

    return violations, passed
