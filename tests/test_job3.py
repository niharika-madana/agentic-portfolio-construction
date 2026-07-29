"""
Tests for Job 3 — Robo-Adviser Checks (SEC IM Guidance Update No. 2017-02).

  3.1  Composition–objective consistency
  3.2  Client-mandate consistency (deterministic + injected-reviewer path)
  3.3  Algorithm-limitation disclosure
  3.4  ETF-provider conflict screen
  + run_compliance() end-to-end with client statements

All deterministic and key-free — the LLM reviewer is only exercised via a stub
callable, never a live API call.

Run from the repo root:
    pytest tests/test_job3.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contracts import (
    ClientProfileSection,
    ClientStatement,
    ComplianceInput,
    ComplianceStatus,
    HumanCapitalSection,
    MacroRegimeSection,
    PipelineDestination,
    RegimeChangeFlag,
    Severity,
    StatementKind,
)
from agents.compliance import job3_robo_adviser_checks as job3
from agents.compliance.job3_robo_adviser_checks import (
    MandateFinding,
    check_algorithm_limitation_disclosure,
    check_client_mandate,
    check_composition_objective,
    check_etf_provider_conflict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exclusion(subject: str, quote: str = None) -> ClientStatement:
    return ClientStatement(
        kind=StatementKind.HARD_CONSTRAINT,
        summary=f"Client excludes {subject}",
        quote=quote or f"I don't want to invest in {subject}.",
        destinations=[PipelineDestination.UNIVERSE_EXCLUSION],
        subject=subject,
    )


def _ci(
    *,
    objective: str = "growth",
    portfolio: dict = None,
    statements: list = None,
    regime_confidence: float = 0.82,
) -> ComplianceInput:
    portfolio = portfolio if portfolio is not None else {"SPY": 0.6, "AGG": 0.4}
    return ComplianceInput(
        client_profile=ClientProfileSection(
            persona="test_persona", age=42, risk_tolerance="moderate",
            time_horizon_years=15, liquidity_needs="low",
            investment_objective=objective,
        ),
        human_capital=HumanCapitalSection(
            industry="technology", income_stability="low",
            income_equity_correlation=0.75, income_equity_beta=1.20,
            implicit_equity_exposure=0.90, rsu_concentration=0.40,
            human_capital_type="equity-like", employer_sector="technology",
            has_pension=False, estimated_hc_pv=2_500_000.0,
        ),
        macro_regime=MacroRegimeSection(
            current_regime="Rate Hike Cycle",
            regime_confidence=regime_confidence, regime_volatility="high",
        ),
        proposed_portfolio=dict(portfolio),
        allocation_rationale={t: "rationale" for t in portfolio},
        regime_change_flag=RegimeChangeFlag(detected=False),
        client_statements=statements or [],
    )


# ---------------------------------------------------------------------------
# 3.1 — Composition–objective consistency
# ---------------------------------------------------------------------------

def test_3_1_preservation_high_equity_is_high_severity():
    ci = _ci(objective="preservation", portfolio={"SPY": 0.90, "AGG": 0.10})
    violations, passed = check_composition_objective(ci)
    assert not passed
    assert any(v.severity == Severity.HIGH for v in violations)


def test_3_1_preservation_mild_overshoot_is_medium():
    ci = _ci(objective="preservation", portfolio={"SPY": 0.50, "AGG": 0.50})
    violations, _ = check_composition_objective(ci)
    assert violations and all(v.severity == Severity.MEDIUM for v in violations)


def test_3_1_growth_balanced_passes():
    ci = _ci(objective="growth", portfolio={"SPY": 0.60, "AGG": 0.40})
    violations, passed = check_composition_objective(ci)
    assert not violations
    assert "check_3_1_composition_objective" in passed


def test_3_1_growth_too_defensive_flags():
    ci = _ci(objective="growth", portfolio={"AGG": 0.80, "SPY": 0.20})
    violations, _ = check_composition_objective(ci)
    assert any("below" in v.description for v in violations)


def test_3_1_income_without_income_assets_flags():
    ci = _ci(objective="income", portfolio={"SPY": 0.50, "GLD": 0.50})
    violations, _ = check_composition_objective(ci)
    assert any("income-producing" in v.description.lower() for v in violations)


def test_3_1_income_with_bonds_passes():
    ci = _ci(objective="income", portfolio={"SPY": 0.40, "AGG": 0.40, "GLD": 0.20})
    violations, passed = check_composition_objective(ci)
    assert not violations
    assert "check_3_1_composition_objective" in passed


# ---------------------------------------------------------------------------
# 3.2 — Client-mandate consistency
# ---------------------------------------------------------------------------

def test_3_2_no_statements_passes_vacuously():
    violations, passed = check_client_mandate(_ci(statements=[]))
    assert not violations
    assert "check_3_2_client_mandate" in passed


def test_3_2_direct_sector_exclusion_fails():
    # Client excludes energy; portfolio directly holds XLE.
    ci = _ci(portfolio={"SPY": 0.5, "XLE": 0.3, "AGG": 0.2},
             statements=[_exclusion("energy")])
    violations, passed = check_client_mandate(ci)
    assert not passed
    assert any(v.severity == Severity.HIGH and "XLE" in v.description for v in violations)


def test_3_2_direct_ticker_exclusion_fails():
    ci = _ci(portfolio={"XLK": 0.5, "AGG": 0.5},
             statements=[_exclusion("XLK", "No technology sector fund, please.")])
    violations, _ = check_client_mandate(ci)
    assert any(v.severity == Severity.HIGH for v in violations)


def test_3_2_indirect_exposure_defers_to_disclosure_under_tiered():
    # "weapons" has no ETF proxy; only broad-market SPY may hold it.
    ci = _ci(portfolio={"SPY": 0.6, "AGG": 0.4}, statements=[_exclusion("weapons")])
    violations, passed = check_client_mandate(ci)
    # Under the default 'tiered' policy 3.2 does not hard-fail indirect exposure.
    assert not violations
    assert "check_3_2_client_mandate" in passed


def test_3_2_strict_policy_fails_indirect(monkeypatch):
    monkeypatch.setattr(job3, "EXCLUSION_POLICY", "strict")
    ci = _ci(portfolio={"SPY": 0.6, "AGG": 0.4}, statements=[_exclusion("weapons")])
    violations, _ = check_client_mandate(ci)
    assert any(v.severity == Severity.HIGH for v in violations)


def test_3_2_injected_reviewer_finding_becomes_violation():
    # Deterministic pass finds nothing for "pollution"; the reviewer catches it.
    ci = _ci(portfolio={"SPY": 0.5, "XLE": 0.5}, statements=[_exclusion("pollution")])

    def reviewer(statements, portfolio):
        return [MandateFinding(
            ticker="XLE", subject="pollution", quote=statements[0].quote,
            directness="direct", reason="XLE is a fossil-fuel-heavy sector fund.",
        )]

    violations, _ = check_client_mandate(ci, mandate_reviewer=reviewer)
    assert any(v.severity == Severity.HIGH and "XLE" in v.description for v in violations)


def test_3_2_reviewer_does_not_double_count():
    # Reviewer re-reports the same conflict the deterministic pass already found.
    ci = _ci(portfolio={"SPY": 0.5, "XLE": 0.5}, statements=[_exclusion("energy")])

    def reviewer(statements, portfolio):
        return [MandateFinding(
            ticker="XLE", subject="energy", quote=statements[0].quote,
            directness="direct", reason="duplicate",
        )]

    violations, _ = check_client_mandate(ci, mandate_reviewer=reviewer)
    assert len([v for v in violations if "XLE" in v.description]) == 1


# ---------------------------------------------------------------------------
# 3.3 — Algorithm-limitation disclosure
# ---------------------------------------------------------------------------

def test_3_3_low_regime_confidence_requires_disclosure():
    ci = _ci(regime_confidence=0.30)
    violations, _ = check_algorithm_limitation_disclosure(ci)
    assert any("confidence" in v.description.lower() and v.severity == Severity.LOW
               for v in violations)


def test_3_3_unhonourable_exclusion_requires_disclosure():
    ci = _ci(portfolio={"SPY": 0.6, "AGG": 0.4}, statements=[_exclusion("weapons")])
    violations, _ = check_algorithm_limitation_disclosure(ci)
    assert any("cannot be fully honoured" in v.description for v in violations)


def test_3_3_clean_portfolio_passes():
    ci = _ci(regime_confidence=0.82, statements=[])
    violations, passed = check_algorithm_limitation_disclosure(ci)
    assert not violations
    assert "check_3_3_algorithm_limitation_disclosure" in passed


# ---------------------------------------------------------------------------
# 3.4 — ETF-provider conflict screen
# ---------------------------------------------------------------------------

def test_3_4_no_declared_conflicts_passes():
    violations, passed = check_etf_provider_conflict(_ci())
    assert not violations
    assert "check_3_4_etf_provider_conflict" in passed


def test_3_4_flags_conflicted_issuer(monkeypatch):
    monkeypatch.setattr(job3, "_CONFLICTED_PROVIDERS", {"SSGA"})
    # SPY is issued by SSGA in the screen's issuer map.
    ci = _ci(portfolio={"SPY": 0.6, "AGG": 0.4})
    violations, _ = check_etf_provider_conflict(ci)
    assert any(v.severity == Severity.MEDIUM and "SSGA" in v.description for v in violations)


# ---------------------------------------------------------------------------
# End-to-end — run_compliance with client statements
# ---------------------------------------------------------------------------

def test_run_compliance_fails_on_direct_mandate_breach():
    from agents.compliance.compliance_agent import run_compliance
    from tests.test_compliance import _make_risk_output  # reuse Job-1 fixture

    portfolio = {"SPY": 0.5, "XLE": 0.3, "AGG": 0.2}
    ci = _ci(portfolio=portfolio, statements=[_exclusion("energy")])
    # Rebuild rationale/risk output to match this portfolio's tickers.
    risk_output = _make_risk_output(portfolio=portfolio)

    result = run_compliance(ci, risk_output)
    assert result.compliance_status == ComplianceStatus.FAIL
    assert not result.clearance
    assert any(v.check == "check_3_2_client_mandate" for v in result.violations)
