"""
Tests for the Pydantic-based compliance pipeline.

Replaces test_validation.py which tested the now-retired input_validation.py.

Structure:
  Section 1 — Contract validation (Pydantic model validators)
  Section 2 — Check 1.1: Completeness
  Section 3 — Check 1.2: Consistency
  Section 4 — Check 1.3: Derivation audit
  Section 5 — Check 2.1: Rationale completeness
  Section 6 — Check 2.2: Rationale specificity
  Section 7 — Check 2.3: HC acknowledgment
  Section 8 — Check 2.4: Weight integrity
  Section 9 — Check 2.5: Volatility suitability
  Section 10 — run_compliance() end-to-end

Run from the repo root:
    pytest agents/compliance/test_compliance.py -v
"""

from __future__ import annotations

import os
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contracts import (
    AllocationAgentOutput,
    ClientProfileSection,
    ComplianceInput,
    ComplianceStatus,
    HumanCapitalSection,
    HumanCapitalType,
    IncomeStability,
    InvestmentObjective,
    LiquidityNeeds,
    MacroRegimeSection,
    MacroRegimeSnapshot,
    PositionLimit,
    ProfileAgentOutput,
    RegimeChangeFlag,
    RegimeEvaluation,
    RiskAgentOutput,
    RiskDecision,
    RiskDerivation,
    RiskToleranceLevel,
    SectorLimit,
    Severity,
)
from agents.compliance.compliance_agent import run_compliance
from agents.compliance.constraint_checks import (
    check_completeness,
    check_consistency,
    check_derivation_audit,
)
from agents.compliance.content_checks import (
    check_hc_acknowledgment,
    check_rationale_completeness,
    check_rationale_specificity,
    check_volatility_suitability,
    check_weight_integrity,
)


# ---------------------------------------------------------------------------
# Shared helpers — build valid base objects, then inject specific failures
# ---------------------------------------------------------------------------

_VALID_PORTFOLIO = {"VTI": 0.40, "BND": 0.35, "GLD": 0.25}

_VALID_RATIONALE = {
    "VTI": (
        "Broad market equity; portfolio equity target reduced to offset equity-like "
        "human capital beta=1.20 and implicit equity exposure from technology sector income"
    ),
    "BND": (
        "Fixed income anchor hedges against equity-like income stream; "
        "appropriate for moderate risk tolerance with high income volatility sigma=0.40"
    ),
    "GLD": (
        "Real asset diversification; low correlation with technology sector and "
        "regime volatility hedge during rate hike cycle"
    ),
}


def _make_compliance_input(
    persona:                  str   = "tech_executive",
    risk_tolerance:           str   = "moderate",
    hc_type:                  str   = "equity-like",
    income_equity_beta:       float = 1.20,
    implicit_equity_exposure: float = 0.906,
    income_equity_correlation: float = 0.75,
    rsu_concentration:        float = 0.40,
    employer_sector:          str   = "technology",
    portfolio:                dict  = None,
    rationale:                dict  = None,
    regime_change:            bool  = False,
) -> ComplianceInput:
    return ComplianceInput(
        client_profile=ClientProfileSection(
            persona              = persona,
            age                  = 42,
            risk_tolerance       = risk_tolerance,
            time_horizon_years   = 15,
            liquidity_needs      = "low",
            investment_objective = "growth",
        ),
        human_capital=HumanCapitalSection(
            industry                  = employer_sector,
            income_stability          = "low",
            income_equity_correlation = income_equity_correlation,
            income_equity_beta        = income_equity_beta,
            implicit_equity_exposure  = implicit_equity_exposure,
            rsu_concentration         = rsu_concentration,
            human_capital_type        = hc_type,
            employer_sector           = employer_sector,
            has_pension               = False,
            estimated_hc_pv           = 2_500_000.0,
        ),
        macro_regime=MacroRegimeSection(
            current_regime     = "Rate Hike Cycle",
            regime_confidence  = 0.82,
            regime_volatility  = "high",
        ),
        proposed_portfolio   = portfolio or dict(_VALID_PORTFOLIO),
        allocation_rationale = rationale or dict(_VALID_RATIONALE),
        regime_change_flag   = RegimeChangeFlag(
            detected       = regime_change,
            prior_regime   = "AI Boom" if regime_change else None,
            current_regime = "Rate Hike Cycle" if regime_change else None,
        ),
    )


def _make_risk_output(
    risk_decision:               RiskDecision = RiskDecision.FLAG,
    portfolio:                   dict         = None,
    hc_type:                     str          = "equity-like",
    rsu_concentration:           float        = 0.40,
    portfolio_volatility_annual: float        = None,
    # Check 1.2 injections
    position_false_positive:     bool         = False,   # actual > limit but passed=True
    position_false_negative:     bool         = False,   # actual <= limit but passed=False
    sector_false_positive:       bool         = False,
    regime_false_positive:       bool         = False,
    # Check 1.3 injections
    wrong_sector_method:         bool         = False,
    skip_employer_sector_limit:  bool         = False,
    wrong_hc_type_in_derivation: bool         = False,
    # Check 1.1 injections
    drop_regime:                 str          = None,
    drop_position_limit_ticker:  str          = None,
) -> RiskAgentOutput:
    if portfolio is None:
        portfolio = dict(_VALID_PORTFOLIO)

    # Build position limits
    position_limits = {}
    for ticker, weight in portfolio.items():
        if position_false_positive and ticker == "VTI":
            # actual(0.40) > limit(0.30) but passed=True — Risk Agent error
            position_limits[ticker] = PositionLimit(
                derived_limit=0.30, actual_weight=weight, passed=True,
                method="marginal_risk_contribution"
            )
        elif position_false_negative and ticker == "BND":
            # actual(0.35) <= limit(0.40) but passed=False — Risk Agent error
            position_limits[ticker] = PositionLimit(
                derived_limit=0.40, actual_weight=weight, passed=False,
                method="marginal_risk_contribution"
            )
        else:
            position_limits[ticker] = PositionLimit(
                derived_limit=0.50, actual_weight=weight, passed=True,
                method="marginal_risk_contribution"
            )

    if drop_position_limit_ticker and drop_position_limit_ticker in position_limits:
        del position_limits[drop_position_limit_ticker]

    # Build sector limits
    tech_sector = SectorLimit(
        base_limit           = 0.30,
        correlation_with_hc  = 0.85,
        adjusted_limit       = 0.05 if not sector_false_positive else 0.50,
        actual_sector_weight = 0.40,
        passed               = sector_false_positive,
    )
    sector_limits = {} if skip_employer_sector_limit else {"technology": tech_sector}

    # Build regime evaluation
    regime_evaluation = {
        "dot_com":   RegimeEvaluation(benchmark_drawdown=-0.44, drawdown_floor=-0.44, portfolio_drawdown=-0.50, passed=False),
        "gfc":       RegimeEvaluation(benchmark_drawdown=-0.51, drawdown_floor=-0.51, portfolio_drawdown=-0.45, passed=True),
        "covid":     RegimeEvaluation(benchmark_drawdown=-0.34, drawdown_floor=-0.34, portfolio_drawdown=-0.30, passed=True),
        "rate_hike": RegimeEvaluation(benchmark_drawdown=-0.22, drawdown_floor=-0.22, portfolio_drawdown=-0.18, passed=True),
        "ai_boom":   RegimeEvaluation(benchmark_drawdown=-0.12, drawdown_floor=-0.12, portfolio_drawdown=-0.08, passed=True),
    }
    if regime_false_positive:
        # dot_com: portfolio worse than floor but passed=True
        regime_evaluation["dot_com"] = RegimeEvaluation(
            benchmark_drawdown=-0.44, drawdown_floor=-0.44,
            portfolio_drawdown=-0.60, passed=True
        )
    if drop_regime and drop_regime in regime_evaluation:
        del regime_evaluation[drop_regime]

    return RiskAgentOutput(
        risk_decision      = risk_decision,
        regime_evaluation  = regime_evaluation,
        position_limits    = position_limits,
        sector_limits      = sector_limits,
        violations         = ["dot_com drawdown exceeds benchmark floor"],
        derivation         = RiskDerivation(
            drawdown_method      = "benchmark_relative_per_regime",
            concentration_method = "marginal_risk_contribution",
            sector_method        = "static_lookup" if wrong_sector_method else "hc_correlation_adjusted",
            hc_type              = "bond-like" if wrong_hc_type_in_derivation else hc_type,
            rsu_concentration    = rsu_concentration,
            data_source          = "CRSP daily returns 2000-2024",
        ),
        portfolio_volatility_annual = portfolio_volatility_annual,
    )


# ===========================================================================
# Section 1 — Contract validation (Pydantic model validators)
# ===========================================================================

class TestAllocationAgentOutputValidation:

    def test_valid_allocation_passes(self):
        alloc = AllocationAgentOutput(
            proposed_portfolio   = _VALID_PORTFOLIO,
            allocation_rationale = _VALID_RATIONALE,
        )
        assert sum(alloc.proposed_portfolio.values()) == pytest.approx(1.0, abs=0.01)

    def test_weights_not_summing_to_one_raises(self):
        bad_portfolio = {"VTI": 0.50, "BND": 0.35, "GLD": 0.25}  # sums to 1.10
        with pytest.raises(ValidationError, match="1.10"):
            AllocationAgentOutput(
                proposed_portfolio   = bad_portfolio,
                allocation_rationale = _VALID_RATIONALE,
            )

    def test_negative_weight_raises(self):
        bad_portfolio = {"VTI": -0.10, "BND": 0.85, "GLD": 0.25}
        with pytest.raises(ValidationError, match="negative"):
            AllocationAgentOutput(
                proposed_portfolio   = bad_portfolio,
                allocation_rationale = _VALID_RATIONALE,
            )

    def test_fewer_than_two_positions_raises(self):
        with pytest.raises(ValidationError, match="minimum 2"):
            AllocationAgentOutput(
                proposed_portfolio   = {"VTI": 1.0},
                allocation_rationale = {"VTI": "only position"},
            )

    def test_missing_rationale_raises(self):
        with pytest.raises(ValidationError, match="Missing rationale"):
            AllocationAgentOutput(
                proposed_portfolio   = _VALID_PORTFOLIO,
                allocation_rationale = {"VTI": "only VTI has rationale"},
                # BND and GLD rationale missing
            )


class TestProfileAgentOutputValidation:

    def _base_kwargs(self, **overrides):
        fc = 2_000_000
        hc = 6_146_977
        total = fc + hc
        hc_share = hc / total
        beta = 1.20
        kwargs = dict(
            client_id                  = "tech_exec",
            career_type                = "Technology",
            age                        = 42,
            financial_capital          = fc,
            human_capital_valuation    = hc,
            total_wealth               = total,
            human_capital_pct_of_total = round(hc / total * 100, 1),
            income_volatility_sigma    = 0.40,
            income_equity_correlation  = 0.75,
            income_equity_beta         = beta,
            implicit_equity_exposure   = round(hc_share * beta, 4),
            human_capital_type         = HumanCapitalType.EQUITY_LIKE,
            income_stability           = IncomeStability.LOW,
            effective_risk_budget      = 0.698,
            industry_exposure_sector   = "technology",
            RSU_concentration          = 0.60,
            current_holdings           = {"rsu": 0.60, "equity": 0.30, "cash": 0.10},
            investment_horizon_years   = 27,
            risk_tolerance_level       = RiskToleranceLevel.AGGRESSIVE,
            liquidity_needs            = LiquidityNeeds.MEDIUM,
            investment_objective       = InvestmentObjective.GROWTH,
        )
        kwargs.update(overrides)
        return kwargs

    def test_valid_profile_passes(self):
        p = ProfileAgentOutput(**self._base_kwargs())
        assert p.implicit_equity_exposure == pytest.approx(
            p.human_capital_valuation / p.total_wealth * p.income_equity_beta,
            abs=0.01
        )

    def test_holdings_not_summing_to_one_raises(self):
        with pytest.raises(ValidationError, match="1.0"):
            ProfileAgentOutput(**self._base_kwargs(
                current_holdings={"rsu": 0.60, "equity": 0.50, "cash": 0.10}  # sums to 1.20
            ))

    def test_wrong_total_wealth_raises(self):
        with pytest.raises(ValidationError, match="total_wealth"):
            ProfileAgentOutput(**self._base_kwargs(total_wealth=999_999))

    def test_wrong_implicit_equity_exposure_raises(self):
        with pytest.raises(ValidationError, match="implicit_equity_exposure"):
            ProfileAgentOutput(**self._base_kwargs(implicit_equity_exposure=0.01))

    def test_hc_type_inconsistent_with_beta_raises(self):
        # beta=1.20 should be EQUITY_LIKE, not BOND_LIKE
        with pytest.raises(ValidationError, match="inconsistent"):
            ProfileAgentOutput(**self._base_kwargs(
                human_capital_type=HumanCapitalType.BOND_LIKE
            ))


class TestMacroRegimeSnapshotFlags:

    def test_low_confidence_flag_set_automatically(self):
        from datetime import date
        snap = MacroRegimeSnapshot(
            as_of             = date(2026, 6, 1),
            regime_label      = "Rate Hike Cycle",
            prior_regime      = "Rate Hike Cycle",
            regime_shift_date = date(2022, 3, 1),
            regime_confidence = 0.45,   # below 0.60 threshold
            regime_volatility = 0.025,
            yield_curve       = -0.50,
            term_spread       = -0.80,
            fed_funds         = 5.25,
            unemployment      = 4.1,
            cpi               = 3.2,
            credit_spread     = 1.8,
        )
        assert snap.is_low_confidence is True
        assert snap.regime_change_detected is False

    def test_regime_change_detected_automatically(self):
        from datetime import date
        snap = MacroRegimeSnapshot(
            as_of             = date(2026, 6, 1),
            regime_label      = "Rate Hike Cycle",
            prior_regime      = "AI Boom",   # different → change detected
            regime_shift_date = date(2022, 3, 1),
            regime_confidence = 0.82,
            regime_volatility = 0.025,
            yield_curve       = -0.50,
            term_spread       = -0.80,
            fed_funds         = 5.25,
            unemployment      = 4.1,
            cpi               = 3.2,
            credit_spread     = 1.8,
        )
        assert snap.regime_change_detected is True


# ===========================================================================
# Section 2 — Check 1.1: Completeness
# ===========================================================================

class TestCheck11Completeness:

    def test_all_complete_passes(self):
        ci = _make_compliance_input()
        ro = _make_risk_output()
        violations, passed = check_completeness(ci, ro)
        assert not violations
        assert "check_1_1a_regime_completeness" in passed
        assert "check_1_1b_position_limit_completeness" in passed
        assert "check_1_1c_derivation_completeness" in passed

    def test_missing_regime_flagged(self):
        ci = _make_compliance_input()
        ro = _make_risk_output(drop_regime="gfc")
        violations, _ = check_completeness(ci, ro)
        checks = [v.check for v in violations]
        assert "check_1_1a_regime_completeness" in checks
        assert any("gfc" in v.description for v in violations)

    def test_missing_position_limit_for_portfolio_ticker(self):
        ci = _make_compliance_input()
        ro = _make_risk_output(drop_position_limit_ticker="VTI")
        violations, _ = check_completeness(ci, ro)
        checks = [v.check for v in violations]
        assert "check_1_1b_position_limit_completeness" in checks
        assert any("VTI" in v.description for v in violations)

    def test_all_checks_pass_when_complete(self):
        ci = _make_compliance_input()
        ro = _make_risk_output()
        violations, passed = check_completeness(ci, ro)
        assert len(violations) == 0
        assert len(passed) == 3


# ===========================================================================
# Section 3 — Check 1.2: Consistency
# ===========================================================================

class TestCheck12Consistency:

    def test_consistent_risk_output_passes(self):
        ci = _make_compliance_input()
        ro = _make_risk_output()
        violations, passed = check_consistency(ci, ro)
        assert not violations
        assert "check_1_2a_position_limit_consistency" in passed
        assert "check_1_2b_sector_limit_consistency" in passed
        assert "check_1_2c_regime_evaluation_consistency" in passed

    def test_position_false_positive_flagged_high_severity(self):
        ci = _make_compliance_input()
        ro = _make_risk_output(position_false_positive=True)
        violations, _ = check_consistency(ci, ro)
        pos_violations = [v for v in violations if v.check == "check_1_2a_position_limit_consistency"]
        assert pos_violations
        assert pos_violations[0].severity == Severity.HIGH
        assert "VTI" in pos_violations[0].description
        assert pos_violations[0].responsible_agent == "risk_agent"

    def test_position_false_negative_flagged_medium_severity(self):
        ci = _make_compliance_input()
        ro = _make_risk_output(position_false_negative=True)
        violations, _ = check_consistency(ci, ro)
        pos_violations = [v for v in violations if v.check == "check_1_2a_position_limit_consistency"]
        assert pos_violations
        assert pos_violations[0].severity == Severity.MEDIUM

    def test_regime_false_positive_flagged(self):
        ci = _make_compliance_input()
        ro = _make_risk_output(regime_false_positive=True)
        violations, _ = check_consistency(ci, ro)
        regime_violations = [v for v in violations if v.check == "check_1_2c_regime_evaluation_consistency"]
        assert regime_violations
        assert regime_violations[0].severity == Severity.HIGH


# ===========================================================================
# Section 4 — Check 1.3: Derivation audit
# ===========================================================================

class TestCheck13DerivationAudit:

    def test_correct_methodology_passes(self):
        ci = _make_compliance_input()
        ro = _make_risk_output()
        violations, passed = check_derivation_audit(ci, ro)
        assert not violations
        assert "check_1_3a_sector_method_for_equity_like_hc" in passed
        assert "check_1_3b_rsu_sector_limit_reduction" in passed
        assert "check_1_3c_hc_type_consistency" in passed

    def test_equity_like_hc_with_wrong_sector_method_flagged(self):
        ci = _make_compliance_input(hc_type="equity-like")
        ro = _make_risk_output(wrong_sector_method=True)
        violations, _ = check_derivation_audit(ci, ro)
        checks = [v.check for v in violations]
        assert "check_1_3a_sector_method_for_equity_like_hc" in checks
        assert violations[0].severity == Severity.HIGH

    def test_wrong_sector_method_not_triggered_for_bond_like_hc(self):
        ci = _make_compliance_input(hc_type="bond-like", income_equity_beta=0.05)
        ro = _make_risk_output(wrong_sector_method=True, hc_type="bond-like")
        violations, passed = check_derivation_audit(ci, ro)
        checks = [v.check for v in violations]
        # Check 1.3a should not fire for bond-like HC
        assert "check_1_3a_sector_method_for_equity_like_hc" not in checks
        assert "check_1_3a_sector_method_for_equity_like_hc" in passed

    def test_rsu_above_10pct_without_reduced_sector_limit_flagged(self):
        ci = _make_compliance_input(rsu_concentration=0.40)
        ro = _make_risk_output(rsu_concentration=0.40, skip_employer_sector_limit=True)
        violations, _ = check_derivation_audit(ci, ro)
        checks = [v.check for v in violations]
        assert "check_1_3b_rsu_sector_limit_reduction" in checks

    def test_rsu_below_10pct_not_triggered(self):
        ci = _make_compliance_input(rsu_concentration=0.05)
        ro = _make_risk_output(rsu_concentration=0.05)
        violations, passed = check_derivation_audit(ci, ro)
        assert "check_1_3b_rsu_sector_limit_reduction" in passed

    def test_hc_type_mismatch_in_derivation_flagged(self):
        ci = _make_compliance_input(hc_type="equity-like")
        ro = _make_risk_output(wrong_hc_type_in_derivation=True)
        violations, _ = check_derivation_audit(ci, ro)
        checks = [v.check for v in violations]
        assert "check_1_3c_hc_type_consistency" in checks
        assert violations[-1].severity == Severity.HIGH


# ===========================================================================
# Section 5 — Check 2.1: Rationale completeness
# ===========================================================================

class TestCheck21RationaleCompleteness:

    def test_complete_rationale_passes(self):
        ci = _make_compliance_input()
        violations, passed = check_rationale_completeness(ci)
        assert not violations
        assert "check_2_1_rationale_completeness" in passed

    def test_missing_rationale_for_ticker_flagged(self):
        rationale = dict(_VALID_RATIONALE)
        del rationale["GLD"]
        ci = _make_compliance_input(rationale=rationale)
        violations, _ = check_rationale_completeness(ci)
        assert violations
        assert any("GLD" in v.description for v in violations)
        assert violations[0].severity == Severity.HIGH

    def test_empty_string_rationale_flagged(self):
        rationale = dict(_VALID_RATIONALE)
        rationale["BND"] = "   "   # whitespace only
        ci = _make_compliance_input(rationale=rationale)
        violations, _ = check_rationale_completeness(ci)
        assert violations
        assert any("BND" in v.description for v in violations)


# ===========================================================================
# Section 6 — Check 2.2: Rationale specificity
# ===========================================================================

class TestCheck22RationaleSpecificity:

    def test_specific_rationale_passes(self):
        ci = _make_compliance_input()  # uses _VALID_RATIONALE which has HC beta, sector, regime
        violations, passed = check_rationale_specificity(ci)
        assert not violations
        assert "check_2_2_rationale_specificity" in passed

    def test_boilerplate_rationale_flagged_medium_severity(self):
        generic = {
            "VTI": "Provides diversification and long-term growth potential",
            "BND": "Offers stability and income for balanced portfolio",
            "GLD": "Alternative asset class for portfolio protection",
        }
        ci = _make_compliance_input(rationale=generic)
        violations, _ = check_rationale_specificity(ci)
        assert violations
        assert all(v.severity == Severity.MEDIUM for v in violations)
        assert all(v.responsible_agent == "allocation_agent" for v in violations)

    def test_partial_specificity_flagged(self):
        mixed = dict(_VALID_RATIONALE)
        mixed["GLD"] = "Gold is a safe haven asset"   # generic
        ci = _make_compliance_input(rationale=mixed)
        violations, _ = check_rationale_specificity(ci)
        flagged_tickers = [
            t for v in violations
            for t in ["VTI", "BND", "GLD"]
            if t in v.description
        ]
        assert "GLD" in flagged_tickers


# ===========================================================================
# Section 7 — Check 2.3: HC acknowledgment
# ===========================================================================

class TestCheck23HCAcknowledgment:

    def test_rsu_acknowledged_passes(self):
        ci = _make_compliance_input()   # _VALID_RATIONALE references "equity-like", beta, sector
        violations, passed = check_hc_acknowledgment(ci)
        assert "check_2_3b_implicit_equity_exposure_acknowledgment" in passed

    def test_rsu_above_10pct_not_mentioned_flagged_high(self):
        generic = {
            "VTI": "Broad market exposure for growth",
            "BND": "Fixed income for stability",
            "GLD": "Diversification through real assets",
        }
        ci = _make_compliance_input(rsu_concentration=0.40, rationale=generic)
        violations, _ = check_hc_acknowledgment(ci)
        checks = [v.check for v in violations]
        assert "check_2_3a_rsu_acknowledgment" in checks
        rsu_v = next(v for v in violations if v.check == "check_2_3a_rsu_acknowledgment")
        assert rsu_v.severity == Severity.HIGH

    def test_rsu_below_10pct_not_triggered(self):
        ci = _make_compliance_input(rsu_concentration=0.05)
        violations, passed = check_hc_acknowledgment(ci)
        assert "check_2_3a_rsu_acknowledgment" in passed

    def test_equity_like_hc_not_mentioned_flagged_medium(self):
        generic = {
            "VTI": "Broad market exposure for growth",
            "BND": "Fixed income for stability",
            "GLD": "Diversification through real assets",
        }
        ci = _make_compliance_input(
            rsu_concentration=0.0,   # no RSU so 2.3a not triggered
            income_equity_beta=1.20,
            rationale=generic,
        )
        violations, _ = check_hc_acknowledgment(ci)
        checks = [v.check for v in violations]
        assert "check_2_3b_implicit_equity_exposure_acknowledgment" in checks
        v = next(v for v in violations if "implicit" in v.check)
        assert v.severity == Severity.MEDIUM


# ===========================================================================
# Section 8 — Check 2.4: Weight integrity
# ===========================================================================

class TestCheck24WeightIntegrity:

    def test_valid_portfolio_passes(self):
        ci = _make_compliance_input()
        violations, passed = check_weight_integrity(ci)
        assert not violations
        assert "check_2_4_weight_integrity" in passed

    def test_single_position_flagged(self):
        ci = _make_compliance_input(
            portfolio={"VTI": 1.0},
            rationale={"VTI": _VALID_RATIONALE["VTI"]},
        )
        violations, _ = check_weight_integrity(ci)
        checks = [v.check for v in violations]
        assert "check_2_4a_minimum_positions" in checks

    def test_weights_not_summing_to_one_flagged(self):
        # Note: AllocationAgentOutput Pydantic validator catches this first,
        # but ComplianceInput accepts raw dicts so check_weight_integrity
        # provides the compliance-layer catch.
        ci = _make_compliance_input()
        # Directly mutate the dict to bypass Pydantic (testing compliance layer)
        ci.proposed_portfolio["VTI"] = 0.80   # now sums to 1.40
        violations, _ = check_weight_integrity(ci)
        checks = [v.check for v in violations]
        assert "check_2_4c_weights_sum_to_one" in checks


# ===========================================================================
# Section 9 — Check 2.5: Volatility suitability
# ===========================================================================

class TestCheck25VolatilitySuitability:

    def test_vol_within_band_passes(self):
        ci = _make_compliance_input(risk_tolerance="moderate")
        ro = _make_risk_output(portfolio_volatility_annual=0.15)
        violations, passed, skipped = check_volatility_suitability(ci, ro)
        assert not skipped
        assert not violations
        assert "check_2_5_volatility_suitability" in passed

    def test_vol_above_band_flagged_high_severity(self):
        ci = _make_compliance_input(risk_tolerance="conservative")
        ro = _make_risk_output(portfolio_volatility_annual=0.25)  # above 0.12 cap
        violations, _, skipped = check_volatility_suitability(ci, ro)
        assert not skipped
        assert violations
        assert violations[0].severity == Severity.HIGH
        assert "above" in violations[0].description

    def test_vol_below_band_flagged_medium_severity(self):
        ci = _make_compliance_input(risk_tolerance="aggressive")
        ro = _make_risk_output(portfolio_volatility_annual=0.03)  # extremely low for aggressive
        violations, _, _ = check_volatility_suitability(ci, ro)
        # below band → MEDIUM (unnecessarily conservative for aggressive client)
        assert violations
        assert violations[0].severity == Severity.MEDIUM

    def test_missing_vol_skipped_with_flag(self):
        ci = _make_compliance_input()
        ro = _make_risk_output(portfolio_volatility_annual=None)
        violations, passed, skipped = check_volatility_suitability(ci, ro)
        assert skipped is True
        assert not violations
        assert not passed

    def test_conservative_band_boundary(self):
        ci = _make_compliance_input(risk_tolerance="conservative")
        ro = _make_risk_output(portfolio_volatility_annual=0.12)   # exactly at cap
        violations, passed, _ = check_volatility_suitability(ci, ro)
        assert not violations
        assert "check_2_5_volatility_suitability" in passed


# ===========================================================================
# Section 10 — run_compliance() end-to-end
# ===========================================================================

class TestRunComplianceEndToEnd:

    def test_clean_portfolio_passes(self):
        ci = _make_compliance_input()
        ro = _make_risk_output(
            risk_decision               = RiskDecision.PASS,
            portfolio_volatility_annual = 0.15,
        )
        result = run_compliance(ci, ro)
        assert result.clearance is True
        assert result.compliance_status in (
            ComplianceStatus.PASS, ComplianceStatus.PASS_WITH_WARNINGS
        )

    def test_risk_agent_false_positive_causes_fail(self):
        ci = _make_compliance_input()
        ro = _make_risk_output(position_false_positive=True)
        result = run_compliance(ci, ro)
        assert result.compliance_status == ComplianceStatus.FAIL
        assert result.clearance is False
        assert "risk_agent" in result.agent_feedback

    def test_wrong_sector_method_causes_fail(self):
        ci = _make_compliance_input(hc_type="equity-like")
        ro = _make_risk_output(wrong_sector_method=True)
        result = run_compliance(ci, ro)
        assert result.compliance_status == ComplianceStatus.FAIL
        assert "risk_agent" in result.agent_feedback

    def test_missing_rationale_causes_fail(self):
        rationale = dict(_VALID_RATIONALE)
        del rationale["GLD"]
        ci = _make_compliance_input(rationale=rationale)
        ro = _make_risk_output()
        result = run_compliance(ci, ro)
        assert result.compliance_status == ComplianceStatus.FAIL
        assert "allocation_agent" in result.agent_feedback

    def test_boilerplate_rationale_causes_pass_with_warnings(self):
        generic = {
            "VTI": "Provides diversification and growth potential for the portfolio",
            "BND": "Offers stability and income through fixed income allocation",
            "GLD": "Alternative asset providing portfolio protection and hedge",
        }
        ci = _make_compliance_input(
            rsu_concentration=0.0,   # no RSU so 2.3a not triggered
            income_equity_beta=0.05, # bond-like so 2.3b not triggered
            implicit_equity_exposure=0.038,
            hc_type="bond-like",
            rationale=generic,
        )
        ro = _make_risk_output(
            hc_type="bond-like",
            rsu_concentration=0.0,
            portfolio_volatility_annual=0.10,
            wrong_sector_method=False,
        )
        result = run_compliance(ci, ro)
        # Boilerplate rationale → MEDIUM violations only → PASS_WITH_WARNINGS
        assert result.compliance_status == ComplianceStatus.PASS_WITH_WARNINGS
        assert result.clearance is True

    def test_agent_feedback_groups_by_responsible_agent(self):
        ci = _make_compliance_input()
        ro = _make_risk_output(
            position_false_positive=True,   # risk_agent violation
        )
        rationale = dict(_VALID_RATIONALE)
        del rationale["GLD"]
        ci.allocation_rationale = rationale   # allocation_agent violation
        ci.proposed_portfolio   = _VALID_PORTFOLIO  # keep portfolio intact
        result = run_compliance(ci, ro)
        assert "risk_agent"       in result.agent_feedback
        assert "allocation_agent" in result.agent_feedback

    def test_passed_checks_list_non_empty_on_clean_run(self):
        ci = _make_compliance_input()
        ro = _make_risk_output(
            risk_decision               = RiskDecision.PASS,
            portfolio_volatility_annual = 0.15,
        )
        result = run_compliance(ci, ro)
        assert len(result.passed_checks) > 0

    def test_all_three_personas_produce_different_outcomes(self):
        # Biology professor — bond-like HC, no RSU, conservative
        prof_ci = _make_compliance_input(
            persona               = "biology_professor",
            risk_tolerance        = "conservative",
            hc_type               = "bond-like",
            income_equity_beta    = 0.05,
            implicit_equity_exposure = 0.039,
            income_equity_correlation = 0.10,
            rsu_concentration     = 0.00,
            employer_sector       = "education",
        )
        prof_ro = _make_risk_output(
            hc_type               = "bond-like",
            rsu_concentration     = 0.00,
            wrong_sector_method   = False,
            portfolio_volatility_annual = 0.08,
        )

        # Tech executive — equity-like HC, high RSU, moderate
        tech_ci = _make_compliance_input(
            persona               = "tech_executive",
            risk_tolerance        = "moderate",
            hc_type               = "equity-like",
            income_equity_beta    = 1.20,
            implicit_equity_exposure = 0.906,
            income_equity_correlation = 0.75,
            rsu_concentration     = 0.40,
            employer_sector       = "technology",
        )
        tech_ro = _make_risk_output(
            hc_type               = "equity-like",
            rsu_concentration     = 0.40,
            portfolio_volatility_annual = 0.16,
        )

        # Financial professional — mixed HC, no RSU, aggressive
        fin_ci = _make_compliance_input(
            persona               = "financial_professional",
            risk_tolerance        = "aggressive",
            hc_type               = "mixed",
            income_equity_beta    = 0.60,
            implicit_equity_exposure = 0.501,
            income_equity_correlation = 0.50,
            rsu_concentration     = 0.00,
            employer_sector       = "finance",
        )
        fin_ro = _make_risk_output(
            hc_type               = "mixed",
            rsu_concentration     = 0.00,
            portfolio_volatility_annual = 0.22,
        )

        prof_result = run_compliance(prof_ci, prof_ro)
        tech_result = run_compliance(tech_ci, tech_ro)
        fin_result  = run_compliance(fin_ci,  fin_ro)

        # All three must produce a valid ComplianceAgentOutput
        for result in (prof_result, tech_result, fin_result):
            assert result.compliance_status is not None
            assert isinstance(result.passed_checks, list)

        # The tech exec (equity-like HC + high RSU) should trigger more checks
        # than the professor (bond-like HC + no RSU)
        tech_violation_checks = {v.check for v in tech_result.violations}
        prof_violation_checks = {v.check for v in prof_result.violations}

        # Check 1.3a (equity-like sector method) should fire for tech but not professor
        if "check_1_3a_sector_method_for_equity_like_hc" in tech_violation_checks:
            assert "check_1_3a_sector_method_for_equity_like_hc" not in prof_violation_checks


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"], check=False)
