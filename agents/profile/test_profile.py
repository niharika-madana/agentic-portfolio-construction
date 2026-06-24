"""
test_profile.py — Profile Agent unit tests
===========================================
No live API calls — all tests use hardcoded fixtures.

Coverage:
    1–3.  compute_human_capital — formula, edge cases
    4–6.  build_profile — formula checks (implicit_equity_exposure uses hc_share×β)
    7–9.  build_profile — enum lowercasing, total_wealth formula
    10.   lookup_hc_beta — returns (float, float) for each HC type
    11–13. to_profile_agent_output — round-trip, raises without beta,
           Pydantic catches wrong formula

Run with: pytest agents/profile/test_profile.py -v
"""

import pytest

from .hc_beta_table import lookup_hc_beta
from .human_capital import (
    INCOME_VOLATILITY_SIGMA,
    build_profile,
    compute_human_capital,
    to_profile_agent_output,
)

DISCOUNT_RATE = 0.044  # fixed for tests — no FRED call needed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_PERSONA = {
    "client_id": "test_persona",
    "name": "Test Client",
    "age": 40,
    "annual_salary": 100000,
    "years_to_retirement": 25,
    "career_type": "Technology",
    "income_stability": "Low",         # sigma=0.40, hc_type="equity-like"
    "industry_exposure_sector": "Technology",
    "financial_capital": 500000,
    "current_holdings": {"equities": 0.6, "bonds": 0.3, "cash": 0.1},
    "investment_horizon_years": 25,
    "risk_tolerance": "Aggressive",
    "liquidity_needs": "Medium",
    "investment_objective": "Growth",
    "RSU_concentration": 0.0,
}


@pytest.fixture
def valid_persona():
    return dict(VALID_PERSONA)


# ---------------------------------------------------------------------------
# 1–3. compute_human_capital
# ---------------------------------------------------------------------------

def test_hc_positive(valid_persona):
    hc = compute_human_capital(
        valid_persona["annual_salary"],
        valid_persona["years_to_retirement"],
        DISCOUNT_RATE,
    )
    assert hc > 0


def test_hc_zero_years():
    assert compute_human_capital(100000, 0, DISCOUNT_RATE) == 0.0


def test_hc_annuity_formula():
    """HC = salary × [1 − (1+r)^(−n)] / r"""
    salary, n, r = 100000, 25, 0.044
    expected = salary * (1 - (1 + r) ** (-n)) / r
    result = compute_human_capital(salary, n, r)
    assert abs(result - expected) < 1.0


# ---------------------------------------------------------------------------
# 4–6. build_profile — formula checks
# ---------------------------------------------------------------------------

def test_implicit_equity_exposure_uses_beta(valid_persona):
    """implicit_equity_exposure = hc_share × β  (contracts.py formula)."""
    beta = 1.20
    profile = build_profile(valid_persona, DISCOUNT_RATE, beta=beta, correlation=0.75)
    hc      = profile["human_capital_valuation"]
    tw      = profile["total_wealth"]
    expected = round((hc / tw) * beta, 3)
    assert profile["implicit_equity_exposure"] == expected


def test_effective_risk_budget_formula(valid_persona):
    """effective_risk_budget = (FC + HC × (1 − σ)) / total_wealth"""
    profile = build_profile(valid_persona, DISCOUNT_RATE, beta=1.20, correlation=0.75)
    hc    = profile["human_capital_valuation"]
    sigma = INCOME_VOLATILITY_SIGMA[valid_persona["income_stability"]]
    fc    = valid_persona["financial_capital"]
    tw    = profile["total_wealth"]
    expected = round((fc + hc * (1 - sigma)) / tw, 3)
    assert profile["effective_risk_budget"] == expected


def test_portfolio_equity_target_formula(valid_persona):
    """portfolio_equity_target = effective_risk_budget − implicit_equity_exposure"""
    profile = build_profile(valid_persona, DISCOUNT_RATE, beta=1.20, correlation=0.75)
    expected = round(
        profile["effective_risk_budget"] - profile["implicit_equity_exposure"], 3
    )
    assert profile["portfolio_equity_target"] == expected


# ---------------------------------------------------------------------------
# 7–9. build_profile — enum lowercasing, total_wealth formula
# ---------------------------------------------------------------------------

def test_total_wealth_is_fc_plus_hc(valid_persona):
    profile = build_profile(valid_persona, DISCOUNT_RATE, beta=1.20, correlation=0.75)
    assert profile["total_wealth"] == (
        profile["financial_capital"] + profile["human_capital_valuation"]
    )


def test_income_stability_lowercase(valid_persona):
    profile = build_profile(valid_persona, DISCOUNT_RATE, beta=1.20, correlation=0.75)
    assert profile["income_stability"] == "low"   # "Low" → "low"


def test_risk_tolerance_lowercase(valid_persona):
    profile = build_profile(valid_persona, DISCOUNT_RATE, beta=1.20, correlation=0.75)
    assert profile["risk_tolerance_level"] == "aggressive"  # "Aggressive" → "aggressive"


# ---------------------------------------------------------------------------
# 10. lookup_hc_beta — calibrated table replaces OLS regression
# ---------------------------------------------------------------------------

def test_lookup_hc_beta_returns_floats():
    for hc_type in ("bond-like", "mixed", "equity-like"):
        beta, corr = lookup_hc_beta(hc_type)
        assert isinstance(beta, float)
        assert isinstance(corr, float)
        assert 0.0 <= corr <= 1.0


# ---------------------------------------------------------------------------
# 11–13. to_profile_agent_output
# ---------------------------------------------------------------------------

def test_to_profile_agent_output_round_trip(valid_persona):
    """Build with beta → adapter → ProfileAgentOutput without ValidationError."""
    from contracts import ProfileAgentOutput
    profile = build_profile(valid_persona, DISCOUNT_RATE, beta=1.20, correlation=0.75)
    output  = to_profile_agent_output(profile)
    assert isinstance(output, ProfileAgentOutput)
    assert output.client_id == valid_persona["client_id"]
    assert output.income_equity_beta == profile["income_equity_beta"]
    assert output.implicit_equity_exposure == profile["implicit_equity_exposure"]


def test_to_profile_agent_output_raises_without_beta(valid_persona):
    """Adapter must raise ValueError when beta field is None."""
    profile = build_profile(valid_persona, DISCOUNT_RATE, beta=1.20, correlation=0.75)
    profile["income_equity_beta"] = None
    profile["implicit_equity_exposure"] = None
    with pytest.raises(ValueError, match="OLS beta estimation"):
        to_profile_agent_output(profile)


def test_pydantic_catches_wrong_formula(valid_persona):
    """
    Manually set implicit_equity_exposure using the OLD formula (hc×σ/tw).
    The contracts.py model_validator enforces hc_share×β and must reject this.
    """
    from pydantic import ValidationError
    beta = 1.20
    profile = build_profile(valid_persona, DISCOUNT_RATE, beta=beta, correlation=0.75)
    # Corrupt to old formula: hc×σ/tw
    hc    = profile["human_capital_valuation"]
    tw    = profile["total_wealth"]
    sigma = profile["income_volatility_sigma"]
    profile["implicit_equity_exposure"] = round(hc * sigma / tw, 3)
    with pytest.raises(ValidationError):
        to_profile_agent_output(profile)
