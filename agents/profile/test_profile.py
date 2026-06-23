"""
test_profile.py — Profile Agent unit tests
===========================================
No live API calls — all tests use hardcoded fixtures.

Coverage:
    1–3.  compute_human_capital — formula, edge cases
    4–6.  build_profile — formula checks (implicit_equity_exposure uses hc_share×β)
    7.    build_profile — None when beta not yet estimated (first pass)
    8.    build_profile — enum values are lowercase
    9.    build_profile — effective_risk_budget and total_wealth formulas
   10–11. validate_raw_persona — holdings sum, missing fields, invalid stability
   12–13. sanity_check — pass and flag cases
   14–16. estimate_hc_beta — FF12 match, CRSP fallback, ff_industry lookup
   17–19. to_profile_agent_output — round-trip, raises without beta,
          Pydantic catches wrong formula
   20–22. baseline personas — all build, HC type labels, equity target ordering

Run with: pytest agents/profile/test_profile.py -v
"""

import numpy as np
import pandas as pd
import pytest

from .beta import estimate_hc_beta, get_ff_industry
from .human_capital import (
    INCOME_VOLATILITY_SIGMA,
    build_profile,
    compute_human_capital,
    sanity_check,
    to_profile_agent_output,
)
from .personas import RAW_PERSONAS, validate_raw_persona

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


@pytest.fixture
def mock_ff_df():
    """Minimal FF12 DataFrame with BusEq column (Technology maps here)."""
    dates = pd.date_range("2014-01-01", periods=120, freq="MS")
    np.random.seed(42)
    return pd.DataFrame({"BusEq": np.random.normal(0.01, 0.05, 120)}, index=dates)


@pytest.fixture
def mock_crsp():
    """Minimal CRSP DataFrame for fallback beta estimation."""
    dates = pd.date_range("2014-01-01", periods=120, freq="MS")
    np.random.seed(42)
    return pd.DataFrame({"vwretd": np.random.normal(0.01, 0.04, 120)}, index=dates)


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
# 7. build_profile — None when no beta (first pass)
# ---------------------------------------------------------------------------

def test_implicit_equity_exposure_none_without_beta(valid_persona):
    """First pass (beta=None) must produce None for implicit_equity_exposure."""
    profile = build_profile(valid_persona, DISCOUNT_RATE)
    assert profile["implicit_equity_exposure"] is None
    assert profile["portfolio_equity_target"] is None


def test_total_wealth_is_fc_plus_hc(valid_persona):
    profile = build_profile(valid_persona, DISCOUNT_RATE)
    assert profile["total_wealth"] == (
        profile["financial_capital"] + profile["human_capital_valuation"]
    )


# ---------------------------------------------------------------------------
# 8. build_profile — enum values are lowercase
# ---------------------------------------------------------------------------

def test_income_stability_lowercase(valid_persona):
    profile = build_profile(valid_persona, DISCOUNT_RATE)
    assert profile["income_stability"] == "low"   # "Low" → "low"


def test_risk_tolerance_lowercase(valid_persona):
    profile = build_profile(valid_persona, DISCOUNT_RATE)
    assert profile["risk_tolerance_level"] == "aggressive"  # "Aggressive" → "aggressive"


def test_liquidity_needs_lowercase(valid_persona):
    profile = build_profile(valid_persona, DISCOUNT_RATE)
    assert profile["liquidity_needs"] == "medium"  # "Medium" → "medium"


def test_investment_objective_lowercase(valid_persona):
    profile = build_profile(valid_persona, DISCOUNT_RATE)
    assert profile["investment_objective"] == "growth"  # "Growth" → "growth"


# ---------------------------------------------------------------------------
# 10–11. validate_raw_persona
# ---------------------------------------------------------------------------

def test_holdings_sum_warning(valid_persona, capsys):
    persona = dict(valid_persona)
    persona["current_holdings"] = {"equities": 0.6, "bonds": 0.3, "cash": 0.05}  # sums to 0.95
    validate_raw_persona(persona)
    assert "holdings sum" in capsys.readouterr().out


def test_holdings_sum_ok(valid_persona, capsys):
    validate_raw_persona(valid_persona)
    assert "holdings sum" not in capsys.readouterr().out


def test_missing_field_triggers_warning(valid_persona, capsys):
    persona = dict(valid_persona)
    del persona["liquidity_needs"]
    validate_raw_persona(persona)
    assert "missing field" in capsys.readouterr().out


def test_invalid_income_stability_returns_false(valid_persona, capsys):
    persona = dict(valid_persona)
    persona["income_stability"] = "VeryHigh"
    result = validate_raw_persona(persona)
    assert result is False
    assert "invalid income_stability" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 12–13. sanity_check
# ---------------------------------------------------------------------------

def test_sanity_check_pass(valid_persona, capsys):
    profile = build_profile(valid_persona, DISCOUNT_RATE)
    sanity_check(profile, valid_persona, DISCOUNT_RATE)
    assert "OK" in capsys.readouterr().out


def test_sanity_check_flag(valid_persona, capsys):
    profile = build_profile(valid_persona, DISCOUNT_RATE)
    profile["human_capital_valuation"] = profile["human_capital_valuation"] * 10
    sanity_check(profile, valid_persona, DISCOUNT_RATE)
    assert "FLAG" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 14–16. estimate_hc_beta
# ---------------------------------------------------------------------------

def test_beta_with_ff12_match(valid_persona, mock_ff_df, mock_crsp):
    result = estimate_hc_beta(valid_persona, mock_ff_df, mock_crsp, seed=42)
    assert result["ff_industry"] == "BusEq"
    assert isinstance(result["income_equity_beta"], float)
    assert isinstance(result["income_equity_correlation"], float)


def test_beta_fallback_to_crsp(valid_persona, mock_ff_df, mock_crsp):
    persona = dict(valid_persona, career_type="UnknownCareer")
    result = estimate_hc_beta(persona, mock_ff_df, mock_crsp, seed=42)
    assert "CRSP" in result["ff_industry"]
    assert isinstance(result["income_equity_beta"], float)


def test_get_ff_industry_known():
    assert get_ff_industry("Technology") == "BusEq"
    assert get_ff_industry("Finance") == "Money"
    assert get_ff_industry("Academia") == "Hlth"


def test_get_ff_industry_unknown():
    assert get_ff_industry("Underwater Basket Weaving") is None


# ---------------------------------------------------------------------------
# 17–19. to_profile_agent_output
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
    """Calling adapter on first-pass profile (beta=None) must raise ValueError."""
    profile = build_profile(valid_persona, DISCOUNT_RATE)  # no beta
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


# ---------------------------------------------------------------------------
# 20–22. Baseline personas
# ---------------------------------------------------------------------------

def test_three_baseline_personas_build():
    for persona in RAW_PERSONAS:
        profile = build_profile(persona, DISCOUNT_RATE)
        assert profile["client_id"] == persona["client_id"]
        assert profile["total_wealth"] > 0
        assert profile["effective_risk_budget"] > 0


def test_professor_bond_like():
    professor = next(p for p in RAW_PERSONAS if "professor" in p["client_id"])
    profile = build_profile(professor, DISCOUNT_RATE)
    assert profile["human_capital_type"] == "bond-like"


def test_tech_exec_equity_like():
    exec_ = next(p for p in RAW_PERSONAS if "tech" in p["client_id"])
    profile = build_profile(exec_, DISCOUNT_RATE)
    assert profile["human_capital_type"] == "equity-like"


def test_professor_higher_equity_target_than_tech_exec():
    """
    Professor's bond-like HC → higher portfolio_equity_target than tech exec.
    Uses realistic betas from the README persona table.
    """
    professor = next(p for p in RAW_PERSONAS if "professor" in p["client_id"])
    exec_     = next(p for p in RAW_PERSONAS if "tech" in p["client_id"])
    prof_profile = build_profile(professor, DISCOUNT_RATE, beta=0.05,  correlation=-0.05)
    exec_profile = build_profile(exec_,     DISCOUNT_RATE, beta=1.20,  correlation=0.75)
    assert prof_profile["portfolio_equity_target"] > exec_profile["portfolio_equity_target"]
