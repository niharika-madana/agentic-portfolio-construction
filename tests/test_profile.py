"""
test_profile.py — Profile Agent unit tests
===========================================
No live API calls — every test uses a fixed discount rate (4.4%) and hardcoded
BLS-grounded persona fixtures.

Beta and correlation are NOT passed in: build_profile() looks them up from the
calibrated hc_beta_table by human-capital type (single-pass architecture). The
fixtures therefore mirror the dicts produced by build_bls_personas().

Coverage (maps to 30JUN_ProfileAgent_Design.md §Unit Tests):
    1.  compute_human_capital — annuity formula
    2.  compute_human_capital — zero/at-retirement horizon
    3.  Software Developer p50 — HC formula, β=0.90, effective_salary incl. bonus,
        implicit_equity_exposure = HC_share × β, validates through the contract
    4.  Biology Professor p50 — bond-like, β=0.05, pension, and
        professor portfolio_equity_target > developer portfolio_equity_target
    5.  effective_risk_budget = (FC + HC×(1−σ)) / total_wealth
    6.  effective earnings include the bonus (HC scales with bonus_rate)
    7.  lookup_hc_beta — returns float β and ρ for each HC type
    8.  contract validator fires when total_wealth is inconsistent
    9.  contract validator fires when β disagrees with human_capital_type
    10. contract validator fires when current_holdings do not sum to 1.0
    11. to_profile_agent_output — round-trip returns a ProfileAgentOutput

Run with: pytest agents/profile/test_profile.py -v
"""

import pytest
from pydantic import ValidationError

from contracts import ProfileAgentOutput

from agents.profile.profile_model import (
    INCOME_VOLATILITY_SIGMA,
    lookup_hc_beta,
    build_profile,
    compute_human_capital,
    to_profile_agent_output,
)

DISCOUNT_RATE = 0.044  # fixed for tests — no FRED call needed


# ---------------------------------------------------------------------------
# Persona fixtures (shape mirrors build_bls_personas() output)
# ---------------------------------------------------------------------------

def _persona(
    *,
    client_id: str,
    career_type: str,
    age: int,
    annual_salary: float,
    bonus_rate: float,
    income_stability: str,
    sector: str,
    financial_capital: float,
    risk_tolerance: str,
    liquidity_needs: str = "medium",
    investment_objective: str = "growth",
    rsu_concentration: float = 0.0,
    has_pension: bool = False,
    current_holdings: dict | None = None,
) -> dict:
    """Build a raw persona dict with effective_salary derived from the bonus."""
    return {
        "client_id": client_id,
        "career_type": career_type,
        "age": age,
        "annual_salary": annual_salary,
        "bonus_rate": bonus_rate,
        "effective_salary": round(annual_salary * (1 + bonus_rate), 2),
        "years_to_retirement": 65 - age,
        "income_stability": income_stability,
        "industry_exposure_sector": sector,
        "financial_capital": financial_capital,
        "current_holdings": current_holdings
        or {"US_equity": 0.50, "intl_equity": 0.15, "bonds": 0.25, "cash": 0.10},
        "investment_horizon_years": 65 - age,
        "risk_tolerance": risk_tolerance,
        "liquidity_needs": liquidity_needs,
        "investment_objective": investment_objective,
        "RSU_concentration": rsu_concentration,
        "has_pension": has_pension,
    }


@pytest.fixture
def software_developer() -> dict:
    """SOC 15-1252, p50 — equity-like, β=0.90, bonus 8.5%."""
    return _persona(
        client_id="bls_15-1252_p50",
        career_type="Technology",
        age=38,
        annual_salary=132270,
        bonus_rate=0.085,
        income_stability="Low",
        sector="Technology",
        financial_capital=90000,
        risk_tolerance="aggressive",
    )


@pytest.fixture
def biology_professor() -> dict:
    """SOC 25-1042, p50 — bond-like, β=0.05, has pension, bonus 4.6%."""
    return _persona(
        client_id="bls_25-1042_p50",
        career_type="Academia",
        age=47,
        annual_salary=83920,
        bonus_rate=0.046,
        income_stability="High",
        sector="Education",
        financial_capital=200000,
        risk_tolerance="moderate",
        liquidity_needs="low",
        has_pension=True,
    )


# ---------------------------------------------------------------------------
# 1–2. compute_human_capital
# ---------------------------------------------------------------------------

def test_hc_annuity_formula():
    """HC = effective_salary × [1 − (1+r)^(−n)] / r."""
    salary, n, r = 143513.0, 27, 0.044
    expected = salary * (1 - (1 + r) ** (-n)) / r
    assert abs(compute_human_capital(salary, n, r) - expected) < 1.0


def test_hc_zero_years_at_retirement():
    assert compute_human_capital(100000, 0, DISCOUNT_RATE) == 0.0


# ---------------------------------------------------------------------------
# 3. Software Developer — known-persona regression test
# ---------------------------------------------------------------------------

def test_software_developer_profile(software_developer):
    profile = build_profile(software_developer, DISCOUNT_RATE)

    # Calibrated beta lookup → equity-like
    assert profile["income_equity_beta"] == 0.90
    assert profile["human_capital_type"] == "equity-like"
    assert profile["income_volatility_sigma"] == 0.40

    # HC uses effective_salary (base × (1 + bonus_rate))
    eff = round(132270 * 1.085, 2)
    expected_hc = compute_human_capital(eff, 65 - 38, DISCOUNT_RATE)
    assert abs(profile["human_capital_valuation"] - expected_hc) < 1.0

    # implicit_equity_exposure = HC_share × β
    hc_share = profile["human_capital_valuation"] / profile["total_wealth"]
    assert profile["implicit_equity_exposure"] == round(hc_share * 0.90, 3)

    # Validates cleanly through the shared contract
    output = to_profile_agent_output(profile)
    assert isinstance(output, ProfileAgentOutput)
    assert output.client_id == "bls_15-1252_p50"


# ---------------------------------------------------------------------------
# 4. Biology Professor — bond-like, and differentiation vs developer
# ---------------------------------------------------------------------------

def test_biology_professor_profile(biology_professor, software_developer):
    prof = build_profile(biology_professor, DISCOUNT_RATE)

    assert prof["income_equity_beta"] == 0.05
    assert prof["human_capital_type"] == "bond-like"
    assert prof["has_pension"] is True
    to_profile_agent_output(prof)  # validates

    dev = build_profile(software_developer, DISCOUNT_RATE)
    # Core thesis: bond-like career leaves far more room for portfolio equity.
    assert prof["portfolio_equity_target"] > dev["portfolio_equity_target"]


# ---------------------------------------------------------------------------
# 5–6. Formula checks
# ---------------------------------------------------------------------------

def test_effective_risk_budget_formula(software_developer):
    profile = build_profile(software_developer, DISCOUNT_RATE)
    hc = profile["human_capital_valuation"]
    fc = software_developer["financial_capital"]
    tw = profile["total_wealth"]
    sigma = INCOME_VOLATILITY_SIGMA[software_developer["income_stability"]]
    assert profile["effective_risk_budget"] == round((fc + hc * (1 - sigma)) / tw, 3)


def test_bonus_increases_human_capital(biology_professor):
    """Effective earnings include the bonus, so HC exceeds a base-salary-only PV."""
    with_bonus = build_profile(biology_professor, DISCOUNT_RATE)
    base_only = compute_human_capital(
        biology_professor["annual_salary"],
        biology_professor["years_to_retirement"],
        DISCOUNT_RATE,
    )
    assert with_bonus["human_capital_valuation"] > base_only


# ---------------------------------------------------------------------------
# 7. Calibrated beta table replaces OLS regression
# ---------------------------------------------------------------------------

def test_lookup_hc_beta_returns_floats():
    for hc_type in ("bond-like", "mixed", "equity-like"):
        cal = lookup_hc_beta(hc_type)
        assert isinstance(cal["beta"], float)
        assert isinstance(cal["correlation"], float)
        assert 0.0 <= cal["correlation"] <= 1.0


# ---------------------------------------------------------------------------
# 8–10. Contract validators reject malformed profiles
# ---------------------------------------------------------------------------

def test_total_wealth_validator_fires(software_developer):
    profile = build_profile(software_developer, DISCOUNT_RATE)
    profile["total_wealth"] = round(profile["total_wealth"] * 1.10, 2)  # 10% off
    with pytest.raises(ValidationError, match="total_wealth"):
        to_profile_agent_output(profile)


def test_hc_type_beta_consistency_validator_fires(software_developer):
    """β=0.10 with an equity-like label must be rejected (β≤0.30 → bond-like)."""
    profile = build_profile(software_developer, DISCOUNT_RATE)
    profile["income_equity_beta"] = 0.10
    hc_share = profile["human_capital_valuation"] / profile["total_wealth"]
    profile["implicit_equity_exposure"] = round(hc_share * 0.10, 3)  # keep IEE valid
    # human_capital_type stays "equity-like" → only the consistency check fires
    with pytest.raises(ValidationError, match="human_capital_type"):
        to_profile_agent_output(profile)


def test_holdings_sum_validator_fires(software_developer):
    bad = _persona(
        client_id="bad_holdings",
        career_type="Technology",
        age=38,
        annual_salary=132270,
        bonus_rate=0.085,
        income_stability="Low",
        sector="Technology",
        financial_capital=90000,
        risk_tolerance="aggressive",
        current_holdings={"US_equity": 0.50, "bonds": 0.20},  # sums to 0.70
    )
    profile = build_profile(bad, DISCOUNT_RATE)
    with pytest.raises(ValidationError, match="current_holdings"):
        to_profile_agent_output(profile)


# ---------------------------------------------------------------------------
# 11. Adapter round-trip
# ---------------------------------------------------------------------------

def test_to_profile_agent_output_round_trip(software_developer):
    profile = build_profile(software_developer, DISCOUNT_RATE)
    output = to_profile_agent_output(profile)
    assert output.income_equity_beta == profile["income_equity_beta"]
    assert output.implicit_equity_exposure == profile["implicit_equity_exposure"]
    assert output.human_capital_type.value == profile["human_capital_type"]
