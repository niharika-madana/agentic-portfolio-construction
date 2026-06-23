"""
human_capital.py — Profile Agent
==================================
Human capital valuation, profile construction, and adapter to
contracts.py ProfileAgentOutput.

Exports:
  - INCOME_VOLATILITY_SIGMA
  - HUMAN_CAPITAL_TYPE
  - compute_human_capital()
  - build_profile()
  - to_profile_agent_output()
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contracts import ProfileAgentOutput

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INCOME_VOLATILITY_SIGMA = {
    "High":   0.05,   # tenured/government — very stable
    "Medium": 0.20,   # bonus-driven, market-correlated
    "Low":    0.40,   # RSU/layoff risk — highly variable
}

HUMAN_CAPITAL_TYPE = {
    "High":   "bond-like",
    "Medium": "mixed",
    "Low":    "equity-like",
}


# ---------------------------------------------------------------------------
# Human capital valuation
# ---------------------------------------------------------------------------

def compute_human_capital(annual_salary, years_to_retirement, discount_rate):
    """
    Annuity present-value formula:
        HC = Salary × [1 − (1 + r)^(−n)] / r

    where r = FRED DGS10 (10Y Treasury yield), n = years to retirement.
    """
    if years_to_retirement <= 0:
        return 0.0
    pv = annual_salary * (1 - (1 + discount_rate) ** (-years_to_retirement)) / discount_rate
    return round(pv, 2)


# ---------------------------------------------------------------------------
# Profile construction
# ---------------------------------------------------------------------------

def build_profile(persona, discount_rate, beta, correlation):
    """
    Compute all derived fields and return a profile dict.

    beta and correlation are always required — they come from
    hc_beta_table.lookup_hc_beta() before this function is called.
    There is no two-pass architecture; this is always a single call.

    Formulas (per contracts.py and README.md):
        implicit_equity_exposure = hc_share × β
        effective_risk_budget    = (FC + HC × (1 − σ)) / total_wealth
        portfolio_equity_target  = effective_risk_budget − implicit_equity_exposure

    Enum string values are lowercased to match contracts.py enum definitions.
    """
    hc = compute_human_capital(
        persona["annual_salary"],
        persona["years_to_retirement"],
        discount_rate,
    )
    total_wealth = persona["financial_capital"] + hc
    sigma   = INCOME_VOLATILITY_SIGMA[persona["income_stability"]]
    hc_type = HUMAN_CAPITAL_TYPE[persona["income_stability"]]
    hc_share = hc / total_wealth

    implicit_equity_exposure = round(hc_share * beta, 3)
    effective_risk_budget    = round(
        (persona["financial_capital"] + hc * (1 - sigma)) / total_wealth, 3
    )
    portfolio_equity_target  = round(effective_risk_budget - implicit_equity_exposure, 3)

    return {
        "client_id":                  persona["client_id"],
        "career_type":                persona["career_type"],
        "age":                        persona["age"],
        "financial_capital":          persona["financial_capital"],
        "human_capital_valuation":    hc,
        "total_wealth":               total_wealth,
        "human_capital_pct_of_total": round(hc / total_wealth * 100, 1),
        "income_volatility_sigma":    sigma,
        "human_capital_type":         hc_type,
        "income_equity_beta":         beta,
        "income_equity_correlation":  correlation,
        "implicit_equity_exposure":   implicit_equity_exposure,
        "effective_risk_budget":      effective_risk_budget,
        "portfolio_equity_target":    portfolio_equity_target,   # reference only; not in contracts.py
        "industry_exposure_sector":   persona["industry_exposure_sector"],
        "income_stability":           persona["income_stability"].lower(),
        "current_holdings":           persona["current_holdings"],
        "investment_horizon_years":   persona["investment_horizon_years"],
        "risk_tolerance_level":       persona["risk_tolerance"].lower(),
        "liquidity_needs":            persona["liquidity_needs"].lower(),
        "investment_objective":       persona["investment_objective"].lower(),
        "RSU_concentration":          persona["RSU_concentration"],
        "has_pension":                persona.get("has_pension", False),
    }


# ---------------------------------------------------------------------------
# Adapter — converts build_profile() dict to ProfileAgentOutput
# ---------------------------------------------------------------------------

def to_profile_agent_output(profile: dict) -> ProfileAgentOutput:
    """
    Convert a finished build_profile() dict into a validated ProfileAgentOutput.

    Only call this from the second build pass — after OLS beta estimation.
    Raises ValueError if beta or implicit_equity_exposure is None.
    Raises pydantic.ValidationError if any contracts.py constraint is violated
    (wrong formula, enum mismatch, holdings not summing to 1.0, etc.).
    """
    if profile.get("income_equity_beta") is None:
        raise ValueError(
            f"{profile['client_id']}: cannot build ProfileAgentOutput before "
            "OLS beta estimation — income_equity_beta is None. "
            "Call build_profile() with beta= set."
        )
    if profile.get("implicit_equity_exposure") is None:
        raise ValueError(
            f"{profile['client_id']}: implicit_equity_exposure is None — "
            "ensure build_profile() was called with beta set."
        )

    return ProfileAgentOutput(
        client_id                  = profile["client_id"],
        career_type                = profile["career_type"],
        age                        = profile["age"],
        financial_capital          = profile["financial_capital"],
        human_capital_valuation    = profile["human_capital_valuation"],
        total_wealth               = profile["total_wealth"],
        human_capital_pct_of_total = profile["human_capital_pct_of_total"],
        income_volatility_sigma    = profile["income_volatility_sigma"],
        income_equity_correlation  = profile["income_equity_correlation"],
        income_equity_beta         = profile["income_equity_beta"],
        implicit_equity_exposure   = profile["implicit_equity_exposure"],
        human_capital_type         = profile["human_capital_type"],
        income_stability           = profile["income_stability"],
        effective_risk_budget      = profile["effective_risk_budget"],
        industry_exposure_sector   = profile["industry_exposure_sector"],
        RSU_concentration          = profile["RSU_concentration"],
        has_pension                = profile.get("has_pension", False),
        current_holdings           = profile["current_holdings"],
        investment_horizon_years   = profile["investment_horizon_years"],
        risk_tolerance_level       = profile["risk_tolerance_level"],
        liquidity_needs            = profile["liquidity_needs"],
        investment_objective       = profile["investment_objective"],
        # portfolio_equity_target is NOT a field in ProfileAgentOutput —
        # the orchestrator computes it inline as:
        #   profile.effective_risk_budget - profile.implicit_equity_exposure
    )
