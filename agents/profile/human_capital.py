"""
human_capital.py — HC valuation, profile assembly, and contract adapter.

Core formulas (25JUN_ProfileAgent_Design.md):

  Effective Annual Earnings = Annual Salary × (1 + bonus_rate)
  HC = Effective Earnings × [1 − (1 + r)^(−n)] / r        (annuity PV)

  HC_share                 = HC / Total Wealth
  Implicit Equity Exposure = HC_share × β
  Effective Risk Budget    = (FC + HC × (1 − σ)) / Total Wealth
  Portfolio Equity Target  = Effective Risk Budget − Implicit Equity Exposure

Single-pass architecture: β is known (from the calibrated table) before
build_profile() runs. to_profile_agent_output() validates against the shared
contracts.ProfileAgentOutput so the result drops straight into the orchestrator.
"""

from __future__ import annotations

from contracts import ProfileAgentOutput

from agents.profile.hc_beta_table import lookup_hc_beta, sigma_for_stability, hc_type_for_stability

# contracts.IncomeStability is lowercase ("high"/"medium"/"low"); personas use
# the capitalised lookup-table keys ("High"/"Medium"/"Low").
_STABILITY_TO_CONTRACT = {"High": "high", "Medium": "medium", "Low": "low"}


def compute_human_capital(
    effective_salary: float, years_to_retirement: int, discount_rate: float
) -> float:
    """
    Present value of the future earnings stream (base + bonus), discounted at
    the FRED DGS10 10Y Treasury yield.

        HC = effective_salary × [1 − (1 + r)^(−n)] / r

    Returns 0.0 for a non-positive horizon (already at/after retirement).
    """
    if years_to_retirement <= 0:
        return 0.0
    pv = effective_salary * (1 - (1 + discount_rate) ** (-years_to_retirement)) / discount_rate
    return round(pv, 2)


def build_profile(persona: dict, discount_rate: float) -> dict:
    """
    Derive all computed fields from a raw BLS persona dict, returning a flat
    dict ready for Pydantic validation against contracts.ProfileAgentOutput.
    """
    hc = compute_human_capital(
        persona["effective_salary"],
        persona["years_to_retirement"],
        discount_rate,
    )
    fc = persona["financial_capital"]
    total_wealth = round(fc + hc, 2)

    sigma = sigma_for_stability(persona["income_stability"])
    hc_type = hc_type_for_stability(persona["income_stability"])
    cal = lookup_hc_beta(hc_type)
    beta = cal["beta"]
    correlation = cal["correlation"]

    hc_share = hc / total_wealth
    implicit_equity_exposure = round(hc_share * beta, 3)
    effective_risk_budget = round((fc + hc * (1 - sigma)) / total_wealth, 3)
    portfolio_equity_target = round(effective_risk_budget - implicit_equity_exposure, 3)

    return {
        "client_id":                  persona["client_id"],
        "career_type":                persona["career_type"],
        "age":                        persona["age"],
        "financial_capital":          fc,
        "human_capital_valuation":    hc,
        "total_wealth":               total_wealth,
        "human_capital_pct_of_total": round(hc / total_wealth * 100, 1),
        "income_volatility_sigma":    sigma,
        "income_equity_beta":         beta,
        "income_equity_correlation":  correlation,
        "implicit_equity_exposure":   implicit_equity_exposure,
        "human_capital_type":         hc_type,
        "income_stability":           _STABILITY_TO_CONTRACT[persona["income_stability"]],
        "effective_risk_budget":      effective_risk_budget,
        "portfolio_equity_target":    portfolio_equity_target,
        "industry_exposure_sector":   persona["industry_exposure_sector"],
        "RSU_concentration":          persona["RSU_concentration"],
        "has_pension":                persona["has_pension"],
        "bonus_rate":                 persona["bonus_rate"],
        "current_holdings":           persona["current_holdings"],
        "investment_horizon_years":   persona["investment_horizon_years"],
        "risk_tolerance_level":       persona["risk_tolerance"],
        "liquidity_needs":            persona["liquidity_needs"],
        "investment_objective":       persona["investment_objective"],
    }


def to_profile_agent_output(profile_dict: dict) -> ProfileAgentOutput:
    """
    Validate a profile dict through contracts.ProfileAgentOutput.

    Raises pydantic.ValidationError immediately if any formula constraint is
    violated (implicit_equity_exposure, holdings sum, β/hc_type consistency,
    total-wealth consistency).
    """
    return ProfileAgentOutput(**profile_dict)
