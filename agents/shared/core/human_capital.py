from __future__ import annotations

import numpy as np

from contracts import RiskProfile


_RISK_AVERSION: dict[RiskProfile, float] = {
    RiskProfile.CONSERVATIVE: 5.0,
    RiskProfile.MODERATE:     3.0,
    RiskProfile.AGGRESSIVE:   2.0,
}


def merton_risky_share(
    expected_excess_return: float,
    portfolio_volatility: float,
    risk_profile: RiskProfile,
) -> float:
    """
    Optimal fraction of financial wealth in the risky portfolio (Merton 1971).
        alpha = (mu - r_f) / (gamma * sigma^2)
    Capped at 1.0 — no leverage.
    """
    gamma = _RISK_AVERSION[risk_profile]
    alpha = expected_excess_return / (gamma * portfolio_volatility ** 2)
    return float(min(alpha, 1.0))


def compute_w_fin(
    alpha: float,
    human_capital_pv: float,
    financial_wealth: float,
    income_beta: float,
) -> float:
    """
    Optimal risky weight on financial portfolio accounting for human capital.

    Risk-free HC (BMS 1992):  w_fin = alpha * (1 + H/W)
    Risky HC (Campbell & Viceira 2002):
        w_fin = alpha * (1 + H/W) - (H/W) * income_beta

    Clipped to [0, 1] — no leverage, no shorting the risky portfolio.
    """
    ratio = human_capital_pv / financial_wealth
    w_fin = alpha * (1.0 + ratio) - ratio * income_beta
    return float(np.clip(w_fin, 0.0, 1.0))


def total_wealth(financial_wealth: float, human_capital_pv: float) -> float:
    return financial_wealth + human_capital_pv


def hc_fraction(financial_wealth: float, human_capital_pv: float) -> float:
    return human_capital_pv / total_wealth(financial_wealth, human_capital_pv)


def effective_equity_exposure(
    portfolio_equity_weight: float,
    financial_wealth: float,
    human_capital_pv: float,
    income_beta: float,
) -> float:
    tw = total_wealth(financial_wealth, human_capital_pv)
    financial_equity = portfolio_equity_weight * financial_wealth
    hc_equity_equiv  = income_beta * human_capital_pv
    return (financial_equity + hc_equity_equiv) / tw


def employer_concentration(
    employer_financial_weight: float,
    financial_wealth: float,
    human_capital_pv: float,
) -> float:
    tw = total_wealth(financial_wealth, human_capital_pv)
    employer_financial = employer_financial_weight * financial_wealth
    return (employer_financial + human_capital_pv) / tw


def economic_sector_exposures(
    financial_weights: dict[str, float],
    sectors: dict[str, str],
    financial_wealth: float,
    human_capital_pv: float,
    employer_sector: str,
) -> dict[str, float]:
    tw = total_wealth(financial_wealth, human_capital_pv)
    sector_values: dict[str, float] = {}
    for ticker, weight in financial_weights.items():
        sector = sectors.get(ticker, "Unknown")
        sector_values[sector] = sector_values.get(sector, 0.0) + weight * financial_wealth
    sector_values[employer_sector] = sector_values.get(employer_sector, 0.0) + human_capital_pv
    return {sector: value / tw for sector, value in sector_values.items()}
