"""
adapters.py — bridges between the public contracts and internal pipeline types.

Conversions:
  ProfileAgentOutput + market cap weights → AllocationInput (for the BL optimizer)
  AllocationOutput                        → AllocationAgentOutput (for Compliance)

Data sourcing note:
  All historical price/return data comes from WRDS/CRSP via data.fetch.wrds.
  ETF_SECTORS is static — Compustat does not classify ETFs by GICS sector.
"""

from __future__ import annotations

from contracts import (
    AllocationAgentOutput, AllocationConstraint, AllocationInput, AllocationOutput,
    HumanCapitalInput, InstrumentUniverse, ProfileAgentOutput, RiskProfile,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Static ETF universe configuration
# ---------------------------------------------------------------------------

DEFAULT_TICKERS: list[str] = [
    # Broad equity
    "SPY", "IWM", "EFA", "EEM",
    # Fixed income
    "AGG", "TLT", "IEF", "SHY", "HYG", "LQD", "TIP",
    # Real assets — DJP excluded (ETN, not in CRSP)
    "GLD", "VNQ",
    # Sector ETFs
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE",
    # Cash proxy
    "BIL",
]

# ETFs are not classified in Compustat — this is the canonical sector source.
ETF_SECTORS: dict[str, str] = {
    "SPY":  "Broad Market",
    "IWM":  "Broad Market",
    "EFA":  "Broad Market",
    "EEM":  "Broad Market",
    "AGG":  "Fixed Income",
    "TLT":  "Fixed Income",
    "IEF":  "Fixed Income",
    "SHY":  "Fixed Income",
    "HYG":  "Fixed Income",
    "LQD":  "Fixed Income",
    "TIP":  "Fixed Income",
    "GLD":  "Real Assets",
    "VNQ":  "Real Estate",
    "XLK":  "Information Technology",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLE":  "Energy",
    "XLI":  "Industrials",
    "XLC":  "Communication Services",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "BIL":  "Cash",
}

# Sector proxy ETF — used as employer_ticker when the employer is not publicly traded.
_SECTOR_PROXY_ETF: dict[str, str] = {
    "Information Technology":  "XLK",
    "Health Care":             "XLV",
    "Education":               "XLV",
    "Financials":              "XLF",
    "Energy":                  "XLE",
    "Industrials":             "XLI",
    "Consumer Discretionary":  "XLY",
    "Consumer Staples":        "XLP",
    "Communication Services":  "XLC",
    "Utilities":               "XLU",
    "Real Estate":             "XLRE",
    "Government":              "SHY",
    "Nonprofit":               "SHY",
}


def _employer_ticker(sector: str) -> str:
    return _SECTOR_PROXY_ETF.get(sector, "SPY")


# ---------------------------------------------------------------------------
# ProfileAgentOutput → AllocationInput
# ---------------------------------------------------------------------------

def profile_to_allocation_input(
    profile: ProfileAgentOutput,
    discount_rate: float,
    market_cap_weights: dict[str, float],
    tickers: list[str] | None = None,
    flag_constraints: list[AllocationConstraint] | None = None,
    flag_iteration: int = 0,
) -> AllocationInput:
    """
    Convert ProfileAgentOutput into the AllocationInput the BL optimizer expects.

    Args:
        profile:            Output of the Profile Agent.
        discount_rate:      FRED DGS10 rate (from data.fetch.fred.latest_dgs10).
        market_cap_weights: CRSP-derived ticker weights from data.fetch.wrds.load_market_cap_weights.
        tickers:            ETF universe to use; defaults to DEFAULT_TICKERS.
        flag_constraints:   Injected by Risk Agent on FLAG re-entry.
        flag_iteration:     Current loop count (0 = first run).
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
    if flag_constraints is None:
        flag_constraints = []

    emp_ticker = _employer_ticker(profile.industry_exposure_sector)

    hc = HumanCapitalInput(
        present_value       = profile.human_capital_valuation,
        employer_ticker     = emp_ticker,
        employer_sector     = profile.industry_exposure_sector,
        income_volatility   = profile.income_volatility_sigma,
        income_beta         = profile.income_equity_beta,
        human_capital_type  = profile.human_capital_type.value,
        rsu_concentration   = profile.RSU_concentration,
        years_to_retirement = profile.investment_horizon_years,
        discount_rate       = discount_rate,
    )

    equity_target = profile.portfolio_equity_target
    if equity_target is None:
        equity_target = profile.effective_risk_budget - profile.implicit_equity_exposure

    user_profile = UserProfile(
        financial_wealth        = profile.financial_capital,
        human_capital           = hc,
        risk_profile            = RiskProfile(profile.risk_tolerance_level.value),
        portfolio_equity_target = equity_target,
    )

    # Use CRSP market cap weights; fall back to equal weight for any missing tickers
    n = len(tickers)
    weights = {t: market_cap_weights.get(t, 1.0 / n) for t in tickers}
    total   = sum(weights.values())
    weights = {t: w / total for t, w in weights.items()}

    universe = InstrumentUniverse(
        tickers            = tickers,
        market_cap_weights = weights,
        sectors            = {t: ETF_SECTORS[t] for t in tickers},
    )

    return AllocationInput(
        user_profile     = user_profile,
        universe         = universe,
        flag_constraints = flag_constraints,
        flag_iteration   = flag_iteration,
    )


# ---------------------------------------------------------------------------
# AllocationOutput → AllocationAgentOutput  (for Compliance)
# ---------------------------------------------------------------------------

def _position_rationale(w, sector: str, hc_type: str) -> str:
    """
    Distinct, position-specific rationale from one ticker's BL decomposition.

    Deterministic — every position cites its own equilibrium weight, factor-view
    tilt and human-capital offset, so each rationale is unique and grounded.
    This satisfies Compliance Check 2.2 (client-specific) and Check 2.6
    (differentiated across positions). When the Allocation Agent later generates
    per-ticker LLM prose from the same decomposition (SCOPE §3.9), it supersedes
    this deterministic floor.
    """
    if w.human_capital_offset > 0.001:
        offset = f"tilted up from its market-cap weight to diversify the client's {hc_type} human capital"
    elif w.human_capital_offset < -0.001:
        offset = f"tilted down to avoid compounding the client's {hc_type} human capital exposure"
    else:
        offset = f"held near its market-cap weight given the client's {hc_type} human capital profile"
    return (
        f"{w.ticker} ({sector}) — {w.total_weight:.1%} of the risky sleeve: "
        f"equilibrium {w.equilibrium_baseline:.1%}, factor-view tilt {w.view_tilt:+.1%}, "
        f"human capital offset {w.human_capital_offset:+.1%}; {offset}."
    )


def allocation_output_to_agent_output(
    allocation_output: AllocationOutput,
) -> AllocationAgentOutput:
    """
    Bridge the internal BL AllocationOutput into the Compliance-facing
    AllocationAgentOutput format.

    Each position gets its own rationale built from that position's weight
    decomposition — not one portfolio-level narrative copied onto every ticker,
    which fails Compliance Check 2.6 under Reg BI's per-recommendation care
    obligation. The portfolio-level LLM narrative stays on
    AllocationOutput.rationale for the executive summary.
    """
    weights = {w.ticker: w.total_weight for w in allocation_output.weights}
    hc_type = allocation_output.allocation_input.user_profile.human_capital.human_capital_type

    rationale = {
        w.ticker: _position_rationale(w, ETF_SECTORS.get(w.ticker, "Unknown"), hc_type)
        for w in allocation_output.weights
    }

    return AllocationAgentOutput(
        proposed_portfolio          = weights,
        allocation_rationale        = rationale,
        revision                    = allocation_output.allocation_input.flag_iteration,
        prior_risk_flags            = [],
        prior_compliance_violations = [],
    )
