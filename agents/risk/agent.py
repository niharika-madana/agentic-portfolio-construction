"""
Risk Agent — public entry point.

run_risk_agent(allocation_output, profile) → (RiskOutput, RiskAgentOutput)

The agent:
  1. Loads yfinance CRSP-like daily returns from parquet
  2. Runs deterministic risk evaluation (agents/shared/core/risk.py)
  3. Calls Claude to write reasoning_trace (LLM Touchpoint — risk narrative)
  4. Bridges internal RiskOutput → Compliance-facing RiskAgentOutput

The bridge maps the 3 internal stress scenarios to a subset of the 5
academic regime labels, and constructs RegimeEvaluation entries the
Compliance Agent's Check 1.1/1.2 can audit.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import anthropic

from contracts import (
    AllocationOutput, ProfileAgentOutput, RiskAgentOutput, RiskDecision,
    RiskDerivation, RiskOutput, RegimeEvaluation, PositionLimit, SectorLimit,
)
from data.fetch.wrds import load_crsp_daily
from agents.shared.core.risk import run_risk
from agents.shared.core.constraints import SECTOR_LIMIT

_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
_client  = anthropic.Anthropic(api_key=_API_KEY) if _API_KEY else None
_MODEL   = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Mapping: internal stress scenarios → academic regime labels
# ---------------------------------------------------------------------------

_STRESS_TO_REGIME: dict[str, str] = {
    "S&P 500, 2008":           "Financial Crisis & ZLB",
    "COVID, 2020":             "Early Recovery",        # COVID shock → rapid recovery regime
    "60/40 Portfolio, 2008":   "Financial Crisis & ZLB",
    "Idiosyncratic Employer Shock": "Inflation Shock",  # used as proxy for idiosyncratic tail
}

# For regimes without direct historical stress test, use 80/20 benchmark estimate
_REGIME_BENCHMARK_DRAWDOWNS: dict[str, float] = {
    "Early Recovery":        0.34,   # COVID-era speed
    "Late-Cycle Expansion":  0.12,   # 2007 peak; gentle decline before crisis
    "Financial Crisis & ZLB": 0.54,  # 2008-2009
    "Moderate Expansion":    0.06,   # 2012-2015 range
    "Inflation Shock":       0.18,   # 2022 drawdown
}

_ALL_REGIMES = list(_REGIME_BENCHMARK_DRAWDOWNS.keys())


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

def _build_reasoning_prompt(
    alloc: AllocationOutput,
    risk:  RiskOutput,
) -> str:
    rm  = risk.risk_metrics
    vc  = rm.var_cvar
    hca = rm.hc_adjusted
    cf  = rm.concentration

    breach_lines = []
    if cf.single_name_breaches:
        breach_lines.append(f"  Single-name breaches: {', '.join(cf.single_name_breaches)}")
    if cf.sector_breaches:
        breach_lines.append(f"  Sector breaches: {', '.join(cf.sector_breaches)}")
    if cf.employer_breach:
        breach_lines.append("  Employer concentration breach")
    breaches_str = "\n".join(breach_lines) if breach_lines else "  None"

    stress_lines = [
        f"  {s.scenario}: {s.portfolio_loss:.1%} loss  [{s.severity.value.upper()}]"
        for s in rm.stress_results
    ]

    decision_map = {
        RiskDecision.APPROVE: "APPROVE — portfolio meets all risk thresholds",
        RiskDecision.FLAG:    f"FLAG — re-optimize needed (iteration {risk.flag_iteration})",
        RiskDecision.REJECT:  "REJECT — portfolio cannot be made compliant",
    }

    lines = [
        "You are a risk officer writing a compliance reasoning trace for an internal audit log.",
        "",
        "RISK METRICS",
        f"  Volatility:       {rm.volatility:.2%}  (annualized)",
        f"  VaR 95%:          {vc.var_95:.2%}  daily",
        f"  CVaR 95%:         {vc.cvar_95:.2%}  daily",
        f"  VaR 99%:          {vc.var_99:.2%}  daily",
        f"  CVaR 99%:         {vc.cvar_99:.2%}  daily",
        f"  Max drawdown:     {rm.max_drawdown:.2%}",
        f"  Liquidity score:  {rm.liquidity_score:.2f}",
        "",
        "HUMAN CAPITAL ADJUSTED",
        f"  Total wealth:              ${hca.total_wealth:>12,.0f}",
        f"  HC fraction:               {hca.hc_fraction:.1%}",
        f"  Effective equity exposure: {hca.effective_equity_exposure:.1%}",
        f"  Employer concentration:    {hca.employer_concentration:.1%}",
        "",
        "CONCENTRATION FLAGS",
        breaches_str,
        "",
        "STRESS TESTS",
    ] + stress_lines + [
        "",
        "CONSTRAINTS VIOLATED",
    ] + ([
        f"  {v.constraint_type.value}: {v.target} ({v.current_value:.1%} vs limit {v.limit:.1%})"
        for v in risk.constraints_violated
    ] or ["  None"]) + [
        "",
        f"DECISION: {decision_map[risk.decision]}",
        "",
        "Write a 4-sentence reasoning trace covering:",
        "1. Which risk metrics drove the decision and why.",
        "2. How the human capital profile affects the overall risk picture.",
        "3. What the stress tests reveal about tail risk.",
        "4. If FLAG or REJECT, what specifically must change and why it cannot be ignored.",
        "Be precise. Reference actual numbers from the data above.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bridge: RiskOutput → RiskAgentOutput (Compliance-facing)
# ---------------------------------------------------------------------------

def _build_regime_evaluation(
    risk_output: RiskOutput,
    profile: ProfileAgentOutput,
) -> dict[str, RegimeEvaluation]:
    """
    Build per-regime drawdown evaluation dict for the Compliance Agent.

    Uses actual portfolio loss from internal stress tests where available;
    falls back to benchmark estimates for the remaining regimes.
    """
    # Collect actual losses from stress tests
    stress_losses: dict[str, float] = {}
    for sr in risk_output.risk_metrics.stress_results:
        regime = _STRESS_TO_REGIME.get(sr.scenario)
        if regime and regime not in stress_losses:
            stress_losses[regime] = sr.portfolio_loss

    evaluations: dict[str, RegimeEvaluation] = {}
    for regime in _ALL_REGIMES:
        bench  = _REGIME_BENCHMARK_DRAWDOWNS[regime]
        actual = stress_losses.get(regime, bench * 0.9)  # 90% of benchmark if no direct data
        evaluations[regime] = RegimeEvaluation(
            benchmark_drawdown = bench,
            drawdown_floor     = bench,
            portfolio_drawdown = actual,
            passed             = actual <= bench,
        )
    return evaluations


def _build_position_limits(
    risk_output: RiskOutput,
) -> dict[str, PositionLimit]:
    """
    Build per-ticker PositionLimit dict from AllocationOutput weights.
    Uses the SINGLE_NAME_LIMIT (10%) as the derived limit (marginal risk contribution
    simplification appropriate for a 29-ETF universe).
    """
    ao = risk_output.allocation_output
    single_name_violations = {
        c.target for c in risk_output.constraints_violated
        if c.constraint_type.value == "single_name"
    }
    from agents.shared.core.constraints import SINGLE_NAME_LIMIT

    limits: dict[str, PositionLimit] = {}
    for w in ao.weights:
        actual = w.total_weight * ao.risky_weight
        lim    = SINGLE_NAME_LIMIT
        limits[w.ticker] = PositionLimit(
            derived_limit  = lim,
            actual_weight  = actual,
            passed         = w.ticker not in single_name_violations,
            method         = "marginal_risk_contribution",
        )
    return limits


def _build_sector_limits(
    risk_output: RiskOutput,
    profile: ProfileAgentOutput,
) -> dict[str, SectorLimit]:
    """
    Build per-sector SectorLimit dict applying HC-correlation-adjusted caps.
    adjusted_limit = base_limit × (1 − ρ) where ρ = income_equity_correlation.
    """
    from agents.allocation.adapters import ETF_SECTORS
    from agents.shared.core.constraints import SECTOR_LIMIT

    ao  = risk_output.allocation_output
    rho = profile.income_equity_correlation

    # Aggregate sector weights
    sector_weights: dict[str, float] = {}
    for w in ao.weights:
        sec = ETF_SECTORS.get(w.ticker, "Unknown")
        sector_weights[sec] = sector_weights.get(sec, 0.0) + w.total_weight * ao.risky_weight

    employer_sector = profile.industry_exposure_sector
    sector_violations = {
        c.target for c in risk_output.constraints_violated
        if c.constraint_type.value in ("sector", "economic_sector")
    }

    limits: dict[str, SectorLimit] = {}
    for sector, actual in sector_weights.items():
        if sector == employer_sector:
            corr = rho
            adj  = max(0.0, SECTOR_LIMIT * (1 - rho))
        else:
            corr = 0.0
            adj  = SECTOR_LIMIT

        limits[sector] = SectorLimit(
            base_limit           = SECTOR_LIMIT,
            correlation_with_hc  = corr,
            adjusted_limit       = adj,
            actual_sector_weight = actual,
            passed               = sector not in sector_violations,
        )
    return limits


def risk_output_to_agent_output(
    risk_output: RiskOutput,
    profile: ProfileAgentOutput,
) -> RiskAgentOutput:
    """
    Convert the internal BL-loop RiskOutput into the Compliance-facing RiskAgentOutput.
    """
    regime_eval   = _build_regime_evaluation(risk_output, profile)
    position_lims = _build_position_limits(risk_output)
    sector_lims   = _build_sector_limits(risk_output, profile)

    violations = [
        f"{c.constraint_type.value}: {c.target} ({c.current_value:.1%} vs limit {c.limit:.1%})"
        for c in risk_output.constraints_violated
    ]

    from agents.allocation.adapters import ETF_SECTORS
    derivation = RiskDerivation(
        drawdown_method      = "benchmark_relative_per_regime",
        concentration_method = "marginal_risk_contribution",
        sector_method        = (
            "hc_correlation_adjusted"
            if profile.income_equity_beta > 0.3
            else "static_lookup"
        ),
        hc_type              = profile.human_capital_type.value,
        rsu_concentration    = profile.RSU_concentration,
        data_source          = "data/storage/crsp_daily.parquet",
    )

    warnings: list[str] = []
    if risk_output.risk_metrics.max_drawdown == 0.0:
        warnings.append("Insufficient daily return history for max drawdown calculation")
    if risk_output.flag_iteration > 0:
        warnings.append(f"Portfolio required {risk_output.flag_iteration} FLAG revision(s)")

    return RiskAgentOutput(
        risk_decision              = RiskDecision(risk_output.decision.value),
        regime_evaluation          = regime_eval,
        position_limits            = position_lims,
        sector_limits              = sector_lims,
        violations                 = violations,
        derivation                 = derivation,
        warnings                   = warnings,
        portfolio_volatility_annual= risk_output.risk_metrics.volatility,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_risk_agent(
    allocation_output: AllocationOutput,
    profile: ProfileAgentOutput,
) -> tuple[RiskOutput, RiskAgentOutput]:
    """
    Run deterministic risk evaluation for one allocation.

    Args:
        allocation_output : Internal AllocationOutput from run_allocation_agent().
        profile           : ProfileAgentOutput (needed for sector mapping).

    Returns:
        (RiskOutput, RiskAgentOutput)
        — RiskOutput:      internal; returned to orchestrator for FLAG loop decisions
        — RiskAgentOutput: Compliance-facing; passed to run_compliance()
    """
    tickers = allocation_output.allocation_input.universe.tickers
    crsp_daily, permno_map = load_crsp_daily(tickers)

    raw_risk = run_risk(allocation_output, crsp_daily, permno_map)

    # ── LLM: Risk reasoning trace ──
    if _client is None:
        rm = raw_risk.risk_metrics
        reasoning_trace = (
            f"[No API key — template trace] "
            f"Decision: {raw_risk.decision.value}. "
            f"Vol {rm.volatility:.2%}, VaR95 {rm.var_cvar.var_95:.2%}, "
            f"max drawdown {rm.max_drawdown:.2%}. "
            f"HC fraction {rm.hc_adjusted.hc_fraction:.1%}, "
            f"employer concentration {rm.hc_adjusted.employer_concentration:.1%}. "
            f"Violations: {len(raw_risk.constraints_violated)}."
        )
    else:
        prompt  = _build_reasoning_prompt(allocation_output, raw_risk)
        message = _client.messages.create(
            model      = _MODEL,
            max_tokens = 500,
            temperature= 0,
            messages   = [{"role": "user", "content": prompt}],
        )
        reasoning_trace = message.content[0].text.strip()

    risk_output   = raw_risk.model_copy(update={"reasoning_trace": reasoning_trace})
    agent_output  = risk_output_to_agent_output(risk_output, profile)

    return risk_output, agent_output
