from __future__ import annotations

import os
import numpy as np
import pandas as pd
import anthropic

from portfolio_system.schemas import AllocationOutput, RiskDecision, RiskOutput
from portfolio_system.core.risk import run_risk

_MODEL = "claude-sonnet-4-6"


def _get_client() -> anthropic.Anthropic | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key) if key else None


def _build_prompt(alloc: AllocationOutput, risk: RiskOutput) -> str:
    rm   = risk.risk_metrics
    vc   = rm.var_cvar
    hca  = rm.hc_adjusted
    cf   = rm.concentration

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

    regime_cap_note = (
        f"  Market regime:    {rm.market_regime.value.upper()}  "
        f"(effective cap: {rm.effective_drawdown_cap:.1%}, "
        f"base cap widened by {rm.effective_drawdown_cap / rm.max_drawdown * 100 - 100:.0f}%)"
        if rm.market_regime.value != "normal" else
        f"  Market regime:    NORMAL  (standard cap: {rm.effective_drawdown_cap:.1%})"
    )

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
        regime_cap_note,
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
    ] + ([f"  {v.constraint_type.value}: {v.target} ({v.current_value:.1%} vs limit {v.limit:.1%})"
          for v in risk.constraints_violated] or ["  None"]) + [
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


def run_risk_agent(
    allocation_output: AllocationOutput,
    crsp_daily: pd.DataFrame,
    permno_map: dict[str, int],
) -> RiskOutput:
    risk_output = run_risk(allocation_output, crsp_daily, permno_map)

    _client = _get_client()
    if _client is None:
        rm = risk_output.risk_metrics
        reasoning_trace = (
            f"[No API key — template trace] "
            f"Decision: {risk_output.decision.value}. "
            f"Vol {rm.volatility:.2%}, VaR95 {rm.var_cvar.var_95:.2%}, "
            f"max drawdown {rm.max_drawdown:.2%} vs cap {rm.effective_drawdown_cap:.1%} "
            f"(regime: {rm.market_regime.value}). "
            f"Employer concentration {rm.hc_adjusted.employer_concentration:.1%}. "
            f"Violations: {len(risk_output.constraints_violated)}."
        )
        return risk_output.model_copy(update={"reasoning_trace": reasoning_trace})

    prompt = _build_prompt(allocation_output, risk_output)
    message = _client.messages.create(
        model=_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    reasoning_trace = message.content[0].text.strip()

    return risk_output.model_copy(update={"reasoning_trace": reasoning_trace})
