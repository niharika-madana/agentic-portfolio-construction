from __future__ import annotations

import os
import pandas as pd
import anthropic

from portfolio_system.schemas import AllocationInput, AllocationOutput
from portfolio_system.core.allocation import run_allocation

_MODEL = "claude-sonnet-4-6"


def _get_client() -> anthropic.Anthropic | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key) if key else None


def _build_prompt(inp: AllocationInput, out: AllocationOutput) -> str:
    up  = inp.user_profile
    hc  = up.human_capital
    top = sorted(out.weights, key=lambda w: w.total_weight, reverse=True)[:5]
    fe  = out.portfolio_statistics.factor_exposures

    lines = [
        "You are a quantitative portfolio analyst writing a client-facing rationale.",
        "",
        "CLIENT PROFILE",
        f"  Financial wealth:    ${up.financial_wealth:>12,.0f}",
        f"  Human capital (PV): ${hc.present_value:>12,.0f}",
        f"  Risk profile:        {up.risk_profile.value}",
        f"  Years to retirement: {hc.years_to_retirement}",
        f"  Employer sector:     {hc.employer_sector}",
        f"  Income beta:         {hc.income_beta:.2f}",
        "",
        "PORTFOLIO DECISION",
        f"  Risky weight:   {out.risky_weight:.1%} of financial wealth",
        f"  Safe weight:    {out.safe_weight:.1%} of financial wealth",
        f"  Expected return:{out.portfolio_statistics.expected_return:.2%}  (annualized)",
        f"  Volatility:     {out.portfolio_statistics.volatility:.2%}  (annualized)",
        f"  Sharpe ratio:   {out.portfolio_statistics.sharpe_ratio:.2f}",
        "",
        "TOP 5 HOLDINGS (within risky sleeve)",
    ] + [f"  {w.ticker}: {w.total_weight:.1%}  (equilibrium: {w.equilibrium_baseline:.1%})" for w in top] + [
        "",
        "FACTOR EXPOSURES",
        f"  Market beta: {fe.market_beta:.2f}",
        f"  SMB (size):  {fe.smb:.2f}",
        f"  HML (value): {fe.hml:.2f}",
        f"  Momentum:    {fe.mom:.2f}",
        "",
        "Write a 4-sentence rationale covering:",
        "1. Why the risky/safe split is appropriate given the human capital profile.",
        "2. Which factor views drove the largest tilts away from market-cap weights.",
        "3. Key risk characteristics (volatility, Sharpe, factor exposures).",
        "4. Any notable concentration or sector bets and their justification.",
        "Be specific with numbers. Do not invent figures not shown above.",
    ]
    return "\n".join(lines)


def run_allocation_agent(
    allocation_input: AllocationInput,
    crsp_monthly: pd.DataFrame,
    ff_factors: pd.DataFrame,
    risk_free_rate: float,
    permno_map: dict[str, int],
) -> AllocationOutput:
    output = run_allocation(
        allocation_input, crsp_monthly, ff_factors, risk_free_rate, permno_map
    )

    _client = _get_client()
    if _client is None:
        top = sorted(output.weights, key=lambda w: w.total_weight, reverse=True)[:3]
        top_str = ", ".join(f"{w.ticker} {w.total_weight:.1%}" for w in top)
        rationale = (
            f"[No API key — template rationale] "
            f"Risky weight {output.risky_weight:.1%} based on BMS human capital model. "
            f"Top holdings: {top_str}. "
            f"Expected return {output.portfolio_statistics.expected_return:.2%}, "
            f"volatility {output.portfolio_statistics.volatility:.2%}, "
            f"Sharpe {output.portfolio_statistics.sharpe_ratio:.2f}."
        )
        return output.model_copy(update={"rationale": rationale})

    prompt = _build_prompt(allocation_input, output)
    message = _client.messages.create(
        model=_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    rationale = message.content[0].text.strip()

    return output.model_copy(update={"rationale": rationale})
