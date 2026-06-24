"""
Allocation Agent — public entry point.

run_allocation_agent(profile, discount_rate, ff_factors, flag_constraints, flag_iteration)
    → AllocationOutput        (internal, passed to Risk Agent)
    → AllocationAgentOutput   (Compliance-facing)

The agent:
  1. Converts ProfileAgentOutput → AllocationInput via adapters.py
  2. Loads CRSP monthly returns and FF risk factors from parquet cache
  3. Runs the Black-Litterman optimizer (agents/shared/core/allocation.py)
  4. Calls Claude to generate rationale text (LLM Touchpoint 1)
  5. Returns both the internal AllocationOutput and the Compliance-facing AllocationAgentOutput

Data source: WRDS/CRSP via data.fetch.wrds (parquet cache must be populated first).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import anthropic
import pandas as pd

from contracts import (
    AllocationAgentOutput, AllocationConstraint, AllocationInput,
    AllocationOutput, ProfileAgentOutput,
)
from agents.allocation.adapters import (
    DEFAULT_TICKERS,
    allocation_output_to_agent_output,
    profile_to_allocation_input,
)
from agents.shared.core.allocation import run_allocation
from data.fetch.wrds import load_crsp_monthly, load_ff_factors, load_market_cap_weights

_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
_client  = anthropic.Anthropic(api_key=_API_KEY) if _API_KEY else None
_MODEL   = "claude-sonnet-4-6"


def _build_rationale_prompt(inp: AllocationInput, out: AllocationOutput) -> str:
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
        f"  Risky weight:    {out.risky_weight:.1%} of financial wealth",
        f"  Safe weight:     {out.safe_weight:.1%} of financial wealth",
        f"  Expected return: {out.portfolio_statistics.expected_return:.2%}  (annualized)",
        f"  Volatility:      {out.portfolio_statistics.volatility:.2%}  (annualized)",
        f"  Sharpe ratio:    {out.portfolio_statistics.sharpe_ratio:.2f}",
        "",
        "TOP 5 HOLDINGS (within risky sleeve)",
    ] + [
        f"  {w.ticker}: {w.total_weight:.1%}  (equilibrium: {w.equilibrium_baseline:.1%},"
        f" view tilt: {w.view_tilt:+.1%}, HC offset: {w.human_capital_offset:+.1%})"
        for w in top
    ] + [
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
        "Reference the client's income beta, HC present value, and at least two specific tickers.",
        "Be precise with numbers. Do not invent figures not shown above.",
    ]
    return "\n".join(lines)


def run_allocation_agent(
    profile: ProfileAgentOutput,
    discount_rate: float,
    ff_factors: pd.DataFrame | None = None,
    flag_constraints: list[AllocationConstraint] | None = None,
    flag_iteration: int = 0,
    tickers: list[str] | None = None,
) -> tuple[AllocationOutput, AllocationAgentOutput]:
    """
    Run Black-Litterman allocation for one client profile.

    Args:
        profile:          ProfileAgentOutput from the Profile Agent.
        discount_rate:    FRED DGS10 rate (annual decimal).
        ff_factors:       Pre-loaded FF risk factors DataFrame. Auto-loaded if None.
        flag_constraints: FLAG constraints from Risk Agent on re-entry.
        flag_iteration:   Loop count (0 = first run).
        tickers:          ETF universe. Defaults to DEFAULT_TICKERS.

    Returns:
        (AllocationOutput, AllocationAgentOutput)
        — AllocationOutput is passed to the Risk Agent
        — AllocationAgentOutput is passed to the Compliance Agent

    Requires:
        data/storage/crsp_monthly.parquet and data/storage/permno_map.json to exist.
        Run data.fetch.wrds.fetch_crsp_monthly() once to populate the cache.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    if ff_factors is None:
        ff_factors = load_ff_factors()

    # Load CRSP and filter to tickers that have PERMNOs (some ETNs may be absent)
    crsp_monthly, permno_map = load_crsp_monthly(tickers)
    available = [t for t in tickers if t in permno_map]
    if len(available) < len(tickers):
        missing = [t for t in tickers if t not in permno_map]
        print(f"[allocation] Excluding {missing} — no CRSP PERMNO found")
        tickers = available

    mkt_weights = load_market_cap_weights(tickers)

    allocation_input = profile_to_allocation_input(
        profile,
        discount_rate      = discount_rate,
        market_cap_weights = mkt_weights,
        tickers            = tickers,
        flag_constraints   = flag_constraints or [],
        flag_iteration     = flag_iteration,
    )

    rf_monthly    = ff_factors["rf"].mean()
    risk_free_rate = float(rf_monthly * 12)

    raw_output = run_allocation(
        allocation_input, crsp_monthly, ff_factors, risk_free_rate, permno_map
    )

    # ── LLM Touchpoint 1: Allocation rationale ──
    if _client is None:
        top = sorted(raw_output.weights, key=lambda w: w.total_weight, reverse=True)[:3]
        top_str = ", ".join(f"{w.ticker} {w.total_weight:.1%}" for w in top)
        rationale = (
            f"[No API key — template rationale] "
            f"Risky weight {raw_output.risky_weight:.1%} based on BMS human capital model "
            f"(income beta {allocation_input.user_profile.human_capital.income_beta:.2f}). "
            f"Top holdings: {top_str}. "
            f"Expected return {raw_output.portfolio_statistics.expected_return:.2%}, "
            f"volatility {raw_output.portfolio_statistics.volatility:.2%}, "
            f"Sharpe {raw_output.portfolio_statistics.sharpe_ratio:.2f}."
        )
    else:
        prompt  = _build_rationale_prompt(allocation_input, raw_output)
        message = _client.messages.create(
            model       = _MODEL,
            max_tokens  = 500,
            temperature = 0,
            messages    = [{"role": "user", "content": prompt}],
        )
        rationale = message.content[0].text.strip()

    allocation_output = raw_output.model_copy(update={"rationale": rationale})
    agent_output      = allocation_output_to_agent_output(allocation_output)

    return allocation_output, agent_output
