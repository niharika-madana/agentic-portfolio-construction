from __future__ import annotations

import pandas as pd

from portfolio_system.schemas import (
    AllocationInput, RiskDecision, RiskOutput, UserProfile,
)
from portfolio_system.agents.allocation_agent import run_allocation_agent
from portfolio_system.agents.risk_agent import run_risk_agent

_MAX_FLAG_ITERATIONS = 3


def run_pipeline(
    user_profile: UserProfile,
    universe_tickers: list[str],
    market_cap_weights: dict[str, float],
    sectors: dict[str, str],
    crsp_monthly: pd.DataFrame,
    crsp_daily: pd.DataFrame,
    ff_factors: pd.DataFrame,
    risk_free_rate: float,
    permno_map: dict[str, int],
) -> RiskOutput:
    """
    Full portfolio construction pipeline with FLAG feedback loop.

    Flow: AllocationAgent → RiskAgent → (FLAG → AllocationAgent → ...) → final RiskOutput

    The FLAG loop tightens constraints on each re-entry and caps at 3 iterations.
    After iteration 3, the Risk engine REJECTs automatically.
    """
    from portfolio_system.schemas import InstrumentUniverse

    universe = InstrumentUniverse(
        tickers=universe_tickers,
        market_cap_weights=market_cap_weights,
        sectors=sectors,
    )

    allocation_input = AllocationInput(
        user_profile=user_profile,
        universe=universe,
        flag_constraints=[],
        flag_iteration=0,
    )

    for iteration in range(_MAX_FLAG_ITERATIONS + 1):
        print(f"\n{'='*60}")
        label = "First run" if iteration == 0 else "FLAG re-entry"
        print(f"  ITERATION {iteration + 1}  --  {label}")
        print(f"{'='*60}")

        print("  [1/2] Running allocation agent...")
        allocation_output = run_allocation_agent(
            allocation_input, crsp_monthly, ff_factors, risk_free_rate, permno_map
        )
        print(f"        Risky weight: {allocation_output.risky_weight:.1%}  |  "
              f"E[r]: {allocation_output.portfolio_statistics.expected_return:.2%}  |  "
              f"Vol: {allocation_output.portfolio_statistics.volatility:.2%}")

        print("  [2/2] Running risk agent...")
        risk_output = run_risk_agent(allocation_output, crsp_daily, permno_map)
        print(f"        Decision: {risk_output.decision.value.upper()}")

        if risk_output.decision == RiskDecision.APPROVE:
            return risk_output

        if risk_output.decision == RiskDecision.REJECT:
            return risk_output

        # FLAG — feed violated constraints back into next allocation run
        allocation_input = AllocationInput(
            user_profile=user_profile,
            universe=universe,
            flag_constraints=risk_output.constraints_violated,
            flag_iteration=iteration + 1,
        )

    # Should not reach here — iteration 3 always REJECTs via schema validator
    return risk_output
