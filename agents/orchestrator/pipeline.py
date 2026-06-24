"""
pipeline.py — internal Allocation ↔ Risk FLAG feedback loop.

run_alloc_risk_loop(profile, discount_rate, ff_factors)
    → (AllocationOutput, RiskOutput, RiskAgentOutput, revisions)

Runs at most _MAX_FLAG_ITERATIONS passes:
  AllocationAgent → RiskAgent → FLAG → AllocationAgent (with tighter constraints) → ...

Terminates on APPROVE or REJECT. On REJECT after exhausting retries, returns
the final RiskOutput so the orchestrator can record the failure in AdvisorPackage.
"""

from __future__ import annotations

import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from contracts import (
    AllocationAgentOutput, AllocationOutput, ProfileAgentOutput,
    RiskAgentOutput, RiskDecision, RiskOutput,
)
from agents.allocation.agent import run_allocation_agent
from agents.risk.agent import run_risk_agent
from data.fetch.wrds import load_ff_factors

logger = logging.getLogger(__name__)

_MAX_FLAG_ITERATIONS = 3


def run_alloc_risk_loop(
    profile: ProfileAgentOutput,
    discount_rate: float,
    ff_factors: pd.DataFrame | None = None,
) -> tuple[AllocationOutput, RiskOutput, RiskAgentOutput, AllocationAgentOutput, int]:
    """
    Run the Allocation ↔ Risk FLAG feedback loop for one client profile.

    Args:
        profile:       ProfileAgentOutput from the Profile Agent.
        discount_rate: FRED DGS10 annual rate.
        ff_factors:    Pre-loaded FF risk factors (auto-loaded if None).

    Returns:
        (AllocationOutput, RiskOutput, RiskAgentOutput, AllocationAgentOutput, revisions)
        where revisions = number of FLAG re-entries (0 = first run passed).
    """
    if ff_factors is None:
        ff_factors = load_ff_factors()

    flag_constraints = []
    flag_iteration   = 0
    alloc_output: AllocationOutput | None = None
    risk_output:  RiskOutput | None       = None
    risk_ao:      RiskAgentOutput | None  = None
    alloc_ao:     AllocationAgentOutput | None = None

    for iteration in range(_MAX_FLAG_ITERATIONS + 1):
        label = "First run" if iteration == 0 else f"FLAG re-entry #{iteration}"
        logger.info(f"[pipeline] Iteration {iteration + 1}  ({label})")

        alloc_output, alloc_ao = run_allocation_agent(
            profile          = profile,
            discount_rate    = discount_rate,
            ff_factors       = ff_factors,
            flag_constraints = flag_constraints,
            flag_iteration   = flag_iteration,
        )

        logger.info(
            f"[pipeline] Allocation done — risky {alloc_output.risky_weight:.1%} "
            f"E[r] {alloc_output.portfolio_statistics.expected_return:.2%} "
            f"vol {alloc_output.portfolio_statistics.volatility:.2%}"
        )

        risk_output, risk_ao = run_risk_agent(
            allocation_output = alloc_output,
            profile           = profile,
        )

        logger.info(f"[pipeline] Risk decision: {risk_output.decision.value}")

        if risk_output.decision in (RiskDecision.APPROVE, RiskDecision.REJECT):
            return alloc_output, risk_output, risk_ao, alloc_ao, iteration

        # FLAG — feed tightened constraints back into next allocation run
        flag_constraints = risk_output.constraints_violated
        flag_iteration   = iteration + 1

    # Safety fallback (should not reach here — iteration 3 always REJECTs)
    return alloc_output, risk_output, risk_ao, alloc_ao, _MAX_FLAG_ITERATIONS
