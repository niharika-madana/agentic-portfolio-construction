"""
Allocation Agent — public entry point.

run_allocation_agent(profile, discount_rate, ff_factors, flag_constraints, flag_iteration)
    → AllocationOutput        (internal, passed to Risk Agent)
    → AllocationAgentOutput   (Compliance-facing)

Architecture — LLM builds, math validates:
  1. Converts ProfileAgentOutput → AllocationInput via adapters.py
  2. Loads CRSP monthly returns and FF risk factors from parquet cache
  3. Runs the Black-Litterman optimizer (agents/shared/core/allocation.py) once,
     not as the final answer but as REFERENCE CONTEXT for the LLM — its
     equilibrium weights, factor-view tilts, and full solved sleeve
  4. Calls Claude to propose the risky-sleeve ETF weights directly, given that
     reference and the hard constraint set
  5. Deterministically validates the proposal (agents/allocation/validator.py)
     against the same limits the optimizer used to enforce as hard bounds; on
     failure, feeds the specific violations back and asks the LLM to revise
     (up to _MAX_LLM_REVISIONS times)
  6. Returns both the internal AllocationOutput and the Compliance-facing
     AllocationAgentOutput, built from the LLM's accepted sleeve

What the LLM does NOT control: risky_weight — the fraction of total financial
wealth allocated to the risky sleeve at all — stays fully deterministic. It is
computed from the Merton/BMS human-capital formula (compute_w_fin) and capped
at profile.portfolio_equity_target, exactly as before. That number is the
project's central human-capital thesis (effective_risk_budget minus implicit
equity exposure); the LLM only decides the composition of the sleeve within
the budget deterministic code has already sized. See validator.py's module
docstring for the reasoning.

Data source: WRDS/CRSP via data.fetch.wrds (parquet cache must be populated first).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import anthropic
import pandas as pd

from contracts import (
    AllocationAgentOutput, AllocationConstraint, AllocationInput,
    AllocationOutput, ProfileAgentOutput, WeightDecomposition,
)
from agents.allocation.adapters import (
    DEFAULT_TICKERS,
    allocation_output_to_agent_output,
    profile_to_allocation_input,
)
from agents.allocation.validator import AllocationValidationError, validate_sleeve
from agents.shared.core.allocation import run_allocation
from agents.shared.core.constraints import (
    EMPLOYER_SECTOR_LIMIT, SECTOR_LIMIT, SINGLE_NAME_LIMIT,
)
from data.fetch.wrds import load_crsp_monthly, load_ff_factors, load_market_cap_weights

_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
_client  = anthropic.Anthropic(api_key=_API_KEY) if _API_KEY else None
_MODEL   = "claude-sonnet-4-6"

_MAX_LLM_REVISIONS = 3  # mirrors the orchestrator's own FLAG-loop budget


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_reference_table(reference: AllocationOutput, sectors: dict[str, str]) -> str:
    rows = sorted(reference.weights, key=lambda w: w.total_weight, reverse=True)
    lines = [f"  {'ticker':<6} {'sector':<24} {'equilibrium':>11} {'view tilt':>10} {'optimizer final':>16}"]
    for w in rows:
        sec = sectors.get(w.ticker, "Unknown")
        lines.append(
            f"  {w.ticker:<6} {sec:<24} {w.equilibrium_baseline:>10.1%} "
            f"{w.view_tilt:>+9.1%} {w.total_weight:>15.1%}"
        )
    return "\n".join(lines)


def _build_portfolio_prompt(
    inp:              AllocationInput,
    reference:        AllocationOutput,
    risky_weight:     float,
    flag_constraints: list[AllocationConstraint],
    prior_weights:    dict[str, float] | None,
    prior_violations: list[str],
) -> str:
    up  = inp.user_profile
    hc  = up.human_capital
    universe = inp.universe

    lines = [
        "You are a quantitative portfolio construction agent. You choose the actual "
        "portfolio weights — a deterministic validator checks your proposal afterward "
        "and will ask you to revise if it fails, but nothing downstream overrides the "
        "weights you pick as long as they pass.",
        "",
        "CLIENT PROFILE",
        f"  Financial wealth:            ${up.financial_wealth:>12,.0f}",
        f"  Human capital (PV):          ${hc.present_value:>12,.0f}",
        f"  Risk profile:                {up.risk_profile.value}",
        f"  Years to retirement:         {hc.years_to_retirement}",
        f"  Employer sector:             {hc.employer_sector}",
        f"  Employer proxy ticker:       {hc.employer_ticker}",
        f"  Income beta:                 {hc.income_beta:.2f}",
        f"  Human capital type:          {hc.human_capital_type}",
        f"  RSU concentration:           {hc.rsu_concentration:.1%}",
        f"  Portfolio equity target:     {up.portfolio_equity_target:.1%}  (risk budget minus implicit equity exposure)",
        "",
        f"  NOTE: risky_weight is fixed by deterministic human-capital math at "
        f"{risky_weight:.1%} of total financial wealth — that split is not yours to set. "
        f"Your job is the COMPOSITION of the risky sleeve itself: which approved ETFs to "
        f"hold and in what proportion, summing to 100% of the sleeve (not of total wealth).",
        "",
        "APPROVED UNIVERSE (ticker: sector, market-cap weight)",
    ] + [
        f"  {t}: {universe.sectors[t]}, mkt-cap weight {universe.market_cap_weights[t]:.1%}"
        for t in universe.tickers
    ] + [
        "",
        "HARD CONSTRAINTS — your proposal is rejected if any is violated:",
        "  - weights must be >= 0 (no shorting) and sum to 1.0 (+/- 1%)",
        "  - at least 2 positions held",
        f"  - no single ticker above {SINGLE_NAME_LIMIT:.0%}",
        f"  - never hold {hc.employer_ticker} ({hc.employer_sector} proxy) — the client's own "
        f"employer/sector is already carried through their career, so the portfolio must not add to it",
        f"  - {hc.employer_sector} sector (client's employer sector) capped at {EMPLOYER_SECTOR_LIMIT:.0%} total",
        f"  - every other sector capped at {SECTOR_LIMIT:.0%} total",
        "  - every held position needs its own rationale",
    ]

    for fc in flag_constraints:
        lines.append(f"  - [Risk Agent tightened this] {fc.constraint_type.value} {fc.target}: <= {fc.limit:.1%}")

    lines += [
        "",
        "REFERENCE — a deterministic Black-Litterman optimizer solved the same inputs. "
        "equilibrium = CAPM market-cap-implied weight; view tilt = Fama-French factor-view "
        "adjustment; optimizer final = what the optimizer would have proposed. Use this as a "
        "starting point and sanity check, not an answer key — you may deviate from it, but your "
        "rationale should explain why when you do:",
        _format_reference_table(reference, universe.sectors),
    ]

    if prior_weights is not None:
        lines += [
            "",
            "YOUR PREVIOUS PROPOSAL WAS REJECTED. Violations:",
        ] + [f"  - {v}" for v in prior_violations] + [
            "",
            "Previous proposal (ticker: weight):",
            "  " + ", ".join(f"{t}: {w:.1%}" for t, w in prior_weights.items()),
            "",
            "Revise it to fix every violation above.",
        ]

    lines += [
        "",
        "Respond with ONLY a JSON object, no markdown fences, no other text:",
        '{"weights": {"TICKER": fraction, ...}, "rationale": {"TICKER": "text", ...}}',
        "",
        "Each rationale must be specific to this client — reference at least 2 of: "
        "human capital type, income beta, RSU concentration, portfolio equity target, "
        "risk profile, employer sector. Be precise with numbers; do not invent figures "
        "not shown above. Do not write the same rationale for two different tickers.",
    ]
    return "\n".join(lines)


def _call_llm_for_sleeve(prompt: str) -> tuple[dict[str, float], dict[str, str]]:
    message = _client.messages.create(
        model       = _MODEL,
        max_tokens  = 2000,
        temperature = 0,
        messages    = [{"role": "user", "content": prompt}],
    )
    text = message.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"response was not valid JSON: {e}") from e

    if "weights" not in parsed or "rationale" not in parsed:
        raise ValueError("response JSON must have 'weights' and 'rationale' keys")

    weights   = {t: float(w) for t, w in parsed["weights"].items()}
    rationale = {t: str(r) for t, r in parsed["rationale"].items()}
    return weights, rationale


# ---------------------------------------------------------------------------
# Deterministic fallback (no API key — same behavior as the optimizer-first design)
# ---------------------------------------------------------------------------

def _deterministic_fallback(reference: AllocationOutput) -> AllocationOutput:
    top = sorted(reference.weights, key=lambda w: w.total_weight, reverse=True)[:3]
    top_str = ", ".join(f"{w.ticker} {w.total_weight:.1%}" for w in top)
    up = reference.allocation_input.user_profile
    rationale = (
        f"[No API key — deterministic optimizer used directly, LLM step skipped] "
        f"Risky weight {reference.risky_weight:.1%} based on BMS human capital model "
        f"(income beta {up.human_capital.income_beta:.2f}, "
        f"human capital type '{up.human_capital.human_capital_type}', "
        f"capped at portfolio equity target {up.portfolio_equity_target:.1%}). "
        f"Top holdings: {top_str}. "
        f"Expected return {reference.portfolio_statistics.expected_return:.2%}, "
        f"volatility {reference.portfolio_statistics.volatility:.2%}, "
        f"Sharpe {reference.portfolio_statistics.sharpe_ratio:.2f}."
    )
    return reference.model_copy(update={"rationale": rationale})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_allocation_agent(
    profile: ProfileAgentOutput,
    discount_rate: float,
    ff_factors: pd.DataFrame | None = None,
    flag_constraints: list[AllocationConstraint] | None = None,
    flag_iteration: int = 0,
    tickers: list[str] | None = None,
) -> tuple[AllocationOutput, AllocationAgentOutput]:
    """
    Build the risky-sleeve portfolio for one client profile.

    The LLM proposes the sleeve composition; a deterministic Black-Litterman
    solve of the same inputs is computed first and passed to the LLM as
    reference context, not as the answer. The proposal is validated against
    the same hard bounds the optimizer used to enforce
    (agents/allocation/validator.py) and re-prompted on failure, up to
    _MAX_LLM_REVISIONS times. risky_weight (how much of total wealth is risky
    at all) remains fully deterministic — see module docstring.

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

    Raises:
        AllocationValidationError: the LLM could not produce a sleeve that
            passes the deterministic validator within _MAX_LLM_REVISIONS
            revisions. Carries the last violation list.

    Requires:
        data/storage/crsp_monthly.parquet and data/storage/permno_map.json to exist.
        Run data.fetch.wrds.fetch_crsp_monthly() once to populate the cache.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
    if flag_constraints is None:
        flag_constraints = []

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
        flag_constraints   = flag_constraints,
        flag_iteration     = flag_iteration,
    )

    rf_monthly     = ff_factors["rf"].mean()
    risk_free_rate = float(rf_monthly * 12)

    # ── Deterministic reference solve — context for the LLM, not the answer.
    #    Also the source of risky_weight/safe_weight, which stay deterministic. ──
    reference = run_allocation(
        allocation_input, crsp_monthly, ff_factors, risk_free_rate, permno_map
    )

    if _client is None:
        allocation_output = _deterministic_fallback(reference)
        agent_output       = allocation_output_to_agent_output(allocation_output)
        return allocation_output, agent_output

    # ── LLM Touchpoint 1 (reversed): LLM builds the sleeve, math validates it ──
    up  = allocation_input.user_profile
    universe = allocation_input.universe

    prior_weights: dict[str, float] | None = None
    violations: list[str] = []
    weights: dict[str, float] = {}
    rationale: dict[str, str] = {}

    for attempt in range(_MAX_LLM_REVISIONS + 1):
        prompt = _build_portfolio_prompt(
            allocation_input, reference, reference.risky_weight,
            flag_constraints, prior_weights, violations,
        )
        try:
            weights, rationale = _call_llm_for_sleeve(prompt)
        except ValueError as e:
            violations    = [str(e)]
            prior_weights = prior_weights or {}
            continue

        violations = validate_sleeve(
            weights, rationale, universe.sectors, universe.tickers,
            up.human_capital.employer_ticker, up.human_capital.employer_sector,
            flag_constraints,
        )
        if not violations:
            break
        prior_weights = weights

    if violations:
        raise AllocationValidationError(
            f"LLM failed to produce a compliant portfolio after {_MAX_LLM_REVISIONS} "
            f"revision(s) for client {profile.client_id}",
            last_violations=violations,
        )

    # Renormalize to exact 1.0 — the validator allows +/-1% business tolerance,
    # but AllocationOutput's own contract validator requires 1e-6 exactness.
    held  = {t: w for t, w in weights.items() if w > 1e-6}
    total = sum(held.values())
    held  = {t: w / total for t, w in held.items()}

    ref_by_ticker = {w.ticker: w for w in reference.weights}
    decomposition = [
        WeightDecomposition(
            ticker               = t,
            total_weight         = held.get(t, 0.0),
            equilibrium_baseline = ref_by_ticker[t].equilibrium_baseline,
            view_tilt            = ref_by_ticker[t].view_tilt,
            # Redefined for the LLM-first design: how far the LLM's choice
            # deviates from what the deterministic optimizer would have
            # proposed, net of the equilibrium and factor-view components
            # already shown above — the audit trail for the LLM's own judgment.
            human_capital_offset = held.get(t, 0.0) - (
                ref_by_ticker[t].equilibrium_baseline + ref_by_ticker[t].view_tilt
            ),
        )
        for t in universe.tickers
    ]

    allocation_output = reference.model_copy(update={
        "weights":   decomposition,
        "rationale": (
            f"LLM-built risky sleeve (reference optimizer available for comparison). "
            f"Risky weight {reference.risky_weight:.1%} of financial wealth set deterministically "
            f"from the human-capital model, capped at portfolio equity target "
            f"{up.portfolio_equity_target:.1%}."
        ),
    })

    agent_output = AllocationAgentOutput(
        proposed_portfolio          = held,
        allocation_rationale        = {t: rationale[t] for t in held},
        revision                    = flag_iteration,
        prior_risk_flags            = [],
        prior_compliance_violations = [],
    )

    return allocation_output, agent_output
