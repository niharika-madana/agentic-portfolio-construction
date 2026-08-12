"""
validator.py — deterministic gate on an LLM-proposed portfolio sleeve.

Architecture (reversed from the original optimizer-first design): the LLM now
chooses which tickers to hold and in what proportion (agents/allocation/agent.py).
This module never proposes a weight — it only checks one, against exactly the
limits agents/shared/core/allocation.optimize_weights() used to enforce as hard
bounds when the optimizer was the one choosing weights. A portfolio the LLM
builds is held to the same bar a portfolio the optimizer built would have been.

What stays out of scope here on purpose: risky_weight (how much of total
financial wealth is risky at all) is not something this validator checks,
because the LLM is never asked to set it — it is computed deterministically in
agent.py from the Merton/BMS human-capital formula and capped at
portfolio_equity_target, the one number the project's human-capital thesis is
built around. Letting the LLM choose that number would validate the wrong
thing. What the LLM does choose — which approved ETFs make up the risky sleeve
— is exactly what this module checks.
"""

from __future__ import annotations

from contracts import AllocationConstraint, ConstraintType
from agents.shared.core.constraints import (
    EMPLOYER_SECTOR_LIMIT,
    EMPLOYER_STOCK_LIMIT,
    SECTOR_LIMIT,
    SINGLE_NAME_LIMIT,
)

_MIN_POSITION = 1e-6  # weights at or below this are treated as "not held"


class AllocationValidationError(RuntimeError):
    """Raised when the LLM cannot produce a compliant sleeve within the revision budget."""

    def __init__(self, message: str, last_violations: list[str]):
        super().__init__(message)
        self.last_violations = last_violations


def validate_sleeve(
    weights: dict[str, float],
    rationale: dict[str, str],
    sectors: dict[str, str],
    universe: list[str],
    employer_ticker: str | None,
    employer_sector: str | None,
    flag_constraints: list[AllocationConstraint],
) -> list[str]:
    """
    Check an LLM-proposed sleeve against the same bounds the deterministic
    optimizer used to enforce structurally. Returns human-readable violation
    strings — empty means the sleeve is accepted as-is.
    """
    violations: list[str] = []

    # ── Structural — mirrors AllocationAgentOutput's own contract validator,
    #    checked here first so a rejected proposal gets a specific, actionable
    #    reason fed back to the LLM instead of a raw pydantic ValidationError. ──
    unknown = sorted(t for t in weights if t not in universe)
    if unknown:
        violations.append(
            f"Unknown ticker(s) not in the approved universe: {unknown}. "
            f"Only propose weights for tickers in the universe list."
        )

    negative = {t: w for t, w in weights.items() if w < -1e-9}
    if negative:
        violations.append(f"Negative weight(s) are not allowed (no shorting): {negative}")

    held = {t: w for t, w in weights.items() if w > _MIN_POSITION}
    if len(held) < 2:
        violations.append(f"Only {len(held)} position(s) held; minimum 2 required")

    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        violations.append(f"Weights sum to {total:.4f}; must sum to 1.0 (+/- 0.01)")

    missing_rationale = [t for t in held if not rationale.get(t, "").strip()]
    if missing_rationale:
        violations.append(
            f"Missing or empty rationale for held position(s): {missing_rationale}. "
            f"Every held ticker needs its own rationale."
        )

    # ── Single-name limit, tightened by any prior Risk Agent FLAG round ──
    single_name_limits = {t: SINGLE_NAME_LIMIT for t in universe}
    for fc in flag_constraints:
        if fc.constraint_type == ConstraintType.SINGLE_NAME and fc.target in single_name_limits:
            single_name_limits[fc.target] = min(single_name_limits[fc.target], fc.limit * 0.99)

    for t, w in weights.items():
        lim = single_name_limits.get(t, SINGLE_NAME_LIMIT)
        if w > lim + 1e-9:
            violations.append(f"{t}: weight {w:.3%} exceeds single-name limit {lim:.3%}")

    # ── Employer stock — never hold the client's own employer; it's already
    #    carried via career + equity comp ──
    if employer_ticker and weights.get(employer_ticker, 0.0) > EMPLOYER_STOCK_LIMIT + 1e-9:
        violations.append(
            f"{employer_ticker}: weight {weights[employer_ticker]:.1%} — this is the client's "
            f"own employer's stock and must be held at {EMPLOYER_STOCK_LIMIT:.0%}"
        )

    # ── Sector aggregate — employer's own sector gets the tighter flat cap,
    #    every other sector the general cap ──
    sector_totals: dict[str, float] = {}
    for t, w in weights.items():
        sec = sectors.get(t, "Unknown")
        sector_totals[sec] = sector_totals.get(sec, 0.0) + w

    sector_limits = {sec: SECTOR_LIMIT for sec in sector_totals}
    if employer_sector:
        sector_limits[employer_sector] = min(sector_limits.get(employer_sector, SECTOR_LIMIT), EMPLOYER_SECTOR_LIMIT)
    for fc in flag_constraints:
        if fc.constraint_type == ConstraintType.SECTOR and fc.target in sector_limits:
            sector_limits[fc.target] = min(sector_limits[fc.target], fc.limit * 0.99)

    for sec, w in sector_totals.items():
        lim = sector_limits.get(sec, SECTOR_LIMIT)
        if w > lim + 1e-9:
            reason = " (client's own employer sector)" if sec == employer_sector else ""
            violations.append(f"{sec} sector{reason}: {w:.3%} exceeds limit {lim:.3%}")

    return violations
