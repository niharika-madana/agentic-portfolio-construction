from __future__ import annotations

from portfolio_system.schemas import AllocationConstraint, ConcentrationFlags, ConstraintType


# ── Limits (single source of truth for both agents) ───────────────────────────
# Enforced as optimizer bounds in core/allocation.py
# Evaluated as rule checks   in core/risk.py

SINGLE_NAME_LIMIT     = 0.10   # max weight in any single instrument
SECTOR_LIMIT          = 0.20   # max financial portfolio weight in any GICS sector
ECONOMIC_SECTOR_LIMIT = 0.25   # max total economic weight (portfolio + HC) in any sector
EMPLOYER_LIMIT        = 0.15   # max employer concentration including human capital


# ── Utility ───────────────────────────────────────────────────────────────────

def aggregate_sector_weights(
    weights: dict[str, float],
    sectors: dict[str, str],
) -> dict[str, float]:
    """
    Sum instrument weights by GICS sector.
    Used internally and by core/allocation.py when building sector constraints.
    """
    result: dict[str, float] = {}
    for ticker, w in weights.items():
        sector = sectors.get(ticker, "Unknown")
        result[sector] = result.get(sector, 0.0) + w
    return result


# ── Individual checks ─────────────────────────────────────────────────────────

def check_single_name(weights: dict[str, float]) -> list[AllocationConstraint]:
    """Returns one AllocationConstraint for every ticker exceeding SINGLE_NAME_LIMIT."""
    return [
        AllocationConstraint(
            constraint_type=ConstraintType.SINGLE_NAME,
            target=ticker,
            current_value=w,
            limit=SINGLE_NAME_LIMIT,
        )
        for ticker, w in weights.items()
        if w > SINGLE_NAME_LIMIT
    ]


def check_sector(
    weights: dict[str, float],
    sectors: dict[str, str],
) -> list[AllocationConstraint]:
    """Returns one AllocationConstraint for every GICS sector exceeding SECTOR_LIMIT."""
    sw = aggregate_sector_weights(weights, sectors)
    return [
        AllocationConstraint(
            constraint_type=ConstraintType.SECTOR,
            target=sector,
            current_value=value,
            limit=SECTOR_LIMIT,
        )
        for sector, value in sw.items()
        if value > SECTOR_LIMIT
    ]


def check_economic_sector(
    economic_sector_exposures: dict[str, float],
) -> list[AllocationConstraint]:
    """
    Returns one AllocationConstraint for every sector exceeding ECONOMIC_SECTOR_LIMIT
    on the total economic balance sheet (portfolio + human capital).
    economic_sector_exposures comes from core/human_capital.economic_sector_exposures().
    """
    return [
        AllocationConstraint(
            constraint_type=ConstraintType.ECONOMIC_SECTOR,
            target=sector,
            current_value=exposure,
            limit=ECONOMIC_SECTOR_LIMIT,
        )
        for sector, exposure in economic_sector_exposures.items()
        if exposure > ECONOMIC_SECTOR_LIMIT
    ]


def check_employer(employer_concentration: float) -> list[AllocationConstraint]:
    """
    Returns one AllocationConstraint if employer concentration exceeds EMPLOYER_LIMIT.
    employer_concentration comes from core/human_capital.employer_concentration().
    """
    if employer_concentration > EMPLOYER_LIMIT:
        return [
            AllocationConstraint(
                constraint_type=ConstraintType.EMPLOYER,
                target="employer",
                current_value=employer_concentration,
                limit=EMPLOYER_LIMIT,
            )
        ]
    return []


# ── Composite checks ──────────────────────────────────────────────────────────

def run_all_checks(
    weights: dict[str, float],
    sectors: dict[str, str],
    economic_sector_exposures: dict[str, float],
    employer_concentration: float,
) -> ConcentrationFlags:
    """
    Runs all four concentration checks and returns a ConcentrationFlags object.
    Called by core/risk.py to evaluate the portfolio from the Allocation Agent.
    """
    sn  = check_single_name(weights)
    s   = check_sector(weights, sectors)
    es  = check_economic_sector(economic_sector_exposures)
    emp = check_employer(employer_concentration)

    return ConcentrationFlags(
        single_name_breaches     = [c.target for c in sn],
        sector_breaches          = [c.target for c in s],
        economic_sector_breaches = [c.target for c in es],
        employer_breach          = len(emp) > 0,
    )


def all_violations(
    weights: dict[str, float],
    sectors: dict[str, str],
    economic_sector_exposures: dict[str, float],
    employer_concentration: float,
) -> list[AllocationConstraint]:
    """
    Flat list of every violated constraint as AllocationConstraint objects.
    Used by core/risk.py to populate RiskOutput.constraints_violated on FLAG decisions,
    which are then fed back into AllocationInput.flag_constraints on re-entry.
    """
    return (
        check_single_name(weights)
        + check_sector(weights, sectors)
        + check_economic_sector(economic_sector_exposures)
        + check_employer(employer_concentration)
    )
