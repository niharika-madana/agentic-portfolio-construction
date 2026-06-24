from __future__ import annotations

from contracts import AllocationConstraint, ConcentrationFlags, ConstraintType


SINGLE_NAME_LIMIT     = 0.10
SECTOR_LIMIT          = 0.20
ECONOMIC_SECTOR_LIMIT = 0.25
EMPLOYER_LIMIT        = 0.15


def aggregate_sector_weights(
    weights: dict[str, float],
    sectors: dict[str, str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for ticker, w in weights.items():
        sector = sectors.get(ticker, "Unknown")
        result[sector] = result.get(sector, 0.0) + w
    return result


def check_single_name(weights: dict[str, float]) -> list[AllocationConstraint]:
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


def run_all_checks(
    weights: dict[str, float],
    sectors: dict[str, str],
    economic_sector_exposures: dict[str, float],
    employer_concentration: float,
) -> ConcentrationFlags:
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
    return (
        check_single_name(weights)
        + check_sector(weights, sectors)
        + check_economic_sector(economic_sector_exposures)
        + check_employer(employer_concentration)
    )
