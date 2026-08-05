"""
sector_guard.py — pre-optimizer check that a client's own sector is not doubled up on.

A software developer's career is already a leveraged position in one sector. Buying
XLK on top of it is not diversification, it is concentration with extra steps. The
Risk and Compliance agents both catch this after the fact — Risk via sector limits,
Compliance via checks 1.2b and 1.3b — but "after the fact" means after the optimizer
has already spent the equity budget, and the revision loop then has to claw it back.

This module is the cheap check that runs first: given proposed weights and a
ticker→sector map, does the client's own sector exceed its cap?

    from agents.profile.sector_guard import check_sector_overlap

    check_sector_overlap(profile, weights, sectors)     # raises on violation

Why it lives on the Profile side
--------------------------------
The cap is a property of the client, not of the optimizer: it is their career that
makes tech dangerous for them and harmless for a nurse. The Profile Agent owns the
two inputs that set it (`industry_exposure_sector` and `income_equity_correlation`),
so it owns the guard, and the Allocation Agent calls it. The alternative — Allocation
re-deriving the client's sector risk — is where the two sides drift apart.

The spelling problem this exists to close
-----------------------------------------
`industry_exposure_sector` is matched by string equality against ETF sector labels.
Before 4 Aug the Profile Agent emitted "Technology", "Healthcare" and "Financial
Services" while the Allocation Agent's ETF_SECTORS spells them "Information
Technology", "Health Care" and "Financials". Nothing raised. The employer sector cap
in agents/shared/core/allocation.py simply never matched a ticker, so XLK fell under
the generic 20% sector limit, and _SECTOR_PROXY_ETF fell through to SPY, hedging the
career against the wrong index. contracts.normalize_sector() now fixes the spelling
at the contract boundary; this guard is what makes a remaining mismatch loud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from contracts import (
    NON_INVESTABLE_EMPLOYER_SECTORS,
    ProfileAgentOutput,
    is_investable_sector,
    normalize_sector,
)

# Matches EMPLOYER_SECTOR_LIMIT in agents/shared/core/constraints.py. Duplicated
# rather than imported: agents/shared is the Allocation/Risk side of the pipeline
# and the Profile Agent should not take a dependency on it to answer a question
# about its own client. The test suite asserts the two stay equal.
DEFAULT_EMPLOYER_SECTOR_CAP = 0.10


class SectorOverlapError(ValueError):
    """The proposed portfolio doubles up on the client's own employment sector."""


@dataclass(frozen=True)
class SectorOverlapResult:
    """
    Outcome of one sector-overlap check.

    Carries `applicable` because "passed" and "did not apply" are different
    findings and collapsing them is how a guard quietly stops guarding. A nurse
    has no investable employer sector, so there is nothing to check; a software
    developer whose sector map came back empty also produces no violation, but for
    a reason that should be fixed.
    """

    client_id:     str
    sector:        str
    exposure:      float
    cap:           float
    applicable:    bool
    reason:        str = ""

    @property
    def violated(self) -> bool:
        return self.applicable and self.exposure > self.cap

    def describe(self) -> str:
        if not self.applicable:
            return f"{self.client_id}: sector overlap not applicable — {self.reason}"
        verdict = "EXCEEDS" if self.violated else "within"
        return (
            f"{self.client_id}: '{self.sector}' exposure {self.exposure:.2%} "
            f"{verdict} cap {self.cap:.2%}"
        )


def employer_sector_cap(
    profile: ProfileAgentOutput,
    base_cap: float = DEFAULT_EMPLOYER_SECTOR_CAP,
    correlation_adjusted: bool = True,
) -> float:
    """
    The cap on the client's own sector, tightened by their income correlation.

    SCOPE.md §3.2 states the rule as `adjusted_limit = base_limit × (1 − ρ)`. A
    client whose pay tracks the market at ρ = 0.75 gets 10% × 0.25 = 2.5%; a
    professor at ρ = 0.10 gets 9%. The base cap is the same number for both — what
    differs is how much of their balance sheet is already in it.

    Set `correlation_adjusted=False` for the flat cap the 4 Aug minutes describe
    ("tech ≤ 10% for a tech client"). The adjusted form is the default because the
    flat cap is the special case ρ = 0, which is true of no one in the persona set.
    """
    if not correlation_adjusted:
        return base_cap
    # ρ can be negative in principle, which would loosen rather than tighten the
    # cap. Clamp at the base: a career that hedges the market is a reason to worry
    # less, not a licence to concentrate further.
    return min(base_cap, base_cap * (1.0 - profile.income_equity_correlation))


def sector_exposure(
    weights: Mapping[str, float],
    sectors: Mapping[str, str],
    sector: str,
) -> float:
    """Total proposed weight in `sector`, matching on normalised sector names."""
    target = normalize_sector(sector)
    return sum(
        w for ticker, w in weights.items()
        if normalize_sector(sectors.get(ticker, "")) == target
    )


def check_sector_overlap(
    profile:              ProfileAgentOutput,
    weights:              Mapping[str, float],
    sectors:              Mapping[str, str],
    base_cap:             float = DEFAULT_EMPLOYER_SECTOR_CAP,
    correlation_adjusted: bool = True,
    raise_on_violation:   bool = True,
) -> SectorOverlapResult:
    """
    Verify the proposed portfolio does not exceed the client's own-sector cap.

    Call this after the optimizer produces weights and before they are returned,
    so a violation surfaces as an exception at the point it was created rather
    than as a Risk FLAG one agent later.

    Parameters
    ----------
    weights : ticker → portfolio weight, as proposed by the optimizer.
    sectors : ticker → GICS sector. Pass `AllocationInput.universe.sectors`, or
              agents.allocation.adapters.ETF_SECTORS directly.
    base_cap : cap before correlation adjustment. Configurable per the 4 Aug
              minutes; defaults to the 10% employer sector limit already in
              agents/shared/core/constraints.py.
    correlation_adjusted : apply `× (1 − ρ)` per SCOPE.md §3.2.
    raise_on_violation : raise SectorOverlapError instead of returning a failed
              result. True by default — a guard that has to be checked by the
              caller is one an optimizer refactor can forget about.

    Raises
    ------
    SectorOverlapError
        When the client's own sector exceeds its cap and `raise_on_violation`.
    """
    sector = normalize_sector(profile.industry_exposure_sector)
    cap    = employer_sector_cap(profile, base_cap, correlation_adjusted)

    if sector in NON_INVESTABLE_EMPLOYER_SECTORS:
        return SectorOverlapResult(
            client_id  = profile.client_id,
            sector     = sector,
            exposure   = 0.0,
            cap        = cap,
            applicable = False,
            reason     = f"'{sector}' has no listed sector ETF — nothing to double up on",
        )

    if not is_investable_sector(sector):
        # Not a no-op worth being quiet about: an unmapped sector means the
        # employer cap and the proxy ETF are both falling through downstream.
        return SectorOverlapResult(
            client_id  = profile.client_id,
            sector     = sector,
            exposure   = 0.0,
            cap        = cap,
            applicable = False,
            reason     = (
                f"'{sector}' is not a recognised GICS sector — add it to "
                f"contracts._SECTOR_ALIASES; until then the employer sector cap "
                f"and employer proxy ETF do not apply to this client"
            ),
        )

    exposure = sector_exposure(weights, sectors, sector)
    result   = SectorOverlapResult(
        client_id  = profile.client_id,
        sector     = sector,
        exposure   = exposure,
        cap        = cap,
        applicable = True,
    )

    if result.violated and raise_on_violation:
        raise SectorOverlapError(
            f"{profile.client_id}: proposed portfolio holds {exposure:.2%} in "
            f"'{sector}', the client's own employment sector, against a cap of "
            f"{cap:.2%} "
            f"(base {base_cap:.2%}"
            + (f" × (1 − ρ={profile.income_equity_correlation:.2f})" if correlation_adjusted else "")
            + f"). Their human capital already carries implicit equity exposure of "
            f"{profile.implicit_equity_exposure:.1%} of total wealth; adding "
            f"portfolio weight in the same sector concentrates the risk the "
            f"human-capital framework exists to offset."
        )

    return result
