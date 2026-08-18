"""
Job 3 — Robo-Adviser Checks (Compliance-owned).

Grounded in SEC IM Guidance Update No. 2017-02 ("Robo-Advisers"), the primary
regulatory document governing algorithmic investment advisers under the
Investment Advisers Act of 1940. Where Job 1 audits the Risk Agent and Job 2
audits the recommendation's content, Job 3 audits the recommendation against the
*client's own mandate* and the *adviser's algorithmic obligations*.

Four checks:
  3.1  Composition–objective consistency — proposed_portfolio composition is
       consistent with the client's stated InvestmentObjective.
  3.2  Client-mandate consistency        — HARD_CONSTRAINT statements (e.g. "no
       defense", ESG exclusions) are honoured by the proposed portfolio.
  3.3  Algorithm-limitation disclosure   — when the recommendation rests on a
       model assumption that could mislead (low-confidence regime, or a mandate
       the all-ETF universe cannot fully honour), the required disclosure is flagged.
  3.4  ETF-provider conflict screen      — proposed ETFs are screened against the
       adviser's conflicted / proprietary-product affiliations.

Determinism note
----------------
Every *decision* in this module is deterministic — the SEC IM 2017-02 standard
requires that a compliance conclusion be reproducible and explainable, so a
stochastic model must never set clearance. Check 3.2 may optionally consult an
LLM *reviewer* (injected via `mandate_reviewer`) to catch mandate conflicts the
keyword map misses, but the reviewer only contributes findings — the rule below
maps findings to severity and owns the pass/fail. With no reviewer injected (the
default, and the state under pytest) 3.2 runs purely on deterministic set
membership.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contracts import (
    ClientStatement,
    ComplianceInput,
    ComplianceViolation,
    PipelineDestination,
    ResponsibleAgent,
    Severity,
    StatementKind,
)

# ---------------------------------------------------------------------------
# Exclusion-handling policy — DEFERRED DECISION (2026-07)
# ---------------------------------------------------------------------------
# How should Job 3 treat a client mandate the current all-ETF universe cannot
# fully honour — e.g. "no defense stocks" while the portfolio holds SPY, which
# contains Lockheed/RTX/Boeing? This is the same underlying fact as review §7
# (ETF overlap). The team will settle it once the universe question (drop SPY /
# add look-through / use ESG-screened variants) is decided. Flip this one switch
# then; the check mechanics do not change.
#
#   "tiered"  → directly holding an excluded ETF/sector = FAIL (3.2);
#               unavoidable broad-market indirect exposure = disclosure (3.3).
#   "strict"  → any exposure, direct or via broad-market ETF = FAIL (3.2).
#   "disclose"→ all mandate conflicts = disclosure only, never a hard FAIL.
EXCLUSION_POLICY = "tiered"

# Below this XGBoost regime confidence, IM 2017-02 requires disclosing that the
# recommendation rests on a low-confidence regime classification (Check 3.3).
DISCLOSURE_CONFIDENCE_FLOOR = 0.50

# ---------------------------------------------------------------------------
# Static universe metadata (Compliance keeps its own copy — it must not depend
# on the Allocation Agent's internals to audit the Allocation Agent's output)
# ---------------------------------------------------------------------------

# ticker → GICS sector for the single-name universe. Compliance's own copy, on
# purpose: importing agents/allocation/adapters.ETF_SECTORS would mean auditing
# the Allocation Agent against its own definition of the truth, which is not an
# audit. The cost of that independence is that this map has to be updated when
# the universe changes — see _UNCLASSIFIED_WEIGHT_FLOOR for the guard that makes
# a stale copy fail loudly instead of silently.
_TICKER_SECTOR: dict[str, str] = {
    "GOOGL": "Communication Services", "META": "Communication Services", "NFLX": "Communication Services",
    "AMZN": "Consumer Discretionary",  "TSLA": "Consumer Discretionary",  "HD":   "Consumer Discretionary",
    "WMT":  "Consumer Staples",        "COST": "Consumer Staples",        "PG":   "Consumer Staples",
    "XOM":  "Energy",                  "CVX":  "Energy",                  "COP":  "Energy",
    "BRK":  "Financials",              "JPM":  "Financials",              "V":    "Financials",
    "LLY":  "Health Care",             "JNJ":  "Health Care",             "UNH":  "Health Care",
    "GE":   "Industrials",             "CAT":  "Industrials",             "HON":  "Industrials",
    "NVDA": "Information Technology",  "MSFT": "Information Technology",  "AAPL": "Information Technology",
    "SHW":  "Materials",               "ECL":  "Materials",               "APD":  "Materials",
    "AMT":  "Real Estate",             "WELL": "Real Estate",             "SPG":  "Real Estate",
    "NEE":  "Utilities",               "SO":   "Utilities",               "DUK":  "Utilities",
}

_EQUITY_ETFS: set[str] = {
    # Single names — the current universe.
    *_TICKER_SECTOR,
    # ETF-era instruments. Retained because they remain correct classifications
    # and because a mixed or reverted universe must still audit cleanly; the
    # equity share is a weight sum, so entries for instruments nobody holds
    # contribute nothing.
    "SPY", "IWM", "EFA", "EEM",                              # broad equity
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP",  # GICS sectors
    "XLU", "XLRE", "VNQ",                                    # utilities / real estate
}
_INCOME_ETFS: set[str] = {"AGG", "TLT", "IEF", "SHY", "HYG", "LQD", "TIP"}
_BROAD_MARKET_ETFS: set[str] = {"SPY", "IWM", "EFA", "EEM"}
"""
Instruments whose holdings are not knowable from the ticker alone.

Empty in practice under the single-name universe, and that is the correct
result rather than a gap: a share of AAPL is technology exposure exactly once,
so there is no hidden spillover for Check 3.3 to disclose. The set stays
populated so a reverted or mixed universe is still screened.
"""

_UNCLASSIFIED_WEIGHT_FLOOR = 0.05
"""
Portfolio weight in instruments Compliance cannot classify before Check 3.1
refuses to opine.

This exists because of a real failure: when the universe moved from 24 ETFs to
33 single names, the classification sets above still held only ETF tickers, so
`equity_w` summed to exactly 0.0 for every client and Check 3.1 reported
"equity share 0% is below the 30% floor" against portfolios that were in fact
almost entirely equity. Eight false MEDIUM violations, and nothing in the output
distinguished them from real ones — the check was confidently wrong rather than
unable to answer.

A compliance check that cannot classify what it is auditing must say so. Below
this floor the unknown weight cannot move the equity share across a band
boundary, so the opinion still stands.
"""

# Free-text mandate subject → the sector ETF that would directly express it.
# A subject that maps to nothing here (e.g. "weapons", "tobacco") has no pure
# proxy in the universe and can only be reached indirectly via broad-market
# holdings — exactly the case Check 3.3 exists to disclose.
_SECTOR_ETF: dict[str, str] = {
    "information technology": "XLK", "technology": "XLK", "tech": "XLK",
    "financials": "XLF", "finance": "XLF", "banks": "XLF", "banking": "XLF",
    "health care": "XLV", "healthcare": "XLV", "pharma": "XLV", "pharmaceutical": "XLV",
    "energy": "XLE", "oil": "XLE", "fossil": "XLE", "fossil fuels": "XLE", "gas": "XLE",
    "industrials": "XLI",
    "communication services": "XLC", "communications": "XLC", "media": "XLC",
    "consumer discretionary": "XLY",
    "consumer staples": "XLP",
    "utilities": "XLU",
    "real estate": "XLRE", "reit": "XLRE",
}

# Free-text mandate subject → canonical GICS sector, for the single-name path in
# Check 3.2. Parallel to _SECTOR_ETF above, which answers the same question for
# an ETF universe ("which fund expresses this?"); this one answers "which sector
# did they mean?", after which _TICKER_SECTOR finds every held name in it.
#
# The client's words are not GICS vocabulary — "oil", "fossil fuels" and "gas"
# all mean Energy, and a mandate that misses because the client said "oil"
# instead of "Energy" is a silent compliance failure, which is the whole class of
# bug Check 3.2 exists to prevent.
_SUBJECT_SECTOR: dict[str, str] = {
    "information technology": "Information Technology", "technology": "Information Technology",
    "tech": "Information Technology", "big tech": "Information Technology",
    "semiconductors": "Information Technology", "software": "Information Technology",
    "financials": "Financials", "finance": "Financials", "banks": "Financials",
    "banking": "Financials", "financial services": "Financials", "insurers": "Financials",
    "health care": "Health Care", "healthcare": "Health Care", "pharma": "Health Care",
    "pharmaceutical": "Health Care", "pharmaceuticals": "Health Care", "biotech": "Health Care",
    "energy": "Energy", "oil": "Energy", "oil and gas": "Energy", "gas": "Energy",
    "fossil": "Energy", "fossil fuels": "Energy", "fossil fuel": "Energy",
    "petroleum": "Energy", "hydrocarbons": "Energy", "coal": "Energy",
    "industrials": "Industrials", "manufacturing": "Industrials", "defense": "Industrials",
    "defence": "Industrials", "aerospace": "Industrials",
    "communication services": "Communication Services", "communications": "Communication Services",
    "media": "Communication Services", "social media": "Communication Services",
    "consumer discretionary": "Consumer Discretionary", "retail": "Consumer Discretionary",
    "consumer staples": "Consumer Staples", "tobacco": "Consumer Staples",
    "alcohol": "Consumer Staples",
    "utilities": "Utilities", "materials": "Materials", "chemicals": "Materials",
    "mining": "Materials",
    "real estate": "Real Estate", "reit": "Real Estate", "reits": "Real Estate",
}

# ticker → issuer, for the conflict-of-interest screen (Check 3.4).
_ETF_ISSUER: dict[str, str] = {
    "SPY": "SSGA", "XLK": "SSGA", "XLF": "SSGA", "XLV": "SSGA", "XLE": "SSGA",
    "XLI": "SSGA", "XLC": "SSGA", "XLY": "SSGA", "XLP": "SSGA", "XLU": "SSGA",
    "XLRE": "SSGA", "BIL": "SSGA",
    "IWM": "iShares", "EFA": "iShares", "EEM": "iShares", "AGG": "iShares",
    "TLT": "iShares", "IEF": "iShares", "SHY": "iShares", "HYG": "iShares",
    "LQD": "iShares", "TIP": "iShares",
    "VNQ": "Vanguard",
    "GLD": "World Gold Council",
}

# Issuers the adviser has a compensation / proprietary-product arrangement with.
# Empty by default: this synthetic adviser declares no product affiliations, so
# the screen passes — but the *mechanism* runs, and would flag a conflict the
# moment a real affiliation is listed here. This is where a live adviser's
# proprietary-fund sponsors go.
_CONFLICTED_PROVIDERS: set[str] = set()

# Composition consistency bands — SEC IM 2017-02 requires the generated
# portfolio to be consistent with the client's stated objective. Bands are the
# (min, max) equity share a portfolio of that objective should fall within.
_OBJECTIVE_EQUITY_BANDS: dict[str, tuple[float, float]] = {
    "preservation": (0.00, 0.40),
    "income":       (0.00, 0.60),
    "growth":       (0.30, 1.00),
}
# Beyond this equity share the mismatch is egregious → HIGH, not MEDIUM.
_EGREGIOUS_EQUITY: dict[str, float] = {"preservation": 0.70}
# An income objective should hold at least this much income-producing exposure.
_MIN_INCOME_SHARE: dict[str, float] = {"income": 0.20}

_IM_2017_02 = "SEC IM Guidance Update No. 2017-02 (Robo-Advisers); Investment Advisers Act of 1940"


# ---------------------------------------------------------------------------
# Optional LLM reviewer hook (Check 3.2) — "LLM reviews, rule decides"
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MandateFinding:
    """One mandate conflict surfaced for Check 3.2. Produced deterministically,
    or optionally augmented by an injected LLM reviewer. Severity is assigned by
    the rule in `_finding_to_violation`, never by the finder."""
    ticker:     str
    reason:     str
    directness: str  # "direct" (names/holds the excluded thing) | "indirect" (broad-market spillover)
    subject:    str  # what the client excluded
    quote:      str  # verbatim client statement, for the audit trail


# A reviewer takes the active exclusion statements plus the portfolio and returns
# extra findings. Injected by the caller only when a live LLM is desired; the
# rule below always owns the decision.
MandateReviewer = Callable[[list[ClientStatement], dict[str, float]], list[MandateFinding]]


# ---------------------------------------------------------------------------
# Check 3.1 — Composition–objective consistency
# ---------------------------------------------------------------------------

def check_composition_objective(
    compliance_input: ComplianceInput,
) -> tuple[list[ComplianceViolation], list[str]]:
    """
    The proposed portfolio's composition must be consistent with the client's
    stated InvestmentObjective (SEC IM 2017-02 — the generated portfolio must
    reflect the objective the client selected).

    A PRESERVATION client in 90% equity, or an INCOME client with no
    income-producing holdings, is a composition the algorithm should never
    output. Severity HIGH when the mismatch is egregious, MEDIUM otherwise.

    Note: this reads proposed_portfolio as delivered. Until the safe/cash sleeve
    is represented as an actual holding (review items 1/8), these shares describe
    the risky sleeve; the bands are calibrated accordingly and tighten for free
    once the cash sleeve lands.
    """
    violations: list[ComplianceViolation] = []
    passed: list[str] = []
    check = "check_3_1_composition_objective"

    portfolio = compliance_input.proposed_portfolio
    objective = compliance_input.client_profile.investment_objective.lower()

    equity_w = sum(w for t, w in portfolio.items() if t in _EQUITY_ETFS)
    income_w = sum(w for t, w in portfolio.items() if t in _INCOME_ETFS)

    # Refuse to opine on a portfolio Compliance cannot classify. Without this,
    # an instrument missing from the classification sets is silently treated as
    # non-equity, so a stale map reads as "0% equity" — an assertion about the
    # portfolio rather than an admission about the map. See
    # _UNCLASSIFIED_WEIGHT_FLOOR.
    unclassified = {
        t: w for t, w in portfolio.items()
        if t not in _EQUITY_ETFS and t not in _INCOME_ETFS and t not in _ETF_ISSUER
    }
    unclassified_w = sum(unclassified.values())
    if unclassified_w > _UNCLASSIFIED_WEIGHT_FLOOR:
        violations.append(ComplianceViolation(
            check             = check,
            severity          = Severity.MEDIUM,
            description       = (
                f"{unclassified_w:.0%} of the portfolio is in instruments Compliance "
                f"cannot classify as equity or income ({', '.join(sorted(unclassified))}), "
                f"so composition cannot be checked against the '{objective}' objective. "
                f"This indicates the Compliance universe map is stale relative to the "
                f"Allocation Agent's universe, not that the portfolio is defective."
            ),
            rule_reference    = _IM_2017_02,
            responsible_agent = ResponsibleAgent.COMPLIANCE.value,
            action_required   = (
                "Update _TICKER_SECTOR / _EQUITY_ETFS / _INCOME_ETFS in "
                "job3_robo_adviser_checks.py to cover the current asset universe, "
                "then re-run compliance."
            ),
        ))
        return violations, passed

    band = _OBJECTIVE_EQUITY_BANDS.get(objective)
    if band is not None:
        lo, hi = band
        if equity_w > hi:
            egregious = equity_w > _EGREGIOUS_EQUITY.get(objective, 1.01)
            violations.append(ComplianceViolation(
                check             = check,
                severity          = Severity.HIGH if egregious else Severity.MEDIUM,
                description        = (
                    f"Equity share {equity_w:.0%} exceeds the {hi:.0%} ceiling for a "
                    f"'{objective}' objective — portfolio composition is inconsistent "
                    f"with the client's stated objective."
                ),
                rule_reference    = _IM_2017_02,
                responsible_agent = ResponsibleAgent.ALLOCATION.value,
                action_required   = (
                    f"Reduce equity exposure below {hi:.0%} or reconcile the stated "
                    f"objective with the recommendation."
                ),
            ))
        elif equity_w < lo:
            violations.append(ComplianceViolation(
                check             = check,
                severity          = Severity.MEDIUM,
                description        = (
                    f"Equity share {equity_w:.0%} is below the {lo:.0%} floor for a "
                    f"'{objective}' objective — the portfolio is too defensive to meet "
                    f"the client's stated growth objective."
                ),
                rule_reference    = _IM_2017_02,
                responsible_agent = ResponsibleAgent.ALLOCATION.value,
                action_required   = f"Raise equity exposure above {lo:.0%} for a growth mandate.",
            ))

    min_income = _MIN_INCOME_SHARE.get(objective)
    if min_income is not None and income_w < min_income:
        violations.append(ComplianceViolation(
            check             = check,
            severity          = Severity.MEDIUM,
            description        = (
                f"Income-producing share {income_w:.0%} is below the {min_income:.0%} "
                f"minimum for an 'income' objective — the portfolio does not deliver "
                f"the income the client asked for."
            ),
            rule_reference    = _IM_2017_02,
            responsible_agent = ResponsibleAgent.ALLOCATION.value,
            action_required   = f"Add income-producing holdings to reach ≥{min_income:.0%}.",
        ))

    if not violations:
        passed.append(check)
    return violations, passed


# ---------------------------------------------------------------------------
# Check 3.2 — Client-mandate consistency
# ---------------------------------------------------------------------------

def _active_exclusions(statements: list[ClientStatement]) -> list[ClientStatement]:
    """HARD_CONSTRAINT statements routed to a universe exclusion / sector underweight."""
    exclusion_dests = {
        PipelineDestination.UNIVERSE_EXCLUSION,
        PipelineDestination.SECTOR_UNDERWEIGHT,
    }
    return [
        s for s in statements
        if s.kind == StatementKind.HARD_CONSTRAINT
        and any(d in exclusion_dests for d in s.destinations)
    ]


def _deterministic_findings(
    statements: list[ClientStatement],
    portfolio: dict[str, float],
) -> list[MandateFinding]:
    """Set-membership conflict detection — the deterministic core of Check 3.2."""
    findings: list[MandateFinding] = []
    held = set(portfolio)

    for s in _active_exclusions(statements):
        subject = (s.subject or "").strip()
        subj_lower = subject.lower()

        # 1. Direct: the client named a ticker the portfolio holds.
        if subject.upper() in held:
            findings.append(MandateFinding(
                ticker=subject.upper(), subject=subject, quote=s.quote,
                directness="direct",
                reason=f"Portfolio directly holds {subject.upper()}, which the client excluded.",
            ))
            continue

        # 2a. Direct: the client named a sector whose ETF the portfolio holds.
        sector_etf = _SECTOR_ETF.get(subj_lower)
        if sector_etf and sector_etf in held:
            findings.append(MandateFinding(
                ticker=sector_etf, subject=subject, quote=s.quote,
                directness="direct",
                reason=(
                    f"Portfolio holds {sector_etf}, a direct proxy for the "
                    f"'{subject}' exposure the client excluded."
                ),
            ))
            continue

        # 2b. Direct: the client named a sector and the portfolio holds single
        #     names in it. Under an all-ETF universe a sector exclusion had one
        #     proxy to find; under a single-name universe it has several, and
        #     each held name is its own breach — "no oil and gas" against a book
        #     holding XOM, CVX and COP is three excluded positions, not one.
        #     Reported per ticker so the remediation names what to sell.
        sector = _SUBJECT_SECTOR.get(subj_lower)
        if sector:
            in_sector = sorted(t for t in held if _TICKER_SECTOR.get(t) == sector)
            if in_sector:
                for ticker in in_sector:
                    findings.append(MandateFinding(
                        ticker=ticker, subject=subject, quote=s.quote,
                        directness="direct",
                        reason=(
                            f"Portfolio holds {ticker}, a {sector} name, which falls "
                            f"inside the '{subject}' exposure the client excluded."
                        ),
                    ))
                continue

        # 3. Indirect: no direct proxy held, but broad-market equity may contain
        #    the excluded issuer/sector (e.g. SPY holds defense names).
        broad_held = sorted(held & _BROAD_MARKET_ETFS)
        if broad_held:
            findings.append(MandateFinding(
                ticker=broad_held[0], subject=subject, quote=s.quote,
                directness="indirect",
                reason=(
                    f"No direct proxy for '{subject}' is held, but broad-market "
                    f"holdings {broad_held} may contain excluded issuers; the all-ETF "
                    f"universe cannot honour a security-level exclusion."
                ),
            ))

    return findings


def _finding_to_violation(f: MandateFinding) -> Optional[ComplianceViolation]:
    """Map a mandate finding to a violation under EXCLUSION_POLICY. This is the
    rule that owns the decision — the LLM never reaches here. Returns None when
    the policy says an indirect finding is handled by disclosure (Check 3.3)."""
    check = "check_3_2_client_mandate"

    if f.directness == "direct":
        # A directly held excluded security is a hard breach under every policy
        # except pure disclosure.
        if EXCLUSION_POLICY == "disclose":
            severity = Severity.LOW
        else:
            severity = Severity.HIGH
    else:  # indirect
        if EXCLUSION_POLICY == "strict":
            severity = Severity.HIGH
        else:
            # tiered / disclose: broad-market spillover is a disclosure item,
            # surfaced by Check 3.3 rather than a hard fail here.
            return None

    return ComplianceViolation(
        check             = check,
        severity          = severity,
        description        = f"Client-mandate conflict on {f.ticker}: {f.reason} (client said: \"{f.quote}\")",
        rule_reference    = _IM_2017_02 + "; FINRA Rule 2111 — client-specific suitability",
        responsible_agent = ResponsibleAgent.ALLOCATION.value,
        action_required   = (
            f"Remove {f.ticker} or substitute a holding that honours the client's "
            f"exclusion of '{f.subject}'."
        ),
    )


def check_client_mandate(
    compliance_input: ComplianceInput,
    mandate_reviewer: Optional[MandateReviewer] = None,
) -> tuple[list[ComplianceViolation], list[str]]:
    """
    The proposed portfolio must honour the client's HARD_CONSTRAINT mandates
    (exclusions / ESG screens) captured at intake.

    Deterministic set membership catches direct conflicts always. When an LLM
    `mandate_reviewer` is injected, its findings are merged in to catch conflicts
    the keyword map misses — but `_finding_to_violation` (a rule) assigns every
    severity, so the LLM never sets clearance (SCOPE §3.9). Indirect
    broad-market spillover is deferred to Check 3.3 under the default 'tiered'
    policy; flip EXCLUSION_POLICY to change that.

    Vacuously passes when the client stated no exclusions.
    """
    check = "check_3_2_client_mandate"
    statements = compliance_input.client_statements
    portfolio = compliance_input.proposed_portfolio

    if not _active_exclusions(statements):
        return [], [check]

    findings = _deterministic_findings(statements, portfolio)
    if mandate_reviewer is not None:
        # Merge, de-duplicating on (ticker, subject) so the LLM can't double-count
        # a conflict the deterministic pass already found.
        seen = {(f.ticker, f.subject.lower()) for f in findings}
        for f in mandate_reviewer(_active_exclusions(statements), portfolio):
            if (f.ticker, f.subject.lower()) not in seen:
                findings.append(f)
                seen.add((f.ticker, f.subject.lower()))

    violations = [v for v in (_finding_to_violation(f) for f in findings) if v is not None]

    passed = [check] if not violations else []
    return violations, passed


# ---------------------------------------------------------------------------
# Check 3.3 — Algorithm-limitation disclosure
# ---------------------------------------------------------------------------

def check_algorithm_limitation_disclosure(
    compliance_input: ComplianceInput,
) -> tuple[list[ComplianceViolation], list[str]]:
    """
    SEC IM 2017-02 requires an algorithmic adviser to disclose the limitations
    and key assumptions of its algorithm when they bear on the recommendation.

    This check flags the *portfolio-specific* disclosures this run triggers:
      - a low-confidence regime classification driving the return views;
      - a client exclusion the all-ETF universe can only partially honour
        (broad-market indirect exposure, per the tiered EXCLUSION_POLICY).

    Each is a LOW-severity item (PASS_WITH_WARNINGS): the recommendation stands,
    but the disclosure must be surfaced to the client. Standing methodology
    disclosures (single-period, HC assumptions) belong in the report template,
    not a per-run check, so a clean portfolio passes 3.3.
    """
    violations: list[ComplianceViolation] = []
    passed: list[str] = []
    check = "check_3_3_algorithm_limitation_disclosure"

    # Trigger 1 — low-confidence regime.
    confidence = compliance_input.macro_regime.regime_confidence
    if confidence < DISCLOSURE_CONFIDENCE_FLOOR:
        violations.append(ComplianceViolation(
            check             = check,
            severity          = Severity.LOW,
            description        = (
                f"Regime classification confidence {confidence:.0%} is below the "
                f"{DISCLOSURE_CONFIDENCE_FLOOR:.0%} floor; the return views rest on a "
                f"low-confidence regime and this limitation must be disclosed."
            ),
            rule_reference    = _IM_2017_02,
            responsible_agent = ResponsibleAgent.COMPLIANCE.value,
            action_required   = (
                "Surface an algorithm-limitation disclosure noting the regime "
                "classification is low-confidence for this period."
            ),
        ))

    # Trigger 2 — exclusion mandate the universe can only honour indirectly.
    if EXCLUSION_POLICY in ("tiered", "disclose"):
        held = set(compliance_input.proposed_portfolio)
        indirect = [
            f for f in _deterministic_findings(
                compliance_input.client_statements, compliance_input.proposed_portfolio
            )
            if f.directness == "indirect"
        ]
        if indirect and held & _BROAD_MARKET_ETFS:
            subjects = sorted({f.subject for f in indirect})
            violations.append(ComplianceViolation(
                check             = check,
                severity          = Severity.LOW,
                description        = (
                    f"Client exclusion(s) {subjects} cannot be fully honoured by the "
                    f"all-ETF universe: broad-market holdings may contain excluded "
                    f"issuers. This algorithmic limitation must be disclosed."
                ),
                rule_reference    = _IM_2017_02,
                responsible_agent = ResponsibleAgent.COMPLIANCE.value,
                action_required   = (
                    "Disclose that security-level exclusions cannot be enforced within a "
                    "broad-market ETF universe; consider an ESG-screened or look-through "
                    "universe to honour the mandate directly."
                ),
            ))

    if not violations:
        passed.append(check)
    return violations, passed


# ---------------------------------------------------------------------------
# Check 3.4 — ETF-provider conflict-of-interest screen
# ---------------------------------------------------------------------------

def check_etf_provider_conflict(
    compliance_input: ComplianceInput,
) -> tuple[list[ComplianceViolation], list[str]]:
    """
    SEC IM 2017-02 / Advisers Act: an adviser compensated for recommending its
    own or an affiliated sponsor's products carries a conflict of interest that
    must be disclosed. This screens every held ETF's issuer against the adviser's
    declared affiliations (`_CONFLICTED_PROVIDERS`).

    With no affiliations declared the screen passes cleanly — but the mechanism
    runs on every portfolio and fires the moment a conflicted issuer is held.
    """
    violations: list[ComplianceViolation] = []
    passed: list[str] = []
    check = "check_3_4_etf_provider_conflict"

    for ticker in compliance_input.proposed_portfolio:
        issuer = _ETF_ISSUER.get(ticker)
        if issuer and issuer in _CONFLICTED_PROVIDERS:
            violations.append(ComplianceViolation(
                check             = check,
                severity          = Severity.MEDIUM,
                description        = (
                    f"{ticker} is issued by {issuer}, with which the adviser has a "
                    f"declared compensation/affiliation arrangement — an undisclosed "
                    f"conflict of interest."
                ),
                rule_reference    = _IM_2017_02,
                responsible_agent = ResponsibleAgent.COMPLIANCE.value,
                action_required   = (
                    f"Disclose the adviser's relationship with {issuer}, or substitute a "
                    f"non-affiliated equivalent for {ticker}."
                ),
            ))

    if not violations:
        passed.append(check)
    return violations, passed
