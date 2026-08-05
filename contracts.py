"""
Inter-agent contracts for the five-agent advisory pipeline.

Design rule: every NUMBER in these contracts is produced by deterministic
code (pandas, numpy, closed-form formulas, FRED data). The LLM never
invents a number. Where an LLM participates (rationale text, narrative),
its output is either parsed into a typed field and validated, or clearly
labelled prose that cites numbers computed elsewhere.

Agent flow:
  ProfileAgent   → ProfileAgentOutput
  ResearchAgent  → MacroRegimeSnapshot
  AllocationAgent(profile, macro)         → AllocationAgentOutput
  RiskAgent(allocation, profile, macro)   → RiskAgentOutput
  ComplianceAgent(compliance_input, risk) → ComplianceAgentOutput

The orchestrator assembles ComplianceInput from the upstream outputs and
runs the Risk→Allocation feedback loop and the Compliance→Allocation
feedback loop before returning a final AdvisorPackage.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import ClassVar, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared vocabularies
# ---------------------------------------------------------------------------

GICS_SECTORS: frozenset[str] = frozenset({
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
})
"""
The eleven GICS sectors, spelled exactly as the Allocation Agent's ETF sector
map spells them.

This is a shared vocabulary, not a Profile Agent detail. `industry_exposure_sector`
is matched by STRING EQUALITY on the allocation and risk side — an employer sector
of "Technology" does not match the ETF sector "Information Technology", so the
employer sector cap silently degrades to the generic sector limit and the employer
proxy ETF silently degrades to SPY. Both failures are invisible: no exception, no
warning, just a portfolio that was never actually constrained.

Keep this set and `agents/allocation/adapters.ETF_SECTORS` in agreement.
"""

NON_INVESTABLE_EMPLOYER_SECTORS: frozenset[str] = frozenset({
    "Education",
    "Government",
    "Nonprofit",
    "Legal",
})
"""
Employer sectors with no GICS equivalent, because no listed sector ETF tracks them.

These are legitimate values for `industry_exposure_sector` — a tenured professor's
employer really is not in an investable sector — and the sector-overlap guard is a
deliberate no-op for them. They are enumerated rather than allowed implicitly so
that a genuine typo ("Techonlogy") is still distinguishable from an honest absence.
"""

_SECTOR_ALIASES: dict[str, str] = {
    "technology":         "Information Technology",
    "tech":               "Information Technology",
    "information technology": "Information Technology",
    "it":                 "Information Technology",
    "healthcare":         "Health Care",
    "health care":        "Health Care",
    "financial services": "Financials",
    "finance":            "Financials",
    "financials":         "Financials",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples":   "Consumer Staples",
    "communication services": "Communication Services",
    "energy":             "Energy",
    "industrials":        "Industrials",
    "materials":          "Materials",
    "real estate":        "Real Estate",
    "utilities":          "Utilities",
    "education":          "Education",
    "government":         "Government",
    "nonprofit":          "Nonprofit",
    "legal":              "Legal",
}


def normalize_sector(sector: str) -> str:
    """
    Map an employer sector onto the canonical GICS spelling.

    Case- and spacing-insensitive, so "technology", "Technology" and
    "  TECHNOLOGY " all land on "Information Technology". Unrecognised values are
    returned stripped but otherwise untouched: this normalises vocabulary, it does
    not invent a classification for a sector nobody has mapped yet. Use
    `is_investable_sector()` to tell whether the result can be matched against an
    ETF sector map.
    """
    if not isinstance(sector, str):
        return sector
    return _SECTOR_ALIASES.get(sector.strip().lower(), sector.strip())


def is_investable_sector(sector: str) -> bool:
    """True when `sector` names a GICS sector with a corresponding sector ETF."""
    return normalize_sector(sector) in GICS_SECTORS


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

class LLMRole(str, Enum):
    """
    What the language model actually did in producing a given profile.

    Recorded per profile rather than asserted once in a design document, so the
    claim is auditable against the object that was produced: a profile built from
    the BLS occupation table says NONE because no model ran, and a profile built
    from a transcript by an LLM extractor says CREATOR because a model authored
    the inputs the formulas then consumed.

    The standing project rule is that models create and classify text while
    deterministic code computes and validates numbers (SCOPE.md §2.6). CREATOR
    names the model's half of that split — it authors the structured facts — and
    is never a claim that a model chose a portfolio weight.
    """
    CREATOR = "creator"  # a model authored the structured inputs; math validated them
    NONE    = "none"     # no model involved — BLS table or deterministic rule-based intake

class HumanCapitalType(str, Enum):
    """
    Qualitative label derived from income_equity_beta.
      bond-like   : β ≤ 0.3  (tenured professor, civil servant, nurse)
      mixed       : 0.3 < β ≤ 0.8  (engineer, sales rep, analyst)
      equity-like : β > 0.8  (tech exec with RSUs, broker, founder)
    """
    BOND_LIKE   = "bond-like"
    MIXED       = "mixed"
    EQUITY_LIKE = "equity-like"


class IncomeStability(str, Enum):
    """
    Qualitative label derived from income_volatility_sigma.
      high   : σ ≤ 0.10  (stable salary, minimal bonus)
      medium : 0.10 < σ ≤ 0.25  (bonus-driven salaried role)
      low    : σ > 0.25  (commission, RSU, or owner-operator income)
    """
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class RiskToleranceLevel(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE     = "moderate"
    AGGRESSIVE   = "aggressive"


class LiquidityNeeds(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class InvestmentObjective(str, Enum):
    GROWTH       = "growth"
    INCOME       = "income"
    PRESERVATION = "preservation"


class RiskDecision(str, Enum):
    APPROVE = "APPROVE"
    FLAG    = "FLAG"
    REJECT  = "REJECT"


class StressSeverity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class MarketRegime(str, Enum):
    """
    Current market volatility regime, detected from 60-day rolling vol vs full-sample baseline.
    Widens the drawdown cap during stress to prevent procyclical selling at market bottoms.
    """
    NORMAL   = "normal"    # rolling/baseline vol ratio < 1.5  — standard caps
    ELEVATED = "elevated"  # ratio 1.5–2.0                     — caps widened 25%
    CRISIS   = "crisis"    # ratio ≥ 2.0 (2008/COVID-scale)    — caps widened 50%


class RebalanceDecision(str, Enum):
    """
    Verdict on whether a detected regime change should move the client's book.

    A regime label flip is not by itself a reason to trade. The Research Agent's
    rebalance evaluator (agents/research/rebalance.py) runs four persona-independent
    evidence gates plus one persona-conditioned materiality gate before answering.
    Declining to trade on noise is a result, not a failure.
    """
    NO_CHANGE            = "no_change"             # regime_label == prior_regime; nothing to evaluate
    JUSTIFIED            = "justified"             # evidence gates pass AND the move is material for this client
    CHURN                = "churn"                 # change detected but not worth trading on
    INSUFFICIENT_HISTORY = "insufficient_history"  # too few observations to judge; fail loud rather than pass silently


class ConstraintType(str, Enum):
    SINGLE_NAME          = "single_name"
    SECTOR               = "sector"
    ECONOMIC_SECTOR      = "economic_sector"
    EMPLOYER             = "employer"
    RISK_PROFILE_DOWNGRADE = "risk_profile_downgrade"  # AGGRESSIVE→MODERATE→CONSERVATIVE on critical stress
    RISKY_WEIGHT_CAP     = "risky_weight_cap"  # direct cap on total risky weight, sized to the actual drawdown breach


class RiskProfile(str, Enum):
    """Maps 1-to-1 with RiskToleranceLevel; used by Allocation/Risk core modules."""
    CONSERVATIVE = "conservative"
    MODERATE     = "moderate"
    AGGRESSIVE   = "aggressive"


class ComplianceStatus(str, Enum):
    PASS               = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL               = "FAIL"


class Severity(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"
    NONE   = "NONE"


class ResponsibleAgent(str, Enum):
    PROFILE    = "profile_agent"
    RESEARCH   = "research_agent"
    ALLOCATION = "allocation_agent"
    RISK       = "risk_agent"
    COMPLIANCE = "compliance_agent"
    UNKNOWN    = "unknown"


# ---------------------------------------------------------------------------
# Agent 1 — Profile Agent intake (transcript → typed profile)
# ---------------------------------------------------------------------------

class FactSource(str, Enum):
    """
    Where a field's value came from. Kept distinct because merging them destroys
    the one thing suitability review needs: which constraints the client actually
    asserted, versus which the system decided on their behalf.

    "I'm 47" is STATED. "Your horizon is 16 years" is INFERRED.
    """
    STATED   = "stated"    # the client said it; evidence_quote is mandatory
    INFERRED = "inferred"  # derived from something the client said
    DEFAULT  = "default"   # population default; the client never addressed it
    UNKNOWN  = "unknown"   # never discussed and must not be guessed


class ExtractedField(BaseModel):
    """
    One field pulled from a client conversation, with its provenance.

    For a fiduciary system, provenance is close to mandatory: you must be able to
    show a client the sentence that produced a number in their recommendation.
    The validators below make an unsourced STATED field, or a guessed UNKNOWN
    field, impossible to construct rather than merely discouraged.
    """

    name:  str
    value: Optional[float | str | bool | dict[str, float]] = Field(
        default=None,
        description=(
            "None whenever source is UNKNOWN. The dict form carries a holdings "
            "breakdown (ticker → fraction) — a client who reads positions off a "
            "statement is asserting a structure, not a scalar, and flattening it "
            "would discard the concentration the pipeline exists to find."
        ),
    )
    source:     FactSource
    confidence: float = Field(
        ge=0, le=1,
        description="Extractor's stated confidence. Checked for calibration by intake_eval.py",
    )
    evidence_quote: Optional[str] = Field(
        default=None,
        description="Verbatim span from the transcript that supports the value",
    )
    evidence_line: Optional[int] = Field(
        default=None, ge=0,
        description="0-indexed transcript line the quote came from, for verification",
    )
    follow_up_question: Optional[str] = Field(
        default=None,
        description="What to ask the client. Mandatory when source is UNKNOWN.",
    )

    @model_validator(mode="after")
    def _check_provenance(self) -> "ExtractedField":
        if self.source == FactSource.STATED:
            if not self.evidence_quote:
                raise ValueError(
                    f"'{self.name}' is marked STATED but carries no evidence_quote; "
                    "a stated fact must point at the sentence that produced it"
                )
            if self.value is None:
                raise ValueError(f"'{self.name}' is STATED but has no value")

        if self.source == FactSource.UNKNOWN:
            if self.value is not None:
                raise ValueError(
                    f"'{self.name}' is UNKNOWN but carries value {self.value!r} — "
                    "an undiscussed field must not be guessed"
                )
            if not self.follow_up_question:
                raise ValueError(
                    f"'{self.name}' is UNKNOWN but proposes no follow_up_question; "
                    "the system must say what it needs to ask"
                )
            if self.confidence != 0.0:
                raise ValueError(
                    f"'{self.name}' is UNKNOWN but reports confidence "
                    f"{self.confidence}; it must be 0.0"
                )
        return self


class StatementKind(str, Enum):
    """
    What kind of object a thing the client said actually is.

    A discovery call does not yield one flat list of fields. It yields
    statements of different kinds, and the system has to sort them before it can
    act — a preference and a constraint look alike in prose and behave nothing
    alike in an optimiser.
    """

    HARD_CONSTRAINT  = "hard_constraint"
    """Something the client will not hold. Shrinks the feasible set, at a cost that can be measured and reported."""

    SOFT_PREFERENCE  = "soft_preference"
    """A tilt the client would like reflected. Not a constraint — closer to a Black-Litterman view, and needs a bounded budget before it can be honoured."""

    RISK_FACT        = "risk_fact"
    """
    A fact that arrives dressed as a preference but is really an exposure.

    Priya's ARVX holding is not a preference for her employer's stock; it is a
    concentration, a human-capital beta, and a forced sector underweight — one
    passage, three destinations. This is the category that makes sorting worth
    doing at all.
    """

    SUITABILITY_FACT = "suitability_fact"
    """Horizon, liquidity, account type, and what is in or out of scope. Often stated once and never repeated."""

    CHALLENGE        = "challenge"
    """
    A contradiction worth putting back to the client — a stated risk tolerance
    that disagrees with their actual exposure, or a self-description that
    disagrees with their holdings. Recorded for advisor review; never acted on
    automatically.
    """


class PipelineDestination(str, Enum):
    """Where a classified statement is supposed to end up."""

    UNIVERSE_EXCLUSION  = "universe_exclusion"
    CONCENTRATION_LIMIT = "concentration_limit"
    HUMAN_CAPITAL_BETA  = "human_capital_beta"
    SECTOR_UNDERWEIGHT  = "sector_underweight"
    BL_VIEW             = "bl_view"
    SUITABILITY_RECORD  = "suitability_record"
    SCOPE_BOUNDARY      = "scope_boundary"
    ADVISOR_REVIEW      = "advisor_review"


class ClientStatement(BaseModel):
    """
    One classified thing the client said, with its provenance and destinations.

    Distinct from ExtractedField: a field is a typed value the formulas consume,
    a statement is a piece of the conversation that has to be routed. The same
    sentence can produce both — Priya's ARVX line yields an `RSU_concentration`
    field *and* a RISK_FACT statement bound for three destinations.
    """

    kind:    StatementKind
    summary: str = Field(description="One line, in the system's words, of what this statement commits the client to")
    quote:   str = Field(description="Verbatim span from the transcript. Required — a routed statement must be showable to the client.")
    evidence_line: Optional[int] = Field(default=None, ge=0)

    destinations: list[PipelineDestination] = Field(
        default_factory=list,
        description="Every place this statement must reach. A RISK_FACT commonly has more than one.",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Ticker, sector, or asset the statement concerns, when it names one",
    )
    source:     FactSource = FactSource.STATED
    confidence: float      = Field(ge=0, le=1, default=0.5)

    @model_validator(mode="after")
    def _check_routable(self) -> "ClientStatement":
        if not self.quote.strip():
            raise ValueError(
                f"{self.kind.value} statement carries no quote; a statement that "
                "routes into the pipeline must be showable to the client"
            )
        if self.kind != StatementKind.CHALLENGE and not self.destinations:
            raise ValueError(
                f"{self.kind.value} statement has no destination — classifying it "
                "and then routing it nowhere is the failure this taxonomy exists to prevent"
            )
        return self


class ExtractedProfile(BaseModel):
    """
    The typed output of the intake layer — the 'extract' half of the standing
    rule that language models extract, classify and narrate while deterministic
    code computes.

    Nothing here is a portfolio number. This is what the client said, typed and
    sourced; ProfileAgentOutput is what the deterministic formulas make of it.
    """

    client_id:      str
    transcript_id:  str = Field(description="Identifies the source conversation")
    extractor:      str = Field(description="Which extraction strategy produced this")
    fields:         dict[str, ExtractedField]

    statements: list[ClientStatement] = Field(
        default_factory=list,
        description=(
            "Classified statements from the conversation. Empty for extractors "
            "that only recover typed fields — classification is a distinct "
            "capability from extraction and is reported separately."
        ),
    )

    def statements_of(self, kind: StatementKind) -> list[ClientStatement]:
        return [s for s in self.statements if s.kind == kind]

    def statements_for(self, destination: PipelineDestination) -> list[ClientStatement]:
        """Every statement that must reach a given part of the pipeline."""
        return [s for s in self.statements if destination in s.destinations]

    @property
    def unresolved(self) -> list[str]:
        """Field names the extractor declined to guess."""
        return sorted(
            name for name, f in self.fields.items() if f.source == FactSource.UNKNOWN
        )

    @property
    def stated(self) -> list[str]:
        """Field names the client asserted themselves."""
        return sorted(
            name for name, f in self.fields.items() if f.source == FactSource.STATED
        )

    def value_of(self, name: str):
        """Value for `name`, or None when absent or unknown."""
        field = self.fields.get(name)
        return field.value if field else None


# ---------------------------------------------------------------------------
# Agent 1 — Profile Agent output
# ---------------------------------------------------------------------------

class ProfileAgentOutput(BaseModel):
    """
    Structured intake produced by the Profile Agent.

    Human capital is computed via the annuity PV formula using the live
    FRED DGS10 10-year Treasury yield as the discount rate:

        HC = Annual Salary × [1 − (1 + r)^(−n)] / r

    The four income-risk fields below are all distinct and all required
    by downstream agents:

        income_volatility_sigma    — total earnings uncertainty (σ)
                                     used in the effective_risk_budget formula
        income_equity_correlation  — corr(income, equity market)
                                     used by Risk Agent for HC-adjusted sector limits
        income_equity_beta         — β of income to equity market = corr × (σ_income / σ_market)
                                     the systematic market risk embedded in the career;
                                     this is the primary driver of portfolio differentiation
        implicit_equity_exposure   — hc_share × income_equity_beta
                                     equity risk the client already carries through their
                                     career before a single stock is purchased;
                                     Allocation Agent subtracts this from the target
                                     risky-asset share so the total (career + portfolio)
                                     equity exposure is appropriate for the client's
                                     risk tolerance

    Reference: Ibbotson, Milevsky, Chen, Zhu (2007) — "Lifetime Financial
    Advice: Human Capital, Asset Allocation, and Insurance."
    """

    # ── Identity ──────────────────────────────────────────────────────
    client_id:    str
    career_type:  str
    age:          int = Field(ge=18, le=80)

    # ── Wealth components ─────────────────────────────────────────────
    financial_capital:          float = Field(gt=0, description="Current market value of all financial holdings")
    human_capital_valuation:    float = Field(gt=0, description="PV of future earnings via annuity formula (FRED DGS10 discount rate)")
    total_wealth:               float = Field(gt=0, description="financial_capital + human_capital_valuation")
    human_capital_pct_of_total: float = Field(ge=0, le=100, description="human_capital_valuation / total_wealth × 100")

    # ── Income risk — four distinct measures ──────────────────────────
    income_volatility_sigma:   float = Field(
        ge=0, le=1,
        description=(
            "Annualised standard deviation of earnings shocks (total income risk). "
            "Maps to income_stability label: high σ (>0.25) → 'low' stability. "
            "Used in: effective_risk_budget = (FC + HC×(1−σ)) / total_wealth"
        )
    )
    income_equity_correlation: float = Field(
        ge=-1.0, le=1.0,
        description=(
            "Correlation of annual income changes with equity market returns. "
            "Used by Risk Agent to compute HC-correlation-adjusted sector limits. "
            "Tech exec ≈ 0.75, biology professor ≈ 0.10, financial professional ≈ 0.50."
        )
    )
    income_equity_beta: float = Field(
        ge=-0.5, le=2.0,
        description=(
            "β of income shocks to equity market returns. "
            "β = corr(income, market) × (σ_income / σ_market). "
            "Drives the qualitative human_capital_type label and, more importantly, "
            "the implicit_equity_exposure that the Allocation Agent must offset. "
            "Tech exec ≈ 1.2, financial professional ≈ 0.6, biology professor ≈ 0.05."
        )
    )
    implied_market_volatility: Optional[float] = Field(
        default=None, ge=0, le=1,
        description=(
            "INTERNAL CONSISTENCY CHECK — no downstream agent consumes this; it exists "
            "so the calibration can be audited, not to be traded on. "
            "σ_market implied by inverting the single-factor identity: "
            "σ_market = income_equity_correlation × income_volatility_sigma / income_equity_beta. "
            "Since the 24 Jul review, income_equity_beta is DERIVED from ρ, σ_income and a "
            "single measured SIGMA_MARKET (0.1571 — annualised Fama-French mktrf, 312 months, "
            "2000-2025), so this must round-trip to ≈0.157 for every profile regardless of "
            "human-capital type. It previously returned 10% / 23% / 33% across the three types, "
            "which was one model claiming three different markets. A value away from 0.157 now "
            "means HC_BETA_TABLE has been hand-edited back out of consistency. "
            "None when income_equity_beta ≤ 0, where the identity is undefined."
        )
    )
    implicit_equity_exposure: float = Field(
        description=(
            "hc_share × income_equity_beta. "
            "The fraction of total wealth that is implicitly exposed to equity market risk "
            "through the client's career alone, before the investment portfolio is constructed. "
            "Allocation Agent: target_risky_share_in_portfolio = risk_budget − implicit_equity_exposure. "
            "This is the key number that makes the three personas produce different allocations."
        )
    )

    # ── Derived labels ────────────────────────────────────────────────
    human_capital_type:    HumanCapitalType   # derived from income_equity_beta
    income_stability:      IncomeStability    # derived from income_volatility_sigma
    effective_risk_budget: float = Field(
        ge=0, le=1,
        description="(FC + HC×(1−σ)) / total_wealth — risk capacity of the total balance sheet"
    )
    portfolio_equity_target: Optional[float] = Field(
        default=None,
        description=(
            "effective_risk_budget − implicit_equity_exposure. "
            "The residual equity capacity the Allocation Agent builds around. "
            "Negative when the career already provides more equity exposure than "
            "the total risk budget allows. Optional for backward compatibility — "
            "the Allocation Agent recomputes it from effective_risk_budget and "
            "implicit_equity_exposure if not supplied."
        )
    )

    # ── Career context ────────────────────────────────────────────────
    industry_exposure_sector: str = Field(
        description=(
            "GICS sector of the client's employer, or one of "
            "NON_INVESTABLE_EMPLOYER_SECTORS when no listed sector tracks it. "
            "Normalised to the canonical GICS spelling on construction — see "
            "normalize_sector(). Downstream code matches this by string equality "
            "against ETF sector labels, so the spelling is load-bearing: it selects "
            "the employer proxy ETF and keys the employer sector cap."
        )
    )
    RSU_concentration:        float = Field(ge=0, le=1, description="Fraction of financial holdings in employer RSUs")
    has_pension:              bool  = False
    bonus_rate:               float = Field(
        default=0.0, ge=0, le=1,
        description=(
            "Industry-specific bonus as a fraction of base salary (BLS ECEC Q1 2026). "
            "effective_salary = annual_salary × (1 + bonus_rate) feeds the HC annuity. "
            "Included in the Compliance Agent's total-compensation audit trail."
        )
    )

    # ── Holdings & preferences ────────────────────────────────────────
    current_holdings:         dict[str, float] = Field(description="Asset → weight; must sum to 1.0")
    investment_horizon_years: int   = Field(ge=1, le=50)
    risk_tolerance_level:     RiskToleranceLevel
    liquidity_needs:          LiquidityNeeds
    investment_objective:     InvestmentObjective

    # ── Provenance ────────────────────────────────────────────────────
    llm_role: LLMRole = Field(
        default=LLMRole.NONE,
        description=(
            "What the language model did in producing THIS profile. CREATOR when a "
            "model authored the structured facts (the transcript intake path via an "
            "LLM extractor); NONE when the profile came from the BLS occupation table "
            "or the deterministic rule-based extractor, where no model ran at all. "
            "Every number in the profile is computed by the formulas in "
            "agents/profile/profile_model.py regardless of this value — the flag "
            "records who supplied the inputs, never who computed the outputs."
        ),
    )

    # ── Client mandates (folded in from the intake layer) ─────────────
    client_statements: list[ClientStatement] = Field(
        default_factory=list,
        description=(
            "Classified statements from the intake conversation, carried over from "
            "ExtractedProfile.statements so the mandate survives into the pipeline. "
            "Compliance Job 3 reads HARD_CONSTRAINT and SOFT_PREFERENCE statements to "
            "verify the proposed portfolio honours the client's stated mandates "
            "(e.g. exclusions, ESG). Empty for personas built without a transcript — "
            "Job 3 then passes those checks vacuously."
        ),
    )

    # ── Validators ────────────────────────────────────────────────────

    @field_validator("industry_exposure_sector")
    @classmethod
    def _canonical_sector(cls, v: str) -> str:
        """
        Normalise the employer sector to the canonical GICS spelling.

        Normalisation happens here rather than at every call site because the
        consequence of a near-miss is silent: agents/allocation/adapters.py looks
        the sector up in ETF_SECTORS and _SECTOR_PROXY_ETF by exact string, and a
        miss falls through to SPY and to the generic 20% sector limit without
        raising. "Technology" instead of "Information Technology" therefore costs
        an RSU-heavy client their 10% employer sector cap and gives them a
        broad-market hedge proxy in place of a tech one.

        Unrecognised sectors pass through rather than raising — a sector this map
        has not seen is a mapping gap to fix, not a reason to refuse to build a
        client's profile — but they will not be investable, so
        check_sector_overlap() reports them as unmapped instead of silently
        passing.
        """
        return normalize_sector(v)

    @model_validator(mode="after")
    def _check_holdings_sum(self) -> "ProfileAgentOutput":
        if self.current_holdings:
            total = sum(self.current_holdings.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"current_holdings weights sum to {total:.4f}; must be 1.0 ±0.01"
                )
        return self

    @model_validator(mode="after")
    def _check_total_wealth_consistency(self) -> "ProfileAgentOutput":
        expected = self.financial_capital + self.human_capital_valuation
        if abs(expected - self.total_wealth) / self.total_wealth > 0.005:
            raise ValueError(
                f"total_wealth {self.total_wealth:,.0f} != "
                f"financial_capital ({self.financial_capital:,.0f}) + "
                f"human_capital_valuation ({self.human_capital_valuation:,.0f}) = {expected:,.0f}"
            )
        return self

    @model_validator(mode="after")
    def _check_implicit_equity_exposure(self) -> "ProfileAgentOutput":
        hc_share = self.human_capital_valuation / self.total_wealth
        expected = hc_share * self.income_equity_beta
        if abs(expected - self.implicit_equity_exposure) > 0.01:
            raise ValueError(
                f"implicit_equity_exposure {self.implicit_equity_exposure:.4f} != "
                f"hc_share ({hc_share:.4f}) × income_equity_beta ({self.income_equity_beta:.4f}) "
                f"= {expected:.4f}"
            )
        return self

    @model_validator(mode="after")
    def _check_hc_type_consistent_with_beta(self) -> "ProfileAgentOutput":
        β = self.income_equity_beta
        expected_type = (
            HumanCapitalType.BOND_LIKE   if β <= 0.3  else
            HumanCapitalType.MIXED       if β <= 0.8  else
            HumanCapitalType.EQUITY_LIKE
        )
        if self.human_capital_type != expected_type:
            raise ValueError(
                f"human_capital_type '{self.human_capital_type}' is inconsistent with "
                f"income_equity_beta {β:.2f} (expected '{expected_type}')"
            )
        return self


# ---------------------------------------------------------------------------
# Agent 2 — Research Agent output
# ---------------------------------------------------------------------------

class RegimeChangeEvidence(BaseModel):
    """
    Persona-independent evidence that a detected regime change is structural
    rather than classifier noise. Built by agents/research/rebalance.py from the
    regime sequence, the PELT break dates, and the XGBoost confidence path.

    Motivation: the 1995-2025 smoothed sequence contains 18 regime runs with a
    median length of 5 months, and 10 of them last 6 months or less. Six label
    flips occur between 2011-11 and 2013-10 alone. Rebalancing on
    regime_change_detected would have traded the book on every one of them.

    Every field is computed deterministically — no LLM involvement.
    """

    months_in_regime: int = Field(
        ge=0,
        description="Consecutive months the current label has held, as of the snapshot date",
    )
    min_dwell_months: int = Field(
        gt=0, description="Persistence threshold the run must clear to be actionable"
    )
    run_confidence: float = Field(
        ge=0, le=1,
        description="Mean XGBoost class probability across the current regime run",
    )
    pelt_corroborated: bool = Field(
        description=(
            "True when a PELT structural break falls within the tolerance window of "
            "regime_shift_date. A label flip with no break behind it is a classifier "
            "wobble in a stable macro environment, not a change of regime."
        )
    )
    months_to_nearest_break: Optional[int] = Field(
        default=None,
        description="Distance in months from regime_shift_date to the nearest PELT break; None when no breaks exist",
    )
    historical_reversion_rate: Optional[float] = Field(
        default=None, ge=0, le=1,
        description=(
            "Fraction of past occurrences of this exact prior→current transition that "
            "reverted to the prior regime within the reversion window. None when the "
            "transition has never been observed before."
        )
    )
    transition_sample_size: int = Field(
        ge=0, description="Number of past occurrences of this transition in the sequence"
    )
    gates_passed: list[str] = Field(
        default_factory=list, description="Names of the evidence gates that passed"
    )
    gates_failed: list[str] = Field(
        default_factory=list, description="Names of the evidence gates that failed"
    )

    @property
    def evidence_supports_change(self) -> bool:
        """True when every evidence gate passed."""
        return not self.gates_failed


class RebalanceEvaluation(BaseModel):
    """
    The Week 9 deliverable: given a detected regime change, is rebalancing this
    client's portfolio justified, or is it churn?

    Persona-conditioned. The same regime change can be justified for one client
    and churn for another, because the evidence gates are shared but the
    materiality gate is not: the regime tilt is applied to the client's own
    portfolio_equity_target. A client whose career already consumes their entire
    equity budget (high implicit_equity_exposure) has a target near zero, so no
    regime tilt moves their book enough to pay for the turnover.

    `explanation` is templated from the computed fields by deterministic code.
    It is not LLM-generated — this object is an input to compliance, not prose.
    """

    client_id:      str
    prior_regime:   str
    current_regime: str
    decision:       RebalanceDecision

    evidence: RegimeChangeEvidence

    # ── Materiality (persona-conditioned) ─────────────────────────────
    current_equity_target: float = Field(
        description="The client's portfolio_equity_target under the prior regime, in financial-wealth units"
    )
    proposed_equity_target: float = Field(
        description="Equity target under the current regime = current × (1 + regime tilt)"
    )
    equity_target_delta: float = Field(
        description="proposed − current. Signed; negative is de-risking. The size of the trade being proposed."
    )
    materiality_threshold: float = Field(
        gt=0,
        description="Minimum |equity_target_delta| worth trading on, net of turnover cost",
    )
    is_material: bool = Field(
        description="True when |equity_target_delta| >= materiality_threshold"
    )

    explanation: str = Field(
        description="Deterministically templated summary of which gates fired and why"
    )

    @model_validator(mode="after")
    def _check_delta_consistent(self) -> "RebalanceEvaluation":
        expected = self.proposed_equity_target - self.current_equity_target
        if abs(expected - self.equity_target_delta) > 1e-6:
            raise ValueError(
                f"equity_target_delta {self.equity_target_delta:.6f} != "
                f"proposed ({self.proposed_equity_target:.6f}) − "
                f"current ({self.current_equity_target:.6f}) = {expected:.6f}"
            )
        if self.is_material != (abs(self.equity_target_delta) >= self.materiality_threshold):
            raise ValueError(
                f"is_material {self.is_material} is inconsistent with "
                f"|delta| {abs(self.equity_target_delta):.6f} vs threshold {self.materiality_threshold:.6f}"
            )
        return self

    @model_validator(mode="after")
    def _check_decision_consistent(self) -> "RebalanceEvaluation":
        """JUSTIFIED requires both the evidence gates and the materiality gate."""
        if self.decision == RebalanceDecision.JUSTIFIED:
            if not self.evidence.evidence_supports_change:
                raise ValueError(
                    f"decision JUSTIFIED but evidence gates failed: {self.evidence.gates_failed}"
                )
            if not self.is_material:
                raise ValueError(
                    f"decision JUSTIFIED but equity target moves only "
                    f"{self.equity_target_delta:+.4f} (threshold {self.materiality_threshold:.4f})"
                )
        return self


class MacroRegimeSnapshot(BaseModel):
    """
    Single-date regime snapshot extracted from the Research Agent's
    date-keyed output dict. The orchestrator pulls the most recent entry.

    All numeric fields come from FRED data processed through the
    PELT → K-means → XGBoost pipeline — no LLM involvement at any stage.
    """

    as_of:             date
    regime_label:      str  = Field(description="e.g. 'AI Boom', 'Rate Hike Cycle'")
    prior_regime:      str  = Field(description="Regime from previous month — signals transitions")
    regime_shift_date: date = Field(description="First date of the current regime run")

    # ── Confidence & volatility ───────────────────────────────────────
    regime_confidence: float = Field(ge=0, le=1, description="XGBoost max class probability")
    regime_volatility: float = Field(ge=0, description="6-month rolling std of credit spread — macro stress level")

    # ── Raw FRED signals — DEPRECATED, pending removal ────────────────
    # The 24 Jul minutes ask this contract to "expose only the fields needed for
    # allocation (label, confidence, volatility)"; the 4 Aug minutes repeat it as
    # Research item 5. These six are consumed by NOTHING outside the Research
    # Agent — verified by grep across agents/risk, agents/allocation,
    # agents/compliance and agents/orchestrator — so they are dead weight on a
    # public contract and a schema-drift risk.
    #
    # They are deprecated rather than deleted because tests/test_research.py and
    # tests/test_compliance.py construct snapshots with them (5-6 references
    # each), and tests/ is outside the Profile/Research edit scope.
    #
    # As of 4 Aug they are OPTIONAL. That is the half of the removal that can be
    # done unilaterally: new code — including build_snapshot() — no longer passes
    # them, so nothing further can come to depend on them, while the existing test
    # fixtures that do pass them keep working unchanged. Deleting the six lines
    # outright is then a mechanical change for whoever owns tests/, with no
    # production caller left to update.
    #
    # The full signal matrix remains available in RegimeRecord and in
    # data/outputs/fred_macro_regimes.csv, so nothing is lost by dropping them.
    yield_curve:   Optional[float] = Field(default=None, deprecated=True, description="DEPRECATED — 10Y minus 2Y Treasury spread, pct points (T10Y2Y)")
    term_spread:   Optional[float] = Field(default=None, deprecated=True, description="DEPRECATED — 10Y minus 3M Treasury spread, pct points (T10Y3M)")
    fed_funds:     Optional[float] = Field(default=None, deprecated=True, description="DEPRECATED — Effective Federal Funds Rate, pct (FEDFUNDS)")
    unemployment:  Optional[float] = Field(default=None, ge=0, deprecated=True, description="DEPRECATED — Civilian Unemployment Rate, pct (UNRATE)")
    cpi:           Optional[float] = Field(default=None, deprecated=True, description="DEPRECATED — CPI YoY % change, derived from CPIAUCSL")
    credit_spread: Optional[float] = Field(default=None, ge=0, deprecated=True, description="DEPRECATED — Baa minus 10Y Treasury, pct points (BAA10Y)")

    # ── Derived flags (set automatically by validator) ────────────────
    is_low_confidence:      bool = Field(default=False, description="True when regime_confidence < 0.60")
    regime_change_detected: bool = Field(default=False, description="True when regime_label != prior_regime")

    # ── Rebalance evidence (Week 9 deliverable) ───────────────────────
    regime_change_evidence: Optional[RegimeChangeEvidence] = Field(
        default=None,
        description=(
            "Persona-independent evidence on whether the detected change is structural. "
            "regime_change_detected answers 'did the label move?'; this answers 'should "
            "anyone care?'. Pair with agents.research.rebalance.evaluate_rebalance(snapshot, "
            "profile) to get the persona-conditioned RebalanceEvaluation. Optional: None when "
            "the snapshot is built without the regime sequence and PELT breaks in hand."
        ),
    )

    DEPRECATED_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "yield_curve", "term_spread", "fed_funds", "unemployment", "cpi", "credit_spread",
    })
    """
    The six raw FRED signals above, named so serialisers can drop them.

    Pass as `exclude=` when writing a snapshot to disk — otherwise they serialise
    as explicit nulls, which is a worse artifact than the numbers were: a reader
    cannot tell a field that was deliberately retired from one whose computation
    failed. See agents/research/research_agent._save_outputs.
    """

    def for_allocation(self) -> dict:
        """
        The minimal view the 24 Jul minutes asked for: label, confidence, volatility.

        Consume this rather than the whole snapshot. It is the intended public
        surface once the deprecated raw signals above are removed, so code
        written against it will not need changing when they go.
        """
        return {
            "regime_label":      self.regime_label,
            "regime_confidence": self.regime_confidence,
            "regime_volatility": self.regime_volatility,
        }

    @model_validator(mode="after")
    def _set_derived_flags(self) -> "MacroRegimeSnapshot":
        self.is_low_confidence      = self.regime_confidence < 0.60
        self.regime_change_detected = self.regime_label != self.prior_regime
        return self


# ---------------------------------------------------------------------------
# Agent 3 — Allocation Agent output
# ---------------------------------------------------------------------------

class AllocationAgentOutput(BaseModel):
    """
    Portfolio recommendation from the Allocation Agent.

    The core allocation logic computes the risky weight via the Merton/
    Campbell-Viceira HC-adjusted formula (compute_w_fin, driven by
    income_equity_beta), then caps it at portfolio_equity_target
    (= effective_risk_budget − implicit_equity_exposure) from the profile,
    so the total (career + portfolio) equity exposure never exceeds the
    client's risk budget.

    This is what produces meaningfully different allocations for each
    persona: the tech exec's high implicit_equity_exposure (≈0.90) leaves
    little room for additional equity in the portfolio, while the biology
    professor's low implicit_equity_exposure (≈0.04) allows a much higher
    equity allocation despite their moderate risk tolerance.

    Every rationale string must cite at least one computed number
    (beta, regime, weight, mu/sigma). Generic boilerplate without client
    context is a compliance flag (Check 2.2).
    """

    proposed_portfolio:          dict[str, float] = Field(description="Ticker → weight; must sum to 1.0")
    allocation_rationale:        dict[str, str]   = Field(description="Ticker → rationale citing client-specific numbers")
    revision:                    int              = Field(default=0, ge=0, description="0 = first run; incremented on each re-run")
    prior_risk_flags:            list[str]        = Field(default_factory=list, description="Violations from RiskAgentOutput.violations passed back for revision")
    prior_compliance_violations: list[str]        = Field(default_factory=list, description="Violations from ComplianceAgentOutput passed back for revision")

    @model_validator(mode="after")
    def _validate_portfolio(self) -> "AllocationAgentOutput":
        p = self.proposed_portfolio
        if len(p) < 2:
            raise ValueError(f"Portfolio has {len(p)} position(s); minimum 2 required")
        for ticker, w in p.items():
            if w < 0:
                raise ValueError(f"{ticker} has negative weight {w:.4f}; no short positions allowed")
        total = sum(p.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Portfolio weights sum to {total:.4f}; must be 1.0 ±0.01")
        for ticker in p:
            if ticker not in self.allocation_rationale:
                raise ValueError(f"Missing rationale for portfolio position: {ticker}")
        return self


# ---------------------------------------------------------------------------
# Internal pipeline types — Allocation ↔ Risk feedback loop
# (Not consumed directly by Compliance; bridged via adapters.py)
# ---------------------------------------------------------------------------

class HumanCapitalInput(BaseModel):
    """HC inputs for BMS optimizer. Built from ProfileAgentOutput by adapters.py."""
    present_value:        float = Field(..., gt=0, description="PV of future labor income (H)")
    employer_ticker:      str   = Field(description="Sector-proxy ETF (e.g. XLK) when employer not publicly traded")
    employer_sector:      str   = Field(description="GICS sector of employer")
    income_volatility:    float = Field(..., ge=0, description="Annualised std dev of earnings shocks (σ)")
    income_beta:          float = Field(description="β of income to equity market")
    human_capital_type:   str   = Field(description="'bond-like' | 'mixed' | 'equity-like' — qualitative label derived from income_beta")
    rsu_concentration:    float = Field(default=0.0, ge=0, le=1, description="Fraction of financial holdings in employer RSUs; scales how much of HC is idiosyncratically tied to this one employer vs. diversifiable sector/market risk")
    years_to_retirement:  int   = Field(..., gt=0)
    discount_rate:        float = Field(..., gt=0, description="FRED DGS10 rate used to discount HC")


class UserProfile(BaseModel):
    """Internal user profile consumed by Allocation / Risk core modules."""
    financial_wealth:        float          = Field(..., gt=0, description="Total investable financial wealth (W)")
    human_capital:           HumanCapitalInput
    risk_profile:            RiskProfile
    portfolio_equity_target: float          = Field(
        description=(
            "effective_risk_budget − implicit_equity_exposure, computed by the Profile Agent. "
            "The residual equity capacity the Allocation Agent must respect: the optimizer's "
            "HC-adjusted risky weight (see compute_w_fin) is capped at this value so total "
            "(career + portfolio) equity exposure never exceeds the client's risk budget. "
            "Can be negative when the career alone already exceeds the budget, forcing the "
            "risky weight toward 0."
        )
    )


class InstrumentUniverse(BaseModel):
    """Pre-approved ETF whitelist with market data for the BL optimizer."""
    tickers:             list[str]        = Field(..., min_length=2)
    market_cap_weights:  dict[str, float] = Field(description="ticker → AUM-derived weight, sums to 1.0")
    sectors:             dict[str, str]   = Field(description="ticker → GICS sector")


class AllocationConstraint(BaseModel):
    """A single violated constraint returned by Risk on a FLAG decision."""
    constraint_type: ConstraintType
    target:          str   = Field(description="Ticker or sector name")
    current_value:   float
    limit:           float


class AllocationInput(BaseModel):
    """Allocation Agent input — first run and FLAG re-entry."""
    user_profile:      UserProfile
    universe:          InstrumentUniverse
    flag_constraints:  list[AllocationConstraint] = Field(default_factory=list)
    flag_iteration:    int = Field(default=0, ge=0, le=3)


class WeightDecomposition(BaseModel):
    """Per-instrument weight decomposed into three sources."""
    ticker:               str
    total_weight:         float = Field(..., ge=0.0, le=1.0)
    equilibrium_baseline: float
    view_tilt:            float
    human_capital_offset: float


class FactorExposures(BaseModel):
    """Fama-French three-factor + momentum exposures."""
    market_beta: float
    smb:         float
    hml:         float
    mom:         float


class PortfolioStatistics(BaseModel):
    expected_return: float
    volatility:      float = Field(..., ge=0)
    sharpe_ratio:    float
    factor_exposures: FactorExposures


class AllocationOutput(BaseModel):
    """Allocation Agent → Risk Agent internal handoff."""
    allocation_input:     AllocationInput
    weights:              list[WeightDecomposition]
    risky_weight:         float = Field(..., ge=0.0, le=1.0)
    safe_weight:          float = Field(..., ge=0.0, le=1.0)
    portfolio_statistics: PortfolioStatistics
    rationale:            str

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "AllocationOutput":
        total = sum(w.total_weight for w in self.weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Portfolio weights must sum to 1.0, got {total:.6f}")
        return self

    @model_validator(mode="after")
    def _risky_safe_partition(self) -> "AllocationOutput":
        if abs(self.risky_weight + self.safe_weight - 1.0) > 1e-6:
            raise ValueError("risky_weight + safe_weight must equal 1.0")
        return self


class VaRMetrics(BaseModel):
    var_95:  float
    var_99:  float
    cvar_95: float
    cvar_99: float


class ConcentrationFlags(BaseModel):
    single_name_breaches:     list[str] = Field(default_factory=list)
    sector_breaches:          list[str] = Field(default_factory=list)
    economic_sector_breaches: list[str] = Field(default_factory=list)
    employer_breach:          bool = False


class StressResult(BaseModel):
    scenario:            str
    portfolio_loss:      float = Field(..., ge=0)
    threshold_breached:  bool
    severity:            StressSeverity


class HumanCapitalAdjustedMetrics(BaseModel):
    total_wealth:              float
    hc_fraction:               float = Field(..., ge=0, le=1)
    effective_equity_exposure: float
    economic_sector_exposures: dict[str, float]
    employer_concentration:    float


class RiskMetrics(BaseModel):
    volatility:             float = Field(..., ge=0)
    var_cvar:               VaRMetrics
    max_drawdown:           float = Field(..., ge=0)
    factor_exposures:       FactorExposures
    liquidity_score:        float = Field(..., ge=0, le=1)
    concentration:          ConcentrationFlags
    hc_adjusted:            HumanCapitalAdjustedMetrics
    stress_results:         list[StressResult]
    market_regime:          MarketRegime   = MarketRegime.NORMAL
    effective_drawdown_cap: float          = Field(default=0.20, ge=0, description="Regime-adjusted drawdown cap used for FLAG/REJECT decisions")


class RiskOutput(BaseModel):
    """Internal Risk → Allocation feedback object (FLAG loop). Bridged to RiskAgentOutput for Compliance."""
    decision:             RiskDecision
    allocation_output:    AllocationOutput
    risk_metrics:         RiskMetrics
    reasoning_trace:      str
    constraints_violated: list[AllocationConstraint] = Field(default_factory=list)
    flag_iteration:       int = Field(default=0, ge=0, le=3)

    @model_validator(mode="after")
    def _flag_requires_constraints(self) -> "RiskOutput":
        if self.decision == RiskDecision.FLAG and not self.constraints_violated:
            raise ValueError("FLAG decision must include at least one violated constraint")
        return self

    @model_validator(mode="after")
    def _flag_iteration_not_exhausted(self) -> "RiskOutput":
        if self.decision == RiskDecision.FLAG and self.flag_iteration >= 3:
            raise ValueError("FLAG iteration limit (3) reached — decision must be REJECT, not FLAG")
        return self


# ---------------------------------------------------------------------------
# Agent 4 — Risk Agent output
# ---------------------------------------------------------------------------

class RegimeEvaluation(BaseModel):
    """Per-regime drawdown stress test vs. 80/20 benchmark."""
    benchmark_drawdown: float = Field(description="80/20 benchmark max drawdown in this regime window")
    drawdown_floor:     float = Field(description="Acceptable portfolio drawdown floor (= benchmark_drawdown for standard method)")
    portfolio_drawdown: float = Field(description="Proposed portfolio max drawdown in this regime window")
    passed:             bool


class PositionLimit(BaseModel):
    """Per-ticker concentration limit derived from marginal risk contribution."""
    derived_limit: float = Field(ge=0, le=1, description="Max allowable weight from marginal risk contribution method")
    actual_weight: float = Field(ge=0, le=1)
    passed:        bool
    method:        str   = Field(description="'marginal_risk_contribution' or 'static_lookup'")


class SectorLimit(BaseModel):
    """Per-sector limit adjusted for human capital correlation."""
    base_limit:           float = Field(ge=0, le=1, description="Unadjusted sector concentration cap")
    correlation_with_hc:  float = Field(ge=-1, le=1, description="Sector return correlation with client's income stream")
    adjusted_limit:       float = Field(ge=0, le=1, description="HC-correlation-adjusted sector cap; lower than base when corr is high")
    actual_sector_weight: float = Field(ge=0)
    passed:               bool


class RiskDerivation(BaseModel):
    """Methodology audit trail — Compliance Check 1.3 reads this."""
    drawdown_method:      str   = Field(description="'benchmark_relative_per_regime' or 'static_lookup'")
    concentration_method: str   = Field(description="'marginal_risk_contribution' or 'static_lookup'")
    sector_method:        str   = Field(description="'hc_correlation_adjusted' or 'static_lookup'")
    hc_type:              str   = Field(description="human_capital_type used to select sector_method")
    rsu_concentration:    float = Field(ge=0, le=1)
    data_source:          str   = Field(description="e.g. 'CRSP daily returns 2000-2024'")


class RiskAgentOutput(BaseModel):
    """
    Adversarial stress-test output from the Risk Agent.

    The Compliance Agent audits this output for internal consistency
    (Checks 1.1-1.3) before running its own independent checks.
    If portfolio_volatility_annual is provided, Compliance also runs
    the SUIT-VOL suitability check (Check 2.5).
    """

    risk_decision:     RiskDecision
    regime_evaluation: dict[str, RegimeEvaluation]
    position_limits:   dict[str, PositionLimit]
    sector_limits:     dict[str, SectorLimit]
    violations:        list[str]
    derivation:        RiskDerivation
    warnings:          list[str]         = Field(default_factory=list)

    market_regime:          Optional[MarketRegime] = Field(
        default=None,
        description="Vol-based regime from 60-day rolling vol ratio (NORMAL/ELEVATED/CRISIS)",
    )
    effective_drawdown_cap: Optional[float] = Field(
        default=None, ge=0,
        description="Regime-adjusted drawdown cap used for FLAG/REJECT decisions",
    )

    portfolio_volatility_annual: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Annualised portfolio volatility computed from historical returns. "
            "If provided, Compliance runs Check 2.5 (SUIT-VOL) against the "
            "client's risk tolerance suitability band. "
            "If None, Check 2.5 is skipped with a pipeline warning."
        )
    )


# ---------------------------------------------------------------------------
# Compliance Agent — assembled input
# ---------------------------------------------------------------------------

class ClientProfileSection(BaseModel):
    """
    Profile fields consumed by the Compliance Agent.
    Field names match input_validation.py exactly — do not rename.
    """
    persona:              str  = Field(description="Maps from ProfileAgentOutput.client_id")
    age:                  int
    risk_tolerance:       str  = Field(description="Maps from ProfileAgentOutput.risk_tolerance_level (lowercase)")
    time_horizon_years:   int  = Field(description="Maps from ProfileAgentOutput.investment_horizon_years")
    liquidity_needs:      str
    investment_objective: str


class HumanCapitalSection(BaseModel):
    """
    Human capital fields consumed by the Compliance Agent.
    Field names match input_validation.py exactly — do not rename.

    income_equity_beta is the primary field used in compliance checks
    (drives Check 1.3 derivation audit and Check 2.3 HC acknowledgment).
    income_equity_correlation is retained for the Risk Agent's sector
    limit adjustment audit in Check 1.2.
    """
    industry:                  str   = Field(description="Maps from ProfileAgentOutput.industry_exposure_sector")
    income_stability:          str
    income_equity_correlation: float = Field(ge=-1.0, le=1.0, description="corr(income, equity market) — used in sector limit audit")
    income_equity_beta:        float = Field(ge=-0.5, le=2.0, description="β of income to market — primary HC risk measure for compliance checks")
    implicit_equity_exposure:  float = Field(description="hc_share × β — equity already held via career; passed through for reference in compliance report")
    rsu_concentration:         float = Field(ge=0, le=1, description="Maps from ProfileAgentOutput.RSU_concentration")
    human_capital_type:        str   = Field(description="'bond-like' | 'mixed' | 'equity-like'")
    employer_sector:           str   = Field(description="Same as industry — used for sector limit audit")
    has_pension:               bool  = False
    estimated_hc_pv:           float = Field(description="Maps from ProfileAgentOutput.human_capital_valuation")


class MacroRegimeSection(BaseModel):
    """Macro regime context passed to the Compliance Agent."""
    current_regime:      str
    historical_analogue: Optional[str]  = None
    regime_risk_flags:   list[str]      = Field(default_factory=list)
    regime_confidence:   float          = Field(ge=0, le=1)
    regime_volatility:   str            = Field(description="'low' | 'medium' | 'high' — discretised from MacroRegimeSnapshot.regime_volatility float")


class RegimeChangeFlag(BaseModel):
    """Signals whether a regime transition was detected this run."""
    detected:           bool
    prior_regime:       Optional[str] = None
    current_regime:     Optional[str] = None
    shift_date:         Optional[str] = None
    rebalance_proposed: Optional[str] = None


class ComplianceInput(BaseModel):
    """
    Full aggregated input consumed by run_compliance().
    Assembled by the orchestrator's assemble_compliance_input() function
    from ProfileAgentOutput + MacroRegimeSnapshot + AllocationAgentOutput.
    """
    client_profile:       ClientProfileSection
    human_capital:        HumanCapitalSection
    macro_regime:         MacroRegimeSection
    proposed_portfolio:   dict[str, float]
    allocation_rationale: dict[str, str]
    regime_change_flag:   RegimeChangeFlag
    client_statements:    list[ClientStatement] = Field(
        default_factory=list,
        description=(
            "Client mandate statements threaded through from "
            "ProfileAgentOutput.client_statements. Consumed by Compliance Job 3 "
            "(robo-adviser checks). Defaults to empty so pre-intake callers and "
            "existing tests remain valid."
        ),
    )


# ---------------------------------------------------------------------------
# Agent 5 — Compliance Agent output
# ---------------------------------------------------------------------------

class ComplianceViolation(BaseModel):
    check:             str
    severity:          Severity
    description:       str
    rule_reference:    str
    responsible_agent: str
    action_required:   str


class AgentFeedbackItem(BaseModel):
    check:           str
    action_required: str


class ComplianceAgentOutput(BaseModel):
    compliance_status: ComplianceStatus
    overall_severity:  Severity
    violations:        list[ComplianceViolation]
    passed_checks:     list[str]
    clearance:         bool
    agent_feedback:    dict[str, list[AgentFeedbackItem]]
    recommendation:    str


# ---------------------------------------------------------------------------
# Final pipeline package
# ---------------------------------------------------------------------------

class RunMetadata(BaseModel):
    risk_revisions:          int              = 0
    compliance_revisions:    int              = 0
    final_risk_decision:     RiskDecision
    final_compliance_status: ComplianceStatus
    pipeline_warnings:       list[str]        = Field(default_factory=list)


class AdvisorPackage(BaseModel):
    """Complete output of the five-agent pipeline for one client persona."""
    profile:    ProfileAgentOutput
    macro:      MacroRegimeSnapshot
    allocation: AllocationAgentOutput
    risk:       RiskAgentOutput
    compliance: ComplianceAgentOutput
    metadata:   RunMetadata
