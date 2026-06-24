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
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

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


class ConstraintType(str, Enum):
    SINGLE_NAME     = "single_name"
    SECTOR          = "sector"
    ECONOMIC_SECTOR = "economic_sector"
    EMPLOYER        = "employer"


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
    industry_exposure_sector: str = Field(description="GICS-aligned sector of client's employer")
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

    # ── Validators ────────────────────────────────────────────────────

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

    # ── Raw FRED signals (all from the 6-feature signal matrix) ───────
    yield_curve:   float = Field(description="10Y minus 2Y Treasury spread, pct points (T10Y2Y)")
    term_spread:   float = Field(description="10Y minus 3M Treasury spread, pct points (T10Y3M)")
    fed_funds:     float = Field(description="Effective Federal Funds Rate, pct (FEDFUNDS)")
    unemployment:  float = Field(ge=0, description="Civilian Unemployment Rate, pct (UNRATE)")
    cpi:           float = Field(description="CPI YoY % change, derived from CPIAUCSL")
    credit_spread: float = Field(ge=0, description="Baa minus 10Y Treasury, pct points (BAA10Y)")

    # ── Derived flags (set automatically by validator) ────────────────
    is_low_confidence:      bool = Field(default=False, description="True when regime_confidence < 0.60")
    regime_change_detected: bool = Field(default=False, description="True when regime_label != prior_regime")

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

    The core allocation logic uses implicit_equity_exposure from the
    profile to compute the portfolio equity target:

        portfolio_risky_share = effective_risk_budget − implicit_equity_exposure

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
    years_to_retirement:  int   = Field(..., gt=0)
    discount_rate:        float = Field(..., gt=0, description="FRED DGS10 rate used to discount HC")


class UserProfile(BaseModel):
    """Internal user profile consumed by Allocation / Risk core modules."""
    financial_wealth: float          = Field(..., gt=0, description="Total investable financial wealth (W)")
    human_capital:    HumanCapitalInput
    risk_profile:     RiskProfile


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
    volatility:       float = Field(..., ge=0)
    var_cvar:         VaRMetrics
    max_drawdown:     float = Field(..., ge=0)
    factor_exposures: FactorExposures
    liquidity_score:  float = Field(..., ge=0, le=1)
    concentration:    ConcentrationFlags
    hc_adjusted:      HumanCapitalAdjustedMetrics
    stress_results:   list[StressResult]


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
