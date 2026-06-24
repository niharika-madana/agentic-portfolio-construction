from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class RiskDecision(str, Enum):
    APPROVE = "APPROVE"
    FLAG = "FLAG"
    REJECT = "REJECT"


class StressSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MarketRegime(str, Enum):
    NORMAL   = "normal"    # recent vol within 1.5× long-run baseline
    ELEVATED = "elevated"  # recent vol 1.5–2× baseline — widen caps by 25%
    CRISIS   = "crisis"    # recent vol > 2× baseline  — widen caps by 50%


class ConstraintType(str, Enum):
    SINGLE_NAME = "single_name"
    SECTOR = "sector"
    ECONOMIC_SECTOR = "economic_sector"
    EMPLOYER = "employer"
    RISK_PROFILE_DOWNGRADE = "risk_profile_downgrade"


# ── Human Capital & User Profile ──────────────────────────────────────────────

class HumanCapitalInput(BaseModel):
    """Inputs needed for the BMS human-capital valuation in core/human_capital.py."""
    present_value: float = Field(..., gt=0, description="PV of future labor income (H)")
    employer_ticker: str
    employer_sector: str                          # GICS sector string from Compustat
    income_volatility: float = Field(..., ge=0, description="Std dev of labor income growth")
    income_beta: float = Field(..., description="Correlation of labor income with market")
    years_to_retirement: int = Field(..., gt=0)
    discount_rate: float = Field(..., gt=0, description="Risk-free rate used to discount HC")


class UserProfile(BaseModel):
    financial_wealth: float = Field(..., gt=0, description="Total investable financial wealth (W)")
    human_capital: HumanCapitalInput
    risk_profile: RiskProfile


# ── Instrument Universe ────────────────────────────────────────────────────────

class InstrumentUniverse(BaseModel):
    """Pre-approved instrument whitelist with market data needed by the optimizer."""
    tickers: list[str] = Field(..., min_length=2)
    market_cap_weights: dict[str, float]          # ticker -> mkt-cap weight, sums to 1.0
    sectors: dict[str, str]                       # ticker -> GICS sector


# ── FLAG Loop Constraint Feedback ─────────────────────────────────────────────
# Defined before AllocationInput because AllocationInput references it.

class AllocationConstraint(BaseModel):
    """A single violated constraint returned by Risk on a FLAG decision."""
    constraint_type: ConstraintType
    target: str                                   # ticker or sector name
    current_value: float
    limit: float


# ── Allocation Agent Input ─────────────────────────────────────────────────────

class AllocationInput(BaseModel):
    """Everything the Allocation Agent receives — on first run and on FLAG re-entry."""
    user_profile: UserProfile
    universe: InstrumentUniverse
    flag_constraints: list[AllocationConstraint] = Field(
        default_factory=list,
        description="Constraints injected by Risk on FLAG; empty on first run",
    )
    flag_iteration: int = Field(default=0, ge=0, le=3)


# ── Allocation Agent Output ────────────────────────────────────────────────────

class WeightDecomposition(BaseModel):
    """Per-instrument weight broken into its three sources."""
    ticker: str
    total_weight: float = Field(..., ge=0.0, le=1.0)
    equilibrium_baseline: float   # CAPM market-cap derived
    view_tilt: float              # Fama-French view adjustment
    human_capital_offset: float   # w_fin adjustment from BMS


class FactorExposures(BaseModel):
    """Fama-French three-factor + momentum exposures, computed in core/."""
    market_beta: float
    smb: float    # small-minus-big
    hml: float    # high-minus-low
    mom: float    # momentum


class PortfolioStatistics(BaseModel):
    """Expected statistics computed deterministically in core/allocation.py."""
    expected_return: float
    volatility: float = Field(..., ge=0)
    sharpe_ratio: float
    factor_exposures: FactorExposures


class AllocationOutput(BaseModel):
    """
    Allocation Agent → Risk Agent handoff.
    All numbers originate in core/allocation.py — the LLM only wrote `rationale`.
    """
    allocation_input: AllocationInput
    weights: list[WeightDecomposition]
    risky_weight: float = Field(..., ge=0.0, le=1.0)
    safe_weight: float = Field(..., ge=0.0, le=1.0)
    portfolio_statistics: PortfolioStatistics
    rationale: str

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> AllocationOutput:
        total = sum(w.total_weight for w in self.weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Portfolio weights must sum to 1.0, got {total:.6f}")
        return self

    @model_validator(mode="after")
    def risky_safe_partition(self) -> AllocationOutput:
        if abs(self.risky_weight + self.safe_weight - 1.0) > 1e-6:
            raise ValueError("risky_weight + safe_weight must equal 1.0")
        return self


# ── Risk Agent Metric Types ────────────────────────────────────────────────────

class VaRMetrics(BaseModel):
    """VaR and CVaR at 95% and 99% per Basel III/IV, computed in core/risk.py."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float


class ConcentrationFlags(BaseModel):
    """Which concentration limits were breached (lists are empty when clean)."""
    single_name_breaches: list[str] = Field(default_factory=list)     # tickers > 10%
    sector_breaches: list[str] = Field(default_factory=list)           # sectors > 20%
    economic_sector_breaches: list[str] = Field(default_factory=list)  # total economic > 25%
    employer_breach: bool = False                                       # employer incl. HC > 15%


class StressResult(BaseModel):
    """Result of one historical stress scenario."""
    scenario: str                                 # e.g. "S&P 500, 2008"
    portfolio_loss: float = Field(..., ge=0)      # positive fraction, e.g. 0.54 means -54%
    threshold_breached: bool
    severity: StressSeverity


class HumanCapitalAdjustedMetrics(BaseModel):
    """
    Total economic balance sheet metrics — the core contribution of this project.
    Incorporates human capital per Bodie, Merton & Samuelson (1992).
    """
    total_wealth: float                           # W + H
    hc_fraction: float = Field(..., ge=0, le=1)  # H / (W + H)
    effective_equity_exposure: float              # financial equity + labor income equity equiv.
    economic_sector_exposures: dict[str, float]   # sector -> fraction of total wealth
    employer_concentration: float                 # employer position incl. HC / total wealth


class RiskMetrics(BaseModel):
    """All risk metrics computed deterministically in core/risk.py."""
    volatility: float = Field(..., ge=0)
    var_cvar: VaRMetrics
    max_drawdown: float = Field(..., ge=0)
    factor_exposures: FactorExposures
    liquidity_score: float = Field(..., ge=0, le=1)
    concentration: ConcentrationFlags
    hc_adjusted: HumanCapitalAdjustedMetrics
    stress_results: list[StressResult]
    market_regime: MarketRegime
    effective_drawdown_cap: float = Field(..., ge=0, le=1)


# ── Risk Agent Output / Compliance Input ──────────────────────────────────────

class RiskOutput(BaseModel):
    """
    Risk Agent → Compliance (APPROVE) or → Allocation (FLAG) or → Human Review (REJECT).
    Decision logic lives in core/risk.py; the LLM only wrote `reasoning_trace`.
    """
    decision: RiskDecision
    allocation_output: AllocationOutput
    risk_metrics: RiskMetrics
    reasoning_trace: str
    constraints_violated: list[AllocationConstraint] = Field(default_factory=list)
    flag_iteration: int = Field(default=0, ge=0, le=3)

    @model_validator(mode="after")
    def flag_requires_constraints(self) -> RiskOutput:
        if self.decision == RiskDecision.FLAG and not self.constraints_violated:
            raise ValueError("FLAG decision must include at least one violated constraint")
        return self

    @model_validator(mode="after")
    def flag_iteration_not_exhausted(self) -> RiskOutput:
        if self.decision == RiskDecision.FLAG and self.flag_iteration >= 3:
            raise ValueError(
                "FLAG iteration limit (3) reached — decision must be REJECT, not FLAG"
            )
        return self
