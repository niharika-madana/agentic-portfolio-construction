from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_system.schemas import (
    AllocationConstraint, AllocationOutput, ConcentrationFlags,
    ConstraintType, HumanCapitalAdjustedMetrics, RiskDecision,
    RiskMetrics, RiskOutput, RiskProfile, StressResult, StressSeverity,
    VaRMetrics,
)
from portfolio_system.core.constraints import (
    EMPLOYER_LIMIT, ECONOMIC_SECTOR_LIMIT, SINGLE_NAME_LIMIT,
    run_all_checks, all_violations,
)
from portfolio_system.core.human_capital import (
    total_wealth, hc_fraction as _hc_fraction,
    effective_equity_exposure, employer_concentration as hc_employer_conc,
    economic_sector_exposures as hc_econ_sector,
)


# ── Deterministic thresholds ──────────────────────────────────────────────────

MAX_DRAWDOWN_CAP: dict[RiskProfile, float] = {
    RiskProfile.CONSERVATIVE: 0.15,
    RiskProfile.MODERATE:     0.20,
    RiskProfile.AGGRESSIVE:   0.25,
}

# Stress scenarios anchored to observed historical losses (spec §3)
STRESS_SCENARIOS: list[dict] = [
    {
        "name":           "S&P 500, 2008",
        "start":          "2008-10-01",
        "end":            "2009-03-09",
        "benchmark_loss": 0.54,
    },
    {
        "name":           "COVID, 2020",
        "start":          "2020-02-19",
        "end":            "2020-03-23",
        "benchmark_loss": 0.34,
    },
    {
        "name":           "60/40 Portfolio, 2008",
        "start":          "2008-01-01",
        "end":            "2008-12-31",
        "benchmark_loss": 0.237,
    },
]

# Idiosyncratic employer shock assumptions
_EMPLOYER_STOCK_SHOCK = 0.50   # 50% drop in employer stock price
_EMPLOYER_HC_SHOCK    = 0.30   # 30% hit to human capital (correlated with employer)

# VaR lookback window (trading days)
VAR_LOOKBACK_DAYS = 1260   # ~5 years


# ── Daily returns matrix ──────────────────────────────────────────────────────

def _build_daily_matrix(
    crsp_daily: pd.DataFrame,
    tickers: list[str],
    permno_map: dict[str, int],
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pivots CRSP daily returns into a (T × N) array aligned to tickers.
    Returns (returns_matrix, available_weights_mask) where the mask is True
    for tickers that had data in the period. Missing tickers are silently dropped
    and the caller re-normalises weights accordingly.
    """
    inv_map = {permno_map[t]: t for t in tickers}
    df = crsp_daily[crsp_daily["permno"].isin(inv_map)].copy()
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    if df.empty:
        return np.empty((0, len(tickers))), np.zeros(len(tickers), dtype=bool)

    df["ticker"] = df["permno"].map(inv_map)
    pivot = df.pivot(index="date", columns="ticker", values="ret")

    # Keep only tickers present for the full period
    pivot = pivot.dropna(axis=1, how="any")
    available = [t for t in tickers if t in pivot.columns]
    matrix = pivot[available].values.astype(np.float64)
    mask = np.array([t in available for t in tickers])
    return matrix, mask


def _apply_mask(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return weights for available tickers, renormalized to sum to 1."""
    w = weights[mask]
    total = w.sum()
    return w / total if total > 0 else np.full(w.shape, 1.0 / len(w))


# ── VaR and CVaR ─────────────────────────────────────────────────────────────

def compute_var_cvar(
    weights: np.ndarray,
    daily_returns: np.ndarray,
) -> VaRMetrics:
    """
    Historical simulation VaR and CVaR at 95% and 99% per Basel III/IV.
    weights        : portfolio weights summing to 1 (N,)
    daily_returns  : daily return matrix (T × N)
    Returns losses as positive fractions (e.g. 0.02 = 2% daily VaR).
    """
    port_ret = daily_returns @ weights
    var_95   = float(-np.percentile(port_ret, 5))
    var_99   = float(-np.percentile(port_ret, 1))
    cvar_95  = float(-port_ret[port_ret <= -var_95].mean())
    cvar_99  = float(-port_ret[port_ret <= -var_99].mean())
    return VaRMetrics(var_95=var_95, var_99=var_99, cvar_95=cvar_95, cvar_99=cvar_99)


# ── Max drawdown ──────────────────────────────────────────────────────────────

def compute_max_drawdown(
    weights: np.ndarray,
    daily_returns: np.ndarray,
) -> float:
    """
    Maximum peak-to-trough cumulative loss of the portfolio.
    Returns a positive fraction (e.g. 0.20 = 20% drawdown).
    """
    port_ret    = daily_returns @ weights
    cum_ret     = np.cumprod(1.0 + port_ret)
    rolling_max = np.maximum.accumulate(cum_ret)
    drawdowns   = (cum_ret - rolling_max) / rolling_max
    return float(-drawdowns.min())


# ── Liquidity score ───────────────────────────────────────────────────────────

def compute_liquidity_score(
    weights: np.ndarray,
    tickers: list[str],
    crsp_daily: pd.DataFrame,
    permno_map: dict[str, int],
) -> float:
    """
    Portfolio-weighted average daily dollar volume, normalised to [0, 1].
    Uses log scale to handle the wide range in typical market-cap universes.
    """
    inv_map  = {permno_map[t]: t for t in tickers}
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    subset = crsp_daily[crsp_daily["permno"].isin(inv_map)].copy()
    subset["ticker"] = subset["permno"].map(inv_map)

    dollar_vol = np.zeros(len(tickers))
    for ticker, grp in subset.groupby("ticker"):
        if ticker in ticker_idx:
            dollar_vol[ticker_idx[ticker]] = (grp["vol"] * grp["prc"]).mean()

    log_vol = np.log1p(dollar_vol)
    max_log = log_vol.max()
    normalised = log_vol / max_log if max_log > 0 else np.zeros_like(log_vol)
    return float(weights @ normalised)


# ── Stress testing ────────────────────────────────────────────────────────────

def _severity(portfolio_loss: float, risk_profile: RiskProfile) -> StressSeverity:
    """Assign severity relative to the investor's own drawdown cap."""
    cap = MAX_DRAWDOWN_CAP[risk_profile]
    if portfolio_loss < cap * 0.5:
        return StressSeverity.LOW
    elif portfolio_loss < cap:
        return StressSeverity.MEDIUM
    elif portfolio_loss < cap * 1.5:
        return StressSeverity.HIGH
    else:
        return StressSeverity.CRITICAL


def _historical_portfolio_loss(
    weights: np.ndarray,
    tickers: list[str],
    crsp_daily: pd.DataFrame,
    permno_map: dict[str, int],
    start_date: str,
    end_date: str,
) -> float | None:
    """
    Cumulative portfolio loss during a stress period using actual CRSP returns.
    Returns None if data is unavailable for the period.
    """
    matrix, mask = _build_daily_matrix(
        crsp_daily, tickers, permno_map, start_date, end_date
    )
    if matrix.shape[0] == 0:
        return None
    w_adj     = _apply_mask(weights, mask)
    port_ret  = matrix @ w_adj
    cum_ret   = float(np.prod(1.0 + port_ret) - 1.0)
    return max(-cum_ret, 0.0)   # return as positive loss; 0 if portfolio gained


def compute_stress_results(
    weights: np.ndarray,
    tickers: list[str],
    crsp_daily: pd.DataFrame,
    permno_map: dict[str, int],
    risk_profile: RiskProfile,
    employer_ticker: str,
    employer_financial_weight: float,
    hc_frac: float,
) -> list[StressResult]:
    """
    Runs all stress scenarios. Historical data is used when available;
    falls back to beta-scaled benchmark loss when CRSP coverage is missing.

    Also runs an idiosyncratic employer shock scenario.
    """
    results: list[StressResult] = []
    cap = MAX_DRAWDOWN_CAP[risk_profile]

    for scenario in STRESS_SCENARIOS:
        loss = _historical_portfolio_loss(
            weights, tickers, crsp_daily, permno_map,
            scenario["start"], scenario["end"],
        )
        if loss is None:
            # Fallback: scale benchmark loss by portfolio market beta
            # Beta approximation: portfolio loss ≈ beta × market loss
            # (beta is available from AllocationOutput but we keep this module
            #  self-contained; use 1.0 as a conservative placeholder)
            loss = scenario["benchmark_loss"]

        results.append(StressResult(
            scenario          = scenario["name"],
            portfolio_loss    = loss,
            threshold_breached= loss > cap,
            severity          = _severity(loss, risk_profile),
        ))

    # Idiosyncratic employer shock
    # Employer stock drops _EMPLOYER_STOCK_SHOCK; HC drops _EMPLOYER_HC_SHOCK.
    # Total loss = financial hit + HC hit as fraction of total economic wealth.
    employer_fin_loss = employer_financial_weight * _EMPLOYER_STOCK_SHOCK
    employer_hc_loss  = hc_frac * _EMPLOYER_HC_SHOCK
    total_employer_loss = employer_fin_loss + employer_hc_loss

    results.append(StressResult(
        scenario          = "Idiosyncratic Employer Shock",
        portfolio_loss    = float(np.clip(total_employer_loss, 0.0, 1.0)),
        threshold_breached= total_employer_loss > cap,
        severity          = _severity(total_employer_loss, risk_profile),
    ))

    return results


# ── HC-adjusted metrics ───────────────────────────────────────────────────────

def compute_hc_adjusted_metrics(
    weights: np.ndarray,
    tickers: list[str],
    sectors: dict[str, str],
    financial_wealth: float,
    human_capital_pv: float,
    income_beta: float,
    employer_sector: str,
    employer_ticker: str,
) -> HumanCapitalAdjustedMetrics:
    """
    Total economic balance sheet metrics incorporating human capital
    per Bodie, Merton & Samuelson (1992). Called by run_risk().
    """
    weight_dict = {tickers[i]: float(weights[i]) for i in range(len(tickers))}

    emp_fin_weight = weight_dict.get(employer_ticker, 0.0)
    econ_sectors   = hc_econ_sector(
        weight_dict, sectors, financial_wealth, human_capital_pv, employer_sector
    )
    emp_conc = hc_employer_conc(emp_fin_weight, financial_wealth, human_capital_pv)
    eff_eq   = effective_equity_exposure(
        sum(weight_dict.values()), financial_wealth, human_capital_pv, income_beta
    )

    return HumanCapitalAdjustedMetrics(
        total_wealth              = total_wealth(financial_wealth, human_capital_pv),
        hc_fraction               = _hc_fraction(financial_wealth, human_capital_pv),
        effective_equity_exposure = eff_eq,
        economic_sector_exposures = econ_sectors,
        employer_concentration    = emp_conc,
    )


# ── Decision logic ────────────────────────────────────────────────────────────

def _drawdown_flag_constraints(
    tickers: list[str],
    weights: np.ndarray,
    current_mdd: float,
    cap: float,
) -> list[AllocationConstraint]:
    """
    When MDD exceeds the cap, scale down every ticker's limit proportionally
    so the optimizer is forced to reduce overall risk.
    reduction = cap / current_mdd  (e.g. 0.20 / 0.28 → tighten all by 71%)
    """
    reduction = cap / current_mdd
    return [
        AllocationConstraint(
            constraint_type=ConstraintType.SINGLE_NAME,
            target=t,
            current_value=float(weights[i]),
            limit=float(np.clip(weights[i] * reduction, 0.01, SINGLE_NAME_LIMIT)),
        )
        for i, t in enumerate(tickers)
        if weights[i] > 0
    ]


def make_decision(
    concentration_flags: ConcentrationFlags,
    violations: list[AllocationConstraint],
    max_drawdown: float,
    stress_results: list[StressResult],
    hc_adjusted: HumanCapitalAdjustedMetrics,
    risk_profile: RiskProfile,
    flag_iteration: int,
    tickers: list[str],
    weights: np.ndarray,
) -> tuple[RiskDecision, list[AllocationConstraint]]:
    """
    Deterministic APPROVE / FLAG / REJECT decision.

    REJECT conditions (issue upstream of Allocation — not fixable by reoptimisation):
      1. flag_iteration >= 3 : retries exhausted
      2. HC alone causes structural employer breach: hc_fraction > EMPLOYER_LIMIT
         (no financial reallocation can fix this)
      3. CRITICAL stress with no concentration violations (inherent risk issue)

    FLAG conditions (fixable by sending constraints back to Allocation):
      - Any concentration violation
      - Max drawdown exceeds the risk-profile cap

    APPROVE: no violations, MDD within cap.
    """
    cap = MAX_DRAWDOWN_CAP[risk_profile]
    drawdown_breach = max_drawdown > cap
    critical_stress = any(r.severity == StressSeverity.CRITICAL for r in stress_results)

    # ── REJECT ──
    if flag_iteration >= 3:
        return RiskDecision.REJECT, []

    if hc_adjusted.hc_fraction > EMPLOYER_LIMIT:
        # Human capital alone exceeds employer limit — structurally unfixable
        return RiskDecision.REJECT, []

    if critical_stress and not violations and not drawdown_breach:
        # Extreme stress with no addressable concentration problem
        return RiskDecision.REJECT, []

    # ── FLAG ──
    flag_constraints: list[AllocationConstraint] = list(violations)

    if drawdown_breach:
        flag_constraints += _drawdown_flag_constraints(tickers, weights, max_drawdown, cap)

    if flag_constraints:
        return RiskDecision.FLAG, flag_constraints

    # ── APPROVE ──
    return RiskDecision.APPROVE, []


# ── Main entry point ──────────────────────────────────────────────────────────

def run_risk(
    allocation_output: AllocationOutput,
    crsp_daily: pd.DataFrame,
    permno_map: dict[str, int],
) -> RiskOutput:
    """
    Full deterministic risk evaluation pipeline.

    Steps:
      1. Build daily returns matrix (VAR_LOOKBACK_DAYS)
      2. VaR / CVaR  (historical simulation, Basel III/IV)
      3. Max drawdown
      4. Liquidity score
      5. Stress scenarios  (historical + employer shock)
      6. HC-adjusted balance sheet metrics  (BMS 1992)
      7. Concentration checks  (calls core/constraints.py)
      8. Decision  (APPROVE / FLAG / REJECT)

    reasoning_trace is left as "" — agents/risk_agent.py fills it in.

    Args:
        allocation_output : validated AllocationOutput from Allocation Agent
        crsp_daily        : from data/loaders.load_crsp_daily()
                            (should cover VAR_LOOKBACK_DAYS + all stress periods)
        permno_map        : {ticker: permno} from data/loaders.tickers_to_permnos()
    """
    ao      = allocation_output
    ai      = ao.allocation_input
    up      = ai.user_profile
    hc      = up.human_capital
    universe= ai.universe
    tickers = universe.tickers

    # Actual financial weights = within-risky weights × risky fraction
    weights = np.array([w.total_weight * ao.risky_weight for w in ao.weights])
    weight_dict = {tickers[i]: float(weights[i]) for i in range(len(tickers))}

    # ── 1. Recent returns for VaR / MDD ──
    recent_matrix, mask = _build_daily_matrix(
        crsp_daily, tickers, permno_map,
        start_date=None, end_date=None,
    )

    # Use last VAR_LOOKBACK_DAYS rows if we have more
    if recent_matrix.shape[0] > VAR_LOOKBACK_DAYS:
        recent_matrix = recent_matrix[-VAR_LOOKBACK_DAYS:]

    w_adj = _apply_mask(weights, mask)

    # ── 2. VaR / CVaR ──
    var_cvar = compute_var_cvar(w_adj, recent_matrix) if recent_matrix.shape[0] > 30 \
               else VaRMetrics(var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0)

    # ── 3. Max drawdown ──
    mdd = compute_max_drawdown(w_adj, recent_matrix) if recent_matrix.shape[0] > 30 \
          else 0.0

    # ── 4. Liquidity ──
    liq = compute_liquidity_score(weights, tickers, crsp_daily, permno_map)

    # ── 5. Stress scenarios ──
    employer_fin_weight = weight_dict.get(hc.employer_ticker, 0.0)
    hc_frac             = _hc_fraction(up.financial_wealth, hc.present_value)

    stress_results = compute_stress_results(
        weights, tickers, crsp_daily, permno_map,
        up.risk_profile, hc.employer_ticker,
        employer_fin_weight, hc_frac,
    )

    # ── 6. HC-adjusted metrics ──
    hc_adj = compute_hc_adjusted_metrics(
        weights, tickers, universe.sectors,
        up.financial_wealth, hc.present_value,
        hc.income_beta, hc.employer_sector, hc.employer_ticker,
    )

    # ── 7. Concentration checks ──
    flags      = run_all_checks(weight_dict, universe.sectors,
                                hc_adj.economic_sector_exposures,
                                hc_adj.employer_concentration)
    violations = all_violations(weight_dict, universe.sectors,
                                hc_adj.economic_sector_exposures,
                                hc_adj.employer_concentration)

    # ── 8. Decision ──
    decision, flag_constraints = make_decision(
        concentration_flags=flags,
        violations=violations,
        max_drawdown=mdd,
        stress_results=stress_results,
        hc_adjusted=hc_adj,
        risk_profile=up.risk_profile,
        flag_iteration=ai.flag_iteration,
        tickers=tickers,
        weights=weights,
    )

    risk_metrics = RiskMetrics(
        volatility       = ao.portfolio_statistics.volatility,
        var_cvar         = var_cvar,
        max_drawdown     = mdd,
        factor_exposures = ao.portfolio_statistics.factor_exposures,
        liquidity_score  = liq,
        concentration    = flags,
        hc_adjusted      = hc_adj,
        stress_results   = stress_results,
    )

    return RiskOutput(
        decision            = decision,
        allocation_output   = ao,
        risk_metrics        = risk_metrics,
        reasoning_trace     = "",   # filled by risk_agent.py
        constraints_violated= flag_constraints,
        flag_iteration      = ai.flag_iteration + (1 if decision == RiskDecision.FLAG else 0),
    )
