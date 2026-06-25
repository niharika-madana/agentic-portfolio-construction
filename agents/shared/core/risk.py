from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import (
    AllocationConstraint, AllocationOutput, ConcentrationFlags,
    ConstraintType, HumanCapitalAdjustedMetrics, MarketRegime, RiskDecision,
    RiskMetrics, RiskOutput, RiskProfile, StressResult, StressSeverity,
    VaRMetrics,
)
from agents.shared.core.constraints import (
    EMPLOYER_LIMIT, ECONOMIC_SECTOR_LIMIT, SINGLE_NAME_LIMIT,
    run_all_checks, all_violations,
)
from agents.shared.core.human_capital import (
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

# Crisis multipliers: widen the drawdown cap during high-volatility regimes so
# the system doesn't force procyclical selling at market bottoms.
_REGIME_CAP_MULTIPLIER: dict[MarketRegime, float] = {
    MarketRegime.NORMAL:   1.00,  # standard cap
    MarketRegime.ELEVATED: 1.25,  # 25% wider — elevated but not extreme vol
    MarketRegime.CRISIS:   1.50,  # 50% wider — 2008/COVID-style dislocations
}

# Regime detection parameters
_REGIME_LOOKBACK_DAYS   = 60    # recent window for vol estimation
_ELEVATED_VOL_THRESHOLD = 1.50  # recent/long-run vol ratio → ELEVATED
_CRISIS_VOL_THRESHOLD   = 2.00  # recent/long-run vol ratio → CRISIS

STRESS_SCENARIOS: list[dict] = [
    {"name": "S&P 500, 2008",        "start": "2008-10-01", "end": "2009-03-09", "benchmark_loss": 0.54},
    {"name": "COVID, 2020",           "start": "2020-02-19", "end": "2020-03-23", "benchmark_loss": 0.34},
    {"name": "60/40 Portfolio, 2008", "start": "2008-01-01", "end": "2008-12-31", "benchmark_loss": 0.237},
]

_EMPLOYER_STOCK_SHOCK = 0.50
_EMPLOYER_HC_SHOCK    = 0.30
VAR_LOOKBACK_DAYS     = 1260


# ── Daily returns matrix ──────────────────────────────────────────────────────

def _build_daily_matrix(
    crsp_daily: pd.DataFrame,
    tickers: list[str],
    permno_map: dict[str, int],
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    pivot = pivot.dropna(axis=1, how="any")
    available = [t for t in tickers if t in pivot.columns]
    matrix = pivot[available].values.astype(np.float64)
    mask = np.array([t in available for t in tickers])
    return matrix, mask


def _apply_mask(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    w = weights[mask]
    total = w.sum()
    return w / total if total > 0 else np.full(w.shape, 1.0 / len(w))


# ── VaR and CVaR ─────────────────────────────────────────────────────────────

def compute_var_cvar(
    weights: np.ndarray,
    daily_returns: np.ndarray,
) -> VaRMetrics:
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
    inv_map    = {permno_map[t]: t for t in tickers}
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    subset = crsp_daily[crsp_daily["permno"].isin(inv_map)].copy()
    subset["ticker"] = subset["permno"].map(inv_map)
    dollar_vol = np.zeros(len(tickers))
    for ticker, grp in subset.groupby("ticker"):
        if ticker in ticker_idx:
            prc_col = "prc" if "prc" in grp.columns else "close"
            vol_col = "vol" if "vol" in grp.columns else "volume"
            dollar_vol[ticker_idx[ticker]] = (
                grp[prc_col].abs() * grp[vol_col].abs()
            ).mean()
    log_vol    = np.log1p(dollar_vol)
    max_log    = log_vol.max()
    normalised = log_vol / max_log if max_log > 0 else np.zeros_like(log_vol)
    total = weights.sum()
    w_norm = weights / total if total > 0 else weights
    return float(w_norm @ normalised)


# ── Regime detection ──────────────────────────────────────────────────────────

def detect_regime(port_returns: np.ndarray) -> MarketRegime:
    """
    Classify the current market regime from portfolio return volatility.

    Compares the annualized vol of the most recent _REGIME_LOOKBACK_DAYS trading
    days to the full-sample baseline. When the ratio exceeds the thresholds the
    regime widens the drawdown cap, preventing procyclical selling at market bottoms.
    Reverts to NORMAL automatically as volatility normalises.
    """
    if len(port_returns) < _REGIME_LOOKBACK_DAYS + 30:
        return MarketRegime.NORMAL

    long_vol   = float(port_returns.std())
    recent_vol = float(port_returns[-_REGIME_LOOKBACK_DAYS:].std())
    ratio = recent_vol / long_vol if long_vol > 0 else 1.0

    if ratio >= _CRISIS_VOL_THRESHOLD:
        return MarketRegime.CRISIS
    elif ratio >= _ELEVATED_VOL_THRESHOLD:
        return MarketRegime.ELEVATED
    else:
        return MarketRegime.NORMAL


def effective_cap(risk_profile: RiskProfile, regime: MarketRegime) -> float:
    """Regime-adjusted drawdown cap. Widens during stress; reverts when vol normalises."""
    return MAX_DRAWDOWN_CAP[risk_profile] * _REGIME_CAP_MULTIPLIER[regime]


# ── Stress testing ────────────────────────────────────────────────────────────

def _severity(portfolio_loss: float, cap: float) -> StressSeverity:
    """Assign severity relative to the effective (regime-adjusted) drawdown cap."""
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
    matrix, mask = _build_daily_matrix(crsp_daily, tickers, permno_map, start_date, end_date)
    if matrix.shape[0] == 0:
        return None
    w_available = weights[mask]
    port_ret    = matrix @ w_available
    cum_ret     = float(np.prod(1.0 + port_ret) - 1.0)
    return max(-cum_ret, 0.0)


def compute_stress_results(
    weights: np.ndarray,
    tickers: list[str],
    crsp_daily: pd.DataFrame,
    permno_map: dict[str, int],
    drawdown_cap: float,
    employer_ticker: str,
    employer_financial_weight: float,
    hc_frac: float,
) -> list[StressResult]:
    """
    Runs all stress scenarios against the regime-adjusted effective cap.
    Severity thresholds scale with the cap so a 25% loss in a crisis regime
    (cap=30%) is HIGH not CRITICAL.
    """
    results: list[StressResult] = []

    for scenario in STRESS_SCENARIOS:
        loss = _historical_portfolio_loss(
            weights, tickers, crsp_daily, permno_map,
            scenario["start"], scenario["end"],
        )
        if loss is None:
            loss = scenario["benchmark_loss"]

        results.append(StressResult(
            scenario           = scenario["name"],
            portfolio_loss     = loss,
            threshold_breached = loss > drawdown_cap,
            severity           = _severity(loss, drawdown_cap),
        ))

    employer_fin_loss   = employer_financial_weight * _EMPLOYER_STOCK_SHOCK
    employer_hc_loss    = hc_frac * _EMPLOYER_HC_SHOCK
    total_employer_loss = employer_fin_loss + employer_hc_loss
    results.append(StressResult(
        scenario           = "Idiosyncratic Employer Shock",
        portfolio_loss     = float(np.clip(total_employer_loss, 0.0, 1.0)),
        threshold_breached = total_employer_loss > drawdown_cap,
        severity           = _severity(total_employer_loss, drawdown_cap),
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
    weight_dict    = {tickers[i]: float(weights[i]) for i in range(len(tickers))}
    emp_fin_weight = weight_dict.get(employer_ticker, 0.0)
    econ_sectors   = hc_econ_sector(weight_dict, sectors, financial_wealth, human_capital_pv, employer_sector)
    emp_conc       = hc_employer_conc(emp_fin_weight, financial_wealth, human_capital_pv)
    eff_eq         = effective_equity_exposure(
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

_PROFILE_DOWNGRADE: dict[RiskProfile, RiskProfile] = {
    RiskProfile.AGGRESSIVE: RiskProfile.MODERATE,
    RiskProfile.MODERATE:   RiskProfile.CONSERVATIVE,
}


def _stress_flag_constraints(
    effective_profile: RiskProfile,
) -> list[AllocationConstraint]:
    """
    Translate a critical stress breach into a risk-profile downgrade constraint.
    AGGRESSIVE → MODERATE → CONSERVATIVE, one step per FLAG iteration.
    Returns empty list when already at CONSERVATIVE (caller should REJECT instead).
    """
    target = _PROFILE_DOWNGRADE.get(effective_profile)
    if target is None:
        return []
    return [AllocationConstraint(
        constraint_type=ConstraintType.RISK_PROFILE_DOWNGRADE,
        target=target.value,
        current_value=0.0,
        limit=0.0,
    )]


def _drawdown_flag_constraints(
    tickers: list[str],
    weights: np.ndarray,
    current_mdd: float,
    cap: float,
) -> list[AllocationConstraint]:
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
    effective_risk_profile: RiskProfile,
    regime: MarketRegime,
    flag_iteration: int,
    tickers: list[str],
    weights: np.ndarray,
) -> tuple[RiskDecision, list[AllocationConstraint]]:
    """
    Deterministic APPROVE / FLAG / REJECT decision.

    REJECT conditions (structurally unfixable):
      - hc_fraction > EMPLOYER_LIMIT: HC alone breaches employer limit
      - CRITICAL stress while already at CONSERVATIVE: no further downgrade possible
      - flag_iteration >= 2 with real violations remaining

    FLAG conditions (fed back to Allocation as tighter constraints):
      - Concentration violations        → per-ticker tighter bounds
      - Max drawdown exceeds cap        → proportional single-name tightening
      - CRITICAL stress                 → RISK_PROFILE_DOWNGRADE one step

    Smart convergence: at flag_iteration >= 2, only real violations (drawdown
    breach, concentration, critical stress) trigger REJECT. A carry-forward
    profile downgrade alone does not — the portfolio has converged.
    """
    cap             = effective_cap(effective_risk_profile, regime)
    drawdown_breach = max_drawdown > cap
    critical_stress = any(r.severity == StressSeverity.CRITICAL for r in stress_results)

    # ── Hard structural REJECTs ──
    if hc_adjusted.hc_fraction > EMPLOYER_LIMIT:
        return RiskDecision.REJECT, []

    if critical_stress and effective_risk_profile == RiskProfile.CONSERVATIVE:
        return RiskDecision.REJECT, []

    # ── Build flag constraints ──
    flag_constraints: list[AllocationConstraint] = list(violations)

    if drawdown_breach:
        flag_constraints += _drawdown_flag_constraints(tickers, weights, max_drawdown, cap)

    if critical_stress:
        flag_constraints += _stress_flag_constraints(effective_risk_profile)

    # Carry an existing profile downgrade forward so it survives subsequent FLAG
    # iterations that may only generate other constraint types (e.g. drawdown).
    if effective_risk_profile != risk_profile and not critical_stress:
        flag_constraints.append(AllocationConstraint(
            constraint_type=ConstraintType.RISK_PROFILE_DOWNGRADE,
            target=effective_risk_profile.value,
            current_value=0.0,
            limit=0.0,
        ))

    # ── Iteration limit ──
    # Real violations = drawdown breach, concentration, or critical stress.
    # Profile downgrade carry-forward alone is not a real violation.
    if flag_iteration >= 2:
        real_violations = drawdown_breach or bool(violations) or critical_stress
        if real_violations:
            return RiskDecision.REJECT, []
        else:
            return RiskDecision.APPROVE, []

    if flag_constraints:
        return RiskDecision.FLAG, flag_constraints

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
      2. VaR / CVaR (historical simulation, Basel III/IV)
      3. Max drawdown
      3b. Regime detection (60-day rolling vol vs full-sample baseline)
      4. Liquidity score
      5. Stress scenarios (historical + employer shock, severity vs effective cap)
      6. HC-adjusted balance sheet metrics (BMS 1992)
      7. Concentration checks
      8. Decision (APPROVE / FLAG / REJECT)

    reasoning_trace is left as "" — the agent layer fills it in.
    """
    ao       = allocation_output
    ai       = ao.allocation_input
    up       = ai.user_profile
    hc       = up.human_capital
    universe = ai.universe
    tickers  = universe.tickers

    # Effective risk profile: may be downgraded one step by a prior FLAG iteration.
    # On a fresh pipeline run (flag_constraints=[]) this always equals up.risk_profile.
    effective_profile = up.risk_profile
    for fc in ai.flag_constraints:
        if fc.constraint_type == ConstraintType.RISK_PROFILE_DOWNGRADE:
            effective_profile = RiskProfile(fc.target)

    weights     = np.array([w.total_weight * ao.risky_weight for w in ao.weights])
    weight_dict = {tickers[i]: float(weights[i]) for i in range(len(tickers))}

    recent_matrix, mask = _build_daily_matrix(crsp_daily, tickers, permno_map)
    if recent_matrix.shape[0] > VAR_LOOKBACK_DAYS:
        recent_matrix = recent_matrix[-VAR_LOOKBACK_DAYS:]
    w_adj = _apply_mask(weights, mask)

    var_cvar = (compute_var_cvar(w_adj, recent_matrix)
                if recent_matrix.shape[0] > 30
                else VaRMetrics(var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0))

    mdd = (compute_max_drawdown(w_adj, recent_matrix)
           if recent_matrix.shape[0] > 30 else 0.0)

    # Regime detection — compare recent 60-day vol to full-sample baseline
    if recent_matrix.shape[0] > 30:
        port_ret_full = recent_matrix @ w_adj
        regime = detect_regime(port_ret_full)
    else:
        regime = MarketRegime.NORMAL

    drawdown_cap = effective_cap(effective_profile, regime)

    liq = compute_liquidity_score(weights, tickers, crsp_daily, permno_map)

    employer_fin_weight = weight_dict.get(hc.employer_ticker, 0.0)
    hc_frac             = _hc_fraction(up.financial_wealth, hc.present_value)
    stress_results = compute_stress_results(
        weights, tickers, crsp_daily, permno_map,
        drawdown_cap, hc.employer_ticker,
        employer_fin_weight, hc_frac,
    )

    hc_adj = compute_hc_adjusted_metrics(
        weights, tickers, universe.sectors,
        up.financial_wealth, hc.present_value,
        hc.income_beta, hc.employer_sector, hc.employer_ticker,
    )

    flags      = run_all_checks(weight_dict, universe.sectors,
                                hc_adj.economic_sector_exposures,
                                hc_adj.employer_concentration)
    violations = all_violations(weight_dict, universe.sectors,
                                hc_adj.economic_sector_exposures,
                                hc_adj.employer_concentration)

    decision, flag_constraints = make_decision(
        concentration_flags    = flags,
        violations             = violations,
        max_drawdown           = mdd,
        stress_results         = stress_results,
        hc_adjusted            = hc_adj,
        risk_profile           = up.risk_profile,
        effective_risk_profile = effective_profile,
        regime                 = regime,
        flag_iteration         = ai.flag_iteration,
        tickers                = tickers,
        weights                = weights,
    )

    risk_metrics = RiskMetrics(
        volatility             = ao.portfolio_statistics.volatility,
        var_cvar               = var_cvar,
        max_drawdown           = mdd,
        factor_exposures       = ao.portfolio_statistics.factor_exposures,
        liquidity_score        = liq,
        concentration          = flags,
        hc_adjusted            = hc_adj,
        stress_results         = stress_results,
        market_regime          = regime,
        effective_drawdown_cap = drawdown_cap,
    )

    return RiskOutput(
        decision             = decision,
        allocation_output    = ao,
        risk_metrics         = risk_metrics,
        reasoning_trace      = "",
        constraints_violated = flag_constraints,
        flag_iteration       = ai.flag_iteration + (1 if decision == RiskDecision.FLAG else 0),
    )
