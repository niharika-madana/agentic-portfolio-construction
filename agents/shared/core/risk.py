from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import (
    AllocationConstraint, AllocationOutput, ConcentrationFlags,
    ConstraintType, HumanCapitalAdjustedMetrics, RiskDecision,
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


MAX_DRAWDOWN_CAP: dict[RiskProfile, float] = {
    RiskProfile.CONSERVATIVE: 0.15,
    RiskProfile.MODERATE:     0.20,
    RiskProfile.AGGRESSIVE:   0.25,
}

STRESS_SCENARIOS: list[dict] = [
    {"name": "S&P 500, 2008",        "start": "2008-10-01", "end": "2009-03-09", "benchmark_loss": 0.54},
    {"name": "COVID, 2020",           "start": "2020-02-19", "end": "2020-03-23", "benchmark_loss": 0.34},
    {"name": "60/40 Portfolio, 2008", "start": "2008-01-01", "end": "2008-12-31", "benchmark_loss": 0.237},
]

_EMPLOYER_STOCK_SHOCK = 0.50
_EMPLOYER_HC_SHOCK    = 0.30
VAR_LOOKBACK_DAYS     = 1260


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


def compute_max_drawdown(
    weights: np.ndarray,
    daily_returns: np.ndarray,
) -> float:
    port_ret    = daily_returns @ weights
    cum_ret     = np.cumprod(1.0 + port_ret)
    rolling_max = np.maximum.accumulate(cum_ret)
    drawdowns   = (cum_ret - rolling_max) / rolling_max
    return float(-drawdowns.min())


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
    return float(weights @ normalised)


def _severity(portfolio_loss: float, risk_profile: RiskProfile) -> StressSeverity:
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
    matrix, mask = _build_daily_matrix(crsp_daily, tickers, permno_map, start_date, end_date)
    if matrix.shape[0] == 0:
        return None
    w_adj    = _apply_mask(weights, mask)
    port_ret = matrix @ w_adj
    cum_ret  = float(np.prod(1.0 + port_ret) - 1.0)
    return max(-cum_ret, 0.0)


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
    results: list[StressResult] = []
    cap = MAX_DRAWDOWN_CAP[risk_profile]

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
            threshold_breached = loss > cap,
            severity           = _severity(loss, risk_profile),
        ))

    employer_fin_loss   = employer_financial_weight * _EMPLOYER_STOCK_SHOCK
    employer_hc_loss    = hc_frac * _EMPLOYER_HC_SHOCK
    total_employer_loss = employer_fin_loss + employer_hc_loss
    results.append(StressResult(
        scenario           = "Idiosyncratic Employer Shock",
        portfolio_loss     = float(np.clip(total_employer_loss, 0.0, 1.0)),
        threshold_breached = total_employer_loss > cap,
        severity           = _severity(total_employer_loss, risk_profile),
    ))
    return results


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
    flag_iteration: int,
    tickers: list[str],
    weights: np.ndarray,
) -> tuple[RiskDecision, list[AllocationConstraint]]:
    cap             = MAX_DRAWDOWN_CAP[risk_profile]
    drawdown_breach = max_drawdown > cap
    critical_stress = any(r.severity == StressSeverity.CRITICAL for r in stress_results)

    if flag_iteration >= 3:
        return RiskDecision.REJECT, []
    if hc_adjusted.hc_fraction > EMPLOYER_LIMIT:
        return RiskDecision.REJECT, []
    if critical_stress and not violations and not drawdown_breach:
        return RiskDecision.REJECT, []

    flag_constraints: list[AllocationConstraint] = list(violations)
    if drawdown_breach:
        flag_constraints += _drawdown_flag_constraints(tickers, weights, max_drawdown, cap)

    if flag_constraints:
        return RiskDecision.FLAG, flag_constraints

    return RiskDecision.APPROVE, []


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
      4. Liquidity score
      5. Stress scenarios (historical + employer shock)
      6. HC-adjusted balance sheet metrics (BMS 1992)
      7. Concentration checks
      8. Decision (APPROVE / FLAG / REJECT)

    reasoning_trace is left as "" — the agent layer fills it in.
    """
    ao      = allocation_output
    ai      = ao.allocation_input
    up      = ai.user_profile
    hc      = up.human_capital
    universe= ai.universe
    tickers = universe.tickers

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

    liq = compute_liquidity_score(weights, tickers, crsp_daily, permno_map)

    employer_fin_weight = weight_dict.get(hc.employer_ticker, 0.0)
    hc_frac             = _hc_fraction(up.financial_wealth, hc.present_value)
    stress_results = compute_stress_results(
        weights, tickers, crsp_daily, permno_map,
        up.risk_profile, hc.employer_ticker,
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
        concentration_flags = flags,
        violations          = violations,
        max_drawdown        = mdd,
        stress_results      = stress_results,
        hc_adjusted         = hc_adj,
        risk_profile        = up.risk_profile,
        flag_iteration      = ai.flag_iteration,
        tickers             = tickers,
        weights             = weights,
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
        decision             = decision,
        allocation_output    = ao,
        risk_metrics         = risk_metrics,
        reasoning_trace      = "",
        constraints_violated = flag_constraints,
        flag_iteration       = ai.flag_iteration + (1 if decision == RiskDecision.FLAG else 0),
    )
