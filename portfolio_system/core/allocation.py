from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

from portfolio_system.schemas import (
    AllocationConstraint, AllocationInput, AllocationOutput,
    ConstraintType, FactorExposures, PortfolioStatistics, WeightDecomposition,
    RiskProfile,
)
from portfolio_system.core.constraints import (
    SINGLE_NAME_LIMIT, SECTOR_LIMIT,
)
from portfolio_system.core.human_capital import (
    compute_w_fin, merton_risky_share,
)


TAU             = 0.025   # BL confidence parameter (Walters 2013)
DELTA           = 2.5     # Implied market risk aversion
MONTHS_PER_YEAR = 12


# ── Data preparation ──────────────────────────────────────────────────────────

def build_returns_matrix(
    crsp_monthly: pd.DataFrame,
    tickers: list[str],
    permno_map: dict[str, int],
) -> pd.DataFrame:
    """
    Pivots CRSP monthly returns into a (T × N) DataFrame.
    Columns: tickers. Index: date. Values: monthly total return.
    Rows with any missing return are dropped.
    """
    inv_map = {permno_map[t]: t for t in tickers}
    subset  = crsp_monthly[crsp_monthly["permno"].isin(inv_map)].copy()
    subset["ticker"] = subset["permno"].map(inv_map)
    pivot = subset.pivot(index="date", columns="ticker", values="ret")[tickers]
    return pivot.dropna()


def _align_and_excess(
    returns: pd.DataFrame,
    ff_factors: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inner-joins returns and FF factors on date.
    Subtracts the monthly rf to produce excess returns.
    Returns (excess T×N float64, ff_arr T×4 float64).
    """
    joined  = returns.join(ff_factors[["mktrf", "smb", "hml", "umd", "rf"]], how="inner")
    rf      = joined["rf"].values
    excess  = joined[returns.columns].values - rf[:, None]
    ff_arr  = joined[["mktrf", "smb", "hml", "umd"]].values
    return excess.astype(np.float64), ff_arr.astype(np.float64)


def build_covariance_matrix(excess_returns: np.ndarray) -> np.ndarray:
    """Annualized sample covariance matrix from monthly excess returns (T × N)."""
    return np.cov(excess_returns, rowvar=False) * MONTHS_PER_YEAR


# ── Black-Litterman ───────────────────────────────────────────────────────────

def compute_equilibrium_returns(
    cov: np.ndarray,
    mkt_weights: np.ndarray,
) -> np.ndarray:
    """
    Reverse-optimize annualized CAPM equilibrium excess returns (Sharpe 1964).
        Pi = DELTA × Sigma × w_mkt
    """
    return DELTA * cov @ mkt_weights


def build_ff_views(
    excess_returns: np.ndarray,
    ff_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Black-Litterman views from the Fama-French factor model.

    For each asset i, OLS: r_i = a_i + B_i·F + e_i
    View Q_i = a_i·12 + B_i·(mean(F)·12)  — annualized factor-implied return.

    P = I_N    (one absolute view per asset)
    Omega = diag(annualized residual variances) / tau   (scales view confidence)

    Args:
        excess_returns : (T × N) monthly excess returns
        ff_arr         : (T × 4) monthly FF factors [mktrf, smb, hml, umd]
    """
    T, N = excess_returns.shape
    X        = np.column_stack([np.ones(T), ff_arr])       # (T × 5)
    lambda_F = ff_arr.mean(axis=0) * MONTHS_PER_YEAR       # annualized factor premiums

    Q        = np.zeros(N)
    res_vars = np.zeros(N)

    for i in range(N):
        coeffs, *_ = np.linalg.lstsq(X, excess_returns[:, i], rcond=None)
        Q[i]       = coeffs[0] * MONTHS_PER_YEAR + coeffs[1:] @ lambda_F
        resid      = excess_returns[:, i] - X @ coeffs
        # unbiased residual variance, annualized
        res_vars[i] = (resid @ resid / (T - X.shape[1])) * MONTHS_PER_YEAR

    return np.eye(N), Q, np.diag(res_vars) / TAU


def black_litterman(
    cov: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Black-Litterman posterior returns and covariance.

        A      = (τΣ)⁻¹ + P'Ω⁻¹P
        mu_BL  = A⁻¹ [ (τΣ)⁻¹π + P'Ω⁻¹Q ]
        Σ_BL   = Σ + A⁻¹

    Returns (mu_BL, Sigma_BL).
    """
    M_inv     = np.linalg.inv(TAU * cov)
    Omega_inv = np.linalg.inv(Omega)
    A         = M_inv + P.T @ Omega_inv @ P
    A_inv     = np.linalg.inv(A)
    mu_BL     = A_inv @ (M_inv @ pi + P.T @ Omega_inv @ Q)
    Sigma_BL  = cov + A_inv
    return mu_BL, Sigma_BL


# ── Constrained optimization ──────────────────────────────────────────────────

def _max_employer_financial_weight(
    financial_wealth: float,
    human_capital_pv: float,
    employer_limit: float,
) -> float:
    """
    Max financial weight in employer stock such that total economic employer
    concentration stays within employer_limit.
    Derived from: (w_emp × W + H) / (W + H) ≤ limit
        → w_emp ≤ (limit × (W+H) − H) / W
    """
    tw    = financial_wealth + human_capital_pv
    max_w = (employer_limit * tw - human_capital_pv) / financial_wealth
    return float(np.clip(max_w, 0.0, SINGLE_NAME_LIMIT))


def optimize_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: list[str],
    sectors: dict[str, str],
    flag_constraints: list[AllocationConstraint],
    employer_ticker: str | None = None,
    financial_wealth: float = 0.0,
    human_capital_pv: float = 0.0,
) -> np.ndarray:
    """
    Constrained mean-variance optimization.
    Maximizes w'μ − (DELTA/2)·w'Σw subject to:
      - weights sum to 1
      - 0 ≤ w[i] ≤ SINGLE_NAME_LIMIT (10%)
      - sum of each sector's weights ≤ SECTOR_LIMIT (20%)

    FLAG re-entry (flag_constraints non-empty) tightens limits for previously
    violated constraints. HC-driven constraints (ECONOMIC_SECTOR, EMPLOYER)
    are translated into tighter financial bounds before the optimizer runs.
    """
    N     = len(tickers)
    upper = np.full(N, SINGLE_NAME_LIMIT)
    sector_limits: dict[str, float] = {}

    for fc in flag_constraints:
        if fc.constraint_type == ConstraintType.SINGLE_NAME and fc.target in tickers:
            idx = tickers.index(fc.target)
            upper[idx] = min(upper[idx], fc.limit * 0.99)

        elif fc.constraint_type == ConstraintType.SECTOR:
            sector_limits[fc.target] = fc.limit * 0.99

        elif fc.constraint_type == ConstraintType.EMPLOYER:
            if employer_ticker and employer_ticker in tickers:
                idx   = tickers.index(employer_ticker)
                max_w = _max_employer_financial_weight(
                    financial_wealth, human_capital_pv, fc.limit * 0.99
                )
                upper[idx] = min(upper[idx], max_w)

        elif fc.constraint_type == ConstraintType.ECONOMIC_SECTOR:
            # Tighten the financial sector limit proportionally to keep
            # the total economic exposure (financial + HC) within the limit.
            if financial_wealth + human_capital_pv > 0:
                fin_share = financial_wealth / (financial_wealth + human_capital_pv)
                sector_limits[fc.target] = fc.limit * fin_share * 0.99

    bounds = Bounds(lb=np.zeros(N), ub=upper)

    unique_sectors = list({sectors.get(t, "Unknown") for t in tickers})
    scipy_constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    for sec in unique_sectors:
        lim = sector_limits.get(sec, SECTOR_LIMIT)
        idx = [i for i, t in enumerate(tickers) if sectors.get(t) == sec]
        if idx:
            scipy_constraints.append({
                "type": "ineq",
                "fun": lambda w, ix=idx, lm=lim: lm - w[ix].sum(),
            })

    def neg_utility(w: np.ndarray) -> float:
        return -(float(w @ mu) - (DELTA / 2.0) * float(w @ Sigma @ w))

    def grad(w: np.ndarray) -> np.ndarray:
        return -(mu - DELTA * (Sigma @ w))

    result = minimize(
        neg_utility,
        x0=np.full(N, 1.0 / N),
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=scipy_constraints,
        options={"ftol": 1e-10, "maxiter": 1000},
    )

    if not result.success:
        w = np.clip(np.full(N, 1.0 / N), 0.0, upper)
        return w / w.sum()
    return result.x


# ── Portfolio statistics ──────────────────────────────────────────────────────

def compute_factor_exposures(
    weights: np.ndarray,
    excess_returns: np.ndarray,
    ff_arr: np.ndarray,
) -> FactorExposures:
    """
    OLS: r_portfolio = α + β_mkt·MktRf + β_smb·SMB + β_hml·HML + β_mom·UMD
    Returns the four beta coefficients.
    """
    T        = excess_returns.shape[0]
    port_ret = excess_returns @ weights
    X        = np.column_stack([np.ones(T), ff_arr])
    coeffs, *_ = np.linalg.lstsq(X, port_ret, rcond=None)
    return FactorExposures(
        market_beta=float(coeffs[1]),
        smb=float(coeffs[2]),
        hml=float(coeffs[3]),
        mom=float(coeffs[4]),
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run_allocation(
    allocation_input: AllocationInput,
    crsp_monthly: pd.DataFrame,
    ff_factors: pd.DataFrame,
    risk_free_rate: float,
    permno_map: dict[str, int],
) -> AllocationOutput:
    """
    Full Black-Litterman allocation pipeline.

    Pipeline:
      1. Build annualized covariance from CRSP monthly excess returns
      2. Compute CAPM equilibrium returns from market-cap weights  (Pi = DELTA·Σ·w_mkt)
      3. Build Fama-French factor views                            (P, Q, Omega)
      4. Compute BL posterior                                      (mu_BL, Sigma_BL)
      5a. Optimize with Pi only  → w_eq  (equilibrium baseline)
      5b. Optimize with mu_BL   → w_BL  (with view tilts)
      6. Apply HC w_fin scaling  → risky_weight, safe_weight
      7. Decompose weights and compute portfolio statistics

    rationale is left as "" — agents/allocation_agent.py fills it in.

    Args:
        allocation_input : validated AllocationInput
        crsp_monthly     : from data/loaders.load_crsp_monthly()
        ff_factors       : from data/loaders.load_fama_french_factors()
        risk_free_rate   : annualized risk-free rate (scalar)
        permno_map       : {ticker: permno} from data/loaders.tickers_to_permnos()
    """
    up       = allocation_input.user_profile
    universe = allocation_input.universe
    tickers  = universe.tickers
    hc       = up.human_capital

    # ── 1. Covariance ──
    returns_df      = build_returns_matrix(crsp_monthly, tickers, permno_map)
    excess, ff_arr  = _align_and_excess(returns_df, ff_factors)
    cov             = build_covariance_matrix(excess)

    # ── 2. Equilibrium returns ──
    mkt_w = np.array([universe.market_cap_weights[t] for t in tickers])
    pi    = compute_equilibrium_returns(cov, mkt_w)

    # ── 3. FF views ──
    P, Q, Omega = build_ff_views(excess, ff_arr)

    # ── 4. BL posterior ──
    mu_BL, Sigma_BL = black_litterman(cov, pi, P, Q, Omega)

    # ── 5. Optimize ──
    opt_kwargs = dict(
        tickers=tickers,
        sectors=universe.sectors,
        flag_constraints=allocation_input.flag_constraints,
        employer_ticker=hc.employer_ticker,
        financial_wealth=up.financial_wealth,
        human_capital_pv=hc.present_value,
    )

    # Equilibrium baseline: send views to zero (inflate Omega → prior dominates)
    _, Sigma_eq = black_litterman(cov, pi, P, np.zeros_like(Q), Omega * 1e8)
    w_eq = optimize_weights(pi, Sigma_eq, **opt_kwargs)
    w_BL = optimize_weights(mu_BL, Sigma_BL, **opt_kwargs)

    # Human capital offset = how much FLAG constraints shifted the BL weights.
    # On first run flag_constraints is empty so w_BL_base == w_BL and offset = 0.
    w_BL_base = optimize_weights(
        mu_BL, Sigma_BL,
        tickers=tickers,
        sectors=universe.sectors,
        flag_constraints=[],
    )

    # ── 6. Human capital w_fin ──
    # Use effective risk profile: may be downgraded one step by a FLAG constraint.
    # Reverts to user's stated profile automatically on any fresh pipeline run.
    effective_profile = up.risk_profile
    for fc in allocation_input.flag_constraints:
        if fc.constraint_type == ConstraintType.RISK_PROFILE_DOWNGRADE:
            effective_profile = RiskProfile(fc.target)

    port_vol    = float(np.sqrt(w_BL @ cov @ w_BL))
    port_excess = float(w_BL @ mu_BL)
    alpha       = merton_risky_share(port_excess, port_vol, effective_profile)
    w_fin       = compute_w_fin(alpha, hc.present_value, up.financial_wealth, hc.income_beta)

    # ── 7. Decompose and build output ──
    decomposition = [
        WeightDecomposition(
            ticker=t,
            total_weight=float(w_BL[i]),
            equilibrium_baseline=float(w_eq[i]),
            view_tilt=float(w_BL_base[i] - w_eq[i]),
            human_capital_offset=float(w_BL[i] - w_BL_base[i]),
        )
        for i, t in enumerate(tickers)
    ]

    exp_ret = float(w_BL @ mu_BL) + risk_free_rate
    vol     = float(np.sqrt(w_BL @ cov @ w_BL))
    sharpe  = (exp_ret - risk_free_rate) / vol if vol > 0 else 0.0
    fe      = compute_factor_exposures(w_BL, excess, ff_arr)

    return AllocationOutput(
        allocation_input=allocation_input,
        weights=decomposition,
        risky_weight=float(w_fin),
        safe_weight=float(1.0 - w_fin),
        portfolio_statistics=PortfolioStatistics(
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            factor_exposures=fe,
        ),
        rationale="",
    )
