### Contracts.py
class AllocationInput(BaseModel):
    user_profile:      UserProfile
    universe:          InstrumentUniverse
    flag_constraints:  list[AllocationConstraint] = Field(default_factory=list)
    flag_iteration:    int = Field(default=0, ge=0, le=3)

class AllocationAgentOutput(BaseModel):
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

class AllocationConstraint(BaseModel):
    constraint_type: ConstraintType
    target:          str   = Field(description="Ticker or sector name")
    current_value:   float
    limit:           float  

### human_capital.py
def merton_risky_share(
    expected_excess_return: float,
    portfolio_volatility: float,
    risk_profile: RiskProfile,
)
    gamma = _RISK_AVERSION[risk_profile]
    alpha = expected_excess_return / (gamma * portfolio_volatility ** 2)
    return float(min(alpha, 1.0))

_RISK_AVERSION: dict[RiskProfile, float] = {
    RiskProfile.CONSERVATIVE: 5.0,
    RiskProfile.MODERATE:     3.0,
    RiskProfile.AGGRESSIVE:   2.0,
}

def compute_w_fin(
    alpha: float,
    human_capital_pv: float,
    financial_wealth: float,
    income_beta: float,
) 
    ratio = human_capital_pv / financial_wealth
    w_fin = alpha * (1.0 + ratio) - ratio * income_beta
    return float(np.clip(w_fin, 0.0, 1.0))

### allocation.py
def black_litterman(
    cov: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    M_inv     = np.linalg.inv(TAU * cov)
    Omega_inv = np.linalg.inv(Omega)
    A         = M_inv + P.T @ Omega_inv @ P
    A_inv     = np.linalg.inv(A)
    mu_BL     = A_inv @ (M_inv @ pi + P.T @ Omega_inv @ Q)
    Sigma_BL  = cov + A_inv
    return mu_BL, Sigma_BL

def optimize_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: list[str],
    sectors: dict[str, str],
    flag_constraints: list[AllocationConstraint],
    employer_ticker: str | None = None,
    employer_sector: str | None = None,
    financial_wealth: float = 0.0,
    human_capital_pv: float = 0.0,
) -> np.ndarray:
    N     = len(tickers)
    upper = np.full(N, SINGLE_NAME_LIMIT)
    sector_limits: dict[str, float] = {}
    if employer_ticker and employer_ticker in tickers:
        upper[tickers.index(employer_ticker)] = EMPLOYER_STOCK_LIMIT
    if employer_sector:
        sector_limits[employer_sector] = EMPLOYER_SECTOR_LIMIT
    for fc in flag_constraints:
        if fc.constraint_type == ConstraintType.SINGLE_NAME and fc.target in tickers:
            idx = tickers.index(fc.target)
            upper[idx] = min(upper[idx], fc.limit * 0.99)
        elif fc.constraint_type == ConstraintType.SECTOR:
            sector_limits[fc.target] = min(sector_limits.get(fc.target, SECTOR_LIMIT), fc.limit * 0.99)
        elif fc.constraint_type == ConstraintType.EMPLOYER:
            if employer_ticker and employer_ticker in tickers:
                idx   = tickers.index(employer_ticker)
                max_w = _max_employer_financial_weight(
                    financial_wealth, human_capital_pv, fc.limit * 0.99
                )
                upper[idx] = min(upper[idx], max_w)
        elif fc.constraint_type == ConstraintType.ECONOMIC_SECTOR:
            if financial_wealth + human_capital_pv > 0:
                fin_share = financial_wealth / (financial_wealth + human_capital_pv)
                sector_limits[fc.target] = min(
                    sector_limits.get(fc.target, SECTOR_LIMIT), fc.limit * fin_share * 0.99
                )
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

 equity_cap = float(np.clip(up.portfolio_equity_target, 0.0, 1.0))
    if effective_risk_profile != up.risk_profile:
        equity_cap *= MAX_DRAWDOWN_CAP[effective_risk_profile] / MAX_DRAWDOWN_CAP[up.risk_profile]
    w_fin      = min(w_fin, equity_cap)
for fc in allocation_input.flag_constraints:
        if fc.constraint_type == ConstraintType.RISKY_WEIGHT_CAP:
            w_fin = min(w_fin, fc.limit)

### risk.py (create script)
var_cvar = (compute_var_cvar(w_adj, recent_matrix)
                if recent_matrix.shape[0] > 30
                else VaRMetrics(var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0))

 mdd = (compute_max_drawdown(w_adj, recent_matrix)
           if recent_matrix.shape[0] > 30 else 0.0)

 if recent_matrix.shape[0] > 30:
        port_ret_full = recent_matrix @ w_adj
        regime = detect_regime(port_ret_full)
    else:
        regime = MarketRegime.NORMAL
    drawdown_cap = effective_cap(effective_profile, regime)
    liq = compute_liquidity_score(weights, tickers, crsp_daily, permno_map)
    employer_fin_weight = weight_dict.get(hc.employer_ticker, 0.0)

    stress_results = compute_stress_results(
        weights, tickers, crsp_daily, permno_map,
        drawdown_cap, idiosyncratic_employer_exposure,
    )

hc_adj = compute_hc_adjusted_metrics(
        weights, tickers, universe.sectors,
        up.financial_wealth, hc.present_value,
        hc.income_beta, hc.employer_sector, hc.employer_ticker,
    )

### Decision Gate (risk.py) (make script)
 decision, flag_constraints = make_decision(
        concentration_flags              = flags,
        violations                       = violations,
        max_drawdown                     = mdd,
        stress_results                   = stress_results,
        idiosyncratic_employer_exposure  = idiosyncratic_employer_exposure,
        risk_profile                     = up.risk_profile,
        effective_risk_profile           = effective_profile,
        regime                           = regime,
        flag_iteration                   = ai.flag_iteration,
        tickers                          = tickers,
        weights                          = weights,
        risky_weight                     = ao.risky_weight,
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

### Allocation Risk Loop
   for iteration in range(MAX_FLAG_ITERATIONS):   # 3
    alloc_out, alloc_agent_out = run_allocation_agent(
        profile, discount_rate, ff_factors,
        flag_constraints = flag_constraints,
        flag_iteration   = iteration,
    )
    risk_out, risk_agent_out = run_risk_agent(alloc_out, ...)

    if risk_out.decision != RiskDecision.FLAG:
        break
    flag_constraints = risk_out.constraints_violated 

