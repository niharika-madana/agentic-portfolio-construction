import pytest
import numpy as np
from portfolio_system.core.risk import (
    compute_var_cvar,
    compute_max_drawdown,
    compute_liquidity_score,
    compute_hc_adjusted_metrics,
    _severity,
    make_decision,
    MAX_DRAWDOWN_CAP,
    _EMPLOYER_STOCK_SHOCK,
    _EMPLOYER_HC_SHOCK,
)
from portfolio_system.schemas import (
    MarketRegime, RiskProfile, RiskDecision, StressSeverity,
    ConcentrationFlags, HumanCapitalAdjustedMetrics,
    StressResult, AllocationConstraint, ConstraintType,
)
from portfolio_system.core.constraints import EMPLOYER_LIMIT


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def daily_returns():
    np.random.seed(7)
    T, N = 500, 5
    return np.random.multivariate_normal(
        mean=np.full(N, 0.0003),
        cov =np.eye(N) * 0.0002 + np.full((N, N), 0.00005),
        size=T,
    )


@pytest.fixture
def equal_weights():
    return np.full(5, 0.20)


@pytest.fixture
def clean_hc_adj():
    return HumanCapitalAdjustedMetrics(
        total_wealth=1_500_000,
        hc_fraction=0.10,
        effective_equity_exposure=0.60,
        economic_sector_exposures={"IT": 0.15, "Financials": 0.10},
        employer_concentration=0.12,
    )


# ── compute_var_cvar ──────────────────────────────────────────────────────────

class TestVarCvar:
    def test_cvar_exceeds_var(self, daily_returns, equal_weights):
        vc = compute_var_cvar(equal_weights, daily_returns)
        assert vc.cvar_95 >= vc.var_95
        assert vc.cvar_99 >= vc.var_99

    def test_99_exceeds_95(self, daily_returns, equal_weights):
        vc = compute_var_cvar(equal_weights, daily_returns)
        assert vc.var_99  >= vc.var_95
        assert vc.cvar_99 >= vc.cvar_95

    def test_values_positive(self, daily_returns, equal_weights):
        vc = compute_var_cvar(equal_weights, daily_returns)
        assert vc.var_95  > 0
        assert vc.var_99  > 0
        assert vc.cvar_95 > 0
        assert vc.cvar_99 > 0

    def test_concentrated_portfolio_higher_var(self, daily_returns):
        w_equal = np.full(5, 0.20)
        w_conc  = np.array([0.80, 0.05, 0.05, 0.05, 0.05])
        vc_eq   = compute_var_cvar(w_equal,  daily_returns)
        vc_conc = compute_var_cvar(w_conc,   daily_returns)
        assert vc_conc.var_95 > vc_eq.var_95


# ── compute_max_drawdown ──────────────────────────────────────────────────────

class TestMaxDrawdown:
    def test_always_positive(self, daily_returns, equal_weights):
        mdd = compute_max_drawdown(equal_weights, daily_returns)
        assert mdd > 0

    def test_at_most_one(self, daily_returns, equal_weights):
        mdd = compute_max_drawdown(equal_weights, daily_returns)
        assert mdd <= 1.0

    def test_all_positive_returns_zero_drawdown(self):
        returns = np.full((100, 3), 0.001)
        w = np.array([1/3, 1/3, 1/3])
        assert compute_max_drawdown(w, returns) == pytest.approx(0.0, abs=1e-9)


# ── _severity ─────────────────────────────────────────────────────────────────

class TestSeverity:
    # _severity(loss, cap) — cap is now passed as a float directly
    def test_low_below_half_cap(self):
        assert _severity(0.05, 0.20) == StressSeverity.LOW

    def test_medium_between_half_and_cap(self):
        assert _severity(0.15, 0.20) == StressSeverity.MEDIUM

    def test_high_between_cap_and_1_5x(self):
        assert _severity(0.25, 0.20) == StressSeverity.HIGH

    def test_critical_above_1_5x_cap(self):
        assert _severity(0.35, 0.20) == StressSeverity.CRITICAL

    def test_conservative_cap_is_lower(self):
        # 0.12 is MEDIUM for conservative (cap=0.15, half=0.075)
        assert _severity(0.12, 0.15) == StressSeverity.MEDIUM
        # Same loss is LOW for aggressive (cap=0.25, half=0.125)
        assert _severity(0.12, 0.25) == StressSeverity.LOW

    def test_crisis_regime_widens_cap(self):
        # During crisis the effective cap for moderate is 0.30 (1.5×0.20)
        # A 25% loss that would be HIGH at normal cap is MEDIUM at crisis cap
        assert _severity(0.25, 0.30) == StressSeverity.MEDIUM


# ── make_decision ─────────────────────────────────────────────────────────────

class TestMakeDecision:
    TICKERS = ["A", "B", "C", "D", "E"]
    WEIGHTS = np.full(5, 0.20)

    def _decide(self, **kwargs):
        defaults = dict(
            concentration_flags=ConcentrationFlags(),
            violations=[],
            max_drawdown=0.10,
            stress_results=[],
            hc_adjusted=HumanCapitalAdjustedMetrics(
                total_wealth=1_000_000, hc_fraction=0.10,
                effective_equity_exposure=0.60,
                economic_sector_exposures={"IT": 0.10},
                employer_concentration=0.10,
            ),
            risk_profile=RiskProfile.MODERATE,
            effective_risk_profile=RiskProfile.MODERATE,
            regime=MarketRegime.NORMAL,
            flag_iteration=0,
            tickers=self.TICKERS,
            weights=self.WEIGHTS,
        )
        defaults.update(kwargs)
        return make_decision(**defaults)

    def test_approve_when_clean(self):
        decision, constr = self._decide()
        assert decision == RiskDecision.APPROVE
        assert constr == []

    def test_flag_on_concentration_violation(self):
        viols = [AllocationConstraint(
            constraint_type=ConstraintType.SINGLE_NAME,
            target="A", current_value=0.12, limit=0.10,
        )]
        decision, constr = self._decide(violations=viols)
        assert decision == RiskDecision.FLAG
        assert len(constr) >= 1

    def test_flag_on_drawdown_breach(self):
        # Moderate cap = 0.20, MDD = 0.25 → breach
        decision, constr = self._decide(max_drawdown=0.25)
        assert decision == RiskDecision.FLAG
        assert len(constr) > 0

    def test_reject_iteration_exhausted_with_violations(self):
        # Iteration limit reached AND violations remain → REJECT
        viols = [AllocationConstraint(
            constraint_type=ConstraintType.SINGLE_NAME,
            target="A", current_value=0.12, limit=0.10,
        )]
        decision, _ = self._decide(flag_iteration=2, violations=viols)
        assert decision == RiskDecision.REJECT

    def test_approve_iteration_exhausted_no_violations(self):
        # Iteration limit reached but portfolio converged — no violations → APPROVE
        decision, _ = self._decide(flag_iteration=2)
        assert decision == RiskDecision.APPROVE

    def test_reject_hc_fraction_above_employer_limit(self):
        # hc_fraction > EMPLOYER_LIMIT (0.15) → structurally unfixable
        bad_hc = HumanCapitalAdjustedMetrics(
            total_wealth=1_000_000, hc_fraction=0.20,
            effective_equity_exposure=0.70,
            economic_sector_exposures={"IT": 0.20},
            employer_concentration=0.20,
        )
        decision, _ = self._decide(hc_adjusted=bad_hc)
        assert decision == RiskDecision.REJECT

    def test_flag_critical_stress_generates_profile_downgrade(self):
        from portfolio_system.schemas import ConstraintType
        critical = [StressResult(
            scenario="Crash", portfolio_loss=0.40,
            threshold_breached=True, severity=StressSeverity.CRITICAL,
        )]
        decision, constr = self._decide(stress_results=critical)
        assert decision == RiskDecision.FLAG
        types = [c.constraint_type for c in constr]
        assert ConstraintType.RISK_PROFILE_DOWNGRADE in types

    def test_moderate_downgrades_to_conservative(self):
        from portfolio_system.schemas import ConstraintType
        critical = [StressResult(
            scenario="Crash", portfolio_loss=0.40,
            threshold_breached=True, severity=StressSeverity.CRITICAL,
        )]
        _, constr = self._decide(stress_results=critical, effective_risk_profile=RiskProfile.MODERATE)
        dc = next(c for c in constr if c.constraint_type == ConstraintType.RISK_PROFILE_DOWNGRADE)
        assert dc.target == RiskProfile.CONSERVATIVE.value

    def test_aggressive_downgrades_to_moderate(self):
        from portfolio_system.schemas import ConstraintType
        critical = [StressResult(
            scenario="Crash", portfolio_loss=0.40,
            threshold_breached=True, severity=StressSeverity.CRITICAL,
        )]
        _, constr = self._decide(
            stress_results=critical,
            risk_profile=RiskProfile.AGGRESSIVE,
            effective_risk_profile=RiskProfile.AGGRESSIVE,
        )
        dc = next(c for c in constr if c.constraint_type == ConstraintType.RISK_PROFILE_DOWNGRADE)
        assert dc.target == RiskProfile.MODERATE.value

    def test_reject_critical_stress_already_conservative(self):
        critical = [StressResult(
            scenario="Crash", portfolio_loss=0.40,
            threshold_breached=True, severity=StressSeverity.CRITICAL,
        )]
        decision, _ = self._decide(
            stress_results=critical,
            effective_risk_profile=RiskProfile.CONSERVATIVE,
        )
        assert decision == RiskDecision.REJECT

    def test_flag_takes_priority_over_critical_when_violations_exist(self):
        # If critical stress AND violations → FLAG (violations are addressable)
        critical = [StressResult(
            scenario="Crash", portfolio_loss=0.40,
            threshold_breached=True, severity=StressSeverity.CRITICAL,
        )]
        viols = [AllocationConstraint(
            constraint_type=ConstraintType.SINGLE_NAME,
            target="A", current_value=0.12, limit=0.10,
        )]
        decision, _ = self._decide(stress_results=critical, violations=viols)
        assert decision == RiskDecision.FLAG

    def test_drawdown_flag_contains_all_tickers(self):
        decision, constr = self._decide(max_drawdown=0.30)
        assert decision == RiskDecision.FLAG
        flagged_tickers = {c.target for c in constr}
        assert all(t in flagged_tickers for t in self.TICKERS)

    def test_crisis_regime_widens_cap(self):
        # Moderate normal cap = 20%. Crisis cap = 30%.
        # MDD of 25% should FLAG in NORMAL but APPROVE in CRISIS.
        decision_normal, _ = self._decide(
            max_drawdown=0.25, regime=MarketRegime.NORMAL,
        )
        decision_crisis, _ = self._decide(
            max_drawdown=0.25, regime=MarketRegime.CRISIS,
        )
        assert decision_normal == RiskDecision.FLAG
        assert decision_crisis == RiskDecision.APPROVE

    def test_elevated_regime_partial_widen(self):
        # Elevated cap = 25%. MDD of 22% should FLAG in NORMAL but APPROVE in ELEVATED.
        decision_normal,   _ = self._decide(max_drawdown=0.22, regime=MarketRegime.NORMAL)
        decision_elevated, _ = self._decide(max_drawdown=0.22, regime=MarketRegime.ELEVATED)
        assert decision_normal   == RiskDecision.FLAG
        assert decision_elevated == RiskDecision.APPROVE


# ── compute_hc_adjusted_metrics ───────────────────────────────────────────────

class TestHcAdjustedMetrics:
    def test_sector_exposures_sum_to_one(self):
        weights = np.array([0.40, 0.30, 0.30])
        tickers = ["AAPL", "JPM", "XOM"]
        sectors = {"AAPL": "IT", "JPM": "Financials", "XOM": "Energy"}
        metrics = compute_hc_adjusted_metrics(
            weights, tickers, sectors,
            financial_wealth=500_000, human_capital_pv=500_000,
            income_beta=0.5, employer_sector="IT", employer_ticker="AAPL",
        )
        assert abs(sum(metrics.economic_sector_exposures.values()) - 1.0) < 1e-9

    def test_total_wealth_correct(self):
        weights = np.array([1.0])
        metrics = compute_hc_adjusted_metrics(
            weights, ["A"], {"A": "IT"},
            financial_wealth=300_000, human_capital_pv=700_000,
            income_beta=0.0, employer_sector="IT", employer_ticker="A",
        )
        assert metrics.total_wealth == 1_000_000

    def test_hc_fraction_correct(self):
        weights = np.array([1.0])
        metrics = compute_hc_adjusted_metrics(
            weights, ["A"], {"A": "IT"},
            financial_wealth=500_000, human_capital_pv=500_000,
            income_beta=0.0, employer_sector="IT", employer_ticker="A",
        )
        assert abs(metrics.hc_fraction - 0.50) < 1e-9
