import pytest
import numpy as np
from portfolio_system.core.human_capital import (
    merton_risky_share,
    compute_w_fin,
    total_wealth,
    hc_fraction,
    effective_equity_exposure,
    employer_concentration,
    economic_sector_exposures,
)
from portfolio_system.schemas import RiskProfile


# ── merton_risky_share ────────────────────────────────────────────────────────

class TestMertonRiskyShare:
    def test_moderate_benchmark(self):
        # gamma=3, mu=0.05, sigma=0.15 → 0.05/(3*0.0225) = 0.7407
        alpha = merton_risky_share(0.05, 0.15, RiskProfile.MODERATE)
        assert abs(alpha - 0.7407) < 1e-3

    def test_conservative_lower_than_aggressive(self):
        alpha_c = merton_risky_share(0.05, 0.15, RiskProfile.CONSERVATIVE)
        alpha_a = merton_risky_share(0.05, 0.15, RiskProfile.AGGRESSIVE)
        assert alpha_c < alpha_a

    def test_capped_at_one(self):
        # Very high excess return → would exceed 1.0 without cap
        alpha = merton_risky_share(1.0, 0.10, RiskProfile.AGGRESSIVE)
        assert alpha == 1.0

    def test_higher_volatility_lowers_alpha(self):
        alpha_low  = merton_risky_share(0.05, 0.10, RiskProfile.MODERATE)
        alpha_high = merton_risky_share(0.05, 0.20, RiskProfile.MODERATE)
        assert alpha_low > alpha_high


# ── compute_w_fin ─────────────────────────────────────────────────────────────

class TestComputeWFin:
    def test_risk_free_hc_increases_w_fin(self):
        # income_beta=0 → pure BMS risk-free case: w_fin = alpha*(1+H/W)
        alpha = 0.60
        w = compute_w_fin(alpha, human_capital_pv=200_000,
                          financial_wealth=100_000, income_beta=0.0)
        expected = alpha * (1 + 2.0)   # = 1.80, capped at 1.0
        assert w == 1.0

    def test_risky_hc_reduces_w_fin(self):
        # income_beta > 0 hedges away some equity demand
        alpha = 0.60
        w_rf    = compute_w_fin(alpha, 200_000, 100_000, income_beta=0.0)
        w_risky = compute_w_fin(alpha, 200_000, 100_000, income_beta=0.5)
        assert w_risky < w_rf

    def test_clipped_to_zero(self):
        # Very high income_beta → could go negative; must clip to 0
        w = compute_w_fin(0.10, 500_000, 100_000, income_beta=2.0)
        assert w >= 0.0

    def test_clipped_to_one(self):
        w = compute_w_fin(1.0, 1_000_000, 100_000, income_beta=0.0)
        assert w == 1.0

    def test_no_hc_equals_alpha(self):
        # H=0 → w_fin = alpha*(1+0) - 0*beta = alpha
        alpha = 0.75
        w = compute_w_fin(alpha, human_capital_pv=0.0,
                          financial_wealth=500_000, income_beta=0.3)
        assert abs(w - alpha) < 1e-9


# ── total_wealth / hc_fraction ────────────────────────────────────────────────

class TestBalanceSheet:
    def test_total_wealth(self):
        assert total_wealth(500_000, 1_000_000) == 1_500_000

    def test_hc_fraction_two_thirds(self):
        f = hc_fraction(500_000, 1_000_000)
        assert abs(f - 2 / 3) < 1e-9

    def test_hc_fraction_zero_hc(self):
        assert hc_fraction(500_000, 0.0) == 0.0


# ── effective_equity_exposure ─────────────────────────────────────────────────

class TestEffectiveEquityExposure:
    def test_zero_beta_only_financial(self):
        # income_beta=0 → HC has no equity character
        exp = effective_equity_exposure(
            portfolio_equity_weight=0.60,
            financial_wealth=500_000,
            human_capital_pv=500_000,
            income_beta=0.0,
        )
        # financial equity = 0.6*500k / 1000k = 0.30
        assert abs(exp - 0.30) < 1e-9

    def test_high_beta_increases_exposure(self):
        exp_low  = effective_equity_exposure(0.60, 500_000, 500_000, income_beta=0.0)
        exp_high = effective_equity_exposure(0.60, 500_000, 500_000, income_beta=0.8)
        assert exp_high > exp_low


# ── employer_concentration ────────────────────────────────────────────────────

class TestEmployerConcentration:
    def test_zero_financial_stake(self):
        # No employer stock in portfolio → concentration = H/(W+H)
        conc = employer_concentration(0.0, 500_000, 750_000)
        expected = 750_000 / 1_250_000
        assert abs(conc - expected) < 1e-9

    def test_exceeds_limit_when_hc_large(self):
        from portfolio_system.core.constraints import EMPLOYER_LIMIT
        conc = employer_concentration(0.0, 200_000, 500_000)
        # hc_fraction = 500k/700k = 0.714, way above 15%
        assert conc > EMPLOYER_LIMIT

    def test_both_financial_and_hc(self):
        # (0.05*500k + 300k) / 800k = 325k/800k = 0.40625
        conc = employer_concentration(0.05, 500_000, 300_000)
        assert abs(conc - (325_000 / 800_000)) < 1e-9


# ── economic_sector_exposures ─────────────────────────────────────────────────

class TestEconomicSectorExposures:
    def setup_method(self):
        self.weights = {"AAPL": 0.40, "JPM": 0.35, "XOM": 0.25}
        self.sectors = {
            "AAPL": "Information Technology",
            "JPM":  "Financials",
            "XOM":  "Energy",
        }

    def test_exposures_sum_to_one(self):
        exps = economic_sector_exposures(
            self.weights, self.sectors,
            financial_wealth=1_000_000,
            human_capital_pv=500_000,
            employer_sector="Information Technology",
        )
        assert abs(sum(exps.values()) - 1.0) < 1e-9

    def test_employer_sector_boosted(self):
        exps = economic_sector_exposures(
            self.weights, self.sectors,
            financial_wealth=1_000_000,
            human_capital_pv=500_000,
            employer_sector="Information Technology",
        )
        # IT financial = 0.40*1M = 400k, + 500k HC = 900k out of 1.5M = 0.60
        assert abs(exps["Information Technology"] - 0.60) < 1e-9

    def test_non_employer_sector_unaffected_by_hc(self):
        exps = economic_sector_exposures(
            self.weights, self.sectors,
            financial_wealth=1_000_000,
            human_capital_pv=500_000,
            employer_sector="Information Technology",
        )
        # Financials = 0.35*1M / 1.5M = 350k/1.5M
        assert abs(exps["Financials"] - 350_000 / 1_500_000) < 1e-9
