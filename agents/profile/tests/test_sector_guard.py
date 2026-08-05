"""
Tests for the sector-overlap guard and the sector vocabulary it depends on.

Located under agents/profile/ rather than tests/ because the top-level tests/
directory is outside the Profile/Research edit scope. pyproject sets
testpaths = ["tests"], so a bare `pytest` does not collect these; run them with

    uv run pytest agents/profile/tests

or fold them into the default run by adding "agents/profile/tests" to testpaths
(a one-line change for whoever owns pyproject.toml).
"""

from __future__ import annotations

import pytest

from contracts import (
    GICS_SECTORS,
    NON_INVESTABLE_EMPLOYER_SECTORS,
    HumanCapitalType,
    IncomeStability,
    InvestmentObjective,
    LiquidityNeeds,
    ProfileAgentOutput,
    RiskToleranceLevel,
    is_investable_sector,
    normalize_sector,
)
from agents.profile.sector_guard import (
    DEFAULT_EMPLOYER_SECTOR_CAP,
    SectorOverlapError,
    check_sector_overlap,
    employer_sector_cap,
    sector_exposure,
)

# A tech client with the persona set's actual tech parameters: ρ = 0.75, β = 1.2,
# RSU 0.35. 95% of their wealth is human capital, so implicit equity exposure is
# 1.14 — the case the whole HC framework exists to offset.
_TECH_CLIENT = dict(
    client_id                  = "bls_15-1252_p50",
    career_type                = "Technology",
    age                        = 38,
    financial_capital          = 90_000.0,
    human_capital_valuation    = 1_710_000.0,
    total_wealth               = 1_800_000.0,
    human_capital_pct_of_total = 95.0,
    income_volatility_sigma    = 0.40,
    income_equity_correlation  = 0.75,
    income_equity_beta         = 1.20,
    implicit_equity_exposure   = 1.14,
    human_capital_type         = HumanCapitalType.EQUITY_LIKE,
    income_stability           = IncomeStability.LOW,
    effective_risk_budget      = 0.62,
    industry_exposure_sector   = "Information Technology",
    RSU_concentration          = 0.35,
    current_holdings           = {"US_equity": 0.65, "bonds": 0.25, "cash": 0.10},
    investment_horizon_years   = 27,
    risk_tolerance_level       = RiskToleranceLevel.AGGRESSIVE,
    liquidity_needs            = LiquidityNeeds.MEDIUM,
    investment_objective       = InvestmentObjective.GROWTH,
)

_SECTORS = {
    "XLK": "Information Technology",
    "XLV": "Health Care",
    "XLF": "Financials",
    "SPY": "Broad Market",
    "AGG": "Fixed Income",
}


def _profile(**overrides) -> ProfileAgentOutput:
    return ProfileAgentOutput(**{**_TECH_CLIENT, **overrides})


class TestSectorNormalization:
    """The near-miss spellings that silently disabled the employer sector cap."""

    @pytest.mark.parametrize(
        "raw, canonical",
        [
            ("Technology",         "Information Technology"),
            ("technology",         "Information Technology"),
            ("  TECHNOLOGY  ",     "Information Technology"),
            ("Healthcare",         "Health Care"),
            ("Financial Services", "Financials"),
            ("Industrials",        "Industrials"),
            ("Consumer Discretionary", "Consumer Discretionary"),
        ],
    )
    def test_aliases_map_to_gics(self, raw, canonical):
        assert normalize_sector(raw) == canonical
        assert canonical in GICS_SECTORS

    def test_non_investable_sectors_pass_through(self):
        for sector in NON_INVESTABLE_EMPLOYER_SECTORS:
            assert normalize_sector(sector) == sector
            assert not is_investable_sector(sector)

    def test_unrecognised_sector_is_not_investable_but_survives(self):
        assert normalize_sector("Techonlogy") == "Techonlogy"
        assert not is_investable_sector("Techonlogy")

    def test_profile_normalises_on_construction(self):
        """The contract fixes the spelling so no call site has to remember to."""
        assert _profile(industry_exposure_sector="Technology").industry_exposure_sector == (
            "Information Technology"
        )


class TestEmployerSectorCap:
    def test_correlation_adjustment_matches_scope_doc(self):
        """SCOPE.md §3.2: adjusted_limit = base_limit × (1 − ρ). ρ=0.75 → 2.5%."""
        assert employer_sector_cap(_profile()) == pytest.approx(0.025)

    def test_low_correlation_client_keeps_most_of_the_cap(self):
        professor = _profile(
            income_equity_correlation = 0.10,
            income_equity_beta        = 0.05,
            implicit_equity_exposure  = 0.0475,
            human_capital_type        = HumanCapitalType.BOND_LIKE,
            income_stability          = IncomeStability.HIGH,
            RSU_concentration         = 0.0,
        )
        assert employer_sector_cap(professor) == pytest.approx(0.09)

    def test_flat_cap_when_adjustment_disabled(self):
        assert employer_sector_cap(_profile(), correlation_adjusted=False) == 0.10

    def test_negative_correlation_does_not_loosen_the_cap(self):
        """A market-hedging career is a reason to worry less, not to concentrate."""
        hedged = _profile(
            income_equity_correlation = -0.30,
            income_equity_beta        = 0.05,
            implicit_equity_exposure  = 0.0475,
            human_capital_type        = HumanCapitalType.BOND_LIKE,
        )
        assert employer_sector_cap(hedged) == pytest.approx(DEFAULT_EMPLOYER_SECTOR_CAP)

    def test_matches_the_shared_constraint_module(self):
        """The Profile-side default must not drift from the Allocation-side limit."""
        from agents.shared.core.constraints import EMPLOYER_SECTOR_LIMIT

        assert DEFAULT_EMPLOYER_SECTOR_CAP == EMPLOYER_SECTOR_LIMIT


class TestSectorExposure:
    def test_sums_only_the_target_sector(self):
        weights = {"XLK": 0.20, "XLV": 0.10, "SPY": 0.70}
        assert sector_exposure(weights, _SECTORS, "Information Technology") == pytest.approx(0.20)

    def test_matches_across_spelling_variants(self):
        weights = {"XLK": 0.20, "SPY": 0.80}
        assert sector_exposure(weights, {"XLK": "Technology", "SPY": "Broad Market"},
                               "Information Technology") == pytest.approx(0.20)

    def test_unknown_ticker_contributes_nothing(self):
        assert sector_exposure({"NOPE": 1.0}, _SECTORS, "Information Technology") == 0.0


class TestCheckSectorOverlap:
    def test_tech_heavy_portfolio_for_tech_client_raises(self):
        """The case from the 4 Aug minutes: a tech client handed a tech-heavy book."""
        weights = {"XLK": 0.30, "SPY": 0.50, "AGG": 0.20}
        with pytest.raises(SectorOverlapError, match="Information Technology"):
            check_sector_overlap(_profile(), weights, _SECTORS)

    def test_error_names_the_client_and_both_numbers(self):
        weights = {"XLK": 0.30, "AGG": 0.70}
        with pytest.raises(SectorOverlapError) as exc:
            check_sector_overlap(_profile(), weights, _SECTORS)
        message = str(exc.value)
        assert "bls_15-1252_p50" in message
        assert "30.00%" in message   # the exposure
        assert "2.50%" in message    # the correlation-adjusted cap

    def test_portfolio_within_cap_passes(self):
        weights = {"XLK": 0.02, "SPY": 0.78, "AGG": 0.20}
        result = check_sector_overlap(_profile(), weights, _SECTORS)
        assert result.applicable
        assert not result.violated

    def test_ten_percent_tech_still_fails_a_high_correlation_client(self):
        """
        The minutes' flat "tech ≤ 10%" would pass this book. The correlation
        adjustment is what makes it fail, which is the point of the adjustment.
        """
        weights = {"XLK": 0.10, "SPY": 0.70, "AGG": 0.20}
        assert not check_sector_overlap(
            _profile(), weights, _SECTORS,
            correlation_adjusted=False, raise_on_violation=False,
        ).violated
        assert check_sector_overlap(
            _profile(), weights, _SECTORS, raise_on_violation=False
        ).violated

    def test_raise_on_violation_false_returns_result_instead(self):
        weights = {"XLK": 0.30, "AGG": 0.70}
        result = check_sector_overlap(
            _profile(), weights, _SECTORS, raise_on_violation=False
        )
        assert result.violated
        assert result.exposure == pytest.approx(0.30)
        assert "EXCEEDS" in result.describe()

    def test_configurable_cap(self):
        weights = {"XLK": 0.15, "AGG": 0.85}
        assert not check_sector_overlap(
            _profile(), weights, _SECTORS,
            base_cap=0.60, raise_on_violation=False,
        ).violated

    def test_non_investable_employer_sector_is_a_no_op(self):
        """A professor's employer is not in an investable sector; nothing to check."""
        professor = _profile(
            client_id                 = "bls_25-1042_p50",
            industry_exposure_sector  = "Education",
            income_equity_correlation = 0.10,
            income_equity_beta        = 0.05,
            implicit_equity_exposure  = 0.0475,
            human_capital_type        = HumanCapitalType.BOND_LIKE,
            income_stability          = IncomeStability.HIGH,
            RSU_concentration         = 0.0,
        )
        result = check_sector_overlap(professor, {"XLK": 0.90, "AGG": 0.10}, _SECTORS)
        assert not result.applicable
        assert not result.violated
        assert "no listed sector ETF" in result.reason

    def test_unmapped_sector_reports_the_mapping_gap(self):
        """
        An unrecognised sector must not read as a pass. This is the shape of the
        bug being fixed: the check silently did not apply, and nothing said so.
        """
        result = check_sector_overlap(
            _profile(industry_exposure_sector="Techonlogy"),
            {"XLK": 0.90, "AGG": 0.10},
            _SECTORS,
        )
        assert not result.applicable
        assert "not a recognised GICS sector" in result.reason
        assert "not applicable" in result.describe()


class TestPersonaSectorsAreMatchable:
    """
    Regression test for the bug this work exists to fix.

    Every persona's employer sector must be a string the Allocation Agent can
    actually match — otherwise the employer sector cap and the employer proxy ETF
    both fail open, with no exception and no warning.
    """

    def test_every_target_occupation_sector_is_canonical(self):
        from agents.profile.profile_model import TARGET_OCCUPATIONS

        for occ in TARGET_OCCUPATIONS:
            sector = occ["sector"]
            assert sector in GICS_SECTORS or sector in NON_INVESTABLE_EMPLOYER_SECTORS, (
                f"{occ['label']} sector {sector!r} is not matchable downstream"
            )

    def test_investable_persona_sectors_resolve_to_a_sector_etf(self):
        from agents.allocation.adapters import ETF_SECTORS
        from agents.profile.profile_model import TARGET_OCCUPATIONS

        etf_sectors = set(ETF_SECTORS.values())
        for occ in TARGET_OCCUPATIONS:
            if occ["sector"] in GICS_SECTORS:
                assert occ["sector"] in etf_sectors, (
                    f"{occ['label']} sector {occ['sector']!r} matches no ETF in "
                    f"ETF_SECTORS, so its sector cap will never bind"
                )

    def test_every_persona_sector_has_an_employer_proxy_etf(self):
        """
        A sector missing from _SECTOR_PROXY_ETF falls through to SPY, hedging the
        career against the broad market instead of the sector it is exposed to.
        """
        from agents.allocation.adapters import _SECTOR_PROXY_ETF
        from agents.profile.profile_model import TARGET_OCCUPATIONS

        unmapped = sorted(
            {occ["sector"] for occ in TARGET_OCCUPATIONS}
            - set(_SECTOR_PROXY_ETF)
        )
        assert unmapped == ["Legal"], (
            f"Persona sectors with no employer proxy ETF: {unmapped}. Only 'Legal' "
            f"is a known gap (no listed ETF tracks legal services; it falls back to "
            f"SPY, which is the honest proxy for it)."
        )
