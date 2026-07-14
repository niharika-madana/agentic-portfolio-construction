import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.shared.core.constraints import (
    SINGLE_NAME_LIMIT, SECTOR_LIMIT, ECONOMIC_SECTOR_LIMIT, EMPLOYER_LIMIT,
    aggregate_sector_weights,
    check_single_name, check_sector, check_economic_sector, check_employer,
    run_all_checks, all_violations,
)
from contracts import ConstraintType


SECTORS = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "JPM":  "Financials",
    "GS":   "Financials",
    "XOM":  "Energy",
}


def test_limits_are_correct():
    assert SINGLE_NAME_LIMIT     == 0.10
    assert SECTOR_LIMIT          == 0.20
    assert ECONOMIC_SECTOR_LIMIT == 0.25
    assert EMPLOYER_LIMIT        == 0.15


def test_aggregate_sector_weights_sums():
    weights = {"AAPL": 0.20, "MSFT": 0.15, "JPM": 0.35, "GS": 0.20, "XOM": 0.10}
    sw = aggregate_sector_weights(weights, SECTORS)
    assert abs(sw["Information Technology"] - 0.35) < 1e-9
    assert abs(sw["Financials"] - 0.55) < 1e-9
    assert abs(sw["Energy"] - 0.10) < 1e-9


class TestCheckSingleName:
    def test_no_breach(self):
        assert check_single_name({"AAPL": 0.10, "MSFT": 0.09}) == []

    def test_exactly_at_limit_is_clean(self):
        assert check_single_name({"AAPL": 0.10}) == []

    def test_breach_detected(self):
        violations = check_single_name({"AAPL": 0.12, "MSFT": 0.08})
        assert len(violations) == 1
        assert violations[0].target == "AAPL"
        assert violations[0].current_value == pytest.approx(0.12)
        assert violations[0].limit == SINGLE_NAME_LIMIT
        assert violations[0].constraint_type == ConstraintType.SINGLE_NAME

    def test_multiple_breaches(self):
        weights    = {"AAPL": 0.15, "MSFT": 0.12, "JPM": 0.08}
        violations = check_single_name(weights)
        assert len(violations) == 2


class TestCheckSector:
    def test_no_breach(self):
        assert check_sector({"AAPL": 0.10, "MSFT": 0.09, "JPM": 0.10}, SECTORS) == []

    def test_exactly_at_limit_is_clean(self):
        assert check_sector({"AAPL": 0.10, "MSFT": 0.10}, SECTORS) == []

    def test_breach(self):
        violations = check_sector({"AAPL": 0.12, "MSFT": 0.12, "JPM": 0.10}, SECTORS)
        assert len(violations) == 1
        assert violations[0].target == "Information Technology"
        assert violations[0].constraint_type == ConstraintType.SECTOR


class TestCheckEconomicSector:
    def test_no_breach(self):
        assert check_economic_sector({"IT": 0.25, "Financials": 0.20}) == []

    def test_exactly_at_limit_is_clean(self):
        assert check_economic_sector({"IT": 0.25}) == []

    def test_breach(self):
        violations = check_economic_sector({"IT": 0.30, "Financials": 0.20})
        assert len(violations) == 1
        assert violations[0].target == "IT"
        assert violations[0].constraint_type == ConstraintType.ECONOMIC_SECTOR


class TestCheckEmployer:
    def test_no_breach(self):
        assert check_employer(0.14) == []

    def test_exactly_at_limit_is_clean(self):
        assert check_employer(0.15) == []

    def test_breach(self):
        violations = check_employer(0.18)
        assert len(violations) == 1
        assert violations[0].constraint_type == ConstraintType.EMPLOYER
        assert violations[0].current_value == pytest.approx(0.18)


class TestRunAllChecks:
    def test_clean_portfolio(self):
        weights  = {"AAPL": 0.08, "MSFT": 0.08, "JPM": 0.09}
        econ_exp = {"IT": 0.20, "Financials": 0.09}
        flags    = run_all_checks(weights, SECTORS, econ_exp, 0.10)
        assert flags.single_name_breaches == []
        assert flags.sector_breaches == []
        assert flags.economic_sector_breaches == []
        assert flags.employer_breach is False

    def test_all_breached(self):
        weights  = {"AAPL": 0.12, "MSFT": 0.12, "JPM": 0.10}
        econ_exp = {"Information Technology": 0.30}
        flags    = run_all_checks(weights, SECTORS, econ_exp, 0.20)
        assert "AAPL" in flags.single_name_breaches
        assert "MSFT" in flags.single_name_breaches
        assert "Information Technology" in flags.sector_breaches
        assert "Information Technology" in flags.economic_sector_breaches
        assert flags.employer_breach is True


def test_all_violations_flat_list():
    weights  = {"AAPL": 0.12, "JPM": 0.08}
    econ_exp = {"Information Technology": 0.30}
    viols    = all_violations(weights, SECTORS, econ_exp, 0.20)
    types    = [v.constraint_type for v in viols]
    assert ConstraintType.SINGLE_NAME     in types
    assert ConstraintType.ECONOMIC_SECTOR in types
    assert ConstraintType.EMPLOYER        in types


def test_all_violations_empty_when_clean():
    weights  = {"AAPL": 0.08, "JPM": 0.08}
    econ_exp = {"Information Technology": 0.10, "Financials": 0.08}
    viols    = all_violations(weights, SECTORS, econ_exp, 0.10)
    assert viols == []
