"""
test_profile.py — Profile Agent unit tests (no live API calls).

Run from the repo root:
    pytest agents/profile/test_profile.py -v

Tests cover the pure formulas, the derivation rules, contract validation, and
the consistency invariants the design doc guarantees. The BLS OES loader is
exercised only if the parquet cache or the source xlsx is present; otherwise
those tests skip.
"""

from __future__ import annotations

import math

import pytest

from contracts import ProfileAgentOutput
from agents.profile import hc_beta_table as hcb
from agents.profile import personas as P
from agents.profile.human_capital import (
    build_profile,
    compute_human_capital,
    to_profile_agent_output,
)


# ── A representative bond-like persona used across tests ───────────────────
def _bio_persona() -> dict:
    return {
        "client_id":                "bls_25-1042_p50",
        "soc":                      "25-1042",
        "label":                    "Biology Professor",
        "career_type":              "Academia",
        "age":                      47,
        "annual_salary":            83920.0,
        "bonus_rate":               0.046,
        "effective_salary":         round(83920.0 * 1.046, 2),
        "years_to_retirement":      18,
        "income_stability":         "High",
        "industry_exposure_sector": "Education",
        "financial_capital":        200000.0,
        "current_holdings":         {"US_equity": 0.50, "intl_equity": 0.15, "bonds": 0.25, "cash": 0.10},
        "investment_horizon_years": 18,
        "risk_tolerance":           "moderate",
        "liquidity_needs":          "low",
        "investment_objective":     "growth",
        "RSU_concentration":        0.0,
        "has_pension":              True,
    }


# ── Human capital annuity formula ──────────────────────────────────────────

def test_compute_human_capital_matches_annuity_formula():
    salary, n, r = 100000.0, 20, 0.044
    expected = salary * (1 - (1 + r) ** (-n)) / r
    assert compute_human_capital(salary, n, r) == pytest.approx(expected, abs=0.01)


def test_compute_human_capital_zero_horizon():
    assert compute_human_capital(100000.0, 0, 0.044) == 0.0


# ── Calibrated table ───────────────────────────────────────────────────────

@pytest.mark.parametrize("hc_type,beta", [
    ("bond-like", 0.05), ("mixed", 0.35), ("equity-like", 0.90),
])
def test_beta_within_contract_thresholds(hc_type, beta):
    assert hcb.lookup_hc_beta(hc_type)["beta"] == beta


def test_lookup_hc_beta_unknown_raises():
    with pytest.raises(KeyError):
        hcb.lookup_hc_beta("nonsense")


# ── Derivation rules ───────────────────────────────────────────────────────

def test_risk_tolerance_rules():
    assert P.derive_risk_tolerance("bond-like", 47) == "moderate"
    assert P.derive_risk_tolerance("bond-like", 55) == "conservative"
    assert P.derive_risk_tolerance("mixed", 60) == "moderate"
    assert P.derive_risk_tolerance("equity-like", 38) == "aggressive"
    assert P.derive_risk_tolerance("equity-like", 50) == "moderate"


def test_holdings_sum_to_one():
    for hc_type, age, rt, rsu in [
        ("bond-like", 47, "moderate", 0.0),
        ("equity-like", 38, "aggressive", 0.35),
        ("mixed", 45, "moderate", 0.0),
        ("bond-like", 55, "conservative", 0.0),
    ]:
        holdings = P.derive_current_holdings(hc_type, age, rt, rsu)
        assert math.isclose(sum(holdings.values()), 1.0, abs_tol=0.01)


# ── build_profile invariants ───────────────────────────────────────────────

def test_build_profile_invariants():
    prof = build_profile(_bio_persona(), discount_rate=0.044)

    # total wealth = FC + HC
    assert prof["total_wealth"] == pytest.approx(
        prof["financial_capital"] + prof["human_capital_valuation"], rel=1e-6
    )
    # implicit equity exposure = hc_share × β
    hc_share = prof["human_capital_valuation"] / prof["total_wealth"]
    assert prof["implicit_equity_exposure"] == pytest.approx(
        round(hc_share * prof["income_equity_beta"], 3), abs=0.01
    )
    # portfolio equity target = risk budget − implicit exposure
    assert prof["portfolio_equity_target"] == pytest.approx(
        round(prof["effective_risk_budget"] - prof["implicit_equity_exposure"], 3), abs=1e-9
    )
    # income_stability is mapped to the lowercase contract value
    assert prof["income_stability"] == "high"


def test_build_profile_validates_against_contract():
    prof = build_profile(_bio_persona(), discount_rate=0.044)
    out = to_profile_agent_output(prof)
    assert isinstance(out, ProfileAgentOutput)
    assert out.human_capital_type.value == "bond-like"
    assert out.income_stability.value == "high"
    assert out.bonus_rate == 0.046
    assert out.portfolio_equity_target is not None


def test_equity_like_persona_can_have_negative_target():
    persona = _bio_persona()
    persona.update({
        "client_id":         "bls_15-1252_p50",
        "income_stability":  "Low",
        "RSU_concentration": 0.35,
        "current_holdings":  P.derive_current_holdings("equity-like", 38, "aggressive", 0.35),
        "age":               38,
        "years_to_retirement": 27,
        "effective_salary":  round(132270.0 * 1.085, 2),
        "financial_capital": 90000.0,
    })
    out = to_profile_agent_output(build_profile(persona, discount_rate=0.044))
    assert out.human_capital_type.value == "equity-like"
    # equity-like career carries large implicit exposure → small/negative target
    assert out.portfolio_equity_target < out.effective_risk_budget


# ── BLS persona builder (only if OES data is available) ────────────────────

def test_build_bls_personas_if_data_present():
    from agents.profile.loaders import BLS_PARQUET, BLS_XLSX
    if not (BLS_PARQUET.exists() or BLS_XLSX.exists()):
        pytest.skip("BLS OES data not available in this environment")
    from agents.profile.loaders import load_bls_oes
    personas = P.build_bls_personas(load_bls_oes(), include_percentile_variants=False)
    assert len(personas) >= 1
    assert all("client_id" in p for p in personas)
