"""
hc_beta_table.py — Calibrated income β/ρ table and income-risk constants.

Replaces the deprecated OLS approach (`beta.py`), which regressed synthetic
Gaussian noise against Fama-French sector returns and produced β ≈ 0 for every
client regardless of career type. β and ρ are now looked up from a calibrated
table keyed by human-capital type.

Academic basis:
  Ibbotson, Milevsky, Chen & Zhu (2007), "Lifetime Financial Advice: Human
    Capital, Asset Allocation, and Insurance," CFA Institute Research Foundation.
  Davis & Willen (2000), "Using Financial Assets to Hedge Labor Income Risks,"
    SSRN — income betas from PSID wage data by occupation class (−0.1 to +0.5 for
    most occupations, rising toward +0.9 for finance/technology roles with equity
    compensation).

The three calibrated β values (0.05, 0.35, 0.90) fall squarely inside the
threshold ranges enforced by contracts.ProfileAgentOutput
(_check_hc_type_consistent_with_beta):
    bond-like   : β ≤ 0.30
    mixed       : 0.30 < β ≤ 0.80
    equity-like : β > 0.80
"""

from __future__ import annotations

# ── Income volatility sigma (σ) ────────────────────────────────────────────
# Annualised earnings uncertainty by income-stability label. Used in the
# effective_risk_budget formula: (FC + HC×(1−σ)) / total_wealth.
INCOME_VOLATILITY_SIGMA = {
    "High":   0.05,   # tenured / government — very stable
    "Medium": 0.20,   # bonus-driven, market-correlated
    "Low":    0.40,   # RSU / commission — highly variable
}

# ── Human-capital type by income stability ─────────────────────────────────
HUMAN_CAPITAL_TYPE = {
    "High":   "bond-like",
    "Medium": "mixed",
    "Low":    "equity-like",
}

# ── Calibrated HC beta table ───────────────────────────────────────────────
# β  = income_equity_beta        — systematic sensitivity of income to equities
# ρ  = income_equity_correlation — corr(income changes, equity market returns)
HC_BETA_TABLE = {
    "bond-like":   {"beta": 0.05, "correlation": 0.10},
    "mixed":       {"beta": 0.35, "correlation": 0.40},
    "equity-like": {"beta": 0.90, "correlation": 0.75},
}


def lookup_hc_beta(hc_type: str) -> dict[str, float]:
    """
    Return {'beta': β, 'correlation': ρ} for a human-capital type.

    Raises KeyError with a clear message if the type is unknown.
    """
    if hc_type not in HC_BETA_TABLE:
        raise KeyError(
            f"Unknown human_capital_type '{hc_type}'. "
            f"Expected one of {list(HC_BETA_TABLE)}."
        )
    return dict(HC_BETA_TABLE[hc_type])


def sigma_for_stability(income_stability: str) -> float:
    """Return income_volatility_sigma (σ) for an income-stability label."""
    if income_stability not in INCOME_VOLATILITY_SIGMA:
        raise KeyError(
            f"Unknown income_stability '{income_stability}'. "
            f"Expected one of {list(INCOME_VOLATILITY_SIGMA)}."
        )
    return INCOME_VOLATILITY_SIGMA[income_stability]


def hc_type_for_stability(income_stability: str) -> str:
    """Return the human_capital_type label for an income-stability label."""
    if income_stability not in HUMAN_CAPITAL_TYPE:
        raise KeyError(
            f"Unknown income_stability '{income_stability}'. "
            f"Expected one of {list(HUMAN_CAPITAL_TYPE)}."
        )
    return HUMAN_CAPITAL_TYPE[income_stability]
