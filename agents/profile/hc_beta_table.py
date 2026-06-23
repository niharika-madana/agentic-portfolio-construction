"""
hc_beta_table.py — Profile Agent
==================================
Calibrated income equity beta (β) and correlation (ρ) by human capital type.

Replaces the OLS regression in beta.py, which regressed synthetic noise
against sector returns and produced β ≈ 0 for every client.

Academic basis:
    Ibbotson, Milevsky, Chen, Zhu (2007) — "Lifetime Financial Advice:
    Human Capital, Asset Allocation, and Insurance." CFA Institute.
    https://www.cfainstitute.org/en/research/foundation/2007/lifetime-financial-advice-human-capital-asset-allocation-and-insurance

    Davis & Willen (2000) — "Using Financial Assets to Hedge Labor Income
    Risks: Estimating the Benefits." SSRN. Estimated income betas by
    education/occupation from PSID data; find range −0.1 to 0.5 across
    occupations, rising to ~0.9 for finance/tech with equity compensation.

Calibration rationale:
    bond-like   β=0.05 — government, academia: salary determined by budgets
                          and tenure, essentially uncorrelated with markets.
                          ρ=0.10 — slight positive correlation (broad economic
                          conditions affect public sector funding).

    mixed       β=0.35 — healthcare, legal, engineering: partial market
                          sensitivity through client/billing cycles and
                          annual bonuses, but base salary is stable.
                          ρ=0.40 — moderate correlation.

    equity-like β=0.90 — technology, finance: bonus + RSU compensation
                          tracks equity markets directly; layoff risk spikes
                          in downturns. Contracts.py docstring examples:
                          "Tech exec ≈ 1.2, financial professional ≈ 0.6."
                          We use 0.90 as the representative equity-like value.
                          ρ=0.75 — matches contracts.py: "Tech exec ≈ 0.75."

Consistency check against contracts.py thresholds (line 244):
    bond-like:   β=0.05 ≤ 0.3  ✓
    mixed:       β=0.35 ∈ (0.3, 0.8]  ✓
    equity-like: β=0.90 > 0.8  ✓

Exports:
    HC_BETA_TABLE
    lookup_hc_beta()
"""

HC_BETA_TABLE: dict[str, dict[str, float]] = {
    "bond-like":   {"beta": 0.05, "correlation": 0.10},
    "mixed":       {"beta": 0.35, "correlation": 0.40},
    "equity-like": {"beta": 0.90, "correlation": 0.75},
}


def lookup_hc_beta(hc_type: str) -> tuple[float, float]:
    """
    Return (beta, correlation) for a given human capital type.

    Args:
        hc_type: "bond-like" | "mixed" | "equity-like"

    Returns:
        (income_equity_beta, income_equity_correlation)

    Raises:
        KeyError if hc_type is not one of the three valid values.
    """
    if hc_type not in HC_BETA_TABLE:
        raise KeyError(
            f"Unknown hc_type: '{hc_type}'. "
            f"Valid values: {list(HC_BETA_TABLE.keys())}"
        )
    entry = HC_BETA_TABLE[hc_type]
    return entry["beta"], entry["correlation"]
