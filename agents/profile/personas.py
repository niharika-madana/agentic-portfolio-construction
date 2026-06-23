"""
personas.py — Profile Agent
============================
BLS OES-based persona construction.

Replaces:
  - RAW_PERSONAS (hardcoded salary/wealth dicts)
  - generate_synthetic() (Claude + GPT-4o pipeline)

All salary data comes from BLS OES May 2023.
All financial_capital data comes from SCF 2022 (via loaders.SCF_FINANCIAL_ASSETS).
Qualitative fields (risk_tolerance, liquidity_needs, investment_objective,
current_holdings) are derived from occupation characteristics via documented rules.

Exports:
  - TARGET_OCCUPATIONS
  - SOC_TO_GICS
  - HUMAN_CAPITAL_TYPE
  - build_bls_personas()
"""

import pandas as pd

from .human_capital import HUMAN_CAPITAL_TYPE
from .loaders import get_scf_financial_capital

# ---------------------------------------------------------------------------
# Target occupations — 9 covering all three HC types
# age: approximate median from BLS CPS Table 11 "Median age by occupation"
# ---------------------------------------------------------------------------

TARGET_OCCUPATIONS = [
    # ── Bond-like (σ=0.05, β=0.05, ρ=0.10) ─────────────────────────────
    {
        "soc":              "25-1042",
        "label":            "Biology Professor",
        "career_type":      "Academia",
        "income_stability": "High",
        "age":              47,
        "has_pension":      True,
        "rsu_eligible":     False,
    },
    {
        "soc":              "29-1141",
        "label":            "Registered Nurse",
        "career_type":      "Healthcare",
        "income_stability": "High",
        "age":              43,
        "has_pension":      False,
        "rsu_eligible":     False,
    },
    {
        "soc":              "13-1041",
        "label":            "Compliance Officer",
        "career_type":      "Government",
        "income_stability": "High",
        "age":              44,
        "has_pension":      True,
        "rsu_eligible":     False,
    },
    # ── Mixed (σ=0.20, β=0.35, ρ=0.40) ─────────────────────────────────
    {
        "soc":              "23-1011",
        "label":            "Lawyer",
        "career_type":      "Legal",
        "income_stability": "Medium",
        "age":              46,
        "has_pension":      False,
        "rsu_eligible":     False,
    },
    {
        "soc":              "17-2141",
        "label":            "Mechanical Engineer",
        "career_type":      "Engineering",
        "income_stability": "Medium",
        "age":              44,
        "has_pension":      False,
        "rsu_eligible":     False,
    },
    {
        "soc":              "13-2051",
        "label":            "Financial Analyst",
        "career_type":      "Finance",
        "income_stability": "Medium",
        "age":              40,
        "has_pension":      False,
        "rsu_eligible":     False,
    },
    # ── Equity-like (σ=0.40, β=0.90, ρ=0.75) ───────────────────────────
    {
        "soc":              "15-1252",
        "label":            "Software Developer",
        "career_type":      "Technology",
        "income_stability": "Low",
        "age":              38,
        "has_pension":      False,
        "rsu_eligible":     True,
    },
    {
        "soc":              "11-3021",
        "label":            "IT Manager",
        "career_type":      "Technology",
        "income_stability": "Low",
        "age":              45,
        "has_pension":      False,
        "rsu_eligible":     True,
    },
    {
        "soc":              "11-2022",
        "label":            "Sales Manager",
        "career_type":      "Sales",
        "income_stability": "Low",
        "age":              42,
        "has_pension":      False,
        "rsu_eligible":     False,
    },
]

# ---------------------------------------------------------------------------
# SOC code → GICS-aligned sector string
# ---------------------------------------------------------------------------

SOC_TO_GICS: dict[str, str] = {
    "25-1042": "Education",
    "29-1141": "Health Care",
    "13-1041": "Government / Public Administration",
    "23-1011": "Professional Services",
    "17-2141": "Industrials",
    "13-2051": "Financials",
    "15-1252": "Information Technology",
    "11-3021": "Information Technology",
    "11-2022": "Consumer Discretionary",
}


# ---------------------------------------------------------------------------
# Rule-based field derivation
# ---------------------------------------------------------------------------

def _derive_risk_tolerance(hc_type: str, age: int) -> str:
    """
    Rule: stable income (bond-like HC) offsets portfolio risk budget, allowing
    slightly higher portfolio equity; but older clients near retirement cap this.

    bond-like + age < 50  → moderate (stable base lets them take portfolio risk)
    bond-like + age >= 50 → conservative (approaching retirement, protect gains)
    mixed                  → moderate
    equity-like + age < 45 → aggressive (high income upside, long horizon)
    equity-like + age >= 45 → moderate (income risk already high, balance it)
    """
    if hc_type == "bond-like":
        return "Conservative" if age >= 50 else "Moderate"
    if hc_type == "mixed":
        return "Moderate"
    # equity-like
    return "Aggressive" if age < 45 else "Moderate"


def _derive_liquidity_needs(income_stability: str) -> str:
    """
    Rule: income predictability drives liquidity need.

    High stability (regular salary) → Low liquidity need; income itself is the buffer.
    Medium stability (bonus-driven)  → Medium; bonus timing is unpredictable.
    Low stability (RSU/commission)   → Medium; equity vesting and variable cash flow.
    """
    return {"High": "Low", "Medium": "Medium", "Low": "Medium"}[income_stability]


def _derive_investment_objective(years_to_retirement: int) -> str:
    """
    Rule based on investment horizon:
    > 20 years  → Growth (maximize long-run accumulation)
    10–20 years → Growth (still accumulating, can absorb volatility)
    < 10 years  → Income (shift toward capital preservation)
    """
    if years_to_retirement >= 10:
        return "Growth"
    return "Income"


def _derive_rsu_concentration(occ: dict, percentile_label: str) -> float:
    """
    RSU concentration only applies to equity-like, RSU-eligible occupations.
    Higher salary percentile → more of compensation is in equity.

    p25: 20% RSU (below-median compensation, less equity grant)
    p50: 35% RSU (typical mid-career equity grant)
    p75: 55% RSU (senior IC or manager level, heavy equity comp)
    """
    if not occ["rsu_eligible"]:
        return 0.0
    return {"p25": 0.20, "p50": 0.35, "p75": 0.55}[percentile_label]


def _derive_current_holdings(
    age: int,
    risk_tolerance: str,
    rsu_concentration: float,
) -> dict[str, float]:
    """
    Rule-based asset allocation from SCF 2022 Table 7 "Family Holdings of
    Financial Assets by Selected Characteristics."

    For RSU holders, employer_RSU takes the RSU concentration weight and
    remaining capital is split across broad market assets.

    All weights sum to exactly 1.0.
    """
    rt = risk_tolerance.lower()

    if rsu_concentration > 0:
        remaining = round(1.0 - rsu_concentration, 2)
        return {
            "employer_RSU": rsu_concentration,
            "US_equity":    round(remaining * 0.55, 2),
            "bonds":        round(remaining * 0.25, 2),
            "cash":         round(remaining * 0.20, 2),
        }

    if rt == "aggressive":
        if age < 45:
            return {"US_equity": 0.65, "intl_equity": 0.20, "bonds": 0.10, "cash": 0.05}
        return {"US_equity": 0.55, "intl_equity": 0.20, "bonds": 0.20, "cash": 0.05}

    if rt == "moderate":
        if age < 45:
            return {"US_equity": 0.50, "intl_equity": 0.15, "bonds": 0.25, "cash": 0.10}
        return {"US_equity": 0.40, "intl_equity": 0.15, "bonds": 0.35, "cash": 0.10}

    # conservative
    return {"US_equity": 0.25, "intl_equity": 0.10, "bonds": 0.50, "cash": 0.15}


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_bls_personas(
    bls_data: pd.DataFrame,
    include_percentile_variants: bool = False,
) -> list[dict]:
    """
    Build persona dicts from BLS OES salary data and SCF wealth data.

    For each occupation in TARGET_OCCUPATIONS, creates:
        - 1 persona (median / p50) if include_percentile_variants=False
        - 3 personas (p25, p50, p75) if include_percentile_variants=True

    All qualitative fields (risk_tolerance, liquidity_needs, investment_objective,
    current_holdings) are derived via documented rules, not hardcoded.

    Returns:
        list[dict] — raw persona dicts ready for build_profile()
    """
    personas = []

    for occ in TARGET_OCCUPATIONS:
        soc = occ["soc"]

        if soc not in bls_data.index:
            print(f"WARNING: SOC {soc} ({occ['label']}) not found in BLS data — skipping")
            continue

        row = bls_data.loc[soc]

        # Decide which salary percentiles to generate
        if include_percentile_variants:
            candidates = [
                ("p25", row.get("A_PCT25")),
                ("p50", row.get("A_MEDIAN")),
                ("p75", row.get("A_PCT75")),
            ]
        else:
            candidates = [("p50", row.get("A_MEDIAN"))]

        for pct_label, salary in candidates:
            if pd.isna(salary):
                print(f"  SKIP: {soc} {pct_label} — BLS wage suppressed or unavailable")
                continue

            salary             = int(salary)
            age                = occ["age"]
            years_to_ret       = 65 - age
            hc_type            = HUMAN_CAPITAL_TYPE[occ["income_stability"]]
            risk_tolerance     = _derive_risk_tolerance(hc_type, age)
            liquidity_needs    = _derive_liquidity_needs(occ["income_stability"])
            investment_obj     = _derive_investment_objective(years_to_ret)
            rsu_concentration  = _derive_rsu_concentration(occ, pct_label)
            current_holdings   = _derive_current_holdings(age, risk_tolerance, rsu_concentration)
            financial_capital  = get_scf_financial_capital(age, pct_label)

            persona = {
                "client_id":                f"bls_{soc}_{pct_label}",
                "name":                     f"{occ['label']} ({pct_label.upper()})",
                "age":                      age,
                "annual_salary":            salary,
                "years_to_retirement":      years_to_ret,
                "career_type":              occ["career_type"],
                "income_stability":         occ["income_stability"],   # "High"/"Medium"/"Low"
                "industry_exposure_sector": SOC_TO_GICS[soc],
                "financial_capital":        financial_capital,
                "current_holdings":         current_holdings,
                "investment_horizon_years": years_to_ret,
                "risk_tolerance":           risk_tolerance,            # "Conservative" etc.
                "liquidity_needs":          liquidity_needs,           # "Low" etc.
                "investment_objective":     investment_obj,            # "Growth" etc.
                "RSU_concentration":        rsu_concentration,
                "has_pension":              occ["has_pension"],
            }
            personas.append(persona)
            print(
                f"  {persona['client_id']:30s}  "
                f"salary=${salary:>9,}  "
                f"FC=${financial_capital:>9,}  "
                f"hc_type={hc_type}"
            )

    print(f"\nbuild_bls_personas: {len(personas)} personas constructed")
    return personas
