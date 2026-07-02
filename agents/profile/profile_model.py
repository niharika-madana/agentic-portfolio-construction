"""
profile_model.py — Profile Agent domain model.

Merged from hc_beta_table.py + human_capital.py + personas.py.

Sections (in dependency order):
  1. HC beta / sigma / type tables    (calibrated constants)
  2. HC valuation formula             (annuity PV, build_profile)
  3. BLS persona builder              (reference tables, build_bls_personas)
"""

from __future__ import annotations

import pandas as pd

from contracts import ProfileAgentOutput


# ===========================================================================
# 1. HC beta / sigma / type tables
# ===========================================================================
# Academic basis:
#   Ibbotson, Milevsky, Chen & Zhu (2007), "Lifetime Financial Advice: Human
#     Capital, Asset Allocation, and Insurance," CFA Institute Research Foundation.
#   Davis & Willen (2000), "Using Financial Assets to Hedge Labor Income Risks,"
#     SSRN — income betas from PSID wage data by occupation class.

# Annualised earnings uncertainty by income-stability label.
# Used in: effective_risk_budget = (FC + HC×(1−σ)) / total_wealth
INCOME_VOLATILITY_SIGMA = {
    "High":   0.05,   # tenured / government — very stable
    "Medium": 0.20,   # bonus-driven, market-correlated
    "Low":    0.40,   # RSU / commission — highly variable
}

# Human-capital type by income stability.
HUMAN_CAPITAL_TYPE = {
    "High":   "bond-like",
    "Medium": "mixed",
    "Low":    "equity-like",
}

# Calibrated β and ρ per HC type.
# β = income_equity_beta  — systematic sensitivity of income to equities
# ρ = income_equity_correlation
HC_BETA_TABLE = {
    "bond-like":   {"beta": 0.05, "correlation": 0.10},
    "mixed":       {"beta": 0.35, "correlation": 0.40},
    "equity-like": {"beta": 0.90, "correlation": 0.75},
}


def hc_type_for_stability(income_stability: str) -> str:
    if income_stability not in HUMAN_CAPITAL_TYPE:
        raise KeyError(f"Unknown income_stability '{income_stability}'. Expected one of {list(HUMAN_CAPITAL_TYPE)}.")
    return HUMAN_CAPITAL_TYPE[income_stability]


def sigma_for_stability(income_stability: str) -> float:
    if income_stability not in INCOME_VOLATILITY_SIGMA:
        raise KeyError(f"Unknown income_stability '{income_stability}'. Expected one of {list(INCOME_VOLATILITY_SIGMA)}.")
    return INCOME_VOLATILITY_SIGMA[income_stability]


def lookup_hc_beta(hc_type: str) -> dict[str, float]:
    if hc_type not in HC_BETA_TABLE:
        raise KeyError(f"Unknown human_capital_type '{hc_type}'. Expected one of {list(HC_BETA_TABLE)}.")
    return dict(HC_BETA_TABLE[hc_type])


# ===========================================================================
# 2. HC valuation formula
# ===========================================================================
# Core formulas (ProfileAgent design doc):
#
#   Effective Annual Earnings = Annual Salary × (1 + bonus_rate)
#   HC = Effective Earnings × [1 − (1 + r)^(−n)] / r        (annuity PV)
#
#   HC_share                 = HC / Total Wealth
#   Implicit Equity Exposure = HC_share × β
#   Effective Risk Budget    = (FC + HC × (1 − σ)) / Total Wealth
#   Portfolio Equity Target  = Effective Risk Budget − Implicit Equity Exposure

_STABILITY_TO_CONTRACT = {"High": "high", "Medium": "medium", "Low": "low"}


def compute_human_capital(
    effective_salary: float, years_to_retirement: int, discount_rate: float
) -> float:
    """PV of future earnings stream (base + bonus) discounted at DGS10."""
    if years_to_retirement <= 0:
        return 0.0
    return round(
        effective_salary * (1 - (1 + discount_rate) ** (-years_to_retirement)) / discount_rate,
        2,
    )


def build_profile(persona: dict, discount_rate: float) -> dict:
    """
    Derive all computed fields from a raw BLS persona dict.
    Returns a flat dict ready for Pydantic validation via to_profile_agent_output().
    """
    hc = compute_human_capital(
        persona["effective_salary"],
        persona["years_to_retirement"],
        discount_rate,
    )
    fc = persona["financial_capital"]
    total_wealth = round(fc + hc, 2)

    sigma   = sigma_for_stability(persona["income_stability"])
    hc_type = hc_type_for_stability(persona["income_stability"])
    cal     = lookup_hc_beta(hc_type)
    beta    = cal["beta"]
    correlation = cal["correlation"]

    hc_share                = hc / total_wealth
    implicit_equity_exposure = round(hc_share * beta, 3)
    effective_risk_budget    = round((fc + hc * (1 - sigma)) / total_wealth, 3)
    portfolio_equity_target  = round(effective_risk_budget - implicit_equity_exposure, 3)

    return {
        "client_id":                  persona["client_id"],
        "career_type":                persona["career_type"],
        "age":                        persona["age"],
        "financial_capital":          fc,
        "human_capital_valuation":    hc,
        "total_wealth":               total_wealth,
        "human_capital_pct_of_total": round(hc / total_wealth * 100, 1),
        "income_volatility_sigma":    sigma,
        "income_equity_beta":         beta,
        "income_equity_correlation":  correlation,
        "implicit_equity_exposure":   implicit_equity_exposure,
        "human_capital_type":         hc_type,
        "income_stability":           _STABILITY_TO_CONTRACT[persona["income_stability"]],
        "effective_risk_budget":      effective_risk_budget,
        "portfolio_equity_target":    portfolio_equity_target,
        "industry_exposure_sector":   persona["industry_exposure_sector"],
        "RSU_concentration":          persona["RSU_concentration"],
        "has_pension":                persona["has_pension"],
        "bonus_rate":                 persona["bonus_rate"],
        "current_holdings":           persona["current_holdings"],
        "investment_horizon_years":   persona["investment_horizon_years"],
        "risk_tolerance_level":       persona["risk_tolerance"],
        "liquidity_needs":            persona["liquidity_needs"],
        "investment_objective":       persona["investment_objective"],
    }


def to_profile_agent_output(profile_dict: dict) -> ProfileAgentOutput:
    """Validate a profile dict through contracts.ProfileAgentOutput."""
    return ProfileAgentOutput(**profile_dict)


# ===========================================================================
# 3. BLS persona builder
# ===========================================================================

# ── Target occupations (9 SOC codes) — BLS OES May 2023 ────────────────────
# https://www.bls.gov/oes/2023/may/oes_nat.htm
TARGET_OCCUPATIONS = [
    {"soc": "25-1042", "label": "Biology Professor",   "career_type": "Academia",    "income_stability": "High",   "has_pension": True,  "rsu_eligible": False, "sector": "Education"},
    {"soc": "29-1141", "label": "Registered Nurse",    "career_type": "Healthcare",  "income_stability": "High",   "has_pension": False, "rsu_eligible": False, "sector": "Healthcare"},
    {"soc": "13-1041", "label": "Compliance Officer",  "career_type": "Government",  "income_stability": "High",   "has_pension": True,  "rsu_eligible": False, "sector": "Government"},
    {"soc": "23-1011", "label": "Lawyer",              "career_type": "Legal",       "income_stability": "Medium", "has_pension": False, "rsu_eligible": False, "sector": "Legal"},
    {"soc": "17-2141", "label": "Mechanical Engineer", "career_type": "Engineering", "income_stability": "Medium", "has_pension": False, "rsu_eligible": False, "sector": "Industrials"},
    {"soc": "13-2051", "label": "Financial Analyst",   "career_type": "Finance",     "income_stability": "Medium", "has_pension": False, "rsu_eligible": False, "sector": "Financial Services"},
    {"soc": "15-1252", "label": "Software Developer",  "career_type": "Technology",  "income_stability": "Low",    "has_pension": False, "rsu_eligible": True,  "sector": "Technology"},
    {"soc": "11-3021", "label": "IT Manager",          "career_type": "Technology",  "income_stability": "Low",    "has_pension": False, "rsu_eligible": True,  "sector": "Technology"},
    {"soc": "11-2022", "label": "Sales Manager",       "career_type": "Sales",       "income_stability": "Low",    "has_pension": False, "rsu_eligible": False, "sector": "Consumer Discretionary"},
]

# Representative career-midpoint age per SOC (for HC annuity horizon n = 65 − age).
SOC_AGES = {
    "25-1042": 47,   # Biology Professor
    "29-1141": 38,   # Registered Nurse
    "13-1041": 42,   # Compliance Officer
    "23-1011": 45,   # Lawyer
    "17-2141": 41,   # Mechanical Engineer
    "13-2051": 40,   # Financial Analyst
    "15-1252": 38,   # Software Developer
    "11-3021": 44,   # IT Manager
    "11-2022": 44,   # Sales Manager
}

RSU_BY_PERCENTILE = {"p25": 0.20, "p50": 0.35, "p75": 0.55}
_PCT_COL_MAP      = {"p25": "A_PCT25", "p50": "A_MEDIAN", "p75": "A_PCT75"}

# ── BLS ECEC Q1 2026 — bonus rates (Table 5) ───────────────────────────────
# bonus_rate = supplemental_pay_pct / wages_and_salaries_pct (full-time private)
# Source: https://www.bls.gov/news.release/ecec.t05.htm
BONUS_RATE_TABLE = {
    "25-1042": 0.046,   # Biology Professor   — Education & health (nonunion)
    "29-1141": 0.046,   # Registered Nurse    — Education & health (nonunion)
    "13-1041": 0.060,   # Compliance Officer  — Professional and related
    "23-1011": 0.060,   # Lawyer              — Professional and related
    "17-2141": 0.060,   # Mechanical Engineer — Professional and related
    "13-2051": 0.085,   # Financial Analyst   — Management, business & financial
    "15-1252": 0.085,   # Software Developer  — Management, business & financial
    "11-3021": 0.085,   # IT Manager          — Management, business & financial
    "11-2022": 0.050,   # Sales Manager       — Sales and related
}

# ── SCF 2022 — median family financial assets (Table 6) ────────────────────
# Source: Federal Reserve SCF 2022, Table 6. https://www.federalreserve.gov/publications/files/scf23.pdf
SCF_FINANCIAL_ASSETS = {
    ("25-34", "q2"): 15000,  ("25-34", "q3"): 35000,  ("25-34", "q4"): 90000,
    ("35-44", "q2"): 45000,  ("35-44", "q3"): 90000,  ("35-44", "q4"): 200000,
    ("45-54", "q2"): 75000,  ("45-54", "q3"): 200000, ("45-54", "q4"): 500000,
    ("55-64", "q2"): 100000, ("55-64", "q3"): 300000, ("55-64", "q4"): 750000,
}
SCF_DEFAULT_FINANCIAL_ASSETS = 50000
BLS_TO_SCF_QUARTILE = {"p25": "q2", "p50": "q3", "p75": "q4"}


def get_age_bracket(age: int) -> str:
    if age < 35: return "25-34"
    if age < 45: return "35-44"
    if age < 55: return "45-54"
    return "55-64"


def lookup_financial_capital(age: int, percentile: str) -> float:
    bracket  = get_age_bracket(age)
    quartile = BLS_TO_SCF_QUARTILE[percentile]
    return float(SCF_FINANCIAL_ASSETS.get((bracket, quartile), SCF_DEFAULT_FINANCIAL_ASSETS))


# ── Qualitative-field derivation rules ─────────────────────────────────────

def derive_risk_tolerance(hc_type: str, age: int) -> str:
    if hc_type == "bond-like":
        return "conservative" if age >= 50 else "moderate"
    if hc_type == "mixed":
        return "moderate"
    return "aggressive" if age < 45 else "moderate"


def derive_liquidity_needs(income_stability: str) -> str:
    return "low" if income_stability == "High" else "medium"


def derive_investment_objective(age: int) -> str:
    return "growth" if (65 - age) >= 10 else "income"


def derive_current_holdings(
    hc_type: str, age: int, risk_tolerance: str, rsu_concentration: float
) -> dict[str, float]:
    if rsu_concentration > 0:
        return {
            "employer_RSU": round(rsu_concentration, 2),
            "US_equity":    round((1 - rsu_concentration) * 0.55, 2),
            "bonds":        round((1 - rsu_concentration) * 0.25, 2),
            "cash":         round((1 - rsu_concentration) * 0.20, 2),
        }
    table = {
        ("aggressive", "young"):  {"US_equity": 0.65, "intl_equity": 0.20, "bonds": 0.10, "cash": 0.05},
        ("aggressive", "older"):  {"US_equity": 0.55, "intl_equity": 0.20, "bonds": 0.20, "cash": 0.05},
        ("moderate",   "young"):  {"US_equity": 0.50, "intl_equity": 0.15, "bonds": 0.25, "cash": 0.10},
        ("moderate",   "older"):  {"US_equity": 0.40, "intl_equity": 0.15, "bonds": 0.35, "cash": 0.10},
        ("conservative", "any"):  {"US_equity": 0.25, "intl_equity": 0.10, "bonds": 0.50, "cash": 0.15},
    }
    age_key = "older" if age >= 45 else "young"
    key = (risk_tolerance, "any") if risk_tolerance == "conservative" else (risk_tolerance, age_key)
    return dict(table[key])


def build_bls_personas(
    oes_df: pd.DataFrame, include_percentile_variants: bool = False
) -> list[dict]:
    """
    Build raw persona dicts from BLS OES + SCF + derivation rules.

    Default: one persona per occupation at median (p50) salary.
    include_percentile_variants=True: three per occupation (p25/p50/p75), up to 27.
    Missing SOC codes and suppressed wage cells are skipped with a warning.
    """
    percentiles = ["p25", "p50", "p75"] if include_percentile_variants else ["p50"]
    personas: list[dict] = []

    for occ in TARGET_OCCUPATIONS:
        soc = occ["soc"]
        if soc not in oes_df.index:
            print(f"WARNING: SOC {soc} ({occ['label']}) not found in OES data — skipping")
            continue

        age        = SOC_AGES[soc]
        hc_type    = hc_type_for_stability(occ["income_stability"])
        bonus_rate = BONUS_RATE_TABLE[soc]
        get_age_bracket(age)  # validates age maps to a known bracket

        for pct in percentiles:
            salary = oes_df.loc[soc, _PCT_COL_MAP[pct]]
            if pd.isna(salary):
                print(f"WARNING: {soc} {occ['label']} {pct} wage suppressed — skipping")
                continue

            salary           = float(salary)
            effective_salary = round(salary * (1 + bonus_rate), 2)
            financial_capital = lookup_financial_capital(age, pct)
            rsu_concentration = RSU_BY_PERCENTILE[pct] if occ["rsu_eligible"] else 0.0
            risk_tolerance    = derive_risk_tolerance(hc_type, age)

            personas.append({
                "client_id":                f"bls_{soc}_{pct}",
                "soc":                      soc,
                "label":                    occ["label"],
                "career_type":              occ["career_type"],
                "age":                      age,
                "annual_salary":            salary,
                "bonus_rate":               bonus_rate,
                "effective_salary":         effective_salary,
                "years_to_retirement":      65 - age,
                "income_stability":         occ["income_stability"],
                "industry_exposure_sector": occ["sector"],
                "financial_capital":        float(financial_capital),
                "current_holdings":         derive_current_holdings(hc_type, age, risk_tolerance, rsu_concentration),
                "investment_horizon_years": 65 - age,
                "risk_tolerance":           risk_tolerance,
                "liquidity_needs":          derive_liquidity_needs(occ["income_stability"]),
                "investment_objective":     derive_investment_objective(age),
                "RSU_concentration":        rsu_concentration,
                "has_pension":              occ["has_pension"],
            })

    print(f"Built {len(personas)} personas from {len(TARGET_OCCUPATIONS)} SOC codes")
    return personas
