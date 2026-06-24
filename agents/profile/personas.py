"""
personas.py — BLS-grounded persona construction.

TARGET_OCCUPATIONS defines the nine-occupation default universe. build_bls_personas()
joins BLS OES salary percentiles with SCF financial capital and the documented
qualitative-field derivation rules to produce raw persona dicts, one (or three,
with percentile variants) per occupation.

Every qualitative field is derived from a documented rule (see the design doc's
"Qualitative Field Derivation Rules" section) — none are invented.
"""

from __future__ import annotations

import pandas as pd

from agents.profile.hc_beta_table import hc_type_for_stability
from agents.profile.loaders import (
    BONUS_RATE_TABLE,
    get_age_bracket,
    lookup_financial_capital,
)

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

# ── Representative career-midpoint age per SOC ─────────────────────────────
# Used for the HC annuity horizon n = 65 − age and the SCF lookup.
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

# ── RSU concentration by BLS percentile (RSU-eligible SOCs only) ──────────
RSU_BY_PERCENTILE = {
    "p25": 0.20,
    "p50": 0.35,
    "p75": 0.55,
}

# BLS percentile → OES wage column.
_PCT_COL_MAP = {"p25": "A_PCT25", "p50": "A_MEDIAN", "p75": "A_PCT75"}


# ---------------------------------------------------------------------------
# Qualitative-field derivation rules
# ---------------------------------------------------------------------------

def derive_risk_tolerance(hc_type: str, age: int) -> str:
    """Risk tolerance from HC type and age (design doc table)."""
    if hc_type == "bond-like":
        return "conservative" if age >= 50 else "moderate"
    if hc_type == "mixed":
        return "moderate"
    # equity-like
    return "aggressive" if age < 45 else "moderate"


def derive_liquidity_needs(income_stability: str) -> str:
    """Stable salary → low; variable income → medium."""
    return "low" if income_stability == "High" else "medium"


def derive_investment_objective(age: int) -> str:
    """≥10 years to retirement → growth, else income."""
    return "growth" if (65 - age) >= 10 else "income"


def derive_current_holdings(
    hc_type: str, age: int, risk_tolerance: str, rsu_concentration: float
) -> dict[str, float]:
    """
    Derive a current-holdings dict that sums to exactly 1.0.

    RSU holders: employer_RSU + a 55/25/20 split of the remainder across
    US_equity / bonds / cash. Non-RSU holders: SCF Table 7 holdings by risk
    tolerance and age.
    """
    if rsu_concentration > 0:
        return {
            "employer_RSU": round(rsu_concentration, 2),
            "US_equity":    round((1 - rsu_concentration) * 0.55, 2),
            "bonds":        round((1 - rsu_concentration) * 0.25, 2),
            "cash":         round((1 - rsu_concentration) * 0.20, 2),
        }
    table = {
        ("aggressive", "young"):   {"US_equity": 0.65, "intl_equity": 0.20, "bonds": 0.10, "cash": 0.05},
        ("aggressive", "older"):   {"US_equity": 0.55, "intl_equity": 0.20, "bonds": 0.20, "cash": 0.05},
        ("moderate",   "young"):   {"US_equity": 0.50, "intl_equity": 0.15, "bonds": 0.25, "cash": 0.10},
        ("moderate",   "older"):   {"US_equity": 0.40, "intl_equity": 0.15, "bonds": 0.35, "cash": 0.10},
        ("conservative", "any"):   {"US_equity": 0.25, "intl_equity": 0.10, "bonds": 0.50, "cash": 0.15},
    }
    age_key = "older" if age >= 45 else "young"
    key = (risk_tolerance, "any") if risk_tolerance == "conservative" else (risk_tolerance, age_key)
    return dict(table[key])


# ---------------------------------------------------------------------------
# Persona builder
# ---------------------------------------------------------------------------

def build_bls_personas(
    oes_df: pd.DataFrame, include_percentile_variants: bool = False
) -> list[dict]:
    """
    Build raw persona dicts from BLS OES + SCF + derivation rules.

    Default: one persona per occupation at the median (p50) salary.
    include_percentile_variants=True: three per occupation (p25, p50, p75),
    up to 27 personas.

    Missing SOC codes and suppressed wage cells are skipped with a warning —
    the pipeline never crashes on a single bad row.
    """
    percentiles = ["p25", "p50", "p75"] if include_percentile_variants else ["p50"]

    personas: list[dict] = []
    for occ in TARGET_OCCUPATIONS:
        soc = occ["soc"]
        row = oes_df[oes_df["OCC_CODE"] == soc]
        if row.empty:
            print(f"WARNING: SOC {soc} ({occ['label']}) not found in OES data — skipping")
            continue

        age = SOC_AGES[soc]
        hc_type = hc_type_for_stability(occ["income_stability"])
        bonus_rate = BONUS_RATE_TABLE[soc]
        _ = get_age_bracket(age)  # validates the age maps to a known bracket

        for pct in percentiles:
            salary = row[_PCT_COL_MAP[pct]].values[0]
            if pd.isna(salary):
                print(f"WARNING: {soc} {occ['label']} {pct} wage suppressed — skipping")
                continue

            salary = float(salary)
            effective_salary = round(salary * (1 + bonus_rate), 2)
            financial_capital = lookup_financial_capital(age, pct)
            rsu_concentration = RSU_BY_PERCENTILE[pct] if occ["rsu_eligible"] else 0.0
            risk_tolerance = derive_risk_tolerance(hc_type, age)

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
                "current_holdings":         derive_current_holdings(
                    hc_type, age, risk_tolerance, rsu_concentration
                ),
                "investment_horizon_years": 65 - age,
                "risk_tolerance":           risk_tolerance,
                "liquidity_needs":          derive_liquidity_needs(occ["income_stability"]),
                "investment_objective":     derive_investment_objective(age),
                "RSU_concentration":        rsu_concentration,
                "has_pension":              occ["has_pension"],
            })

    print(f"Built {len(personas)} personas from {len(TARGET_OCCUPATIONS)} SOC codes")
    return personas
