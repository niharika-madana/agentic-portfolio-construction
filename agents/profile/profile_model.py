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
#
# ── Known calibration inconsistency (14 Jul 2026 review) ───────────────────
# σ (INCOME_VOLATILITY_SIGMA), β, and ρ are each calibrated independently from
# a different literature source, for a different downstream purpose:
#   σ → effective_risk_budget;  β → implicit_equity_exposure;  ρ → Risk Agent's
#   HC-adjusted sector limits.
# They are NOT jointly estimated. So the single-factor identity
#   β = ρ × σ_income / σ_market   ⇒   σ_market = ρ × σ_income / β
# does not resolve to one common market volatility across the three tiers:
#   bond-like:   0.10 × 0.05 / 0.05 = 10.0%
#   mixed:       0.40 × 0.20 / 0.35 ≈ 22.9%
#   equity-like: 0.75 × 0.40 / 0.90 ≈ 33.3%
# A single-factor model would require all three to equal one σ_market (~16-20%
# for US equities). The spread is the honest reading: this table is a pragmatic
# calibration, not a strict econometric model.
# Surfaced per-profile as implied_market_volatility (see below) so the
# discrepancy is visible and auditable rather than hidden. Left uncorrected on
# purpose — retuning β or ρ to force consistency would shift
# implicit_equity_exposure and therefore portfolio_equity_target for every
# persona, changing all downstream allocations. That is a team decision, not a
# side effect of adding the diagnostic.
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


def implied_market_volatility(
    sigma: float, beta: float, correlation: float
) -> float | None:
    """
    σ_market implied by the single-factor identity β = ρ × σ_income / σ_market.

    Diagnostic only — no downstream calculation consumes it. Because σ, β and ρ
    are calibrated separately (see the note above HC_BETA_TABLE), this returns a
    different value per HC type rather than one common market volatility.
    Returns None when β ≤ 0, where the identity is undefined.
    """
    if beta <= 0:
        return None
    return round(correlation * sigma / beta, 4)


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
    sigma_market             = implied_market_volatility(sigma, beta, correlation)
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
        "implied_market_volatility":  sigma_market,
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

# ── Income-stability categorization — documented basis ─────────────────────
# Addresses review note: "income_stability is hardcoded, provide proof of
# categorization." Each occupation's High/Medium/Low label is NOT a bare
# hand-assignment — it is derived from three observable, citable labor-economics
# signals, and corroborated by the pipeline's own comp-structure fields
# (bonus_rate, rsu_eligible, has_pension):
#
#   (1) Variable-compensation share — fraction of total pay that is not fixed
#       base salary. Proxied by BLS ECEC supplemental-pay share (Table 5 — the
#       same source as BONUS_RATE_TABLE below) plus equity comp (rsu_eligible)
#       and commission. More variable pay → income co-moves with markets → lower
#       stability → higher σ and higher income_equity_beta.
#   (2) Cyclical employment risk — occupational unemployment level/cyclicality
#       (BLS CPS). Education, healthcare, and government run structurally low,
#       near-acyclical unemployment; professional/technical roles are moderate;
#       technology and sales are the most layoff-cyclical.
#   (3) Institutional job protection — tenure, occupational licensure,
#       civil-service status, and DB pension coverage (has_pension).
#
# Tier definitions (see INCOME_VOLATILITY_SIGMA / HUMAN_CAPITAL_TYPE above):
#   High   (σ=0.05, bond-like):   salaried, licensed/tenured/civil-service,
#                                 minimal variable pay (ECEC supp. ~3.3–4.1%),
#                                 often pensioned.
#   Medium (σ=0.20, mixed):       stable base + meaningful bonus (ECEC supp.
#                                 ~4.1–5.7%), no equity/commission, moderate
#                                 cyclicality.
#   Low    (σ=0.40, equity-like): large equity- or commission-linked variable
#                                 pay (RSU concentration ≥0.35; sales commission)
#                                 and high layoff cyclicality — income tracks
#                                 equity markets.
#
# Sources:
#   BLS ECEC Q1 2026, Table 5 (supplemental-pay share by occupational group):
#     https://www.bls.gov/news.release/ecec.t05.htm
#   BLS CPS, unemployment by occupation: https://www.bls.gov/cps/tables.htm
#   Davis & Willen (2000), occupational income betas by occupation class (SSRN).
#
# Per-occupation justification (the "proof" for each row of TARGET_OCCUPATIONS):
INCOME_STABILITY_BASIS = {
    "25-1042": {"tier": "High",   "drivers": "Tenure-track academic; fixed salary; DB pension; ECEC supp. pay ~3.3%; education employment near-acyclical."},
    "29-1141": {"tier": "High",   "drivers": "Licensed clinician; salaried base; healthcare unemployment structurally low and counter-cyclical; ECEC supp. pay ~3.3%."},
    "13-1041": {"tier": "High",   "drivers": "Civil-service/regulated role; DB pension; government employment acyclical; no equity or commission pay."},
    "23-1011": {"tier": "Medium", "drivers": "Salaried base plus bonus (ECEC prof. supp. ~4.1%); licensed but demand cyclical (firm layoffs); no equity comp."},
    "17-2141": {"tier": "Medium", "drivers": "Salaried plus modest bonus (ECEC prof. supp. ~4.1%); hiring tracks the industrial cycle; no equity comp."},
    "13-2051": {"tier": "Medium", "drivers": "Base plus sizable bonus (ECEC mgmt/finance supp. ~5.7%) that co-moves with markets; no equity grant at analyst level."},
    "15-1252": {"tier": "Low",    "drivers": "RSU-eligible (equity comp, RSU concentration ~0.35); tech employment layoff-cyclical; income tracks company/market equity."},
    "11-3021": {"tier": "Low",    "drivers": "RSU-eligible management equity comp; technology-sector cyclicality; income tracks equity markets."},
    "11-2022": {"tier": "Low",    "drivers": "Commission-linked pay tied to revenue/market conditions; high earnings variance; consumer-discretionary cyclicality."},
}

# ── Target occupations (9 SOC codes) — BLS OES May 2023 ────────────────────
# https://www.bls.gov/oes/2023/may/oes_nat.htm
# income_stability per row is justified in INCOME_STABILITY_BASIS above; the two
# are kept in sync by the integrity check immediately following this table.
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

# Proof-of-categorization integrity check: every occupation's income_stability
# label must match its documented basis in INCOME_STABILITY_BASIS, so the label
# and its justification cannot silently drift apart.
for _occ in TARGET_OCCUPATIONS:
    _basis = INCOME_STABILITY_BASIS.get(_occ["soc"])
    assert _basis is not None, (
        f"No income_stability basis documented for SOC {_occ['soc']} "
        f"({_occ['label']}). Add it to INCOME_STABILITY_BASIS."
    )
    assert _basis["tier"] == _occ["income_stability"], (
        f"income_stability mismatch for SOC {_occ['soc']} ({_occ['label']}): "
        f"TARGET_OCCUPATIONS says {_occ['income_stability']!r}, "
        f"INCOME_STABILITY_BASIS says {_basis['tier']!r}."
    )

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
