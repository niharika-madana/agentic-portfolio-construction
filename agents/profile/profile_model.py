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

from contracts import (
    GICS_SECTORS,
    NON_INVESTABLE_EMPLOYER_SECTORS,
    LLMRole,
    ProfileAgentOutput,
)


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

# Correlation of income shocks with equity returns, ρ, per HC type.
# Calibrated from the literature; the primitive input, not derived.
HC_CORRELATION = {
    "bond-like":   0.10,
    "mixed":       0.40,
    "equity-like": 0.75,
}

# ── Market volatility, measured (resolves 24 Jul review §7) ────────────────
# Previously σ, β and ρ were each calibrated independently, so the single-factor
# identity β = ρ × σ_income / σ_market implied three different market
# volatilities (10.0%, 22.9%, 33.3%) — one model claiming three markets. The
# review's instruction was to pick one and derive the betas from it.
#
# σ_market is now measured, not asserted: the annualised standard deviation of
# the Fama-French excess market return (mktrf), 312 monthly observations,
# 2000-01 to 2025-12, from data/storage/ff_risk_factors.parquet. Same factor and
# same file the ticker betas in agents/research/ticker_betas.py regress against,
# so human-capital beta and asset beta are now denominated in one market.
#
# Held as a constant rather than read at import so profile construction stays
# deterministic and file-free; measure_market_volatility() recomputes it from
# the parquet to verify or refresh.
SIGMA_MARKET = 0.1571

# β is DERIVED, never hand-set: β = ρ × σ_income / σ_market. The identity now
# holds by construction and cannot drift out of agreement again.
#
#   bond-like:    0.10 × 0.05 / 0.1571 = 0.032
#   mixed:        0.40 × 0.20 / 0.1571 = 0.509
#   equity-like:  0.75 × 0.40 / 0.1571 = 1.910
#
# The equity-like row lands at 1.91, near the contract's 2.0 ceiling. That is a
# direct consequence of σ_income = 0.40 for RSU/commission income: a career with
# 40% earnings volatility and 0.75 correlation to equities genuinely carries
# close to 2x market exposure. The number is now a consequence of two calibrated
# inputs rather than a third independent guess, which is the point of the fix.
HC_BETA_TABLE = {
    hc_type: {
        "beta":        round(rho * INCOME_VOLATILITY_SIGMA[stability] / SIGMA_MARKET, 4),
        "correlation": rho,
    }
    for stability, hc_type in HUMAN_CAPITAL_TYPE.items()
    for rho in (HC_CORRELATION[hc_type],)
}


def measure_market_volatility(
    factor_parquet: str | None = None, factor: str = "mktrf"
) -> float:
    """
    Recompute σ_market from the Fama-French factor cache.

    Returns the annualised standard deviation of the monthly excess market
    return. Use to verify or refresh SIGMA_MARKET; not called at import, so a
    missing cache never breaks profile construction.
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd

    path = Path(factor_parquet) if factor_parquet else (
        Path(__file__).resolve().parent.parent.parent
        / "data" / "storage" / "ff_risk_factors.parquet"
    )
    series = pd.read_parquet(path)[factor].dropna()
    return round(float(series.std(ddof=1) * np.sqrt(12)), 4)


def hc_type_for_stability(income_stability: str) -> str:
    if income_stability not in HUMAN_CAPITAL_TYPE:
        raise KeyError(f"Unknown income_stability '{income_stability}'. Expected one of {list(HUMAN_CAPITAL_TYPE)}.")
    return HUMAN_CAPITAL_TYPE[income_stability]


def sigma_for_stability(income_stability: str) -> float:
    if income_stability not in INCOME_VOLATILITY_SIGMA:
        raise KeyError(f"Unknown income_stability '{income_stability}'. Expected one of {list(INCOME_VOLATILITY_SIGMA)}.")
    return INCOME_VOLATILITY_SIGMA[income_stability]


def hc_type_for_beta(beta: float) -> str:
    """
    Human-capital label implied by beta, matching the thresholds enforced by
    ProfileAgentOutput._check_hc_type_consistent_with_beta (contracts.py).

    Normal construction takes the label from income_stability; this is used when
    a perturbed beta must carry its label with it (see build_profile overrides).
    """
    if beta <= 0.3:
        return "bond-like"
    if beta <= 0.8:
        return "mixed"
    return "equity-like"


def lookup_hc_beta(hc_type: str) -> dict[str, float]:
    if hc_type not in HC_BETA_TABLE:
        raise KeyError(f"Unknown human_capital_type '{hc_type}'. Expected one of {list(HC_BETA_TABLE)}.")
    return dict(HC_BETA_TABLE[hc_type])


def implied_market_volatility(
    sigma: float, beta: float, correlation: float
) -> float | None:
    """
    σ_market implied by inverting the single-factor identity β = ρ × σ_income / β.

    INTERNAL CONSISTENCY CHECK — no downstream agent consumes this. Since β is
    now derived from ρ, σ_income and SIGMA_MARKET, inverting the identity must
    return SIGMA_MARKET for every HC type (± rounding). A value that disagrees
    means the table has been hand-edited back out of consistency, which is
    exactly the drift the 24 Jul review flagged.

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


def build_profile(
    persona: dict,
    discount_rate: float,
    overrides: dict[str, float] | None = None,
    llm_role: LLMRole = LLMRole.NONE,
) -> dict:
    """
    Derive all computed fields from a raw BLS persona dict.
    Returns a flat dict ready for Pydantic validation via to_profile_agent_output().

    Parameters
    ----------
    overrides : dict[str, float] | None
        Optional continuous overrides for the table-driven inputs, keyed
        "income_volatility_sigma", "income_equity_correlation" and/or
        "income_equity_beta". Normal profile construction never passes this.

        It exists so agents/profile/sensitivity.py can perturb an input by a
        few percent and re-derive everything downstream through the real
        formulas rather than a reimplementation of them. The table lookups are
        categorical (High/Medium/Low), so without this hook the smallest
        possible perturbation of sigma is a whole tier — far too coarse to
        measure how input error propagates into portfolio weights.

        When beta is not overridden it stays derived from the (possibly
        overridden) sigma and correlation, so the single-factor identity
        beta = rho * sigma_income / SIGMA_MARKET continues to hold.
    llm_role : LLMRole
        Who supplied the inputs in `persona`. Defaults to NONE, which is correct
        for the BLS path — no model runs there. The transcript path passes
        CREATOR when an LLM extractor produced the facts; see
        intake_bridge.build_profile_from_intake(). It never changes a computed
        number, only what the profile records about its own provenance.
    """
    overrides = overrides or {}

    hc = compute_human_capital(
        persona["effective_salary"],
        persona["years_to_retirement"],
        discount_rate,
    )
    fc = persona["financial_capital"]
    total_wealth = round(fc + hc, 2)

    hc_type = hc_type_for_stability(persona["income_stability"])
    cal     = lookup_hc_beta(hc_type)

    sigma       = overrides.get("income_volatility_sigma",
                                sigma_for_stability(persona["income_stability"]))
    correlation = overrides.get("income_equity_correlation", cal["correlation"])

    if "income_equity_beta" in overrides:
        beta = overrides["income_equity_beta"]
    elif overrides:
        # Re-derive so the identity holds for the perturbed inputs too.
        beta = round(correlation * sigma / SIGMA_MARKET, 4)
    else:
        beta = cal["beta"]

    if overrides:
        # ProfileAgentOutput validates human_capital_type against beta's
        # thresholds, so a perturbation large enough to cross a tier boundary
        # must move the label with it or the profile fails validation.
        hc_type = hc_type_for_beta(beta)

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
        "llm_role":                   llm_role,
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
#
# `sector` uses the canonical GICS spelling from contracts.GICS_SECTORS, or one of
# contracts.NON_INVESTABLE_EMPLOYER_SECTORS where no listed sector tracks the
# employer. This is load-bearing, not cosmetic: agents/allocation/adapters.py keys
# _SECTOR_PROXY_ETF and agents/shared/core/allocation.py keys the 10% employer
# sector cap on this exact string. Until 4 Aug these rows read "Technology",
# "Healthcare" and "Financial Services", none of which match an ETF sector, so the
# two RSU-heavy tech personas were hedged against SPY instead of XLK and were
# capped at the generic 20% sector limit instead of 10%. Neither failure raised.
TARGET_OCCUPATIONS = [
    {"soc": "25-1042", "label": "Biology Professor",   "career_type": "Academia",    "income_stability": "High",   "has_pension": True,  "rsu_eligible": False, "sector": "Education"},
    {"soc": "29-1141", "label": "Registered Nurse",    "career_type": "Healthcare",  "income_stability": "High",   "has_pension": False, "rsu_eligible": False, "sector": "Health Care"},
    {"soc": "13-1041", "label": "Compliance Officer",  "career_type": "Government",  "income_stability": "High",   "has_pension": True,  "rsu_eligible": False, "sector": "Government"},
    {"soc": "23-1011", "label": "Lawyer",              "career_type": "Legal",       "income_stability": "Medium", "has_pension": False, "rsu_eligible": False, "sector": "Legal"},
    {"soc": "17-2141", "label": "Mechanical Engineer", "career_type": "Engineering", "income_stability": "Medium", "has_pension": False, "rsu_eligible": False, "sector": "Industrials"},
    {"soc": "13-2051", "label": "Financial Analyst",   "career_type": "Finance",     "income_stability": "Medium", "has_pension": False, "rsu_eligible": False, "sector": "Financials"},
    {"soc": "15-1252", "label": "Software Developer",  "career_type": "Technology",  "income_stability": "Low",    "has_pension": False, "rsu_eligible": True,  "sector": "Information Technology"},
    {"soc": "11-3021", "label": "IT Manager",          "career_type": "Technology",  "income_stability": "Low",    "has_pension": False, "rsu_eligible": True,  "sector": "Information Technology"},
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
    # Sector spelling must be one downstream agents can actually match. Checked at
    # import so a hand-edit reintroducing a near-miss ("Technology") fails loudly
    # rather than silently disabling the employer sector cap and proxy ETF.
    assert (
        _occ["sector"] in GICS_SECTORS
        or _occ["sector"] in NON_INVESTABLE_EMPLOYER_SECTORS
    ), (
        f"TARGET_OCCUPATIONS[{_occ['soc']}] sector {_occ['sector']!r} is neither a "
        f"GICS sector nor a recognised non-investable employer sector. Downstream "
        f"sector caps and employer proxy ETFs match this string exactly and fail "
        f"open when it does not."
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
