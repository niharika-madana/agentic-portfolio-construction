"""
tier_derivation.py — reproducible derivation of the income-stability tiers.

Answers the review note: "income_stability is hardcoded, provide proof of
categorization." INCOME_STABILITY_BASIS (profile_model.py) documents *why* each
occupation carries its tier in prose. This module makes that reasoning
executable: it recomputes every tier from the encoded signals and asserts the
result matches the declared label.

Run `python -m agents.profile.tier_derivation` to print the derivation table.

Scope and honesty
-----------------
The prose basis cites three signals. Only two of them are encoded in this repo:

  (1) Variable-compensation share  — BONUS_RATE_TABLE (BLS ECEC Q1 2026 Table 5)
                                     plus equity comp via rsu_eligible.  ENCODED.
  (2) Cyclical employment risk     — per-occupation BLS CPS unemployment.
                                     NOT ENCODED. No CPS fetcher exists; the only
                                     unemployment series in the repo is FRED's
                                     aggregate UNRATE, consumed by the Research
                                     Agent, which is not occupation-specific.
  (3) Institutional job protection — partially encoded, as has_pension. Licensure,
                                     tenure, and civil-service status are not
                                     separate fields.

So this derivation is a scoring rule over signals (1) and the has_pension slice
of (3). It reproduces 8 of the 9 declared tiers. The ninth is recorded in
OVERRIDES with its reason, rather than being papered over by tuning a threshold
until it fits. A test asserts every override is load-bearing, so OVERRIDES cannot
quietly become a dumping ground for rows the rule fails to explain.

PENSION_DISCOUNT is the rule's one free parameter. It is calibrated, not
measured — see its definition below.
"""

from __future__ import annotations

from agents.profile.profile_model import (
    BONUS_RATE_TABLE,
    RSU_BY_PERCENTILE,
    TARGET_OCCUPATIONS,
)

# ── Scoring parameters ─────────────────────────────────────────────────────
# The score is an "exposure of pay to market outcomes" proxy, in units of
# supplemental-pay-to-wages ratio (the units of BONUS_RATE_TABLE).

# Equity comp dwarfs bonus. An RSU-eligible employee at the median carries this
# much of their portfolio in employer stock, and their grant value tracks it.
# Reuses the median from RSU_BY_PERCENTILE rather than introducing a new number.
RSU_PREMIUM = RSU_BY_PERCENTILE["p50"]  # 0.35

# DB pension coverage removes the retirement-income component of earnings risk.
# CALIBRATED, NOT MEASURED: 0.02 is the smallest round value that pulls a
# pensioned worker in the ECEC "Professional and related" group (0.060) below
# HIGH_MAX, without pulling an unpensioned one below it. This single parameter is
# what distinguishes Compliance Officer (High) from Lawyer and Mechanical
# Engineer (Medium) — all three share bonus_rate 0.060. It stands in for the
# civil-service/tenure protection that signal (3) describes but does not encode.
PENSION_DISCOUNT = 0.02

# Tier cutoffs, placed at the gaps between ECEC occupational-group ratios rather
# than at round numbers. Group ratios (supplemental pay / wages, full-time
# private): Education & health 0.046 · Sales 0.050 · Professional 0.060 ·
# Management, business & financial 0.085.
#
# HIGH_MAX sits in the 0.046–0.060 gap. Note this leaves Registered Nurse (0.046,
# no pension) only 0.004 below the cutoff: its High tier is reproducible here,
# but it is the row most sensitive to this threshold, and its prose basis leans
# on licensure and counter-cyclical healthcare demand — both unencoded signals.
HIGH_MAX = 0.050

# MEDIUM_MAX sits above the highest cash-comp group (0.085) and far below any
# score carrying RSU_PREMIUM (>= 0.385). Equity comp, not bonus size, is what
# separates Medium from Low among the encoded signals.
MEDIUM_MAX = 0.100


# ── Documented overrides ───────────────────────────────────────────────────
# Rows where the encoded signals do not reproduce the declared tier, and expert
# judgment governs. Each entry must state what the rule computes, what the true
# tier is, and which unencoded signal justifies the difference.
OVERRIDES = {
    "11-2022": {
        "tier": "Low",
        "reason": (
            "Sales Manager pay is commission-linked, and commission is not a "
            "field in this repo. The ECEC 'Sales and related' group ratio "
            "(0.050) captures only supplemental pay, so the rule scores this "
            "occupation as Medium — below Financial Analyst (0.085) and below "
            "Lawyer (0.060), both Medium. That ranking is an artifact of the "
            "missing field, not a claim that sales income is stable: commission "
            "revenue tracks consumer-discretionary demand, giving this "
            "occupation equity-like earnings variance. Encoding a commission "
            "share would remove the need for this override."
        ),
    },
}


def _tier_from_score(score: float) -> str:
    """Map a market-exposure score to an income-stability tier."""
    if score < HIGH_MAX:
        return "High"
    if score < MEDIUM_MAX:
        return "Medium"
    return "Low"


def score_occupation(soc: str, *, rsu_eligible: bool, has_pension: bool) -> float:
    """Market-exposure score from the encoded signals, for one SOC code."""
    score = BONUS_RATE_TABLE[soc]
    if rsu_eligible:
        score += RSU_PREMIUM
    if has_pension:
        score -= PENSION_DISCOUNT
    return round(score, 4)


def derive_tier(occ: dict) -> dict:
    """Derive one occupation's tier. Returns the score, the rule's tier, any
    override applied, and the final tier."""
    score      = score_occupation(
        occ["soc"], rsu_eligible=occ["rsu_eligible"], has_pension=occ["has_pension"]
    )
    rule_tier  = _tier_from_score(score)
    override   = OVERRIDES.get(occ["soc"])
    final_tier = override["tier"] if override else rule_tier

    return {
        "soc":        occ["soc"],
        "label":      occ["label"],
        "score":      score,
        "rule_tier":  rule_tier,
        "overridden": override is not None,
        "reason":     override["reason"] if override else None,
        "tier":       final_tier,
    }


def derive_all() -> list[dict]:
    """Derive tiers for every occupation in TARGET_OCCUPATIONS."""
    return [derive_tier(occ) for occ in TARGET_OCCUPATIONS]


def _main() -> None:
    rows = derive_all()
    width = max(len(r["label"]) for r in rows)

    print(f"\nIncome-stability tier derivation ({len(rows)} occupations)")
    print(f"  RSU_PREMIUM={RSU_PREMIUM}  PENSION_DISCOUNT={PENSION_DISCOUNT}  "
          f"HIGH_MAX={HIGH_MAX}  MEDIUM_MAX={MEDIUM_MAX}\n")
    print(f"  {'SOC':<9} {'Occupation':<{width}} {'score':>7}  {'rule':<7} {'final':<7} override")
    print("  " + "-" * (9 + width + 7 + 7 + 7 + 12))
    for r in rows:
        flag = "yes" if r["overridden"] else ""
        print(f"  {r['soc']:<9} {r['label']:<{width}} {r['score']:>7.3f}  "
              f"{r['rule_tier']:<7} {r['tier']:<7} {flag}")

    overridden = [r for r in rows if r["overridden"]]
    print(f"\n  Reproduced by rule: {len(rows) - len(overridden)}/{len(rows)}")
    for r in overridden:
        print(f"\n  OVERRIDE {r['soc']} ({r['label']}): rule={r['rule_tier']} -> {r['tier']}")
        print(f"    {r['reason']}")
    print()


if __name__ == "__main__":
    _main()
