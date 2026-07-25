"""
intake_bridge.py — ExtractedProfile → ProfileAgentOutput.

Without this module the intake layer is a dead end: intake.py produces a typed,
sourced ExtractedProfile and the only thing that reads it is the grader. A
transcript could be measured but could not drive the pipeline — the exact
failure the July 24 review names as the one thing to take away ("a stage
computes something correct and the next stage never receives it").

This closes that. A discovery call becomes a persona dict, which goes through
the same `build_profile()` the BLS personas use, so every number downstream is
still produced by the deterministic formulas. The language model never computes
anything here; it only supplies inputs, and only the inputs it can evidence.

Which source wins
-----------------
The mentor's oral question is "if the client's own words disagree with what your
data says about them, which wins, and why?". The answer this module implements,
and the reasoning behind it:

  Facts about the client's situation — age, salary, balances, employer, holdings.
      The CLIENT WINS. They are the authority on their own circumstances; no
      table knows their account balance. A stated value overrides any default.

  Model parameters — income beta, correlation, income volatility sigma.
      THE MEASURED CALIBRATION WINS. These are estimated from BLS/CRSP data in
      profile_model.py, not observed. When a transcript carries a beta it is
      almost never the client's measurement: in the reference intake the ADVISOR
      proposed 1.2 and the client merely assented, hedging that it might be low.
      Soft assent to someone else's estimate is not evidence, and letting it
      overwrite a calibrated parameter would launder an opinion into a number
      the optimiser treats as fact.

      The disagreement is not discarded — it is recorded in `conflicts` so a
      reviewer can see that the client was told 0.8 while the model uses 0.51.

  Fields the client never discussed.
      NEITHER WINS — they are defaulted, and every default is listed in
      `defaulted_fields`. A suitability file has to distinguish a constraint the
      client asserted from one the system chose for them.

  Fields the client never discussed that cannot be defaulted safely.
      Nothing is built. REQUIRED_FIELDS below cannot be guessed without
      inventing the client, so the bridge refuses and returns the follow-up
      questions instead. This is the refusal-to-guess principle from the
      extractor carried through to profile construction — it would be strange
      to have the extractor honestly decline and then have the bridge fabricate
      the same value one layer down.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from contracts import ExtractedProfile, FactSource, ProfileAgentOutput
from agents.profile.profile_model import (
    HC_BETA_TABLE,
    build_profile,
    hc_type_for_stability,
    to_profile_agent_output,
)

RETIREMENT_AGE = 65
"""
Assumed retirement age when the client does not state a horizon. Used only to
derive years_to_retirement; a stated investment_horizon_years always wins.
"""

REQUIRED_FIELDS = ("age", "annual_salary", "financial_capital", "income_stability")
"""
Fields with no defensible default. Age and salary drive the human-capital
annuity, financial capital sets the wealth split, and income stability selects
the entire beta/sigma calibration row — a wrong guess at any of them changes
every downstream number. If one is missing the bridge declines to build.
"""

FIELD_DEFAULTS = {
    "bonus_rate":              0.0,
    "RSU_concentration":       0.0,
    "has_pension":             False,
    "industry_exposure_sector": "Unknown",
    "risk_tolerance":          "moderate",
    "liquidity_needs":         "medium",
    "investment_objective":    "growth",
}
"""
Population defaults for fields a client may simply not mention. Each one used is
reported in `defaulted_fields` — the point is not to hide the gap but to make it
enumerable.
"""

DEFAULT_HOLDINGS = {"US_equity": 0.6, "intl_equity": 0.2, "bonds": 0.15, "cash": 0.05}
"""
Neutral starting book when current holdings were not discussed. Sums to 1.0 as
ProfileAgentOutput requires. Always reported as defaulted — a real engagement
would pull this from statements, not assume it.
"""


@dataclass
class BridgeResult:
    """
    Outcome of converting one conversation into a profile.

    `profile` is None when required fields were missing; `open_questions` then
    carries what to ask before trying again.
    """

    client_id:       str
    profile:         ProfileAgentOutput | None
    stated_fields:   list[str] = dc_field(default_factory=list)
    defaulted_fields: list[str] = dc_field(default_factory=list)
    open_questions:  list[str] = dc_field(default_factory=list)
    conflicts:       list[str] = dc_field(default_factory=list)

    @property
    def built(self) -> bool:
        return self.profile is not None

    def summary(self) -> str:
        if not self.built:
            return (
                f"{self.client_id}: NOT BUILT — {len(self.open_questions)} required "
                f"field(s) never discussed."
            )
        return (
            f"{self.client_id}: built from {len(self.stated_fields)} stated fact(s), "
            f"{len(self.defaulted_fields)} defaulted, {len(self.conflicts)} conflict(s)"
        )


def _resolve(extracted: ExtractedProfile, name: str):
    """Return (value, source) for a field, or (None, None) when unusable."""
    f = extracted.fields.get(name)
    if f is None or f.source == FactSource.UNKNOWN or f.value is None:
        return None, None
    return f.value, f.source


def _check_calibration_conflicts(
    extracted: ExtractedProfile, income_stability: str
) -> list[str]:
    """
    Compare any transcript-supplied beta against the calibrated one.

    Records the disagreement without acting on it — see the module docstring on
    why a soft-assented advisor estimate does not override a measured parameter.
    """
    conflicts: list[str] = []
    stated_beta, source = _resolve(extracted, "advisor_supplied_beta")
    if stated_beta is None:
        return conflicts

    calibrated = HC_BETA_TABLE[hc_type_for_stability(income_stability)]["beta"]
    try:
        stated_beta = float(stated_beta)
    except (TypeError, ValueError):
        return conflicts

    if abs(stated_beta - calibrated) > 0.05:
        origin = (
            "client asserted" if source == FactSource.STATED
            else "advisor-supplied, client assented"
        )
        conflicts.append(
            f"income_equity_beta: transcript says {stated_beta:.2f} ({origin}); "
            f"calibration derives {calibrated:.3f} from income_stability="
            f"'{income_stability}'. Calibration used — the transcript figure is an "
            f"estimate, not a measurement. Flag for advisor review."
        )
    return conflicts


def to_persona(extracted: ExtractedProfile) -> tuple[dict | None, BridgeResult]:
    """
    Convert an ExtractedProfile into a raw persona dict for build_profile().

    Returns (persona, result). `persona` is None when a required field is
    missing; `result.open_questions` then holds the extractor's follow-ups.
    """
    result = BridgeResult(client_id=extracted.client_id, profile=None)

    missing = []
    for name in REQUIRED_FIELDS:
        value, _ = _resolve(extracted, name)
        if value is None:
            missing.append(name)
            f = extracted.fields.get(name)
            if f is not None and f.follow_up_question:
                result.open_questions.append(f.follow_up_question)
            else:
                result.open_questions.append(f"Need a value for {name}.")

    if missing:
        return None, result

    persona: dict = {"client_id": extracted.client_id}

    for name in ("age", "annual_salary", "financial_capital", "income_stability"):
        value, source = _resolve(extracted, name)
        persona[name] = value
        if source == FactSource.STATED:
            result.stated_fields.append(name)

    for name, default in FIELD_DEFAULTS.items():
        value, source = _resolve(extracted, name)
        if value is None:
            persona[name] = default
            result.defaulted_fields.append(name)
        else:
            persona[name] = value
            if source == FactSource.STATED:
                result.stated_fields.append(name)

    persona["age"] = int(persona["age"])
    persona["annual_salary"] = float(persona["annual_salary"])
    persona["financial_capital"] = float(persona["financial_capital"])
    persona["bonus_rate"] = float(persona["bonus_rate"])
    persona["RSU_concentration"] = float(persona["RSU_concentration"])
    persona["has_pension"] = bool(persona["has_pension"])

    # Derived, not extracted — the arithmetic stays in code, per the standing rule.
    persona["effective_salary"] = round(
        persona["annual_salary"] * (1 + persona["bonus_rate"]), 2
    )
    persona["years_to_retirement"] = max(1, RETIREMENT_AGE - persona["age"])

    horizon, h_source = _resolve(extracted, "investment_horizon_years")
    if horizon is None:
        persona["investment_horizon_years"] = persona["years_to_retirement"]
        result.defaulted_fields.append("investment_horizon_years")
    else:
        persona["investment_horizon_years"] = int(horizon)
        if h_source == FactSource.STATED:
            result.stated_fields.append("investment_horizon_years")

    persona["career_type"] = persona["industry_exposure_sector"]
    persona["current_holdings"] = dict(DEFAULT_HOLDINGS)
    result.defaulted_fields.append("current_holdings")

    result.conflicts = _check_calibration_conflicts(extracted, persona["income_stability"])
    result.stated_fields.sort()
    result.defaulted_fields.sort()
    return persona, result


def build_profile_from_intake(
    extracted: ExtractedProfile, discount_rate: float
) -> BridgeResult:
    """
    Full path: conversation → typed profile → validated ProfileAgentOutput.

    Every number in the returned profile is computed by build_profile() from the
    extracted inputs. The extractor supplies facts; the formulas supply figures.
    """
    persona, result = to_persona(extracted)
    if persona is None:
        return result

    profile_dict = build_profile(persona, discount_rate)
    result.profile = to_profile_agent_output(profile_dict)
    return result


def run(save: bool = False) -> list[BridgeResult]:
    """
    Demonstrate the end-to-end path on the synthetic corpus.

    Uses the offline rule-based extractor so this runs without an API key; pass
    a different extractor to compare how extraction quality propagates into the
    profile.
    """
    from agents.profile.intake import RuleBasedExtractor
    from agents.profile.profile_agent import _get_discount_rate, _load_bls_oes
    from agents.profile.profile_model import build_bls_personas
    from agents.profile.transcript_generator import generate_corpus

    discount_rate = _get_discount_rate()
    personas = build_bls_personas(_load_bls_oes())
    cases = generate_corpus(personas, seeds=(0,))
    extractor = RuleBasedExtractor()

    results: list[BridgeResult] = []
    for case in cases:
        extracted = extractor.extract(case.text, case.client_id, case.transcript_id)
        results.append(build_profile_from_intake(extracted, discount_rate))

    built = [r for r in results if r.built]
    print(f"\n=== Transcript → ProfileAgentOutput ===")
    print(f"{len(built)} of {len(results)} conversations produced a validated profile\n")

    for r in results:
        print(" ", r.summary())
        for c in r.conflicts:
            print(f"      CONFLICT: {c}")
        for q in r.open_questions:
            print(f"      ASK: {q}")

    if built:
        print("\nAgainst the BLS-built profiles for the same personas:")
        by_id = {p["client_id"]: p for p in personas}
        print(f"  {'client':22s} {'source':>10s} {'HC':>14s} {'beta':>7s} {'eq_target':>10s}")
        for r in built[:4]:
            p = r.profile
            bls = build_profile(by_id[p.client_id], discount_rate)
            print(f"  {p.client_id:22s} {'transcript':>10s} "
                  f"{p.human_capital_valuation:>14,.0f} {p.income_equity_beta:>7.3f} "
                  f"{(p.portfolio_equity_target or 0):>10.3f}")
            print(f"  {'':22s} {'BLS':>10s} {bls['human_capital_valuation']:>14,.0f} "
                  f"{bls['income_equity_beta']:>7.3f} {bls['portfolio_equity_target']:>10.3f}")

    return results


if __name__ == "__main__":  # pragma: no cover
    run()
