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

from contracts import (
    ClientStatement,
    ExtractedProfile,
    FactSource,
    LLMRole,
    PipelineDestination,
    ProfileAgentOutput,
)
from agents.profile.profile_model import (
    sigma_for_stability,
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


CONCENTRATION_THRESHOLD = 0.10
"""
Single-name weight above which a holding is flagged as a concentration.

Matches the 10% single-name cap the Allocation Agent already enforces
(SINGLE_NAME_LIMIT), so a position flagged here is one the optimiser would have
to cut anyway. Flagging is not enforcement — the Profile Agent reports, the
optimiser acts.
"""


@dataclass
class RoutedStatements:
    """
    Classified statements sorted by what the Profile Agent can do with them.

    The taxonomy assigns each statement a destination. Without this, that
    assignment is decoration: the statements are classified, carried, and then
    nothing reads them — the exact "a stage computes something correct and the
    next stage never receives it" failure the July 24 review leads with.

    Three of these the Profile Agent acts on itself (concentrations are measured
    against the client's actual holdings; exclusions and scope boundaries are
    recorded on the profile's audit trail). The rest are carried as structured
    output for the agents that own those decisions — `soft_views` is Black-
    Litterman input and belongs to Allocation, `for_advisor_review` belongs to a
    human.
    """

    concentrations:    list[str] = dc_field(default_factory=list)
    sector_underweights: list[str] = dc_field(default_factory=list)
    exclusions:        list[str] = dc_field(default_factory=list)
    out_of_scope:      list[str] = dc_field(default_factory=list)
    soft_views:        list[ClientStatement] = dc_field(default_factory=list)
    for_advisor_review: list[str] = dc_field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([
            self.concentrations, self.sector_underweights, self.exclusions,
            self.out_of_scope, self.soft_views, self.for_advisor_review,
        ])


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
    statements:      list[ClientStatement] = dc_field(default_factory=list)
    routed:          RoutedStatements = dc_field(default_factory=RoutedStatements)

    @property
    def built(self) -> bool:
        return self.profile is not None

    def summary(self) -> str:
        if not self.built:
            return (
                f"{self.client_id}: NOT BUILT — {len(self.open_questions)} required "
                f"field(s) never discussed."
            )
        routed = "" if self.routed.is_empty() else (
            f", {len(self.statements)} statement(s) routed"
        )
        return (
            f"{self.client_id}: built from {len(self.stated_fields)} stated fact(s), "
            f"{len(self.defaulted_fields)} defaulted, {len(self.conflicts)} conflict(s)"
            f"{routed}"
        )


def route_statements(
    statements: list[ClientStatement],
    holdings: dict[str, float] | None,
) -> RoutedStatements:
    """
    Sort classified statements into what the Profile Agent can act on.

    `concentration_limit` statements are checked against the client's actual
    holdings rather than taken at face value. A client saying "that's a lot in
    one name" is a claim; the weight in their book is the evidence. When both
    agree the flag carries a measured number, which is what a suitability file
    needs — and when the holdings do not support the claim, that disagreement is
    worth seeing too.
    """
    routed = RoutedStatements()
    holdings = holdings or {}

    # Several statements commonly name the same position — Priya's ARVX is
    # raised as an employer tie, a concentration and a sector exposure in three
    # separate sentences. Deduplicate on the resolved holding rather than on the
    # classifier's subject string, which varies ("ARVX", "ARVX / biotech
    # sector"), so one position produces one flag.
    seen_positions: set[str] = set()

    for s in statements:
        subject = (s.subject or "").strip()

        if PipelineDestination.CONCENTRATION_LIMIT in s.destinations:
            ticker, weight = _resolve_holding(holdings, subject)
            key = ticker or subject.lower()
            if key not in seen_positions:
                seen_positions.add(key)
                if weight is None:
                    routed.concentrations.append(
                        f"{subject or 'unnamed position'}: raised in conversation, "
                        f"not identifiable in current holdings — confirm with client"
                    )
                elif weight >= CONCENTRATION_THRESHOLD:
                    routed.concentrations.append(
                        f"{ticker}: {weight:.1%} of holdings, above the "
                        f"{CONCENTRATION_THRESHOLD:.0%} single-name limit"
                    )
                else:
                    routed.concentrations.append(
                        f"{ticker}: {weight:.1%} of holdings — below the "
                        f"{CONCENTRATION_THRESHOLD:.0%} limit despite being raised as a concern"
                    )

        if PipelineDestination.SECTOR_UNDERWEIGHT in s.destinations and subject:
            # Prefer a holdings ticker when the subject resolves to one, so
            # "ARVX" and "ARVX / biotech sector" collapse to a single entry.
            ticker, _ = _resolve_holding(holdings, subject)
            entry = ticker or subject
            if entry not in routed.sector_underweights:
                routed.sector_underweights.append(entry)

        if PipelineDestination.UNIVERSE_EXCLUSION in s.destinations:
            routed.exclusions.append(f"{subject or s.summary}: {s.summary}")

        if PipelineDestination.SCOPE_BOUNDARY in s.destinations:
            routed.out_of_scope.append(s.summary)

        if PipelineDestination.BL_VIEW in s.destinations:
            routed.soft_views.append(s)

        if PipelineDestination.ADVISOR_REVIEW in s.destinations:
            routed.for_advisor_review.append(s.summary)

    return routed


def _resolve_holding(
    holdings: dict[str, float], subject: str
) -> tuple[str | None, float | None]:
    """
    Resolve a statement's subject to a holding, returning (ticker, weight).

    The subject is written by a classifier and arrives in several shapes for the
    same position — "ARVX", "ARVX / biotech sector", "her own company stock". An
    exact key lookup catches only the first, so match on whether a holdings key
    appears in the subject or vice versa, and return the canonical ticker so
    callers can deduplicate on it.

    Returns (None, None) when nothing matches — which is itself informative: a
    concentration raised in conversation that does not appear in the book is a
    discrepancy worth surfacing, not a silent no-op.
    """
    if not subject or not holdings:
        return None, None
    needle = subject.upper()
    for ticker, weight in holdings.items():
        t = ticker.upper()
        if t == needle or t in needle or needle in t:
            return ticker, float(weight)
    return None, None


def _normalise_holdings(value) -> dict[str, float] | None:
    """
    Coerce an extracted holdings breakdown into weights summing to 1.0.

    Returns None when nothing usable was extracted, so the caller falls back to
    the neutral book and reports it as defaulted.

    Normalising rather than rejecting an imperfect sum is deliberate: a client
    reading balances off a statement produces figures that sum to 0.98 or 1.03,
    and discarding a real holdings breakdown over rounding would throw away the
    concentration this pipeline exists to find. Anything that is not a positive
    number is dropped; if nothing survives, the caller defaults.
    """
    if not isinstance(value, dict) or not value:
        return None

    clean: dict[str, float] = {}
    for ticker, weight in value.items():
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if w > 0:
            clean[str(ticker)] = w

    total = sum(clean.values())
    if not clean or total <= 0:
        return None
    return {t: round(w / total, 6) for t, w in clean.items()}


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

    holdings, h_src = _resolve(extracted, "current_holdings")
    normalised = _normalise_holdings(holdings)
    if normalised is None:
        persona["current_holdings"] = dict(DEFAULT_HOLDINGS)
        result.defaulted_fields.append("current_holdings")
    else:
        persona["current_holdings"] = normalised
        if h_src == FactSource.STATED:
            result.stated_fields.append("current_holdings")

    result.conflicts = _check_calibration_conflicts(extracted, persona["income_stability"])
    result.stated_fields.sort()
    result.defaulted_fields.sort()
    return persona, result


def _llm_role_for(extractor_name: str) -> LLMRole:
    """
    Which LLMRole the profile should record, given the extractor that produced it.

    Resolved from the extractor class's own `uses_llm` declaration rather than a
    string match on the name, so an extractor added later reports itself correctly
    without anyone remembering to update this function. An unrecognised name is
    reported as CREATOR: an extractor this module has never heard of is more
    likely a new model-backed one than a new deterministic one, and over-reporting
    model involvement is the safe direction to be wrong in for an audit trail.
    """
    from agents.profile import intake

    for obj in vars(intake).values():
        if isinstance(obj, type) and getattr(obj, "name", None) == extractor_name:
            return LLMRole.CREATOR if getattr(obj, "uses_llm", True) else LLMRole.NONE
    return LLMRole.CREATOR


def build_profile_from_intake(
    extracted: ExtractedProfile, discount_rate: float
) -> BridgeResult:
    """
    Full path: conversation → typed profile → validated ProfileAgentOutput.

    Every number in the returned profile is computed by build_profile() from the
    extracted inputs. The extractor supplies facts; the formulas supply figures.
    That split is recorded on the profile itself as `llm_role`, so a reader of the
    output can tell whether a model authored its inputs without knowing which
    extractor was passed in.
    """
    persona, result = to_persona(extracted)

    # Statements are routed whether or not a profile could be built. A
    # conversation that is missing a required field still carries exclusions,
    # scope boundaries and things needing a human — discarding those because a
    # salary was never mentioned would throw away the part that does not depend
    # on it.
    result.statements = list(extracted.statements)
    result.routed = route_statements(
        extracted.statements,
        persona.get("current_holdings") if persona else None,
    )

    if persona is None:
        return result

    profile_dict = build_profile(
        persona,
        discount_rate,
        overrides = _sigma_override(extracted, result),
        llm_role  = _llm_role_for(extracted.extractor),
    )
    result.profile = to_profile_agent_output(profile_dict)

    # Carry the classified statements onto the profile itself, not just onto this
    # BridgeResult. Without this line the mandate stops here: the orchestrator
    # builds ComplianceInput from ProfileAgentOutput (never from BridgeResult), so
    # Compliance Job 3.2 receives an empty client_statements list and reports every
    # exclusion check as passing — the vacuous-pass failure mode, which looks
    # identical to a genuine pass in the report.
    #
    # Assigned rather than passed through build_profile() because statements are
    # not an input to any formula: build_profile computes numbers, and a mandate
    # ("no weapons") has no place in that signature. None of
    # ProfileAgentOutput's four model_validators reference client_statements, so
    # assigning post-construction cannot invalidate the object.
    result.profile.client_statements = list(extracted.statements)
    return result


SIGMA_BOUNDS = (0.01, 0.60)
"""
Admissible range for a transcript-derived income volatility.

A value outside this is not a measurement, it is a misread — 0 means no income
risk at all and >0.6 exceeds anything the calibration table contemplates. Out of
range falls back to the tier.
"""


def _sigma_override(extracted: ExtractedProfile, result: BridgeResult) -> dict | None:
    """
    Use a transcript-derived income volatility in place of the tier lookup.

    `income_stability` maps to three values — 0.05, 0.20, 0.40. That is a coarse
    instrument for the quantity the sensitivity analysis identifies as the most
    consequential input in the whole pipeline: a ±20% error in sigma moves the
    mixed persona's equity share 13.5 points, more than any other field.

    A transcript can carry more than three levels. "A target bonus around 25%
    but all over the place year to year — some years basically zero, other years
    well above target" describes a distribution, and collapsing it to `Low` and
    then to 0.40 discards the width the mentor calls "the single most important
    input to her human-capital valuation".

    So when the client actually describes year-to-year variation, that estimate
    is used and the substitution is recorded. When they do not, the tier stands —
    the extractor is instructed to return null rather than guess, because a
    fabricated sigma is worse than a coarse one.
    """
    value, source = _resolve(extracted, "income_volatility_estimate")
    if value is None:
        return None

    try:
        sigma = float(value)
    except (TypeError, ValueError):
        return None

    low, high = SIGMA_BOUNDS
    if not (low <= sigma <= high):
        result.conflicts.append(
            f"income_volatility_estimate {sigma:.3f} is outside the admissible "
            f"range [{low}, {high}] — ignored, tier value used instead."
        )
        return None

    tier = sigma_for_stability(
        {"high": "High", "medium": "Medium", "low": "Low"}.get(
            str(extracted.value_of("income_stability")).lower(),
            str(extracted.value_of("income_stability")),
        )
    ) if extracted.value_of("income_stability") else None

    if tier is not None and abs(tier - sigma) > 0.02:
        result.conflicts.append(
            f"income_volatility_sigma: transcript supports {sigma:.3f} "
            f"({source.value if source else 'unknown'}); the "
            f"'{extracted.value_of('income_stability')}' tier would give {tier:.2f}. "
            f"Transcript estimate used — it is a description of this client's own "
            f"pay, not a population bucket."
        )
    return {"income_volatility_sigma": sigma}


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
