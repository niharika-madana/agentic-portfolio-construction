"""
intake.py — transcript → typed profile. The 'extract' half of the standing rule.

The standing rule since Week 1 is that language models extract, classify and
narrate, while pandas and closed-form math compute. The project had built
narrate. This is extract: unstructured client speech in, ExtractedProfile out,
with every value carrying the sentence that produced it. No number here reaches
a portfolio without passing through the deterministic formulas in
profile_model.py afterwards.

Three extractors, deliberately
------------------------------
    RuleBasedExtractor    deterministic regex. No API key, no network. It is the
                          floor: whatever the language models do, they have to
                          beat pattern matching to have earned their cost.
    NaiveExtractor        one call, transcript in, flat profile out. This is the
                          control the mentor asked for directly: "you cannot
                          claim your design earns its complexity without it."
    StructuredExtractor   field-by-field with an explicit unknown option,
                          provenance and calibrated confidence.

All three satisfy the same protocol and are scored by the same harness
(intake_eval.py), which is the only way the comparison means anything.

Model backend
-------------
`call_model` is a thin swappable wrapper, per the 14 Jul minutes: the backend can
be pointed at a cheap OpenRouter endpoint without touching extraction logic. When
no API key is configured the LLM extractors raise rather than silently degrade to
the rule-based path — a benchmark that quietly measures the wrong extractor is
worse than one that refuses to run.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from contracts import (
    ClientStatement,
    ExtractedField,
    ExtractedProfile,
    FactSource,
    PipelineDestination,
    StatementKind,
)

# Fields the intake layer is responsible for recovering.
TARGET_FIELDS = (
    "age",
    "annual_salary",
    "bonus_rate",
    "income_stability",
    "financial_capital",
    "industry_exposure_sector",
    "RSU_concentration",
    "has_pension",
    "investment_horizon_years",
    "liquidity_needs",
    "investment_objective",
    "advisor_supplied_beta",
    "risk_tolerance",
    "current_holdings",
    "income_volatility_estimate",
)

FIELD_VALUE_SPACE = {
    "income_stability":         ["High", "Medium", "Low"],
    "has_pension":              [True, False],
    "liquidity_needs":          ["low", "medium", "high"],
    "investment_objective":     ["growth", "income", "preservation"],
    "risk_tolerance":           ["conservative", "moderate", "aggressive"],
}
"""
Allowed values for the categorical fields.

These must be in the prompt. Without them an extractor returns a semantically
correct description — "stable base salary but unpredictable bonus" — that no
exact-match scorer can accept, and the harness reports a capability gap that is
really a specification gap. The downstream formulas key off the exact labels
(profile_model.sigma_for_stability), so free prose is genuinely unusable here,
not merely awkward to grade.
"""


FIELD_UNITS = {
    "annual_salary":            "base salary in dollars, a number (310000, not '$310k')",
    "bonus_rate":               "FRACTION of base salary, not dollars — 25% is 0.25",
    "financial_capital":        "total investable assets in dollars; sum the holdings if the client lists them separately, and mark it inferred because the sum is yours, not theirs",
    "RSU_concentration":        "FRACTION of total financial holdings held in employer stock, between 0 and 1 — $360,000 of employer stock inside $905,000 of holdings is 0.40, NOT 360000",
    "investment_horizon_years": "whole years until the money is drawn on; if the client gives a retirement age, subtract their current age and mark it inferred",
    "age":                      "whole years",
    "income_volatility_estimate": (
        "annualised standard deviation of TOTAL compensation as a FRACTION of "
        "average total comp, between 0 and 1. Estimate it from what the client "
        "describes, not from a table: a base that never moves is near 0.05; a "
        "base plus a bonus that swings from zero to well above target is 0.30-0.45. "
        "Return null unless the client actually describes year-to-year variation — "
        "this is the single most consequential input in the model and a guess is "
        "worse than an honest unknown."
    ),
    "current_holdings":         (
        "a JSON object mapping asset name or ticker to its FRACTION of total "
        'holdings, summing to 1.0 — e.g. {"ARVX": 0.40, "VTI": 0.28, "cash": 0.10}. '
        "Use the client's own tickers where given; do not invent positions."
    ),
}
"""
Units and shape for the numeric fields.

`RSU_concentration` is the one that bites: it reads as a quantity and the
transcript states it as a dollar figure, so an extractor with no unit guidance
returns 360000 for a field the contract bounds to [0, 1]. The profile then fails
validation at the bridge, one layer away from the actual mistake. Stating the
unit in the prompt is cheaper than catching it downstream.
"""


def _value_space_spec() -> str:
    """Render the allowed-values constraint for inclusion in a prompt."""
    lines = [
        f"  {name}: must be exactly one of {values}"
        for name, values in FIELD_VALUE_SPACE.items()
    ]
    return "\n".join(lines)


def _units_spec() -> str:
    """Render the units constraint for inclusion in a prompt."""
    return "\n".join(f"  {name}: {desc}" for name, desc in FIELD_UNITS.items())


FOLLOW_UP_QUESTIONS = {
    "age":                      "How old are you, and when are you planning to stop working?",
    "annual_salary":            "What does your base compensation look like?",
    "bonus_rate":               "Is there a bonus or variable component on top of base?",
    "income_stability":         "How much does your total pay move year to year?",
    "financial_capital":        "What's the total across your investment accounts today?",
    "industry_exposure_sector": "What sector is your employer in?",
    "RSU_concentration":        "Do you hold any company stock or unvested equity?",
    "has_pension":              "Is there a pension or any other guaranteed income?",
    "investment_horizon_years": "When would you expect to start drawing on this money?",
    "liquidity_needs":          "Do you anticipate needing cash from this in the next few years?",
    "investment_objective":     "Is the priority growth, income, or protecting what's there?",
    "advisor_supplied_beta":    "How sensitive do you think your income is to the stock market?",
    "risk_tolerance":           "How would you describe your appetite for risk?",
    "current_holdings":         "What are you holding today, and roughly how much in each?",
    "income_volatility_estimate": "In a bad year versus a good year, how much does your total pay actually differ?",
}


class Extractor(Protocol):
    """Anything the harness can score."""

    name: str

    uses_llm: bool
    """
    Whether this strategy calls a language model.

    Declared by the extractor rather than inferred from its name, because the
    answer travels: it is what sets ProfileAgentOutput.llm_role, which is the
    field the oral defense points at to show which profiles a model touched. A
    name-matching heuristic in the bridge would quietly get this wrong the first
    time someone adds an extractor.
    """

    def extract(self, transcript: str, client_id: str, transcript_id: str) -> ExtractedProfile:
        ...


# ---------------------------------------------------------------------------
# Model wrapper — swappable backend
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("INTAKE_MODEL", "claude-opus-5")
"""
Extraction model. Override per run with INTAKE_MODEL, which is how the harness
sweeps cost/quality — the extractor is scored the same way whatever backend runs
it, so `INTAKE_MODEL=claude-haiku-4-5 python -m agents.profile.intake_eval`
answers "how cheap can the intake layer get before recall drops?" with a number.
"""


def call_model(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 8000) -> str:
    """
    Send a prompt to the configured backend and return raw text.

    Kept deliberately thin so the backend can be swapped (Anthropic today, a
    cheap OpenRouter endpoint tomorrow) without touching any extraction logic.
    Raises when no key is configured rather than falling back, so a benchmark
    can never silently score a different extractor than the one it names.

    Two response-shape details that are easy to get wrong and cost a whole
    harness run when missed:

    - `content` is a list of typed blocks, and on models where thinking is on by
      default the first block is a ThinkingBlock, not the answer. Reading
      `content[0].text` raises AttributeError on every call. Filter by
      `block.type == "text"` instead of indexing.
    - `max_tokens` caps thinking AND response text together, so a budget sized
      for the JSON alone truncates the answer once thinking is on.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No ANTHROPIC_API_KEY configured. The LLM extractors cannot run. "
            "RuleBasedExtractor runs offline and the harness will score it; set a "
            "key (or repoint call_model at an OpenRouter model) to benchmark the "
            "language-model extractors against it."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    if response.stop_reason == "refusal":
        raise RuntimeError(f"Model declined the extraction request ({model}).")
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Response hit max_tokens ({max_tokens}); the JSON is truncated. "
            "Raise max_tokens — it covers thinking plus output."
        )

    text = "".join(b.text for b in response.content if b.type == "text")
    if not text.strip():
        raise RuntimeError(
            f"No text block in response (blocks: {[b.type for b in response.content]})"
        )
    return text


def _parse_json_block(raw: str) -> dict:
    """Pull the first JSON object out of a model response."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    candidate = fenced.group(1) if fenced else raw
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model response: {raw[:200]}")
    return json.loads(candidate[start : end + 1])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _unknown(name: str) -> ExtractedField:
    """An honest non-answer: no value, zero confidence, and a question to ask."""
    return ExtractedField(
        name=name,
        value=None,
        source=FactSource.UNKNOWN,
        confidence=0.0,
        follow_up_question=FOLLOW_UP_QUESTIONS.get(
            name, f"Could you tell me about {name.replace('_', ' ')}?"
        ),
    )


def _is_turn_formatted(lines: list[str]) -> bool:
    """True when the transcript uses explicit SPEAKER: turns."""
    return any(l.startswith(("CLIENT:", "ADVISOR:")) for l in lines)


def _client_voice(lines: list[str]) -> str:
    """
    The text attributable to the client.

    Turn-formatted transcripts give this exactly. Real intake notes often do
    not: the reference discovery call is the advisor's own prose written after
    the fact, with the client's statements reported rather than quoted. In that
    format there is no line-level attribution to be had, so the whole document
    is searched and attribution has to be decided per fact instead of per line —
    which is why `_advisor_voice` narrows to the sentences that mark an
    advisor-originated judgement.

    Reading only `CLIENT:` lines silently returned nothing on the reference
    transcript: 0 of 12 fields, no error.
    """
    if _is_turn_formatted(lines):
        return "\n".join(l for l in lines if l.startswith("CLIENT:"))
    return "\n".join(lines)


_ADVISOR_JUDGEMENT_CUES = (
    "i'd call", "i would call", "i'd put", "i would put", "i said",
    "when i floated", "i floated", "think of it like", "i'd say",
)


def _advisor_voice(lines: list[str]) -> str:
    """
    Text carrying an advisor-originated judgement rather than a client assertion.

    In turn format that is the ADVISOR lines. In prose notes it is the sentences
    where the advisor marks their own opinion — "I'd put her at a moderate risk
    tolerance", "when I floated that ... a beta of about 1.2". Those numbers
    must not be recorded as the client's own, however readily the client agreed.
    """
    if _is_turn_formatted(lines):
        return "\n".join(l for l in lines if l.startswith("ADVISOR:"))
    return "\n".join(
        l for l in lines if any(cue in l.lower() for cue in _ADVISOR_JUDGEMENT_CUES)
    )


def _find_line(lines: list[str], quote: str) -> int | None:
    """Index of the first line containing `quote`, or None."""
    if not quote:
        return None
    needle = quote.strip()[:40]
    for i, line in enumerate(lines):
        if needle and needle in line:
            return i
    return None


# ---------------------------------------------------------------------------
# 1. Rule-based extractor — the floor, runs offline
# ---------------------------------------------------------------------------

_MONEY = r"([\d,]+(?:\.\d+)?)"

_STABILITY_CUES = {
    "High":   ("doesn't move", "does not move", "same figure every year", "steady", "predictable"),
    "Low":    ("all over the place", "could be double", "depends on the equity", "swings wildly"),
    "Medium": ("swings around", "some years", "varies", "moves around"),
}


class RuleBasedExtractor:
    """
    Deterministic pattern matching. No model, no key, no network.

    Present so the language-model extractors have something to beat. If the gap
    turns out to be small, that is a real finding and the honest conclusion is
    that the extra machinery is not doing work.
    """

    name = "rule_based"
    uses_llm = False

    def extract(self, transcript: str, client_id: str, transcript_id: str) -> ExtractedProfile:
        lines = transcript.split("\n")
        client_text = _client_voice(lines)
        fields: dict[str, ExtractedField] = {}

        def stated(name: str, value, quote: str, confidence: float) -> None:
            fields[name] = ExtractedField(
                name=name, value=value, source=FactSource.STATED,
                confidence=confidence, evidence_quote=quote.strip(),
                evidence_line=_find_line(lines, quote),
            )

        # age
        m = re.search(r"I'?m (\d{2})\b", client_text)
        if m:
            stated("age", int(m.group(1)), m.group(0), 0.95)
        else:
            fields["age"] = _unknown("age")

        # salary — "Base is 83,920 a year"
        m = re.search(rf"[Bb]ase is {_MONEY}", client_text)
        if m:
            stated("annual_salary", float(m.group(1).replace(",", "")), m.group(0), 0.9)
        else:
            fields["annual_salary"] = _unknown("annual_salary")

        # bonus — "around 5%"
        m = re.search(r"bonus on top[^.]*?(\d+(?:\.\d+)?)\s*%", client_text)
        if m:
            stated("bonus_rate", float(m.group(1)) / 100.0, m.group(0), 0.8)
        else:
            fields["bonus_rate"] = _unknown("bonus_rate")

        # financial capital — "accounts come to about 200,000"
        m = re.search(rf"accounts come to about {_MONEY}", client_text)
        if m:
            stated("financial_capital", float(m.group(1).replace(",", "")), m.group(0), 0.9)
        else:
            fields["financial_capital"] = _unknown("financial_capital")

        # income stability — qualitative cues
        matched = None
        for label, cues in _STABILITY_CUES.items():
            for cue in cues:
                if cue in client_text.lower():
                    idx = client_text.lower().find(cue)
                    matched = (label, client_text[max(0, idx - 40): idx + 40])
                    break
            if matched:
                break
        if matched:
            stated("income_stability", matched[0], matched[1], 0.7)
        else:
            fields["income_stability"] = _unknown("income_stability")

        # employer sector
        m = re.search(r"employer's in ([A-Za-z &]+?),", client_text)
        if m:
            stated("industry_exposure_sector", m.group(1).strip(), m.group(0), 0.85)
        else:
            fields["industry_exposure_sector"] = _unknown("industry_exposure_sector")

        # RSU concentration
        m = re.search(r"roughly (\d+(?:\.\d+)?)\s*% of it is employer shares", client_text)
        if m:
            stated("RSU_concentration", float(m.group(1)) / 100.0, m.group(0), 0.85)
        else:
            fields["RSU_concentration"] = _unknown("RSU_concentration")

        # pension
        if "pension through work" in client_text.lower():
            idx = client_text.lower().find("pension through work")
            stated("has_pension", True, client_text[max(0, idx - 20): idx + 60], 0.8)
        elif "no pension" in client_text.lower():
            idx = client_text.lower().find("no pension")
            stated("has_pension", False, client_text[idx: idx + 40], 0.8)
        else:
            fields["has_pension"] = _unknown("has_pension")

        # Advisor-supplied estimate. Deliberately read from the ADVISOR turns,
        # not the client's, and recorded INFERRED rather than STATED: the
        # advisor proposed the number and the client only assented to it. A
        # suitability file that records this as the client's own assertion has
        # lost the distinction that matters most.
        advisor_text = _advisor_voice(lines)
        m = re.search(r"sensitivity somewhere around (\d+(?:\.\d+)?)", advisor_text)
        if m:
            fields["advisor_supplied_beta"] = ExtractedField(
                name="advisor_supplied_beta",
                value=float(m.group(1)),
                source=FactSource.INFERRED,
                confidence=0.6,
                evidence_quote=m.group(0),
                evidence_line=_find_line(lines, m.group(0)),
            )

        for name in TARGET_FIELDS:
            fields.setdefault(name, _unknown(name))

        return ExtractedProfile(
            client_id=client_id, transcript_id=transcript_id,
            extractor=self.name, fields=fields,
        )


# ---------------------------------------------------------------------------
# 2. Naive extractor — the control
# ---------------------------------------------------------------------------

_NAIVE_PROMPT = """Read this client discovery call and return a JSON object with these fields:
{fields}

Constrained fields — use exactly these values:
{value_space}

Units — return numbers in these units:
{units}

Return ONLY the JSON object, values only, no explanation.

TRANSCRIPT:
{transcript}
"""


class NaiveExtractor:
    """
    One call, transcript in, flat profile out. No provenance, no unknowns.

    The mentor asked for this directly. It is the control that makes the
    structured extractor's complexity defensible — or shows that it isn't.
    Because it has no unknown option, everything it returns is scored as an
    assertion, which is exactly what makes refusal-to-guess measurable.
    """

    name = "naive"
    uses_llm = True

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def extract(self, transcript: str, client_id: str, transcript_id: str) -> ExtractedProfile:
        raw = call_model(
            _NAIVE_PROMPT.format(
                fields=", ".join(TARGET_FIELDS),
                value_space=_value_space_spec(),
                units=_units_spec(),
                transcript=transcript,
            ),
            model=self.model,
        )
        parsed = _parse_json_block(raw)

        fields: dict[str, ExtractedField] = {}
        for name in TARGET_FIELDS:
            value = parsed.get(name)
            if value is None or (isinstance(value, str) and value.lower() in {"unknown", "n/a", ""}):
                fields[name] = _unknown(name)
            else:
                # No provenance is available, so it cannot be marked STATED —
                # the contract requires an evidence quote for that. INFERRED is
                # the honest label for an unsourced assertion.
                fields[name] = ExtractedField(
                    name=name, value=value, source=FactSource.INFERRED,
                    confidence=0.5,
                )

        return ExtractedProfile(
            client_id=client_id, transcript_id=transcript_id,
            extractor=self.name, fields=fields,
        )


# ---------------------------------------------------------------------------
# 3. Structured extractor — provenance, refusal, calibrated confidence
# ---------------------------------------------------------------------------

_STRUCTURED_PROMPT = """You are extracting a client profile from a discovery call for a fiduciary advisory system.

For EACH field below, return an object with:
  "value"      the extracted value, or null if the client never discussed it
  "source"     "stated" if the CLIENT asserted it themselves,
               "inferred" if you derived it from something they said,
               "unknown" if it was never discussed
  "confidence" 0.0 to 1.0. Use 0.0 for unknown. Be honest: a value you are
               guessing at should score low. Your confidence will be checked
               against your actual accuracy.
  "quote"      the VERBATIM sentence from the transcript supporting the value,
               or null. Required whenever source is "stated". Copy it exactly.

Critical rules:
- If a field was never discussed, return null with source "unknown". Do NOT
  guess a plausible value. Refusing to answer is the correct behaviour.
- If the ADVISOR supplied a number and the client merely agreed, that is
  "inferred", not "stated". Only mark "stated" when the client asserted it.

FIELDS: {fields}

Constrained fields — "value" must be exactly one of the listed options, not a
description. A paraphrase like "stable base but variable bonus" is unusable
downstream even when it is accurate:
{value_space}

Units — return numbers in these units. Getting these wrong fails validation
one layer downstream, where the cause is no longer visible:
{units}

Return ONLY a JSON object mapping each field name to its object.

TRANSCRIPT:
{transcript}
"""


_CLASSIFY_PROMPT = """You are sorting a client discovery call for a fiduciary advisory system.

Typed field extraction is handled separately. Your job is different: identify the
statements in this conversation that must be ROUTED somewhere, and say where.

Return a JSON object with one key, "statements", holding a list. Each entry:

  "kind"         one of:
                   hard_constraint   — something the client will not hold
                   soft_preference   — a tilt they'd like reflected (not binding)
                   risk_fact         — arrives sounding like a preference but is
                                       really an exposure (a concentration, an
                                       income beta, a forced sector underweight)
                   suitability_fact  — horizon, liquidity, account type, and what
                                       is in or out of scope
                   challenge         — a contradiction worth putting back to the
                                       client (stated risk tolerance disagreeing
                                       with actual exposure, self-description
                                       disagreeing with holdings)
  "summary"      one line, in your words, of what this commits the client to
  "quote"        VERBATIM span from the transcript. Copy exactly. Required.
  "subject"      ticker / sector / asset it concerns, or null
  "destinations" list of: universe_exclusion, concentration_limit,
                 human_capital_beta, sector_underweight, bl_view,
                 suitability_record, scope_boundary, advisor_review
  "source"       "stated" if the client asserted it; "inferred" if the advisor
                 supplied it and the client agreed
  "confidence"   0.0 to 1.0

Telling risk_fact from challenge — the pair most easily confused:
- A risk_fact states an EXPOSURE. "40% of my account is my employer's stock."
- A challenge states a CONTRADICTION between two things the client has told you.
  "I'm cautious" alongside "I want this to double in ten years" is a challenge:
  neither half is an exposure, but together they cannot both be satisfied.
- If one passage does both — asserts an exposure AND contradicts something the
  client said earlier — emit TWO statements over the same quote, one of each
  kind. Do not pick whichever seems dominant.

Rules that matter:
- A risk_fact usually has MORE THAN ONE destination. A holding in the client's
  own employer is simultaneously a concentration, a human-capital beta, and a
  reason to underweight that sector. Give every destination that applies.
- Every entry except a challenge must have at least one destination. If you
  cannot say where a statement goes, it is not a statement worth classifying.
- Do not invent statements. Only classify what is actually in the text.
- A challenge is recorded for a human to review, never acted on automatically.

Return ONLY the JSON object.

TRANSCRIPT:
{transcript}
"""


def _parse_statements(raw: str, lines: list[str]) -> list[ClientStatement]:
    """Parse and validate a classification response, dropping unroutable entries."""
    parsed = _parse_json_block(raw)
    out: list[ClientStatement] = []

    for entry in parsed.get("statements", []):
        if not isinstance(entry, dict):
            continue
        quote = (entry.get("quote") or "").strip()
        if not quote:
            continue  # unshowable to the client, so not routable

        try:
            destinations = [
                PipelineDestination(d)
                for d in entry.get("destinations", [])
                if d in PipelineDestination._value2member_map_
            ]
            out.append(ClientStatement(
                kind          = StatementKind(entry["kind"]),
                summary       = entry.get("summary", "").strip() or quote[:80],
                quote         = quote,
                evidence_line = _find_line(lines, quote),
                destinations  = destinations,
                subject       = entry.get("subject"),
                source        = FactSource(str(entry.get("source", "stated")).lower()),
                confidence    = float(entry.get("confidence", 0.5)),
            ))
        except (KeyError, ValueError):
            # A malformed or unroutable classification is dropped rather than
            # coerced — the contract's validators define what routable means.
            continue

    return out


class StructuredExtractor:
    """
    Field-by-field extraction with an explicit unknown option and provenance,
    plus statement classification.

    The design claim being tested: giving the model somewhere honest to put "I
    don't know", and requiring it to cite the sentence, produces a profile a
    fiduciary system can actually defend. The harness decides whether that claim
    survives contact with the answer key.

    Classification is a second call. It is kept separate from field extraction
    because they are different jobs against different targets — one recovers
    typed values the formulas consume, the other sorts prose into things that
    must be routed. Merging them into one prompt made both worse in testing.
    """

    name = "structured"
    uses_llm = True

    def __init__(self, model: str = DEFAULT_MODEL, classify: bool = True):
        self.model = model
        self.classify = classify

    def _classify(self, transcript: str) -> list[ClientStatement]:
        try:
            raw = call_model(
                _CLASSIFY_PROMPT.format(transcript=transcript), model=self.model
            )
            return _parse_statements(raw, transcript.split("\n"))
        except Exception as e:
            print(f"  classification failed: {type(e).__name__}: {str(e)[:80]}")
            return []

    def extract(self, transcript: str, client_id: str, transcript_id: str) -> ExtractedProfile:
        raw = call_model(
            _STRUCTURED_PROMPT.format(
                fields=", ".join(TARGET_FIELDS),
                value_space=_value_space_spec(),
                units=_units_spec(),
                transcript=transcript,
            ),
            model=self.model,
        )
        parsed = _parse_json_block(raw)
        lines = transcript.split("\n")

        fields: dict[str, ExtractedField] = {}
        for name in TARGET_FIELDS:
            entry = parsed.get(name)
            if not isinstance(entry, dict):
                fields[name] = _unknown(name)
                continue

            value  = entry.get("value")
            source = str(entry.get("source", "unknown")).lower()
            quote  = entry.get("quote")

            if value is None or source == "unknown":
                fields[name] = _unknown(name)
                continue

            # A "stated" claim with no quote cannot be constructed under the
            # contract. Rather than discard the extraction, demote it to
            # inferred — the model asserted something it could not source, and
            # the harness should see that as an unsourced assertion.
            resolved = FactSource.STATED if (source == "stated" and quote) else FactSource.INFERRED

            fields[name] = ExtractedField(
                name=name,
                value=value,
                source=resolved,
                confidence=float(entry.get("confidence", 0.5)),
                evidence_quote=quote,
                evidence_line=_find_line(lines, quote) if quote else None,
            )

        return ExtractedProfile(
            client_id=client_id, transcript_id=transcript_id,
            extractor=self.name, fields=fields,
            statements=self._classify(transcript) if self.classify else [],
        )


def available_extractors(include_llm: bool = True) -> list[Extractor]:
    """
    Extractors the harness should run.

    The rule-based floor is always included. The LLM extractors are included
    only when a key is configured, so the harness degrades to a smaller but
    still honest comparison rather than crashing.
    """
    extractors: list[Extractor] = [RuleBasedExtractor()]
    if include_llm and os.getenv("ANTHROPIC_API_KEY"):
        extractors.extend([NaiveExtractor(), StructuredExtractor()])
    return extractors
