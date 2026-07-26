"""
transcript_generator.py — synthetic discovery calls built from a known profile,
so the answer key exists before the transcript does.

Why not generate these with a language model
--------------------------------------------
*Now and Forward* §4 names the trap directly: "generating transcripts from a
prompt and then extracting from them means the same model writes the exam and
grades it, and your accuracy will look excellent for entirely the wrong reason."

This module inverts the dependency. The ground-truth profile comes first, and the
transcript is rendered from it by deterministic templating. The answer key is not
recovered from the transcript afterwards — it is the input that produced it. No
model is involved on the generation side at all, so an extractor cannot score
well by sharing a prior with the generator.

Salience is deliberate
----------------------
Each fact is planted at a chosen salience so recall can be measured against it:

    PROMINENT      discussed across several turns, with numbers repeated
    MENTIONED      stated once, plainly, in a single sentence
    PARENTHETICAL  buried in an aside, never returned to
    OMITTED        never discussed — the extractor must return UNKNOWN and ask

This mirrors the mentor's own example: Priya's ARVX position is prominent, her
condo exclusion is a parenthetical, and both have to survive extraction.

Speaker turns are line-indexed so `evidence_line` in ExtractedField can be
checked against the line that actually carries the fact.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field as dc_field
from enum import Enum
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "data" / "outputs"
TRANSCRIPT_DIR = OUTPUTS_DIR / "transcripts"


class Salience(str, Enum):
    PROMINENT     = "prominent"
    MENTIONED     = "mentioned"
    PARENTHETICAL = "parenthetical"
    OMITTED       = "omitted"


@dataclass
class PlantedFact:
    """One fact placed in a transcript, with everything needed to grade it."""

    name:      str
    value:     float | str | bool | None
    salience:  Salience
    is_stated: bool
    """
    True when the client asserts the value themselves. False when the ADVISOR
    supplies it and the client merely assents — the mentor's point about the
    beta of 1.2 in Priya's call being "the advisor's estimate carrying the
    client's soft assent", which a suitability review must not record as the
    client's own assertion.
    """
    expected_quote: str | None = None
    expected_line:  int | None = None


@dataclass
class PlantedStatement:
    """
    A statement deliberately planted in a transcript, with its expected routing.

    The classification answer key. Same principle as the field answer key: the
    ground truth is the input that produced the transcript, not something read
    back out of it afterwards, so a classifier cannot score well by agreeing
    with whoever wrote the key.

    `required_destinations` is what a correct classification MUST include.
    Extra destinations are not penalised — a risk fact legitimately reaches
    several places, and the taxonomy's point is that missing one is the error,
    not finding one more.
    """

    kind:    str
    subject: str | None
    quote:   str
    required_destinations: tuple[str, ...]


@dataclass
class TranscriptCase:
    """A synthetic discovery call plus the key that generated it."""

    transcript_id: str
    client_id:     str
    text:          str
    lines:         list[str]
    facts:         dict[str, PlantedFact] = dc_field(default_factory=dict)
    planted_statements: list[PlantedStatement] = dc_field(default_factory=list)

    @property
    def omitted(self) -> list[str]:
        return sorted(n for n, f in self.facts.items() if f.salience == Salience.OMITTED)

    def answer_key(self) -> dict:
        return {
            "transcript_id": self.transcript_id,
            "client_id":     self.client_id,
            "facts": {
                n: {
                    "value":          f.value,
                    "salience":       f.salience.value,
                    "is_stated":      f.is_stated,
                    "expected_quote": f.expected_quote,
                    "expected_line":  f.expected_line,
                }
                for n, f in self.facts.items()
            },
        }


# ── Default salience plan ──────────────────────────────────────────────────
# Chosen so every measurable property has something to measure: two prominent
# facts, a spread of single-mention ones, two buried asides, and two omissions
# that the extractor must refuse to guess.
DEFAULT_SALIENCE = {
    "annual_salary":      Salience.PROMINENT,
    "income_stability":   Salience.PROMINENT,
    "financial_capital":  Salience.MENTIONED,
    "age":                Salience.MENTIONED,
    "industry_exposure_sector": Salience.MENTIONED,
    "RSU_concentration":  Salience.MENTIONED,
    "bonus_rate":         Salience.PARENTHETICAL,
    "has_pension":        Salience.PARENTHETICAL,
    "investment_horizon_years": Salience.OMITTED,
    "liquidity_needs":    Salience.OMITTED,
}

_STABILITY_PHRASES = {
    "High": [
        "It's a university line, so the number basically doesn't move.",
        "Same figure every year. That's the whole appeal of the job, honestly.",
    ],
    "Medium": [
        "The base is steady but the bonus swings around a fair bit year to year.",
        "Some years the variable piece is great, some years it's nothing.",
    ],
    "Low": [
        "It's all over the place year to year, really depends on the equity piece.",
        "Honestly it could be double or half depending on how the stock does.",
    ],
}


class TranscriptBuilder:
    """Accumulates speaker turns while recording which line carries which fact."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def say(self, speaker: str, text: str) -> int:
        """Append a turn and return its 0-indexed line number."""
        self.lines.append(f"{speaker}: {text}")
        return len(self.lines) - 1

    def advisor(self, text: str) -> int:
        return self.say("ADVISOR", text)

    def client(self, text: str) -> int:
        return self.say("CLIENT", text)


def _plant(
    case_facts: dict[str, PlantedFact],
    name: str,
    value,
    salience: Salience,
    is_stated: bool = True,
    quote: str | None = None,
    line: int | None = None,
) -> None:
    case_facts[name] = PlantedFact(
        name=name, value=value, salience=salience, is_stated=is_stated,
        expected_quote=quote, expected_line=line,
    )


def generate_transcript(
    persona: dict,
    salience_plan: dict[str, Salience] | None = None,
    seed: int = 0,
) -> TranscriptCase:
    """
    Render a synthetic discovery call from a raw BLS persona dict.

    `persona` is the ground truth. `salience_plan` overrides how prominently each
    field is planted; anything set to OMITTED is left out of the transcript
    entirely and becomes a refusal-to-guess test.
    """
    plan = {**DEFAULT_SALIENCE, **(salience_plan or {})}
    rng  = random.Random(seed)
    b    = TranscriptBuilder()
    facts: dict[str, PlantedFact] = {}

    client_id = persona["client_id"]
    career    = persona.get("career_type", "professional")

    b.advisor("Thanks for making the time. Before anything else I just want to "
              "understand where things stand — nothing prepared, just talk me through it.")
    b.client(f"Sure. I've been in {career} work for a while now, same field the whole time.")

    # ── Age ────────────────────────────────────────────────────────────────
    if plan.get("age") != Salience.OMITTED:
        quote = f"I'm {persona['age']}."
        if plan.get("age") == Salience.PARENTHETICAL:
            quote = f"My partner keeps reminding me I'm {persona['age']} and not getting younger."
        line = b.client(quote)
        _plant(facts, "age", persona["age"], plan.get("age", Salience.MENTIONED),
               True, quote, line)
    else:
        _plant(facts, "age", None, Salience.OMITTED, False)

    # ── Salary — prominent means discussed over several turns ──────────────
    salary = persona["annual_salary"]
    sal_salience = plan.get("annual_salary", Salience.PROMINENT)
    if sal_salience != Salience.OMITTED:
        b.advisor("Let's start with compensation. What does that look like?")
        quote = f"Base is {salary:,.0f} a year."
        line = b.client(quote)
        _plant(facts, "annual_salary", salary, sal_salience, True, quote, line)
        if sal_salience == Salience.PROMINENT:
            b.advisor("And has that been stable, or has it moved around?")
            b.client(f"It's crept up over the years but {salary:,.0f} is where it sits now.")
    else:
        _plant(facts, "annual_salary", None, Salience.OMITTED, False)

    # ── Bonus — parenthetical by default, easy to miss ─────────────────────
    bonus = persona.get("bonus_rate", 0.0)
    bon_salience = plan.get("bonus_rate", Salience.PARENTHETICAL)
    if bon_salience != Salience.OMITTED and bonus:
        # The answer key records what the transcript SAYS, not the persona's
        # underlying float. The client says "around 5%", so 5% is the correct
        # extraction — 4.6% is not recoverable from the conversation, and
        # grading against it penalises an extractor for reading correctly.
        spoken_bonus = round(bonus, 2)
        quote = (f"There's a bonus on top, works out around {spoken_bonus:.0%} most years, "
                 "though I never really count on it.")
        line = b.client(quote)
        _plant(facts, "bonus_rate", spoken_bonus, bon_salience, True, quote, line)
    else:
        _plant(facts, "bonus_rate", None, Salience.OMITTED, False)

    # ── Income stability — prominent, and expressed qualitatively ──────────
    stability = persona["income_stability"]
    stab_salience = plan.get("income_stability", Salience.PROMINENT)
    if stab_salience != Salience.OMITTED:
        b.advisor("How predictable is all of that, year to year?")
        quote = rng.choice(_STABILITY_PHRASES[stability])
        line = b.client(quote)
        _plant(facts, "income_stability", stability, stab_salience, True, quote, line)
        if stab_salience == Salience.PROMINENT:
            b.advisor("So if I had to put a range on it?")
            b.client("I'd rather you didn't pin me to a number, but that's the shape of it.")
    else:
        _plant(facts, "income_stability", None, Salience.OMITTED, False)

    # ── Financial capital ──────────────────────────────────────────────────
    fc = persona["financial_capital"]
    fc_salience = plan.get("financial_capital", Salience.MENTIONED)
    if fc_salience != Salience.OMITTED:
        b.advisor("And what's invested at the moment?")
        quote = f"All in, the investment accounts come to about {fc:,.0f}."
        line = b.client(quote)
        _plant(facts, "financial_capital", fc, fc_salience, True, quote, line)
    else:
        _plant(facts, "financial_capital", None, Salience.OMITTED, False)

    # ── Employer sector ────────────────────────────────────────────────────
    sector = persona.get("industry_exposure_sector", "Unknown")
    sec_salience = plan.get("industry_exposure_sector", Salience.MENTIONED)
    if sec_salience != Salience.OMITTED:
        quote = f"The employer's in {sector}, has been the whole time."
        line = b.client(quote)
        _plant(facts, "industry_exposure_sector", sector, sec_salience, True, quote, line)
    else:
        _plant(facts, "industry_exposure_sector", None, Salience.OMITTED, False)

    # ── RSU concentration ──────────────────────────────────────────────────
    rsu = persona.get("RSU_concentration", 0.0)
    rsu_salience = plan.get("RSU_concentration", Salience.MENTIONED)
    if rsu_salience != Salience.OMITTED and rsu:
        b.advisor("Any of that in company stock?")
        quote = (f"Yes — roughly {rsu:.0%} of it is employer shares I've never sold. "
                 "I know that's a lot in one name.")
        line = b.client(quote)
        _plant(facts, "RSU_concentration", rsu, rsu_salience, True, quote, line)
    else:
        _plant(facts, "RSU_concentration", 0.0 if not rsu else None,
               Salience.OMITTED if not rsu else rsu_salience, False)

    # ── Pension — buried aside ─────────────────────────────────────────────
    pension = persona.get("has_pension", False)
    pen_salience = plan.get("has_pension", Salience.PARENTHETICAL)
    if pen_salience != Salience.OMITTED:
        quote = ("There's a pension through work too, but that's separate, "
                 "I don't really think about it.") if pension else \
                ("No pension, it's all just the accounts.")
        line = b.client(quote)
        _plant(facts, "has_pension", pension, pen_salience, True, quote, line)
    else:
        _plant(facts, "has_pension", None, Salience.OMITTED, False)

    # ── Advisor-supplied estimate the client merely assents to ─────────────
    # Recorded is_stated=False on purpose: this is the advisor's number, and a
    # suitability file must not show it as the client's own assertion.
    b.advisor("Careers like yours tend to track the market fairly closely — "
              "I'd put the sensitivity somewhere around 0.8.")
    assent_line = b.client("That sounds about right, though it might be low in a bad year.")
    _plant(facts, "advisor_supplied_beta", 0.8, Salience.MENTIONED, False,
           "That sounds about right, though it might be low in a bad year.", assent_line)

    # ── Omitted fields — never mentioned, must be refused ──────────────────
    for name in ("investment_horizon_years", "liquidity_needs", "investment_objective"):
        if plan.get(name) == Salience.OMITTED or name not in plan:
            _plant(facts, name, None, Salience.OMITTED, False)

    # ── Statements to classify ─────────────────────────────────────────────
    # Planted deliberately so classification can be SCORED rather than asserted.
    # One of each kind that matters, including the two the taxonomy exists for:
    # a risk fact that must reach several destinations at once, and a challenge
    # where the client's self-description disagrees with their own position.
    planted: list[PlantedStatement] = []

    b.advisor("Anything you'd refuse to hold on principle?")
    excl_quote = "I don't want tobacco or weapons in there. That's firm."
    b.client(excl_quote)
    planted.append(PlantedStatement(
        kind="hard_constraint", subject="tobacco, weapons",
        quote=excl_quote, required_destinations=("universe_exclusion",),
    ))

    pref_quote = "I'd like some clean-energy exposure if it doesn't cost me much."
    b.client(pref_quote)
    planted.append(PlantedStatement(
        kind="soft_preference", subject="clean energy",
        quote=pref_quote, required_destinations=("bl_view",),
    ))

    if rsu:
        # The category the whole taxonomy exists for: employer stock is at once
        # a single-name concentration, an income beta, and a reason to
        # underweight the sector the client already earns their living in.
        risk_quote = (
            f"About {rsu:.0%} of the account is my own company's stock, and my "
            f"salary comes from the same place."
        )
        b.client(risk_quote)
        planted.append(PlantedStatement(
            kind="risk_fact", subject=sector,
            quote=risk_quote,
            required_destinations=("concentration_limit", "human_capital_beta"),
        ))

    scope_quote = "The house isn't part of this, and my partner's accounts are separate."
    b.client(scope_quote)
    planted.append(PlantedStatement(
        kind="suitability_fact", subject=None,
        quote=scope_quote, required_destinations=("scope_boundary",),
    ))

    b.advisor("How would you describe your appetite for risk?")
    # A challenge must be a CONTRADICTION, not a restatement of an exposure.
    # The earlier version of this line ("most of what I own is in the one
    # stock") asserted a concentration — and did so for all nine personas,
    # including the six holding no employer stock at all. That is a risk fact
    # wearing a challenge's label, so a classifier filing it as `risk_fact` was
    # reading it correctly and the answer key was wrong. It cost 7 of 38 on kind
    # accuracy and every one of those was the harness's fault.
    #
    # This version contradicts the client's own stated tolerance against their
    # own stated goal, with no exposure claim attached and nothing that depends
    # on the persona's holdings.
    # The contradiction must also avoid *satisfying* the refusal tests. An
    # earlier version read "...I would like this to roughly double over the next
    # ten years", which hands the extractor an investment horizon and a growth
    # objective — both planted as OMITTED. Structured hallucination jumped from
    # 0% to 52.9% and every one of those was the extractor correctly reading a
    # horizon this generator had put in the transcript.
    #
    # So a planted statement has two constraints, not one: it must be the kind
    # it claims to be, and it must not mention any field planted as OMITTED.
    chal_quote = (
        "Pretty cautious, I'd say. Though honestly a 30% drop wouldn't "
        "bother me all that much."
    )
    b.client(chal_quote)
    planted.append(PlantedStatement(
        kind="challenge", subject=None,
        quote=chal_quote, required_destinations=("advisor_review",),
    ))

    b.advisor("That's a good picture to start from. I'll put some numbers together.")

    text = "\n".join(b.lines)
    case = TranscriptCase(
        transcript_id=f"synth_{client_id}_s{seed}",
        client_id=client_id,
        text=text,
        lines=b.lines,
        facts=facts,
        planted_statements=planted,
    )
    _assert_omissions_are_omitted(case)
    return case


# Words that would let an extractor legitimately recover a field the answer key
# says was never discussed. Keyed by the OMITTED field they would satisfy.
# Phrases, deliberately, not bare words. A first version tripped on "years" and
# fired on "works out around 5% most years" — a bonus statement with nothing to
# do with an investment horizon. A guard that cries wolf gets switched off, so
# each cue has to be specific enough that its presence really would let an
# extractor recover the omitted field.
_OMISSION_TRIPWIRES = {
    "investment_horizon_years": (
        "over the next", "next ten years", "retire in", "retire at",
        "stop working", "time horizon", "until about",
    ),
    "investment_objective":     (
        "roughly double", "grow it", "income from this", "preserve capital",
        "capital preservation",
    ),
    "liquidity_needs":          (
        "need cash", "draw on it", "draw on this", "need the money",
    ),
}


def _assert_omissions_are_omitted(case: TranscriptCase) -> None:
    """
    Fail loudly when a transcript discusses a field its answer key calls OMITTED.

    A field planted as OMITTED is a refusal test: the extractor is supposed to
    return unknown, and is marked as hallucinating if it returns a value. That
    only holds if the transcript genuinely never mentions it — otherwise the
    extractor is reading correctly and the harness scores it as fabricating.

    This is not hypothetical. A planted challenge that read "I would like this
    to roughly double over the next ten years" pushed structured hallucination
    from 0% to 52.9% in one run, entirely because the generator had put a
    horizon and an objective into a transcript that claimed to have neither.
    A silent 52-point swing in the headline metric is exactly the failure this
    check exists to make noisy.
    """
    lowered = case.text.lower()
    for field, cues in _OMISSION_TRIPWIRES.items():
        planted = case.facts.get(field)
        if planted is None or planted.salience != Salience.OMITTED:
            continue
        hits = [c for c in cues if c in lowered]
        if hits:
            raise AssertionError(
                f"{case.transcript_id}: '{field}' is planted as OMITTED — a "
                f"refusal test — but the transcript contains {hits}. An extractor "
                f"reading that would be scored as hallucinating. Reword the "
                f"transcript or stop planting this field as omitted."
            )


def generate_corpus(
    personas: list[dict],
    seeds: tuple[int, ...] = (0, 1),
    salience_plan: dict[str, Salience] | None = None,
) -> list[TranscriptCase]:
    """Render a transcript per (persona, seed)."""
    return [
        generate_transcript(p, salience_plan=salience_plan, seed=s)
        for p in personas
        for s in seeds
    ]


def save_corpus(cases: list[TranscriptCase], directory: Path | None = None) -> Path:
    """Write each transcript as .txt and the combined answer key as JSON."""
    directory = directory or TRANSCRIPT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    for case in cases:
        (directory / f"{case.transcript_id}.txt").write_text(case.text)

    key_path = directory / "answer_key.json"
    key_path.write_text(
        json.dumps({c.transcript_id: c.answer_key() for c in cases}, indent=2)
    )
    print(f"Saved {len(cases)} transcripts + answer key → {directory}")
    return key_path


def run(save: bool = True) -> list[TranscriptCase]:
    """Build the corpus from the BLS personas."""
    from agents.profile.profile_agent import _load_bls_oes
    from agents.profile.profile_model import build_bls_personas

    personas = build_bls_personas(_load_bls_oes())
    cases = generate_corpus(personas)

    counts: dict[str, int] = {}
    for c in cases:
        for f in c.facts.values():
            counts[f.salience.value] = counts.get(f.salience.value, 0) + 1

    print(f"\n=== Synthetic transcript corpus ===")
    print(f"{len(cases)} transcripts from {len(personas)} personas")
    print(f"Planted facts by salience: {counts}")
    print(f"Omitted (refusal-to-guess tests) per transcript: {cases[0].omitted}")
    print(f"\n--- sample: {cases[0].transcript_id} ---")
    print(cases[0].text)

    if save:
        save_corpus(cases)
    return cases


if __name__ == "__main__":  # pragma: no cover
    run()
