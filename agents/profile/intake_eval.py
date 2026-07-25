"""
intake_eval.py — the extraction harness. Evaluation is the graded deliverable.

The standing rule since Week 7 is that any prompt, model or tooling change cites
the harness number it improved. This is that harness for the intake layer.

Every extractor implementing the `Extractor` protocol is scored against the same
answer key on the same five properties from *Now and Forward* §4:

    recall_by_salience   Facts discussed at length versus mentioned once in
                         passing. A system that only recovers what was repeated
                         is not reading the conversation, it is reading the
                         headline.
    provenance           Does every extracted field point at a sentence, and does
                         that sentence actually appear in the transcript and
                         support the value? Checked, not taken on trust.
    stated_vs_inferred   "I'm 47" is the client's. An advisor-supplied estimate
                         the client merely agreed with is not. Conflating them
                         means you can no longer tell which constraints the
                         client actually asserted.
    refusal_to_guess     Fed a field that was never discussed, does the extractor
                         return unknown and raise a follow-up question, or invent
                         a plausible number? The easiest failure to plant and the
                         hardest to fake past.
    calibration          Bucketed by stated confidence, is accuracy in the bucket
                         near the confidence? Most systems emit confidence and
                         never check it.

The answer key comes from transcript_generator.py, where the ground-truth profile
produced the transcript rather than being recovered from it — so no extractor can
score well by sharing a prior with the generator.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path

import pandas as pd

from contracts import ExtractedProfile, FactSource
from agents.profile.transcript_generator import (
    Salience,
    TranscriptCase,
    generate_corpus,
)

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "data" / "outputs"
REPORT_JSON  = OUTPUTS_DIR / "intake_eval_report.json"
REPORT_CSV   = OUTPUTS_DIR / "intake_eval_cells.csv"

NUMERIC_TOLERANCE = 0.02
"""Relative tolerance for numeric matches — 2%, to absorb rounding in phrasing."""

CALIBRATION_BUCKETS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))


@dataclass
class FieldOutcome:
    """Grade for one (transcript, field) pair."""

    transcript_id: str
    extractor:     str
    field:         str
    salience:      str

    truth_value:     object
    extracted_value: object
    correct:         bool

    truth_is_stated:     bool
    extracted_source:    str
    source_correct:      bool

    confidence:          float
    has_quote:           bool
    quote_in_transcript: bool
    quote_supports:      bool

    refused:          bool
    should_refuse:    bool
    refusal_correct:  bool
    asked_follow_up:  bool


@dataclass
class ExtractorScore:
    """Aggregate scores for one extractor."""

    extractor: str
    n_cells:   int
    recall_by_salience:  dict[str, float] = dc_field(default_factory=dict)
    provenance_rate:     float | None = None
    quote_verified_rate: float | None = None
    stated_accuracy:     float | None = None
    refusal_precision:   float | None = None
    refusal_recall:      float | None = None
    hallucination_rate:  float | None = None
    calibration:         list[dict] = dc_field(default_factory=list)
    calibration_error:   float | None = None
    overall_accuracy:    float | None = None


def _values_match(truth, got) -> bool:
    """Compare a ground-truth value with an extracted one, tolerantly."""
    if truth is None or got is None:
        return truth is None and got is None
    if isinstance(truth, bool) or isinstance(got, bool):
        return bool(truth) == bool(got)
    if isinstance(truth, (int, float)) and isinstance(got, (int, float)):
        if truth == 0:
            return abs(got) < 1e-9
        return abs(float(got) - float(truth)) / abs(float(truth)) <= NUMERIC_TOLERANCE
    return str(truth).strip().lower() == str(got).strip().lower()


def grade(case: TranscriptCase, extracted: ExtractedProfile) -> list[FieldOutcome]:
    """Grade one extraction against the transcript's answer key."""
    outcomes: list[FieldOutcome] = []
    transcript_lower = case.text.lower()

    for name, planted in case.facts.items():
        got = extracted.fields.get(name)
        should_refuse = planted.salience == Salience.OMITTED

        if got is None:
            # The extractor did not address the field at all. Treated as a
            # silent omission: correct only when nothing was there to find, and
            # never credited as a refusal, since no follow-up was raised.
            outcomes.append(FieldOutcome(
                transcript_id=case.transcript_id, extractor=extracted.extractor,
                field=name, salience=planted.salience.value,
                truth_value=planted.value, extracted_value=None,
                correct=should_refuse,
                truth_is_stated=planted.is_stated, extracted_source="absent",
                source_correct=False, confidence=0.0,
                has_quote=False, quote_in_transcript=False, quote_supports=False,
                refused=False, should_refuse=should_refuse, refusal_correct=False,
                asked_follow_up=False,
            ))
            continue

        refused = got.source == FactSource.UNKNOWN
        correct = should_refuse if refused else _values_match(planted.value, got.value)

        quote = (got.evidence_quote or "").strip()
        has_quote = bool(quote)
        # Provenance is verified against the transcript, not trusted: a quote
        # the model composed rather than copied fails here.
        quote_in_transcript = has_quote and quote.lower() in transcript_lower
        quote_supports = bool(
            quote_in_transcript
            and planted.expected_quote
            and (
                quote.lower() in planted.expected_quote.lower()
                or planted.expected_quote.lower() in quote.lower()
            )
        )

        source_correct = (
            refused if should_refuse
            else (got.source == FactSource.STATED) == planted.is_stated
        )

        outcomes.append(FieldOutcome(
            transcript_id=case.transcript_id, extractor=extracted.extractor,
            field=name, salience=planted.salience.value,
            truth_value=planted.value, extracted_value=got.value,
            correct=correct,
            truth_is_stated=planted.is_stated, extracted_source=got.source.value,
            source_correct=source_correct, confidence=got.confidence,
            has_quote=has_quote, quote_in_transcript=quote_in_transcript,
            quote_supports=quote_supports,
            refused=refused, should_refuse=should_refuse,
            refusal_correct=(refused == should_refuse),
            asked_follow_up=bool(got.follow_up_question),
        ))

    return outcomes


def score(outcomes: list[FieldOutcome], extractor: str) -> ExtractorScore:
    """Aggregate graded cells into the five reported properties."""
    cells = [o for o in outcomes if o.extractor == extractor]
    if not cells:
        return ExtractorScore(extractor=extractor, n_cells=0)

    def rate(subset: list[FieldOutcome], attr: str) -> float | None:
        return round(sum(getattr(o, attr) for o in subset) / len(subset), 4) if subset else None

    # Recall by salience — only over facts that were actually present.
    recall: dict[str, float] = {}
    for level in ("prominent", "mentioned", "parenthetical"):
        present = [o for o in cells if o.salience == level]
        if present:
            recall[level] = round(sum(o.correct for o in present) / len(present), 4)

    answered = [o for o in cells if not o.refused and o.extracted_source != "absent"]
    stated_truth = [o for o in cells if o.salience != "omitted"]

    omitted = [o for o in cells if o.should_refuse]
    refusals = [o for o in cells if o.refused]

    # Hallucination: asserted a value for something never discussed.
    hallucinated = [o for o in omitted if not o.refused and o.extracted_value is not None]

    calibration_rows: list[dict] = []
    weighted_error = 0.0
    total_in_buckets = 0
    for low, high in CALIBRATION_BUCKETS:
        bucket = [o for o in answered if low <= o.confidence < high]
        if not bucket:
            continue
        mean_conf = sum(o.confidence for o in bucket) / len(bucket)
        accuracy  = sum(o.correct for o in bucket) / len(bucket)
        calibration_rows.append({
            "bucket": f"{low:.1f}-{high:.1f}", "n": len(bucket),
            "mean_confidence": round(mean_conf, 4), "accuracy": round(accuracy, 4),
            "gap": round(accuracy - mean_conf, 4),
        })
        weighted_error += abs(accuracy - mean_conf) * len(bucket)
        total_in_buckets += len(bucket)

    return ExtractorScore(
        extractor           = extractor,
        n_cells             = len(cells),
        recall_by_salience  = recall,
        provenance_rate     = rate(answered, "has_quote"),
        quote_verified_rate = rate(answered, "quote_in_transcript"),
        stated_accuracy     = rate(stated_truth, "source_correct"),
        refusal_precision   = rate(refusals, "should_refuse"),
        refusal_recall      = rate(omitted, "refused"),
        hallucination_rate  = round(len(hallucinated) / len(omitted), 4) if omitted else None,
        calibration         = calibration_rows,
        calibration_error   = round(weighted_error / total_in_buckets, 4) if total_in_buckets else None,
        overall_accuracy    = rate(cells, "correct"),
    )


def run(save: bool = True, include_llm: bool = True) -> dict:
    """Build the corpus, run every available extractor, score, and report."""
    from agents.profile.intake import available_extractors
    from agents.profile.profile_agent import _load_bls_oes
    from agents.profile.profile_model import build_bls_personas

    personas = build_bls_personas(_load_bls_oes())
    cases    = generate_corpus(personas)
    extractors = available_extractors(include_llm=include_llm)

    all_outcomes: list[FieldOutcome] = []
    failures: list[str] = []

    for extractor in extractors:
        for case in cases:
            try:
                extracted = extractor.extract(
                    case.text, case.client_id, case.transcript_id
                )
                all_outcomes.extend(grade(case, extracted))
            except Exception as e:
                failures.append(f"{extractor.name} / {case.transcript_id}: {e}")

    scores = [score(all_outcomes, e.name) for e in extractors]
    report = {
        "n_transcripts": len(cases),
        "n_personas":    len(personas),
        "extractors":    [asdict(s) for s in scores],
        "failures":      failures,
    }

    _print_report(report, extractors)

    if save and all_outcomes:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(o) for o in all_outcomes]).to_csv(REPORT_CSV, index=False)
        REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nSaved → {REPORT_CSV}")
        print(f"Saved → {REPORT_JSON}")

    return report


def _print_report(report: dict, extractors) -> None:
    print("\n=== Intake extraction harness ===")
    print(f"{report['n_transcripts']} transcripts from {report['n_personas']} personas "
          f"| {len(report['extractors'])} extractor(s) scored")

    names = [e.name for e in extractors]
    if "naive" not in names or "structured" not in names:
        print(
            "\nNOTE: only the offline rule-based extractor ran. The naive and "
            "structured language-model extractors need ANTHROPIC_API_KEY (or a "
            "reconfigured call_model backend). Without them the naive-vs-designed "
            "comparison the mentor asked for is not yet answered."
        )

    def fmt(v):
        return "   n/a" if v is None else f"{v:6.1%}"

    print(f"\n{'metric':26s}" + "".join(f"{s['extractor']:>14s}" for s in report["extractors"]))
    rows = [
        ("overall accuracy",     lambda s: s["overall_accuracy"]),
        ("recall: prominent",    lambda s: s["recall_by_salience"].get("prominent")),
        ("recall: mentioned",    lambda s: s["recall_by_salience"].get("mentioned")),
        ("recall: parenthetical",lambda s: s["recall_by_salience"].get("parenthetical")),
        ("provenance (has quote)", lambda s: s["provenance_rate"]),
        ("quote verified",       lambda s: s["quote_verified_rate"]),
        ("stated vs inferred",   lambda s: s["stated_accuracy"]),
        ("refusal recall",       lambda s: s["refusal_recall"]),
        ("refusal precision",    lambda s: s["refusal_precision"]),
        ("hallucination rate",   lambda s: s["hallucination_rate"]),
        ("calibration error",    lambda s: s["calibration_error"]),
    ]
    for label, getter in rows:
        print(f"{label:26s}" + "".join(f"{fmt(getter(s)):>14s}" for s in report["extractors"]))

    for s in report["extractors"]:
        if s["calibration"]:
            print(f"\nCalibration — {s['extractor']}:")
            print(f"  {'bucket':>10s} {'n':>4s} {'mean conf':>10s} {'accuracy':>9s} {'gap':>7s}")
            for row in s["calibration"]:
                print(f"  {row['bucket']:>10s} {row['n']:>4d} {row['mean_confidence']:>10.2f} "
                      f"{row['accuracy']:>9.2f} {row['gap']:>+7.2f}")

    if report["failures"]:
        print(f"\n{len(report['failures'])} extraction failure(s):")
        for f in report["failures"][:5]:
            print(f"  {f}")


if __name__ == "__main__":  # pragma: no cover
    run()
