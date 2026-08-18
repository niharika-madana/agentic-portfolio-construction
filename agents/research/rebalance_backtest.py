"""
rebalance_backtest.py — evaluation harness for the rebalance evaluator.

The standing rule since Week 7 is that any change cites the harness number it
improved. This is that harness for the Week 9 deliverable: replay the evaluator
over every regime transition in the 1995-2025 sequence and score its verdicts
against what actually happened next.

Ground truth
------------
A transition is DURABLE when the destination regime went on to hold for at least
DURABLE_MONTHS months. It is NOISE otherwise. This is knowable after the fact
from the sequence itself, which is what makes the harness possible without any
labelling work.

Scoring
-------
Each transition is evaluated at DECISION_LAG months after the shift date — the
evaluator is allowed to wait and see, which is the entire point of the
persistence gate, but it must commit before the outcome window opens.

    true negative   noise correctly declined      churn avoided
    true positive   durable change acted on       rebalance justified
    false positive  noise acted on                churn — the failure that costs money
    false negative  durable change declined       opportunity missed

The baseline is the current system: rebalance whenever regime_change_detected is
True, which acts on every transition and therefore takes every false positive.
Reporting the evaluator against that baseline is the point — "a system that
correctly declines to rebalance on noise is a better result than one that always
acts."

Caveat, stated plainly: PELT and XGBoost are fitted on the full sample upstream,
so the label path and break locations at date t embed post-t information. The
gates are truncated point-in-time (see build_evidence) but that upstream leak
remains, and it flatters the structural_break and confidence gates. A
walk-forward refit is future work; the churn/durable split reported here should
be read as an upper bound on live performance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from agents.research.rebalance import (
    RegimeRun,
    build_evidence,
    regime_runs,
)

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "data" / "outputs"
SEQUENCE_JSON = OUTPUTS_DIR / "regime_sequence.json"
BACKTEST_JSON = OUTPUTS_DIR / "rebalance_backtest.json"

DURABLE_MONTHS = 12
"""Destination run length at or above which a transition counts as durable."""

DECISION_LAG = 3
"""
Months after the shift date at which the evaluator must commit. Matches
MIN_DWELL_MONTHS: the evaluator may wait exactly as long as its own persistence
gate requires, and no longer.
"""


@dataclass
class TransitionOutcome:
    """One historical transition, the verdict, and what actually happened."""

    shift_date:       str
    prior_regime:     str
    current_regime:   str
    actual_months:    int
    durable:          bool
    evidence_passed:  bool
    gates_failed:     list[str]
    months_in_regime: int
    run_confidence:   float
    pelt_corroborated: bool
    reversion_rate:   float | None
    outcome:          str  # true_positive | true_negative | false_positive | false_negative


def _shift_by_months(stamp: pd.Timestamp, months: int) -> pd.Timestamp:
    return (stamp + pd.DateOffset(months=months)).normalize()


def replay(
    labels:      pd.Series,
    confidence:  pd.Series,
    break_dates: list[pd.Timestamp],
    durable_months: int = DURABLE_MONTHS,
    decision_lag:   int = DECISION_LAG,
) -> list[TransitionOutcome]:
    """
    Replay the evidence gates over every transition in the sequence.

    The final run is skipped: it is still in progress, so its durability is not
    yet knowable and scoring it would count an unfinished run as noise.
    """
    runs: list[RegimeRun] = regime_runs(labels)
    outcomes: list[TransitionOutcome] = []

    for i in range(1, len(runs) - 1):  # skip the in-progress final run
        prior, current = runs[i - 1], runs[i]
        as_of = _shift_by_months(current.start, decision_lag)
        if as_of > labels.index[-1]:
            continue

        evidence = build_evidence(
            labels            = labels,
            confidence        = confidence,
            break_dates       = break_dates,
            regime_label      = current.label,
            prior_regime      = prior.label,
            regime_shift_date = current.start.date(),
            as_of             = as_of.date(),
        )

        durable  = current.months >= durable_months
        acted    = evidence.evidence_supports_change

        if durable and acted:
            outcome = "true_positive"
        elif not durable and not acted:
            outcome = "true_negative"
        elif not durable and acted:
            outcome = "false_positive"
        else:
            outcome = "false_negative"

        outcomes.append(
            TransitionOutcome(
                shift_date        = str(current.start.date()),
                prior_regime      = prior.label,
                current_regime    = current.label,
                actual_months     = current.months,
                durable           = durable,
                evidence_passed   = acted,
                gates_failed      = list(evidence.gates_failed),
                months_in_regime  = evidence.months_in_regime,
                run_confidence    = evidence.run_confidence,
                pelt_corroborated = evidence.pelt_corroborated,
                reversion_rate    = evidence.historical_reversion_rate,
                outcome           = outcome,
            )
        )

    return outcomes


def score(outcomes: list[TransitionOutcome]) -> dict:
    """Confusion matrix, derived rates, and the comparison against always-act."""
    counts = {k: 0 for k in ("true_positive", "true_negative", "false_positive", "false_negative")}
    for o in outcomes:
        counts[o.outcome] += 1

    n = len(outcomes)
    tp, tn, fp, fn = (counts[k] for k in
                      ("true_positive", "true_negative", "false_positive", "false_negative"))
    n_durable = tp + fn
    n_noise   = tn + fp

    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    # Baseline: act on every detected change. It catches every durable
    # transition and eats every noise transition as a false positive.
    gate_failure_counts: dict[str, int] = {}
    for o in outcomes:
        for g in o.gates_failed:
            gate_failure_counts[g] = gate_failure_counts.get(g, 0) + 1

    return {
        "n_transitions":       n,
        "n_durable":           n_durable,
        "n_noise":             n_noise,
        "durable_months":      DURABLE_MONTHS,
        "decision_lag_months": DECISION_LAG,
        "confusion":           counts,
        "accuracy":            rate(tp + tn, n),
        "churn_avoided_rate":  rate(tn, n_noise),      # specificity
        "durable_capture_rate": rate(tp, n_durable),   # sensitivity
        "precision":           rate(tp, tp + fp),
        "baseline_always_act": {
            "accuracy":            rate(n_durable, n),
            "churn_avoided_rate":  0.0 if n_noise else None,
            "durable_capture_rate": 1.0 if n_durable else None,
            "precision":           rate(n_durable, n),
            "trades_taken":        n,
        },
        "trades_taken":        tp + fp,
        "trades_avoided":      tn + fn,
        "gate_failure_counts": gate_failure_counts,
    }


def run_backtest(save: bool = True) -> dict:
    """
    Load the persisted regime sequence, replay the evaluator, and report.

    Requires data/outputs/regime_sequence.json — run run_research_agent() first.
    Break dates are recovered from the PELT run when the feature matrix is
    available; otherwise the structural_break gate is evaluated against an empty
    break list and will fail for every transition, which the summary flags.
    """
    if not SEQUENCE_JSON.exists():
        raise FileNotFoundError(
            f"{SEQUENCE_JSON} not found — run agents.research.research_agent."
            "run_research_agent() first."
        )

    seq = json.loads(SEQUENCE_JSON.read_text())
    labels     = pd.Series({pd.Timestamp(k): v["regime_label"] for k, v in seq.items()}).sort_index()
    confidence = pd.Series({pd.Timestamp(k): v["regime_confidence"] for k, v in seq.items()}).sort_index()

    break_dates = _recover_break_dates()

    outcomes = replay(labels, confidence, break_dates)
    summary  = score(outcomes)
    summary["n_pelt_breaks"] = len(break_dates)
    if not break_dates:
        summary["warning"] = (
            "No PELT breaks available — the structural_break gate failed by default "
            "for every transition. Re-run with the feature matrix present."
        )

    report = {"summary": summary, "transitions": [asdict(o) for o in outcomes]}

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        BACKTEST_JSON.write_text(json.dumps(report, indent=2))
        print(f"Saved → {BACKTEST_JSON}")

    _print_report(report)
    return report


def _recover_break_dates() -> list[pd.Timestamp]:
    """
    Re-derive PELT breaks from the cached feature matrix, or return [].

    The backtest is a separate entry point from run_research_agent(), so the
    breaks are not in hand; recomputing them from the persisted parquet keeps
    the harness runnable on its own.
    """
    parquet = PROJECT_ROOT / "data" / "storage" / "fred_macro_regimes.parquet"
    if not parquet.exists():
        return []
    try:
        from agents.research.pipeline import SIGNAL_COLS, detect_change_points
        from agents.research.research_agent import PELT_MODEL, PELT_PEN

        features_df = pd.read_parquet(parquet)
        missing = [c for c in SIGNAL_COLS if c not in features_df.columns]
        if missing:
            return []
        break_dates, _ = detect_change_points(
            features_df, list(SIGNAL_COLS), pen=PELT_PEN, model=PELT_MODEL
        )
        return list(break_dates)
    except Exception as e:  # pragma: no cover — optional dependency path
        print(f"PELT break recovery skipped ({e}).")
        return []


def _print_report(report: dict) -> None:
    s = report["summary"]
    c = s["confusion"]
    b = s["baseline_always_act"]

    print("\n=== Rebalance evaluator — historical replay ===")
    print(f"{s['n_transitions']} transitions | {s['n_durable']} durable "
          f"(>= {s['durable_months']} months) | {s['n_noise']} noise")
    print(f"Committed at {s['decision_lag_months']} months after each shift date.\n")

    print(f"  true negative  (churn avoided)      {c['true_negative']:3d}")
    print(f"  true positive  (durable acted on)   {c['true_positive']:3d}")
    print(f"  false positive (churn traded)       {c['false_positive']:3d}")
    print(f"  false negative (durable missed)     {c['false_negative']:3d}\n")

    def pct(v):
        return "  n/a" if v is None else f"{v:5.1%}"

    print(f"{'':22s} {'evaluator':>10s} {'always-act':>12s}")
    print(f"{'accuracy':22s} {pct(s['accuracy']):>10s} {pct(b['accuracy']):>12s}")
    print(f"{'churn avoided':22s} {pct(s['churn_avoided_rate']):>10s} {pct(b['churn_avoided_rate']):>12s}")
    print(f"{'durable captured':22s} {pct(s['durable_capture_rate']):>10s} {pct(b['durable_capture_rate']):>12s}")
    print(f"{'precision':22s} {pct(s['precision']):>10s} {pct(b['precision']):>12s}")
    print(f"{'trades taken':22s} {s['trades_taken']:>10d} {b['trades_taken']:>12d}")

    if s.get("gate_failure_counts"):
        print("\nGate failures across all transitions:")
        for gate, count in sorted(s["gate_failure_counts"].items(), key=lambda kv: -kv[1]):
            print(f"  {gate:22s} {count:3d}")

    if s.get("warning"):
        print(f"\nWARNING: {s['warning']}")

    print("\nPer-transition detail:")
    for t in report["transitions"]:
        verdict = "ACT " if t["evidence_passed"] else "HOLD"
        truth   = "durable" if t["durable"] else "noise  "
        flag    = "  <-- " + t["outcome"] if t["outcome"].startswith("false") else ""
        print(f"  {t['shift_date']}  {verdict}  {truth}  {t['actual_months']:3d}mo  "
              f"{t['prior_regime'][:18]:18s} → {t['current_regime'][:18]:18s}"
              f"{flag}")


if __name__ == "__main__":  # pragma: no cover
    run_backtest()
