"""
rebalance.py — the Week 9 deliverable: regime-change detection with a rebalance
evaluation that answers whether a detected change justifies trading, or is churn.

`MacroRegimeSnapshot.regime_change_detected` answers "did the label move?". It is
a one-line string comparison and it is not a trading signal. This module answers
the question that matters: should anyone act on it?

The case for the deliverable
----------------------------
The smoothed 1995-2025 sequence contains 18 regime runs with a median length of
5 months; 10 of the 18 last 6 months or fewer. Between 2011-11 and 2013-10 the
label flips six times, with runs of 2, 1, 3, 1, 2 and 3 months. A system that
rebalanced whenever regime_change_detected went True would have turned the book
over six times in twenty months on what is, structurally, one long stretch of
post-GFC normalisation.

How the verdict is reached
--------------------------
Four persona-independent evidence gates, then one persona-conditioned
materiality gate. All five are deterministic; no LLM participates.

  persistence          the new label has held at least MIN_DWELL_MONTHS months
  confidence           mean XGBoost probability across the run clears CONFIDENCE_FLOOR
  structural_break     a PELT break sits within BREAK_TOLERANCE_MONTHS of the shift date
  reversion_base_rate  this transition has not historically reverted more than
                       MAX_REVERSION_RATE of the time
  materiality          the client's own equity target moves by at least
                       MIN_MATERIAL_DELTA once the regime tilt is applied

The structural_break gate is the one that does the most work: the PELT breaks are
computed from the multivariate signal matrix independently of the XGBoost labels,
so a label flip with no break behind it is a classifier wobble inside a stable
macro environment rather than a change of regime.

The materiality gate is what makes the answer client-specific. The regime tilt
(agents/research/regime_returns.py) is a fraction of the client's own
portfolio_equity_target, so a client whose career already consumes their entire
equity budget — high implicit_equity_exposure, target near zero — cannot have a
material rebalance no matter how real the regime change is. There is nothing to
move. The same transition is justified for the biology professor and churn for
the tech executive, and that asymmetry is the point of the total-wealth model.

Units
-----
portfolio_equity_target arrives in TOTAL-WEALTH units (it is
effective_risk_budget − implicit_equity_exposure, both fractions of financial +
human capital). Turnover is paid in FINANCIAL-wealth units, so materiality is
judged after converting:

    financial_units = total_wealth_units × (financial + human) / financial

This is the same conversion applied in agents/shared/core/allocation.py's
_max_employer_financial_weight. It is applied here locally so this module's
numbers are correct on their own terms; it is not a fix to the allocation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from contracts import (
    MacroRegimeSnapshot,
    ProfileAgentOutput,
    RebalanceDecision,
    RebalanceEvaluation,
    RegimeChangeEvidence,
)
from agents.research.regime_returns import RegimeReturnStats, regime_tilt

# ── Evidence gate thresholds ───────────────────────────────────────────────

MIN_DWELL_MONTHS = 3
"""
Months the new label must hold before the change is actionable. Set against the
observed run-length distribution: 6 of 18 historical runs are shorter than this,
and every one of them sits inside the 2011-2013 churn cluster.
"""

CONFIDENCE_FLOOR = 0.60
"""Mean class probability across the run. Matches MacroRegimeSnapshot.is_low_confidence."""

BREAK_TOLERANCE_MONTHS = 6
"""
How far a PELT structural break may sit from the label shift date and still
corroborate it. The label path is smoothed by a centred 6-month majority vote,
so a shift date can legitimately lag the underlying break by up to half a window.
"""

MAX_REVERSION_RATE = 0.50
"""
Reject when a transition historically reverted more often than this. A coin-flip
transition carries no information about the destination being durable.
"""

REVERSION_WINDOW_MONTHS = 12
"""Horizon over which a transition counts as having reverted."""

MIN_TRANSITION_SAMPLE = 2
"""
Occurrences of a specific prior→current transition needed before its own
reversion rate is used. Below this the base rate falls back to the destination
regime's short-run frequency, which is better powered (2-5 runs per regime
versus 1-3 per transition).
"""

MIN_DESTINATION_SAMPLE = 3
"""
Prior runs of the destination regime needed before the fallback base rate is
trusted. Without this floor a single short prior run yields a reversion rate of
100% and vetoes the transition outright — which is what happened to the March
2008 move into Financial Crisis & ZLB, a transition no one would call churn.
Below the floor the base rate is reported as unavailable and the gate passes:
absence of evidence is not evidence of churn.
"""

MIN_MATERIAL_DELTA = 0.02
"""
Minimum absolute change in the equity target, in financial-wealth units, worth
trading on. Two percentage points of the portfolio: below that, turnover and
spread consume the benefit of the tilt.
"""


@dataclass(frozen=True)
class RegimeRun:
    """One contiguous stretch of a single regime label."""

    label:  str
    start:  pd.Timestamp
    end:    pd.Timestamp
    months: int


# ---------------------------------------------------------------------------
# Sequence statistics
# ---------------------------------------------------------------------------

def regime_runs(labels: pd.Series) -> list[RegimeRun]:
    """Collapse a month-by-month label series into contiguous runs."""
    labels = labels.sort_index()
    if labels.empty:
        return []

    runs: list[RegimeRun] = []
    current = labels.iloc[0]
    start   = labels.index[0]
    prev    = labels.index[0]
    months  = 1

    for stamp, label in list(labels.items())[1:]:
        if label != current:
            runs.append(RegimeRun(current, start, prev, months))
            current, start, months = label, stamp, 1
        else:
            months += 1
        prev = stamp

    runs.append(RegimeRun(current, start, labels.index[-1], months))
    return runs


def transition_reversion_rate(
    runs: list[RegimeRun],
    prior_regime: str,
    current_regime: str,
    window_months: int = REVERSION_WINDOW_MONTHS,
    exclude_last: bool = True,
) -> tuple[float | None, int]:
    """
    Historical reversion rate for a specific prior→current transition.

    A transition reverts when the sequence returns to `prior_regime` within
    `window_months` of the switch. Returns (rate, n_occurrences); rate is None
    when the transition has never been observed.

    `exclude_last` drops the final run so the transition currently under
    evaluation does not vote on its own base rate.
    """
    considered = runs[:-1] if exclude_last and len(runs) > 1 else runs

    occurrences = reversions = 0
    for i in range(1, len(considered)):
        if considered[i - 1].label != prior_regime or considered[i].label != current_regime:
            continue
        occurrences += 1

        # Walk forward from the switch until the window closes.
        elapsed = 0
        for j in range(i, len(considered)):
            if j > i and considered[j].label == prior_regime and elapsed <= window_months:
                reversions += 1
                break
            elapsed += considered[j].months
            if elapsed > window_months:
                break

    if occurrences == 0:
        return None, 0
    return reversions / occurrences, occurrences


def destination_short_run_rate(
    runs: list[RegimeRun],
    regime: str,
    window_months: int = REVERSION_WINDOW_MONTHS,
    exclude_last: bool = True,
) -> float | None:
    """
    Fallback base rate: the fraction of past runs of `regime` that lasted less
    than `window_months`. Used when a specific transition is too rare to have a
    reversion rate of its own.
    """
    considered = runs[:-1] if exclude_last and len(runs) > 1 else runs
    lengths = [r.months for r in considered if r.label == regime]
    if len(lengths) < MIN_DESTINATION_SAMPLE:
        return None
    return sum(1 for m in lengths if m < window_months) / len(lengths)


def months_since(shift_date: date, as_of: date) -> int:
    """Whole months between two dates, floored at zero."""
    months = (as_of.year - shift_date.year) * 12 + (as_of.month - shift_date.month)
    return max(0, months) + 1  # inclusive of the shift month itself


def nearest_break_distance(
    shift_date: date, break_dates: list[pd.Timestamp]
) -> int | None:
    """Months from `shift_date` to the closest PELT break, or None when none exist."""
    if not break_dates:
        return None
    distances = [
        abs((pd.Timestamp(b).year - shift_date.year) * 12
            + (pd.Timestamp(b).month - shift_date.month))
        for b in break_dates
    ]
    return int(min(distances))


# ---------------------------------------------------------------------------
# Evidence gates (persona-independent)
# ---------------------------------------------------------------------------

def build_evidence(
    labels:      pd.Series,
    confidence:  pd.Series,
    break_dates: list[pd.Timestamp],
    regime_label:      str,
    prior_regime:      str,
    regime_shift_date: date,
    as_of:             date,
) -> RegimeChangeEvidence:
    """
    Run the four persona-independent evidence gates.

    Parameters
    ----------
    labels : pd.Series
        DatetimeIndex → smoothed regime label, full history.
    confidence : pd.Series
        DatetimeIndex → per-month XGBoost max class probability.
    break_dates : list[pd.Timestamp]
        PELT structural breaks, from detect_change_points(). Computed from the
        signal matrix independently of the labels — this independence is what
        makes the structural_break gate informative.

    Point-in-time discipline
    ------------------------
    Everything is truncated at `as_of` before any gate is evaluated, so a
    historical replay (rebalance_backtest.py) cannot see the future it is being
    asked to predict. One in-sample artefact remains and is not removable here:
    PELT and XGBoost are both fitted on the full sample upstream, so the break
    locations and label path at date t embed information from after t. That
    inflates the structural_break and confidence gates in the backtest and is
    disclosed rather than hidden — a walk-forward refit is future work.
    """
    as_of_ts = pd.Timestamp(as_of)

    # Point-in-time truncation — no gate may see past `as_of`.
    labels     = labels[labels.index <= as_of_ts].sort_index()
    confidence = confidence[confidence.index <= as_of_ts].sort_index()
    break_dates = [b for b in break_dates if pd.Timestamp(b) <= as_of_ts]

    runs = regime_runs(labels)

    months_in_regime = months_since(regime_shift_date, as_of)

    run_mask = (labels.index >= pd.Timestamp(regime_shift_date)) & (labels.index <= as_of_ts)
    run_conf = confidence[confidence.index.isin(labels.index[run_mask])]
    run_confidence = float(run_conf.mean()) if len(run_conf) else 0.0
    run_confidence = float(np.clip(run_confidence, 0.0, 1.0))

    break_distance = nearest_break_distance(regime_shift_date, break_dates)
    corroborated = break_distance is not None and break_distance <= BREAK_TOLERANCE_MONTHS

    reversion_rate, sample_size = transition_reversion_rate(
        runs, prior_regime, regime_label
    )
    used_fallback = sample_size < MIN_TRANSITION_SAMPLE
    if used_fallback:
        reversion_rate = destination_short_run_rate(runs, regime_label)

    passed: list[str] = []
    failed: list[str] = []

    def gate(name: str, ok: bool) -> None:
        (passed if ok else failed).append(name)

    gate("persistence", months_in_regime >= MIN_DWELL_MONTHS)
    gate("confidence", run_confidence >= CONFIDENCE_FLOOR)
    gate("structural_break", corroborated)
    # No base rate at all (a genuinely unprecedented destination) is not evidence
    # against the change — the gate passes and the absence shows up in the
    # explanation via transition_sample_size.
    gate(
        "reversion_base_rate",
        reversion_rate is None or reversion_rate <= MAX_REVERSION_RATE,
    )

    return RegimeChangeEvidence(
        months_in_regime          = months_in_regime,
        min_dwell_months          = MIN_DWELL_MONTHS,
        run_confidence            = round(run_confidence, 4),
        pelt_corroborated         = corroborated,
        months_to_nearest_break   = break_distance,
        historical_reversion_rate = None if reversion_rate is None else round(reversion_rate, 4),
        transition_sample_size    = 0 if used_fallback else sample_size,
        gates_passed              = passed,
        gates_failed              = failed,
    )


# ---------------------------------------------------------------------------
# Materiality gate (persona-conditioned) and the verdict
# ---------------------------------------------------------------------------

def _equity_target_total_wealth(profile: ProfileAgentOutput) -> float:
    """
    The client's equity target in TOTAL-WEALTH units.

    Mirrors the Allocation Agent's fallback: recompute from effective_risk_budget
    and implicit_equity_exposure when the profile does not carry the field.
    """
    if profile.portfolio_equity_target is not None:
        return float(profile.portfolio_equity_target)
    return float(profile.effective_risk_budget - profile.implicit_equity_exposure)


def to_financial_units(target_total_wealth: float, profile: ProfileAgentOutput) -> float:
    """
    Convert a total-wealth equity target into an achievable portfolio weight.

        financial = total_wealth_units × (financial + human) / financial

    then clipped to [0, 1]. Turnover is paid out of financial wealth, so the
    materiality gate has to judge the trade in these units — and the clip is not
    cosmetic, because the portfolio genuinely cannot hold less than 0% or more
    than 100% equity (AllocationAgentOutput forbids shorts and requires weights
    to sum to 1).

    The clip binds hard for every current BLS persona. With human capital 6-29x
    financial capital, a total-wealth target of +0.48 maps to +8.9 in financial
    units and a target of −0.25 maps to −6.7. Both saturate. See
    corner_solution() — this is the dominant finding of the rebalance work, not
    an artefact of the conversion.
    """
    financial = profile.financial_capital
    human     = profile.human_capital_valuation
    if financial <= 0:
        return 0.0
    return float(np.clip(target_total_wealth * (financial + human) / financial, 0.0, 1.0))


def corner_solution(profile: ProfileAgentOutput) -> str | None:
    """
    Report whether the client's equity target saturates, and at which bound.

    Returns "fully invested" (pinned at 100% equity), "fully defensive" (pinned
    at 0%), or None when the target sits strictly inside the achievable range
    and a regime tilt can actually move the book.

    A client at a corner cannot be rebalanced on a regime signal of any size:
    the constraint, not the regime, is setting their allocation. Reporting this
    is more useful than reporting a zero delta, because the two have completely
    different causes.
    """
    raw = _equity_target_total_wealth(profile)
    financial, human = profile.financial_capital, profile.human_capital_valuation
    if financial <= 0:
        return "fully defensive"

    unclipped = raw * (financial + human) / financial
    # Tilts are bounded by TILT_CAP, so a target beyond this margin cannot be
    # pulled back inside the achievable range by any regime.
    from agents.research.regime_returns import TILT_CAP

    if unclipped >= 1.0 / (1.0 - TILT_CAP):
        return "fully invested"
    if unclipped <= 0.0:
        return "fully defensive"
    return None


def evaluate_rebalance(
    snapshot:     MacroRegimeSnapshot,
    profile:      ProfileAgentOutput,
    regime_stats: dict[str, RegimeReturnStats],
    evidence:     RegimeChangeEvidence | None = None,
) -> RebalanceEvaluation:
    """
    Answer "justified or churn?" for one client under one regime snapshot.

    Parameters
    ----------
    snapshot : MacroRegimeSnapshot
        Must carry regime_change_evidence, or `evidence` must be supplied.
    profile : ProfileAgentOutput
        Supplies the client's own equity target — the persona conditioning.
    regime_stats : dict[str, RegimeReturnStats]
        CRSP-derived per-regime tilts from compute_regime_stats(). An empty dict
        yields zero tilts, so nothing is ever material and every change is
        reported as churn rather than acted on with a fabricated number.

    Returns
    -------
    RebalanceEvaluation
    """
    evidence = evidence or snapshot.regime_change_evidence
    if evidence is None:
        raise ValueError(
            "evaluate_rebalance requires regime change evidence — either attach it to "
            "the snapshot (run_research_agent does this) or pass it explicitly."
        )

    prior   = snapshot.prior_regime
    current = snapshot.regime_label

    # Tilts are fractions of the client's own equity target, so the whole
    # materiality question scales with how much room the client actually has.
    target_tw   = _equity_target_total_wealth(profile)
    current_eq  = round(
        to_financial_units(target_tw * (1.0 + regime_tilt(regime_stats, prior)), profile), 6
    )
    proposed_eq = round(
        to_financial_units(target_tw * (1.0 + regime_tilt(regime_stats, current)), profile), 6
    )
    delta       = proposed_eq - current_eq
    is_material = abs(delta) >= MIN_MATERIAL_DELTA

    if not snapshot.regime_change_detected:
        decision = RebalanceDecision.NO_CHANGE
    elif not evidence.evidence_supports_change:
        decision = RebalanceDecision.CHURN
    elif not regime_stats:
        # Evidence is sound but no tilt could be derived — say so instead of
        # silently passing (review §7 on unmapped regimes failing loud).
        decision = RebalanceDecision.INSUFFICIENT_HISTORY
    elif not is_material:
        decision = RebalanceDecision.CHURN
    else:
        decision = RebalanceDecision.JUSTIFIED

    return RebalanceEvaluation(
        client_id              = profile.client_id,
        prior_regime           = prior,
        current_regime         = current,
        decision               = decision,
        evidence               = evidence,
        current_equity_target  = current_eq,
        proposed_equity_target = proposed_eq,
        equity_target_delta    = delta,
        materiality_threshold  = MIN_MATERIAL_DELTA,
        is_material            = is_material,
        explanation            = _explain(
            decision, prior, current, evidence, delta, profile, corner_solution(profile)
        ),
    )


def _explain(
    decision:  RebalanceDecision,
    prior:     str,
    current:   str,
    evidence:  RegimeChangeEvidence,
    delta:     float,
    profile:   ProfileAgentOutput,
    corner:    str | None = None,
) -> str:
    """Template the verdict from computed fields. Deterministic — not LLM prose."""
    if decision == RebalanceDecision.NO_CHANGE:
        return (
            f"No regime change: {current} has held for "
            f"{evidence.months_in_regime} months. No rebalance evaluated."
        )

    head = f"{prior} → {current}: "

    if decision == RebalanceDecision.CHURN and evidence.gates_failed:
        reasons = []
        if "persistence" in evidence.gates_failed:
            reasons.append(
                f"the new label has held only {evidence.months_in_regime} month(s), "
                f"short of the {evidence.min_dwell_months}-month minimum"
            )
        if "confidence" in evidence.gates_failed:
            reasons.append(
                f"mean classifier confidence across the run is "
                f"{evidence.run_confidence:.0%}, below the {CONFIDENCE_FLOOR:.0%} floor"
            )
        if "structural_break" in evidence.gates_failed:
            near = (
                f"nearest is {evidence.months_to_nearest_break} months away"
                if evidence.months_to_nearest_break is not None
                else "no structural breaks were detected at all"
            )
            reasons.append(
                f"no PELT structural break corroborates the shift ({near}) — the macro "
                f"signal matrix did not move, so this is a classifier wobble"
            )
        if "reversion_base_rate" in evidence.gates_failed:
            basis = (
                f"across {evidence.transition_sample_size} past occurrence(s)"
                if evidence.transition_sample_size
                else "based on the destination regime's historical run lengths"
            )
            reasons.append(
                f"this transition reverted {evidence.historical_reversion_rate:.0%} of the "
                f"time within {REVERSION_WINDOW_MONTHS} months {basis}"
            )
        return head + "CHURN — " + "; ".join(reasons) + ". No trade recommended."

    if decision == RebalanceDecision.CHURN:
        if corner is not None:
            hc_multiple = profile.human_capital_valuation / profile.financial_capital
            return (
                head + f"CHURN — the change is structurally supported, but "
                f"{profile.client_id} is at a corner solution ({corner}). Human capital is "
                f"{hc_multiple:.0f}x financial capital, so the total-wealth equity target of "
                f"{_equity_target_total_wealth(profile):+.2f} saturates the achievable "
                f"0-100% portfolio range. No regime tilt of any size moves this book — the "
                f"constraint is setting the allocation, not the regime."
            )
        return (
            head + f"CHURN — the change is structurally supported, but "
            f"{profile.client_id}'s equity target moves only {delta:+.2%} of financial "
            f"wealth, under the {MIN_MATERIAL_DELTA:.0%} threshold. The client's implicit "
            f"equity exposure of {profile.implicit_equity_exposure:.0%} leaves too little "
            f"portfolio equity capacity for the tilt to be worth the turnover."
        )

    if decision == RebalanceDecision.INSUFFICIENT_HISTORY:
        return (
            head + "INSUFFICIENT HISTORY — the change is structurally supported, but no "
            "regime-conditional market statistics were available to size a tilt. "
            "Not treated as justified."
        )

    return (
        head + f"JUSTIFIED — held {evidence.months_in_regime} months at "
        f"{evidence.run_confidence:.0%} mean confidence, corroborated by a PELT structural "
        f"break {evidence.months_to_nearest_break} month(s) from the shift date. "
        f"{profile.client_id}'s equity target moves {delta:+.2%} of financial wealth, "
        f"above the {MIN_MATERIAL_DELTA:.0%} turnover threshold."
    )
