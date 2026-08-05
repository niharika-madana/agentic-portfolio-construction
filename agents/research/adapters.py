"""
adapters.py — Pydantic validation and the downstream MacroRegimeSnapshot.

Two schemas:

  RegimeRecord          research-internal per-row schema; validates every month
                        of the smoothed regime sequence before it is written to
                        disk. Carries vix and indpro for inspection.
  MacroRegimeSnapshot   the shared downstream contract (contracts.py), built for
                        the most recent month and consumed by the Allocation and
                        Risk agents via the orchestrator. Its is_low_confidence
                        and regime_change_detected flags are derived by the
                        contract's own validator.

add_derived_fields() computes regime_volatility (6-month rolling std of the
credit spread), prior_regime, and regime_shift_date on the feature frame.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from contracts import MacroRegimeSnapshot

VALID_REGIME_LABELS = {
    "Early Recovery",
    "Late-Cycle Expansion",
    "Financial Crisis & ZLB",
    "Moderate Expansion",
    "Inflation Shock",
}


class RegimeRecord(BaseModel):
    """Per-row validation schema for the full regime sequence."""
    regime_label:      str
    prior_regime:      str
    regime_shift_date: str
    regime_confidence: float = Field(ge=0.0, le=1.0)
    regime_volatility: float = Field(ge=0.0)
    yield_curve:       float
    term_spread:       float
    fed_funds:         float
    unemployment:      float
    cpi:               float
    credit_spread:     float
    vix:               float
    indpro:            float

    @field_validator("regime_label")
    @classmethod
    def _label_valid(cls, v):
        if v not in VALID_REGIME_LABELS:
            raise ValueError(f"Invalid regime label: '{v}'")
        return v

    @field_validator("prior_regime")
    @classmethod
    def _prior_valid(cls, v):
        if v != "None" and v not in VALID_REGIME_LABELS:
            raise ValueError(f"Invalid prior_regime: '{v}'")
        return v


def add_derived_fields(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime_volatility, prior_regime, and regime_shift_date to the smoothed
    feature frame. Requires regime_label_smoothed to be present.
    """
    features_df["regime_volatility"] = (
        features_df["credit_spread"].rolling(6, min_periods=1).std().fillna(0).round(4)
    )
    features_df["prior_regime"] = (
        features_df["regime_label_smoothed"].shift(1).fillna("None")
    )

    shift_dates, current_start = [], features_df.index[0]
    smoothed = features_df["regime_label_smoothed"]
    for i, date_idx in enumerate(features_df.index):
        if i == 0:
            current_start = date_idx
        elif smoothed.iloc[i] != smoothed.iloc[i - 1]:
            current_start = date_idx
        shift_dates.append(current_start)
    features_df["regime_shift_date"] = shift_dates
    return features_df


def build_regime_sequence(features_df: pd.DataFrame) -> dict[str, dict]:
    """
    Validate every month against RegimeRecord and return a date-keyed dict of
    validated records. Rows that fail validation are logged and skipped.
    """
    regime_sequence: dict[str, dict] = {}
    passed = failed = 0

    for date_idx, row in features_df.iterrows():
        record = {
            "regime_label":      row["regime_label_smoothed"],
            "prior_regime":      row["prior_regime"],
            "regime_shift_date": str(pd.Timestamp(row["regime_shift_date"]).date()),
            "regime_confidence": round(float(row["regime_confidence"]), 3),
            "regime_volatility": round(float(row["regime_volatility"]), 4),
            "yield_curve":       round(float(row["yield_curve"]), 4),
            "term_spread":       round(float(row["term_spread"]), 4),
            "fed_funds":         round(float(row["fed_funds"]), 4),
            "unemployment":      round(float(row["unemployment"]), 4),
            "cpi":               round(float(row["cpi"]), 4),
            "credit_spread":     round(float(row["credit_spread"]), 4),
            "vix":               round(float(row["vix"]), 4),
            "indpro":            round(float(row["indpro"]), 4),
        }
        try:
            RegimeRecord(**record)
            regime_sequence[str(date_idx.date())] = record
            passed += 1
        except Exception as e:
            print(f"  VALIDATION FAILED [{date_idx.date()}]: {e}")
            failed += 1

    print(f"Validation complete — Passed: {passed} | Failed: {failed}")
    return regime_sequence


def build_snapshot(
    features_df: pd.DataFrame,
    break_dates: list | None = None,
) -> MacroRegimeSnapshot:
    """
    Build the most-recent-month MacroRegimeSnapshot (contracts.py) for the
    orchestrator. is_low_confidence and regime_change_detected are derived by
    the contract's validator.

    When `break_dates` (the PELT structural breaks from detect_change_points) is
    supplied, the snapshot also carries regime_change_evidence — the four
    persona-independent gates that decide whether a detected change is worth
    acting on. Without the breaks the structural_break gate cannot be evaluated,
    so the evidence block is omitted rather than half-filled.
    """
    last_date = features_df.index[-1]
    row = features_df.loc[last_date]

    evidence = None
    if break_dates is not None:
        from agents.research.rebalance import build_evidence

        evidence = build_evidence(
            labels            = features_df["regime_label_smoothed"],
            confidence        = features_df["regime_confidence"],
            break_dates       = list(break_dates),
            regime_label      = row["regime_label_smoothed"],
            prior_regime      = row["prior_regime"],
            regime_shift_date = pd.Timestamp(row["regime_shift_date"]).date(),
            as_of             = pd.Timestamp(last_date).date(),
        )

    # The six raw FRED signals (yield_curve, term_spread, fed_funds, unemployment,
    # cpi, credit_spread) are deliberately NOT passed. They are deprecated on the
    # contract and consumed by nothing downstream — the 24 Jul and 4 Aug minutes
    # both ask this snapshot to carry only label, confidence and volatility. The
    # full signal matrix stays available in RegimeRecord, regime_sequence.json and
    # data/outputs/fred_macro_regimes.csv, so nothing is lost by omitting them
    # here; omitting them is what stops new consumers appearing before the fields
    # can be deleted outright.
    return MacroRegimeSnapshot(
        regime_change_evidence = evidence,
        as_of             = pd.Timestamp(last_date).date(),
        regime_label      = row["regime_label_smoothed"],
        prior_regime      = row["prior_regime"],
        regime_shift_date = pd.Timestamp(row["regime_shift_date"]).date(),
        regime_confidence = round(float(row["regime_confidence"]), 3),
        regime_volatility = round(float(row["regime_volatility"]), 4),
    )
