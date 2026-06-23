"""
adapters.py — Research Agent
==============================
Converts the Research Agent's internal regime_sequence dict output
into a contracts.py MacroRegimeSnapshot for the orchestrator.

This is the Research Agent's equivalent of the Profile Agent's
to_profile_agent_output() — it bridges internal data structures
to the typed inter-agent contract.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import date

from contracts import MacroRegimeSnapshot


def to_macro_regime_snapshot(
    regime_sequence: dict,
    as_of: date = None,
) -> MacroRegimeSnapshot:
    """
    Convert the Research Agent's regime_sequence dict to a MacroRegimeSnapshot.

    The regime_sequence is date-keyed (YYYY-MM-DD strings → regime record dicts),
    produced by build_regime_sequence() and validated against RegimeRecord before
    being written to disk. This function picks the most recent validated entry
    and maps it to the contracts.py MacroRegimeSnapshot type.

    Args:
        regime_sequence: date-keyed dict from build_regime_sequence()
        as_of:           snapshot date (defaults to the latest key in the dict)

    Returns:
        MacroRegimeSnapshot — validated, ready for the orchestrator's run_pipeline()

    Raises:
        ValueError  — if regime_sequence is empty
        KeyError    — if as_of date is not present in regime_sequence
        ValidationError — if the record fails MacroRegimeSnapshot constraints
    """
    if not regime_sequence:
        raise ValueError(
            "regime_sequence is empty — Research Agent produced no validated rows. "
            "Check that pull_fred_data() succeeded and build_regime_sequence() "
            "did not fail all Pydantic validation rows."
        )

    if as_of is None:
        latest_key = max(regime_sequence.keys())
    else:
        latest_key = as_of.isoformat()
        if latest_key not in regime_sequence:
            raise KeyError(
                f"Date {latest_key} not found in regime_sequence. "
                f"Available range: {min(regime_sequence)} → {max(regime_sequence)}"
            )

    row           = regime_sequence[latest_key]
    snapshot_date = date.fromisoformat(latest_key)

    # MacroRegimeSnapshot's @model_validator sets is_low_confidence and
    # regime_change_detected automatically from confidence and prior_regime —
    # no need to pass them.
    return MacroRegimeSnapshot(
        as_of             = snapshot_date,
        regime_label      = row["regime_label"],
        prior_regime      = row["prior_regime"],
        regime_shift_date = date.fromisoformat(row["regime_shift_date"]),
        regime_confidence = row["regime_confidence"],
        regime_volatility = row["regime_volatility"],
        yield_curve       = row["yield_curve"],
        term_spread       = row["term_spread"],
        fed_funds         = row["fed_funds"],
        unemployment      = row["unemployment"],
        cpi               = row["cpi"],
        credit_spread     = row["credit_spread"],
        # vix and indpro are in RegimeRecord but not in MacroRegimeSnapshot
    )


def load_regime_snapshot_from_json(
    json_path: str,
    as_of: date = None,
) -> MacroRegimeSnapshot:
    """
    Load a saved regime_sequence.json from disk and return the snapshot
    for the given date (or the latest date if as_of is None).

    Convenience wrapper for the orchestrator when the Research Agent
    has already run and saved its output.
    """
    import json

    with open(json_path, "r") as f:
        regime_sequence = json.load(f)

    return to_macro_regime_snapshot(regime_sequence, as_of=as_of)
