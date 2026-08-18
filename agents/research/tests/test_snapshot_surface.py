"""
Tests for the pruned MacroRegimeSnapshot surface (4 Aug minutes, Research item 5).

The six raw FRED signals are deprecated. They cannot be deleted from the contract
yet — tests/test_research.py and tests/test_compliance.py still construct
snapshots with them and tests/ is outside this edit scope — so these tests pin the
half that is done: nothing in production populates them any more, and they are
absent from the artifact on disk.

Run with `uv run pytest agents/research/tests`.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from contracts import MacroRegimeSnapshot

_MINIMAL = dict(
    as_of             = date(2025, 12, 1),
    regime_label      = "Late-Cycle Expansion",
    prior_regime      = "Late-Cycle Expansion",
    regime_shift_date = date(2023, 8, 1),
    regime_confidence = 0.902,
    regime_volatility = 0.0549,
)


class TestDeprecatedFieldsAreOptional:
    def test_snapshot_builds_without_them(self):
        """The whole point: new code need not supply the deprecated signals."""
        snap = MacroRegimeSnapshot(**_MINIMAL)
        assert snap.regime_label == "Late-Cycle Expansion"
        assert all(
            getattr(snap, f) is None for f in MacroRegimeSnapshot.DEPRECATED_FIELDS
        )

    def test_old_callers_still_work(self):
        """Backward compatible — the tests/ fixtures that pass them must not break."""
        snap = MacroRegimeSnapshot(
            **_MINIMAL,
            yield_curve=0.71, term_spread=0.51, fed_funds=3.72,
            unemployment=4.4, cpi=2.6533, credit_spread=1.72,
        )
        assert snap.credit_spread == 1.72

    def test_deprecated_set_matches_the_documented_six(self):
        assert MacroRegimeSnapshot.DEPRECATED_FIELDS == {
            "yield_curve", "term_spread", "fed_funds",
            "unemployment", "cpi", "credit_spread",
        }

    def test_derived_flags_still_computed(self):
        """Dropping the raw signals must not disturb the flags the orchestrator reads."""
        snap = MacroRegimeSnapshot(**{**_MINIMAL, "regime_confidence": 0.42,
                                      "prior_regime": "Inflation Shock"})
        assert snap.is_low_confidence
        assert snap.regime_change_detected

    def test_for_allocation_is_the_intended_surface(self):
        assert MacroRegimeSnapshot(**_MINIMAL).for_allocation() == {
            "regime_label":      "Late-Cycle Expansion",
            "regime_confidence": 0.902,
            "regime_volatility": 0.0549,
        }


class TestSerialisedSnapshot:
    def test_excluding_deprecated_fields_omits_rather_than_nulls(self):
        payload = json.loads(
            MacroRegimeSnapshot(**_MINIMAL).model_dump_json(
                exclude=MacroRegimeSnapshot.DEPRECATED_FIELDS
            )
        )
        for name in MacroRegimeSnapshot.DEPRECATED_FIELDS:
            assert name not in payload
        assert payload["regime_label"] == "Late-Cycle Expansion"

    def test_written_snapshot_on_disk_is_pruned(self):
        """The artifact the minutes name explicitly: macro_regime_snapshot.json."""
        from agents.research.research_agent import SNAPSHOT_OUTPUT

        if not SNAPSHOT_OUTPUT.exists():
            pytest.skip("no snapshot written yet — run the Research Agent first")

        payload = json.loads(SNAPSHOT_OUTPUT.read_text())
        present = MacroRegimeSnapshot.DEPRECATED_FIELDS & payload.keys()
        assert not present, f"deprecated fields still on disk: {sorted(present)}"
        for required in ("regime_label", "regime_confidence", "regime_volatility"):
            assert required in payload

    def test_disk_snapshot_still_validates(self):
        from agents.research.research_agent import SNAPSHOT_OUTPUT

        if not SNAPSHOT_OUTPUT.exists():
            pytest.skip("no snapshot written yet — run the Research Agent first")

        MacroRegimeSnapshot.model_validate(json.loads(SNAPSHOT_OUTPUT.read_text()))


class TestBuildSnapshotDoesNotPopulateThem:
    def test_adapter_leaves_deprecated_fields_unset(self):
        """
        build_snapshot() is the only production constructor. If it stops passing
        the deprecated fields, no new consumer can appear before they are deleted.
        """
        import inspect

        from agents.research import adapters

        source = inspect.getsource(adapters.build_snapshot)
        constructor = source.split("return MacroRegimeSnapshot(", 1)[1]
        for name in MacroRegimeSnapshot.DEPRECATED_FIELDS:
            assert f"{name} " not in constructor and f"{name}=" not in constructor, (
                f"build_snapshot() still populates deprecated field {name!r}"
            )
