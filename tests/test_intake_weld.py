"""
Verify the intake -> profile weld.

Classified client statements (mandates like "no weapons") must survive from the
intake layer onto ProfileAgentOutput.client_statements, which is what the
orchestrator threads into ComplianceInput and Compliance Job 3 reads. Without the
weld the statements stop at the BridgeResult and never reach compliance.

No API key needed — the ExtractedProfile is built directly, standing in for what
an extractor would return.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contracts import (
    ClientStatement,
    ExtractedField,
    ExtractedProfile,
    FactSource,
    PipelineDestination,
    StatementKind,
)
from agents.profile.intake_bridge import build_profile_from_intake


def _field(name, value):
    return ExtractedField(
        name=name, value=value, source=FactSource.STATED,
        confidence=0.9, evidence_quote=f"stated {name}",
    )


def _required_fields():
    # The four fields the bridge refuses to build without.
    return {
        "age":               _field("age", 38),
        "annual_salary":     _field("annual_salary", 180000.0),
        "financial_capital": _field("financial_capital", 300000.0),
        "income_stability":  _field("income_stability", "Medium"),
    }


def _extracted(statements):
    return ExtractedProfile(
        client_id="test_client", transcript_id="t1", extractor="manual",
        fields=_required_fields(), statements=statements,
    )


def test_mandate_reaches_profile_output():
    mandate = ClientStatement(
        kind=StatementKind.HARD_CONSTRAINT,
        summary="Client excludes weapons manufacturers",
        quote="I really don't want to invest in weapons companies.",
        destinations=[PipelineDestination.UNIVERSE_EXCLUSION],
        subject="weapons",
    )
    result = build_profile_from_intake(_extracted([mandate]), discount_rate=0.044)

    assert result.built
    assert len(result.profile.client_statements) == 1
    carried = result.profile.client_statements[0]
    assert carried.subject == "weapons"
    assert carried.kind == StatementKind.HARD_CONSTRAINT


def test_no_statements_yields_empty_list():
    result = build_profile_from_intake(_extracted([]), discount_rate=0.044)
    assert result.built
    assert result.profile.client_statements == []
