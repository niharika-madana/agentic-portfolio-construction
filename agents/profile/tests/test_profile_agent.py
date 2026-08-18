"""
Tests for the llm_role provenance flag, validate=True, and the persona archive.

Run with `uv run pytest agents/profile/tests` — see test_sector_guard.py for why
these live here rather than under tests/.
"""

from __future__ import annotations

import json

import pytest

from contracts import LLMRole, ProfileAgentOutput
from agents.profile import profile_agent
from agents.profile.profile_agent import (
    ProfileValidationError,
    archive_profiles,
    assert_complete,
    build_profiles,
    expected_persona_count,
    verify_archive,
)
from agents.profile.profile_model import TARGET_OCCUPATIONS, build_profile

DISCOUNT_RATE = 0.044


def _persona(soc: str, label: str, sector: str, stability: str, **overrides) -> dict:
    base = dict(
        client_id                = f"bls_{soc}_p50",
        career_type              = label,
        age                      = 40,
        annual_salary            = 120_000.0,
        bonus_rate               = 0.05,
        effective_salary         = 126_000.0,
        years_to_retirement      = 25,
        income_stability         = stability,
        industry_exposure_sector = sector,
        financial_capital        = 150_000.0,
        current_holdings         = {"US_equity": 0.50, "intl_equity": 0.15,
                                    "bonds": 0.25, "cash": 0.10},
        investment_horizon_years = 25,
        risk_tolerance           = "moderate",
        liquidity_needs          = "medium",
        investment_objective     = "growth",
        RSU_concentration        = 0.0,
        has_pension              = False,
    )
    base.update(overrides)
    return base


@pytest.fixture
def personas() -> list[dict]:
    """One valid persona per target occupation, so counts line up with the real run."""
    return [
        _persona(occ["soc"], occ["label"], occ["sector"], occ["income_stability"])
        for occ in TARGET_OCCUPATIONS
    ]


@pytest.fixture
def profiles(personas) -> list[ProfileAgentOutput]:
    return build_profiles(personas, DISCOUNT_RATE)


class TestLLMRoleFlag:
    def test_defaults_to_none(self, personas):
        """No model runs on the BLS path, so the profile must not claim one did."""
        profile = build_profile(personas[0], DISCOUNT_RATE)
        assert profile["llm_role"] is LLMRole.NONE

    def test_survives_validation_and_serialisation(self, personas):
        out = ProfileAgentOutput(**build_profile(personas[0], DISCOUNT_RATE))
        assert out.llm_role is LLMRole.NONE
        assert out.model_dump(mode="json")["llm_role"] == "none"

    def test_creator_is_recorded_when_passed(self, personas):
        profile = build_profile(personas[0], DISCOUNT_RATE, llm_role=LLMRole.CREATOR)
        assert ProfileAgentOutput(**profile).llm_role is LLMRole.CREATOR

    def test_llm_role_does_not_change_any_number(self, personas):
        """The flag is provenance. It must not touch a computed field."""
        plain   = build_profile(personas[0], DISCOUNT_RATE)
        flagged = build_profile(personas[0], DISCOUNT_RATE, llm_role=LLMRole.CREATOR)
        assert {k: v for k, v in plain.items() if k != "llm_role"} == {
            k: v for k, v in flagged.items() if k != "llm_role"
        }

    def test_rule_based_extractor_reports_no_model(self):
        from agents.profile.intake_bridge import _llm_role_for

        assert _llm_role_for("rule_based") is LLMRole.NONE

    @pytest.mark.parametrize("extractor", ["structured", "naive"])
    def test_llm_extractors_report_creator(self, extractor):
        from agents.profile.intake_bridge import _llm_role_for

        assert _llm_role_for(extractor) is LLMRole.CREATOR

    def test_unknown_extractor_over_reports_rather_than_under_reports(self):
        """Wrong in the safe direction for an audit trail."""
        from agents.profile.intake_bridge import _llm_role_for

        assert _llm_role_for("some_future_extractor") is LLMRole.CREATOR

    def test_every_extractor_declares_uses_llm(self):
        from agents.profile import intake

        for name in ("RuleBasedExtractor", "NaiveExtractor", "StructuredExtractor"):
            assert isinstance(getattr(intake, name).uses_llm, bool)


class TestValidation:
    def test_expected_count_matches_target_occupations(self):
        assert expected_persona_count() == len(TARGET_OCCUPATIONS) == 9
        assert expected_persona_count(include_percentile_variants=True) == 27

    def test_complete_set_passes(self, profiles):
        assert_complete(profiles)  # does not raise

    def test_missing_persona_is_named(self, profiles):
        with pytest.raises(ProfileValidationError, match="bls_15-1252_p50"):
            assert_complete([p for p in profiles if p.client_id != "bls_15-1252_p50"])

    def test_missing_persona_reports_counts(self, profiles):
        with pytest.raises(ProfileValidationError, match="Expected 9 personas, got 8"):
            assert_complete(profiles[:-1])

    def test_unexpected_persona_is_rejected(self, profiles, personas):
        stray = build_profiles(
            [_persona("99-9999", "Astronaut", "Industrials", "Medium")], DISCOUNT_RATE
        )
        with pytest.raises(ProfileValidationError, match="bls_99-9999_p50"):
            assert_complete(profiles + stray)

    def test_strict_build_raises_instead_of_skipping(self, personas):
        """The silent-skip path the 4 Aug minutes call out."""
        broken = personas + [
            _persona("13-2051", "Broken", "Financials", "Medium",
                     client_id="bls_broken_p50",
                     current_holdings={"US_equity": 0.9})  # sums to 0.9, not 1.0
        ]
        # Default: skipped with a printed warning, run still "succeeds".
        assert len(build_profiles(broken, DISCOUNT_RATE)) == len(personas)
        # strict: the same input is a failure.
        with pytest.raises(ProfileValidationError, match="bls_broken_p50"):
            build_profiles(broken, DISCOUNT_RATE, strict=True)

    def test_validate_rejected_on_transcript_path(self):
        with pytest.raises(ValueError, match="transcript path"):
            profile_agent.run_profile_agent(
                transcripts={"c1": "some text"}, validate=True, save=False
            )


class TestArchive:
    @pytest.fixture(autouse=True)
    def _isolate_archive(self, tmp_path, monkeypatch):
        """Never write to the real data/outputs/profiles during a test run."""
        archive = tmp_path / "profiles"
        monkeypatch.setattr(profile_agent, "ARCHIVE_DIR", archive)
        monkeypatch.setattr(profile_agent, "CHECKSUM_FILE", archive / "SHA256SUMS")
        return archive

    def test_writes_one_file_per_persona(self, profiles, _isolate_archive):
        archive_profiles(profiles)
        written = sorted(p.name for p in _isolate_archive.glob("*.json"))
        assert written == sorted(f"{p.client_id}.json" for p in profiles)

    def test_archived_json_round_trips_through_the_contract(self, profiles, _isolate_archive):
        archive_profiles(profiles)
        path = _isolate_archive / f"{profiles[0].client_id}.json"
        restored = ProfileAgentOutput.model_validate(json.loads(path.read_text()))
        assert restored == profiles[0]

    def test_manifest_covers_every_file(self, profiles, _isolate_archive):
        archive_profiles(profiles)
        manifest = (_isolate_archive / "SHA256SUMS").read_text().splitlines()
        assert len(manifest) == len(profiles)
        assert all(len(line.split("  ")[0]) == 64 for line in manifest)

    def test_verify_passes_on_a_fresh_archive(self, profiles):
        archive_profiles(profiles)
        assert verify_archive() == []

    def test_verify_detects_a_modified_profile(self, profiles, _isolate_archive):
        archive_profiles(profiles)
        target = _isolate_archive / f"{profiles[0].client_id}.json"
        target.write_text(target.read_text().replace('"age"', '"AGE"'))
        assert verify_archive() == [target.name]

    def test_verify_detects_a_deleted_profile(self, profiles, _isolate_archive):
        archive_profiles(profiles)
        target = _isolate_archive / f"{profiles[0].client_id}.json"
        target.unlink()
        assert verify_archive() == [target.name]

    def test_hashes_are_stable_across_runs(self, profiles, _isolate_archive):
        archive_profiles(profiles)
        first = (_isolate_archive / "SHA256SUMS").read_text()
        archive_profiles(profiles)
        assert (_isolate_archive / "SHA256SUMS").read_text() == first

    def test_stale_profiles_are_cleared(self, profiles, _isolate_archive):
        archive_profiles(profiles)
        stale = _isolate_archive / "bls_retired_p50.json"
        stale.write_text("{}")
        archive_profiles(profiles)
        assert not stale.exists()

    def test_verify_without_a_manifest_raises(self):
        with pytest.raises(FileNotFoundError):
            verify_archive()
