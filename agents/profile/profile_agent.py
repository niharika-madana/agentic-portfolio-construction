"""
profile_agent.py: Profile Agent entry point (Agent 2 of 5).

run_profile_agent() builds and validates one ProfileAgentOutput per BLS occupation (median salary by default, or p25/p50/p75 with
include_percentile_variants=True) and optionally persists them:

    data/outputs/profiles_all.json        — list of validated profiles
    data/outputs/profiles/<client_id>.json — one archived profile per persona
    data/outputs/profiles/SHA256SUMS      — integrity manifest for the above
    data/storage/profiles_all.parquet     — consumed by Allocation/Risk/Compliance

The returned list[ProfileAgentOutput] feeds straight into orchestrator.run_all(personas, macro).

Pass validate=True to turn the build into an assertion: every expected persona must
be present and must pass Pydantic validation, or the run raises. See
run_profile_agent() for why that is opt-in rather than the default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from contracts import ProfileAgentOutput

from agents.profile.profile_model import (
    TARGET_OCCUPATIONS,
    build_profile,
    to_profile_agent_output,
    build_bls_personas,
)

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
STORAGE_DIR  = PROJECT_ROOT / "data" / "storage"
OUTPUTS_DIR  = PROJECT_ROOT / "data" / "outputs"

_BLS_PARQUET = STORAGE_DIR / "bls_oes.parquet"
_FALLBACK_DISCOUNT_RATE = 0.044


class ProfileValidationError(RuntimeError):
    """A validate=True run did not produce the full, valid persona set."""


def _get_discount_rate(api_key: str | None = None) -> float:
    """Return DGS10 as a decimal — cache → live FRED → 4.4% fallback."""
    try:
        from data.fetch.fred import latest_dgs10
        rate = latest_dgs10(fred_api_key=api_key, fallback=_FALLBACK_DISCOUNT_RATE)
        if rate != _FALLBACK_DISCOUNT_RATE:
            return float(rate)
    except Exception:
        pass

    if api_key:
        try:
            import requests
            url = (
                "https://api.stlouisfed.org/fred/series/observations"
                f"?series_id=DGS10&api_key={api_key}"
                "&sort_order=desc&limit=1&file_type=json"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            rate = float(resp.json()["observations"][0]["value"]) / 100.0
            print(f"FRED DGS10 (10Y Treasury): {rate:.4f}")
            return rate
        except Exception as e:
            print(f"FRED fetch failed ({e}); using fallback rate {_FALLBACK_DISCOUNT_RATE}")

    print(f"DGS10 unavailable; using fallback discount rate {_FALLBACK_DISCOUNT_RATE}")
    return _FALLBACK_DISCOUNT_RATE


def _load_bls_oes() -> pd.DataFrame:
    """Read BLS OES from shared parquet cache; fetch if missing. OCC_CODE is the index."""
    if not _BLS_PARQUET.exists():
        from data.fetch.bls import fetch_bls_oes
        fetch_bls_oes()
    oes_df = pd.read_parquet(_BLS_PARQUET)
    print(f"Loaded BLS OES from parquet cache: {_BLS_PARQUET}")
    return oes_df

JSON_OUTPUT    = OUTPUTS_DIR / "profiles_all.json"
PARQUET_OUTPUT = STORAGE_DIR / "profiles_all.parquet"
ARCHIVE_DIR    = OUTPUTS_DIR / "profiles"
CHECKSUM_FILE  = ARCHIVE_DIR / "SHA256SUMS"


def build_profiles(
    raw_personas: list[dict],
    discount_rate: float,
    strict: bool = False,
) -> list[ProfileAgentOutput]:
    """
    Validate every persona, skipping (and logging) any that fail validation.
    Returns the list of successfully validated ProfileAgentOutput objects.

    Parameters
    ----------
    strict : bool
        Re-raise instead of skipping. The skip-and-log default is right for
        exploratory runs over percentile variants, where one suppressed BLS wage
        cell should not lose the other 26 personas. It is wrong for a build whose
        output is about to be archived as evidence, because the failure mode is a
        short list that still looks like a successful run — see
        run_profile_agent(validate=True).
    """
    outputs: list[ProfileAgentOutput] = []
    for persona in raw_personas:
        try:
            profile = build_profile(persona, discount_rate)
            outputs.append(to_profile_agent_output(profile))
        except Exception as e:  # pydantic.ValidationError or derivation error
            if strict:
                raise ProfileValidationError(
                    f"{persona.get('client_id', '?')} failed validation: {e}"
                ) from e
            print(f"SKIP: {persona.get('client_id', '?')} — {e}")
    print(f"Validated {len(outputs)} / {len(raw_personas)} profiles successfully")
    return outputs


def expected_persona_count(include_percentile_variants: bool = False) -> int:
    """How many personas a complete BLS run should produce."""
    return len(TARGET_OCCUPATIONS) * (3 if include_percentile_variants else 1)


def assert_complete(
    outputs: list[ProfileAgentOutput],
    include_percentile_variants: bool = False,
) -> None:
    """
    Assert the BLS run produced the full persona set, naming what is missing.

    build_bls_personas() skips missing SOC codes and suppressed wage cells with a
    printed warning, and build_profiles() skips validation failures the same way.
    Both are reasonable in isolation and jointly produce the failure this guards
    against: a run that prints two warnings into a long log, returns seven
    profiles, and is archived as though it were nine.

    Raises
    ------
    ProfileValidationError
        With the client_ids that are missing, not just a count.
    """
    expected_ids = {
        f"bls_{occ['soc']}_{pct}"
        for occ in TARGET_OCCUPATIONS
        for pct in (["p25", "p50", "p75"] if include_percentile_variants else ["p50"])
    }
    actual_ids = {p.client_id for p in outputs}

    missing = sorted(expected_ids - actual_ids)
    if missing:
        raise ProfileValidationError(
            f"Expected {len(expected_ids)} personas, got {len(actual_ids)}. "
            f"Missing: {', '.join(missing)}. A SOC code absent from the BLS OES "
            f"table or a suppressed wage cell is skipped with a warning — check "
            f"the log above for 'not found in OES data' or 'wage suppressed'."
        )

    unexpected = sorted(actual_ids - expected_ids)
    if unexpected:
        raise ProfileValidationError(
            f"Unexpected personas not in TARGET_OCCUPATIONS: {', '.join(unexpected)}"
        )

    # Every object is already a validated ProfileAgentOutput by construction —
    # to_profile_agent_output() is the only way one is made. Re-validating the
    # serialised form is the part that is not implied: it proves the profile
    # survives the JSON round-trip that the archive, the parquet export and every
    # downstream agent actually read, rather than only existing in memory.
    for profile in outputs:
        try:
            ProfileAgentOutput.model_validate(profile.model_dump(mode="json"))
        except Exception as e:
            raise ProfileValidationError(
                f"{profile.client_id} does not survive a JSON round-trip: {e}"
            ) from e

    print(f"Validated persona set complete: {len(actual_ids)} / {len(expected_ids)}")


def archive_profiles(outputs: list[ProfileAgentOutput]) -> Path:
    """
    Write one JSON file per persona to data/outputs/profiles/ plus a SHA256SUMS
    manifest, and return the archive directory.

    One file per persona rather than one combined file because the point is
    regression testing and evidence: a diff on bls_15-1252_p50.json says exactly
    which client changed, where a diff on the combined array says "something in a
    9-element list moved". The manifest uses the `sha256sum -c` format, so
    integrity can be checked with a standard tool and no project code:

        cd data/outputs/profiles && sha256sum -c SHA256SUMS

    Hashes are taken over the exact bytes written, with sorted keys and a trailing
    newline, so the same profiles always hash the same way regardless of field
    insertion order.
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Clear stale files so a persona removed from TARGET_OCCUPATIONS does not
    # linger in the archive and get re-hashed as though it were still current.
    for stale in ARCHIVE_DIR.glob("*.json"):
        stale.unlink()

    digests: list[tuple[str, str]] = []
    for profile in sorted(outputs, key=lambda p: p.client_id):
        payload = json.dumps(
            profile.model_dump(mode="json"), indent=2, sort_keys=True
        ) + "\n"
        path = ARCHIVE_DIR / f"{profile.client_id}.json"
        path.write_text(payload)
        digests.append((hashlib.sha256(payload.encode()).hexdigest(), path.name))

    CHECKSUM_FILE.write_text(
        "".join(f"{digest}  {name}\n" for digest, name in digests)
    )
    print(f"Archived {len(digests)} profiles → {ARCHIVE_DIR} (+ SHA256SUMS)")
    return ARCHIVE_DIR


def verify_archive() -> list[str]:
    """
    Re-hash the archived profiles against SHA256SUMS.

    Returns the names of files whose contents no longer match their recorded
    digest, plus any that are missing entirely. An empty list means the archive is
    intact. Present so the integrity claim is checkable from Python — the manifest
    is also readable by `sha256sum -c`.
    """
    if not CHECKSUM_FILE.exists():
        raise FileNotFoundError(f"No manifest at {CHECKSUM_FILE}; run archive_profiles() first")

    mismatched: list[str] = []
    for line in CHECKSUM_FILE.read_text().splitlines():
        if not line.strip():
            continue
        expected_digest, name = line.split("  ", 1)
        path = ARCHIVE_DIR / name
        if not path.exists():
            mismatched.append(name)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
            mismatched.append(name)
    return mismatched


def save_profiles(outputs: list[ProfileAgentOutput], archive: bool = True) -> None:
    """Persist profiles to JSON, the per-persona archive, and (if available) parquet."""
    profiles_as_dicts = [out.model_dump(mode="json") for out in outputs]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUTPUT, "w") as f:
        json.dump(profiles_as_dicts, f, indent=2)
    print(f"Saved {len(profiles_as_dicts)} profiles → {JSON_OUTPUT}")

    if archive:
        archive_profiles(outputs)

    try:
        import pandas as pd

        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(profiles_as_dicts)
        # current_holdings is a dict column — serialise to JSON string for parquet.
        df["current_holdings"] = df["current_holdings"].apply(json.dumps)
        df.to_parquet(PARQUET_OUTPUT, index=False)
        print(f"Saved {len(df)} profiles → {PARQUET_OUTPUT}")
    except Exception as e:  # pragma: no cover - optional dependency path
        print(f"Parquet export skipped ({e}); JSON output is authoritative.")


def build_profiles_from_transcripts(
    transcripts:   dict[str, str],
    discount_rate: float,
    extractor=None,
) -> tuple[list[ProfileAgentOutput], list]:
    """
    Build profiles from client conversations instead of BLS occupation data.

    This is the "input should stop being a JSON file" path. Extraction is an
    LLM (or regex) job; every number in the returned profiles is still computed
    by build_profile() from the extracted inputs.

    Parameters
    ----------
    transcripts : dict[str, str]
        client_id → raw transcript text.
    extractor : Extractor | None
        Defaults to RuleBasedExtractor, which needs no API key. Pass
        StructuredExtractor() for the language-model path.

    Returns
    -------
    (profiles, results)
        `results` carries one BridgeResult per transcript, including the
        conversations that could NOT be built and the questions to ask before
        retrying. Callers should surface those rather than dropping them —
        a conversation that produced no profile is a client to follow up with,
        not an error to swallow.
    """
    from agents.profile.intake import RuleBasedExtractor
    from agents.profile.intake_bridge import build_profile_from_intake

    extractor = extractor or RuleBasedExtractor()

    profiles: list[ProfileAgentOutput] = []
    results = []
    for client_id, text in transcripts.items():
        extracted = extractor.extract(text, client_id, f"transcript_{client_id}")
        result = build_profile_from_intake(extracted, discount_rate)
        results.append(result)
        if result.built:
            profiles.append(result.profile)
        else:
            print(f"SKIP {client_id}: {len(result.open_questions)} required field(s) "
                  f"never discussed — {result.open_questions}")

    print(f"Built {len(profiles)} / {len(transcripts)} profiles from transcripts")
    return profiles, results


def run_profile_agent(
    fred_api_key: str | None = None,
    include_percentile_variants: bool = False,
    save: bool = True,
    transcripts: dict[str, str] | None = None,
    extractor=None,
    validate: bool = False,
) -> list[ProfileAgentOutput]:
    """
    End-to-end Profile Agent run.

    Parameters
    ----------
    fred_api_key : str | None
        FRED API key for the live DGS10 discount rate. If None (or the call
        fails), the cached DGS10 parquet is used, then a 4.4% fallback.
    include_percentile_variants : bool
        False (default) → 9 personas at p50; True → up to 27 (p25/p50/p75).
    save : bool
        Persist JSON + archive + parquet outputs when True.
    transcripts : dict[str, str] | None
        client_id → raw discovery-call text. When supplied, profiles are built
        from the conversations instead of the BLS occupation table — the intake
        path from *Now and Forward* §2. BLS remains the default so existing
        callers are unaffected.
    extractor : Extractor | None
        Extraction strategy for the transcript path; defaults to the offline
        RuleBasedExtractor. Ignored when `transcripts` is None.
    validate : bool
        Turn the run into an assertion. Every expected persona must be built,
        must pass Pydantic validation, and must survive a JSON round-trip, or the
        call raises ProfileValidationError naming what is missing.

        Opt-in rather than the default because the two behaviours serve different
        callers. Exploratory work over percentile variants genuinely wants the
        partial result — losing 26 good personas to one suppressed BLS wage cell
        helps nobody. Anything whose output will be archived, demoed or defended
        wants the opposite, because the failure mode there is silent: a run that
        prints a warning into a long log and returns a short list still looks like
        a successful run.

        Not supported on the transcript path, where an incomplete result is the
        expected outcome rather than a fault — a conversation that never mentioned
        salary produces open questions to ask the client, and BridgeResult already
        carries them.

    Returns
    -------
    list[ProfileAgentOutput]

    Raises
    ------
    ProfileValidationError
        When `validate` and the persona set is incomplete or invalid.
    ValueError
        When `validate` is combined with `transcripts`.
    """
    if validate and transcripts:
        raise ValueError(
            "validate=True is not supported on the transcript path: an incomplete "
            "profile set is the expected outcome there, not a fault. Inspect the "
            "BridgeResult.open_questions from build_profiles_from_transcripts() "
            "instead."
        )

    discount_rate = _get_discount_rate(fred_api_key)
    print(f"Discount rate (FRED DGS10): {discount_rate:.4f}")

    if transcripts:
        outputs, _ = build_profiles_from_transcripts(
            transcripts, discount_rate, extractor=extractor
        )
    else:
        oes_df = _load_bls_oes()
        raw_personas = build_bls_personas(
            oes_df, include_percentile_variants=include_percentile_variants
        )
        outputs = build_profiles(raw_personas, discount_rate, strict=validate)
        if validate:
            assert_complete(outputs, include_percentile_variants)

    if save:
        save_profiles(outputs)

        if validate:
            corrupted = verify_archive()
            if corrupted:
                raise ProfileValidationError(
                    f"Archive failed its own checksum immediately after being "
                    f"written: {', '.join(corrupted)}"
                )
            print(f"Archive verified against {CHECKSUM_FILE.name}")

    return outputs


if __name__ == "__main__":  # pragma: no cover
    profiles = run_profile_agent(validate=True)
    print(f"\n{len(profiles)} profiles generated")
