"""
profile_agent.py: Profile Agent entry point (Agent 2 of 5).

run_profile_agent() builds and validates one ProfileAgentOutput per BLS occupation (median salary by default, or p25/p50/p75 with
include_percentile_variants=True) and optionally persists them:

    data/outputs/profiles_all.json        — list of validated profiles
    data/storage/profiles_all.parquet     — consumed by Allocation/Risk/Compliance

The returned list[ProfileAgentOutput] feeds straight into orchestrator.run_all(personas, macro).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from contracts import ProfileAgentOutput

from agents.profile.profile_model import build_profile, to_profile_agent_output, build_bls_personas

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
STORAGE_DIR  = PROJECT_ROOT / "data" / "storage"
OUTPUTS_DIR  = PROJECT_ROOT / "data" / "outputs"

_BLS_PARQUET = STORAGE_DIR / "bls_oes.parquet"
_FALLBACK_DISCOUNT_RATE = 0.044


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


def build_profiles(raw_personas: list[dict], discount_rate: float) -> list[ProfileAgentOutput]:
    """
    Validate every persona, skipping (and logging) any that fail validation.
    Returns the list of successfully validated ProfileAgentOutput objects.
    """
    outputs: list[ProfileAgentOutput] = []
    for persona in raw_personas:
        try:
            profile = build_profile(persona, discount_rate)
            outputs.append(to_profile_agent_output(profile))
        except Exception as e:  # pydantic.ValidationError or derivation error
            print(f"SKIP: {persona.get('client_id', '?')} — {e}")
    print(f"Validated {len(outputs)} / {len(raw_personas)} profiles successfully")
    return outputs


def save_profiles(outputs: list[ProfileAgentOutput]) -> None:
    """Persist profiles to JSON and (if pandas/pyarrow are available) parquet."""
    profiles_as_dicts = [out.model_dump(mode="json") for out in outputs]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUTPUT, "w") as f:
        json.dump(profiles_as_dicts, f, indent=2)
    print(f"Saved {len(profiles_as_dicts)} profiles → {JSON_OUTPUT}")

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
        Persist JSON + parquet outputs when True.
    transcripts : dict[str, str] | None
        client_id → raw discovery-call text. When supplied, profiles are built
        from the conversations instead of the BLS occupation table — the intake
        path from *Now and Forward* §2. BLS remains the default so existing
        callers are unaffected.
    extractor : Extractor | None
        Extraction strategy for the transcript path; defaults to the offline
        RuleBasedExtractor. Ignored when `transcripts` is None.

    Returns
    -------
    list[ProfileAgentOutput]
    """
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
        outputs = build_profiles(raw_personas, discount_rate)

    if save:
        save_profiles(outputs)
    return outputs


if __name__ == "__main__":  # pragma: no cover
    profiles = run_profile_agent()
    print(f"\n{len(profiles)} profiles generated")
