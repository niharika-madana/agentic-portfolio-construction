"""
profile_agent.py — Profile Agent entry point (Agent 1 of 5).

run_profile_agent() builds and validates one ProfileAgentOutput per BLS
occupation (median salary by default, or p25/p50/p75 with
include_percentile_variants=True) and optionally persists them:

    agents/profile/profiles_all.json      — list of validated profiles
    data/storage/profiles_all.parquet     — consumed by Allocation/Risk/Compliance

The returned list[ProfileAgentOutput] feeds straight into
orchestrator.run_all(personas, macro).
"""

from __future__ import annotations

import json
from pathlib import Path

from contracts import ProfileAgentOutput

from agents.profile.human_capital import build_profile, to_profile_agent_output
from agents.profile.loaders import get_discount_rate, load_bls_oes
from agents.profile.personas import build_bls_personas

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
STORAGE_DIR = PROJECT_ROOT / "data" / "storage"

JSON_OUTPUT = _THIS_DIR / "profiles_all.json"
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

    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
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


def run_profile_agent(
    fred_api_key: str | None = None,
    include_percentile_variants: bool = False,
    save: bool = True,
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

    Returns
    -------
    list[ProfileAgentOutput]
    """
    discount_rate = get_discount_rate(fred_api_key)
    print(f"Discount rate (FRED DGS10): {discount_rate:.4f}")

    oes_df = load_bls_oes()
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
