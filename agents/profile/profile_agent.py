"""
profile_agent.py — Profile Agent
==================================
Entry point: run_profile_agent() → list[ProfileAgentOutput]

Single-pass pipeline:
    1. FRED DGS10           → discount_rate
    2. BLS OES (May 2023)   → salary distributions by SOC code
    3. SCF 2022 (static)    → financial_capital by age × salary percentile
    4. build_bls_personas() → raw persona dicts (all fields from data)
    5. lookup_hc_beta()     → β and ρ from calibrated table (no OLS)
    6. build_profile()      → single call, all fields computed
    7. to_profile_agent_output() → Pydantic validation via contracts.py
    8. Save JSON outputs

Returns list[ProfileAgentOutput] ready for orchestrator.run_pipeline().
"""

import json
import os

from .hc_beta_table import lookup_hc_beta
from .human_capital import HUMAN_CAPITAL_TYPE, build_profile, to_profile_agent_output
from .loaders import get_discount_rate_from_fred, load_bls_oes
from .personas import build_bls_personas

OUTPUT_DIR = "agents/profile"


def run_profile_agent(
    fred_api_key=None,
    include_percentile_variants=False,
    output_dir=OUTPUT_DIR,
):
    """
    Run the full Profile Agent pipeline.

    Args:
        fred_api_key:                 FRED API key (optional; uses 4.4% fallback if None)
        include_percentile_variants:  If True, generates p25/p50/p75 variants per
                                      occupation (27 profiles). Default: p50 only (9 profiles).
        output_dir:                   Directory to save output JSON files.

    Returns:
        list[ProfileAgentOutput] — validated Pydantic objects, pass directly to orchestrator.
    """
    # Step 1 — Discount rate
    discount_rate = get_discount_rate_from_fred(fred_api_key)

    # Step 2 — BLS OES salary data
    bls_data = load_bls_oes()

    # Step 3+4 — Build persona dicts from BLS + SCF
    print("\n=== Building BLS Personas ===")
    raw_personas = build_bls_personas(bls_data, include_percentile_variants)

    if not raw_personas:
        raise RuntimeError(
            "No personas built — check BLS SOC codes against loaded data. "
            "Run load_bls_oes() and verify the SOC codes in TARGET_OCCUPATIONS."
        )

    # Step 5+6+7 — Single pass: beta lookup → build_profile → Pydantic adapter
    print("\n=== Building Profiles ===")
    typed_profiles = []
    profile_dicts  = []

    for persona in raw_personas:
        hc_type            = HUMAN_CAPITAL_TYPE[persona["income_stability"]]
        beta, correlation  = lookup_hc_beta(hc_type)
        profile            = build_profile(persona, discount_rate, beta, correlation)
        profile_dicts.append(profile)

        # Pydantic validation — raises ValidationError if any constraint fails
        typed_profiles.append(to_profile_agent_output(profile))

        print(
            f"  {profile['client_id']:35s}  "
            f"HC=${profile['human_capital_valuation']:>10,.0f}  "
            f"implicit_eq={profile['implicit_equity_exposure']:.3f}  "
            f"eq_target={profile['portfolio_equity_target']:+.3f}"
        )

    print(f"\n{len(typed_profiles)} ProfileAgentOutput objects validated")

    # Step 8 — Save outputs
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "profiles_all.json"), "w") as f:
        json.dump(profile_dicts, f, indent=2)

    print(f"Saved {len(profile_dicts)} profiles → {output_dir}/profiles_all.json")

    return typed_profiles
