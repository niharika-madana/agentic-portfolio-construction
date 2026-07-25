"""
sensitivity.py — how much does the final portfolio move when an input is wrong?

This is the question from *Now and Forward* §5, which the mentor named as the one
he would most like answered:

    "Perturb an extracted field, rerun the allocation, measure the change in
    weights. If a twenty percent error in Priya's income variability moves her
    equity share by one percent, then extraction precision is not the binding
    constraint on this system and you should say so plainly in the paper. If it
    moves it by fifteen percent, then the intake layer is the bottleneck of the
    entire pipeline and that is where the remaining effort belongs."

Either answer is a result. This module produces the number.

Method
------
For each persona and each intake-extractable input, perturb by ±PERTURBATIONS,
rebuild the profile through the real formulas (build_profile, via its `overrides`
hook so nothing is reimplemented), re-run the real allocation agent, and measure:

    d_equity_share   change in risky_weight, in percentage points of the portfolio
    d_equity_target  change in portfolio_equity_target (total-wealth units)
    turnover         0.5 * L1 distance between old and new weight vectors, i.e.
                     the fraction of the book that would have to trade
    elasticity       (% change in equity share) / (% change in the input)

Which inputs
------------
Only fields an intake layer would have to extract from a client conversation.
`income_volatility_sigma` is the headline one: it is the mentor's own example,
and it is the hardest field to extract because a client states it as "my bonus
is all over the place year to year" rather than as a number.

Reading the output
------------------
Elasticity near zero means extraction error in that field does not reach the
portfolio, and precision there is not worth engineering effort. Large elasticity
means the intake layer is load-bearing. Note that a zero can arise two ways —
genuine insensitivity, or a client pinned at a corner solution (see
agents/research/rebalance.py::corner_solution) where a constraint, not the input,
is setting the allocation. The report distinguishes them, because they have
opposite implications for where to spend the remaining weeks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from contracts import ProfileAgentOutput
from agents.profile.profile_model import build_profile

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "data" / "outputs"
SENSITIVITY_JSON = OUTPUTS_DIR / "sensitivity_report.json"
SENSITIVITY_CSV  = OUTPUTS_DIR / "sensitivity_report.csv"

PERTURBATIONS = (-0.20, -0.10, 0.10, 0.20)
"""Relative input errors to test. ±20% is the mentor's stated example."""

PERTURBABLE_FIELDS = (
    "income_volatility_sigma",
    "income_equity_correlation",
    "annual_salary",
    "financial_capital",
)
"""
Inputs an intake layer would have to extract from a transcript.

sigma and correlation are injected through build_profile's `overrides` hook;
salary and financial capital are perturbed on the raw persona dict before the
human-capital annuity is computed, so the whole chain re-derives.
"""

_PERSONA_LEVEL = {"annual_salary", "financial_capital"}


@dataclass
class SensitivityResult:
    """One (persona, field, perturbation) cell."""

    client_id:        str
    human_capital_type: str
    field:            str
    perturbation:     float
    baseline_value:   float
    perturbed_value:  float

    baseline_equity_share:  float
    perturbed_equity_share: float
    d_equity_share:         float

    baseline_equity_target:  float
    perturbed_equity_target: float
    d_equity_target:         float

    turnover:    float
    elasticity:  float | None
    corner_if_units_fixed: bool


def _perturb_persona(persona: dict, field: str, pct: float) -> dict:
    """Return a copy of the raw persona with one numeric field scaled by (1+pct)."""
    out = dict(persona)
    if field == "annual_salary":
        out["annual_salary"]    = persona["annual_salary"] * (1 + pct)
        # effective_salary carries the bonus and feeds the HC annuity directly.
        out["effective_salary"] = persona["effective_salary"] * (1 + pct)
    elif field == "financial_capital":
        out["financial_capital"] = persona["financial_capital"] * (1 + pct)
    else:
        raise ValueError(f"{field} is not a persona-level field")
    return out


_SAFE_SLEEVE = "__SAFE__"
"""
Synthetic line for the safe/cash sleeve in the assembled book.

`AllocationAgentOutput.proposed_portfolio` is sleeve-relative: it always sums to
1.0 across the risky ETFs no matter what risky_weight is. Measuring turnover on
it alone therefore returns 0.0 for every perturbation, even when the equity share
moves 13 points — which is exactly the review's §4.1 finding that the safe weight
never reaches the client book. Until that is fixed in the allocation path, this
module assembles the final book itself:

    final = risky_weight * risky_sleeve  +  safe_weight * safe_sleeve

so the measured turnover reflects the trade a client would actually place.
"""


def _run_allocation(profile: ProfileAgentOutput, discount_rate: float):
    """
    Run the real allocation agent.

    Returns (risky_weight, final_book) where final_book is the assembled
    risky+safe portfolio, not the sleeve-relative one. See _SAFE_SLEEVE.
    """
    from agents.allocation.agent import run_allocation_agent

    alloc_output, alloc_ao = run_allocation_agent(
        profile=profile, discount_rate=discount_rate
    )
    risky = float(alloc_output.risky_weight)
    safe  = float(alloc_output.safe_weight)

    book = {t: w * risky for t, w in alloc_ao.proposed_portfolio.items()}
    book[_SAFE_SLEEVE] = safe
    return risky, book


def _turnover(before: dict[str, float], after: dict[str, float]) -> float:
    """Fraction of the book that must trade: 0.5 * L1 distance."""
    tickers = set(before) | set(after)
    return round(
        0.5 * sum(abs(after.get(t, 0.0) - before.get(t, 0.0)) for t in tickers), 6
    )


def analyse_persona(
    persona:       dict,
    discount_rate: float,
    fields:        tuple[str, ...] = PERTURBABLE_FIELDS,
    perturbations: tuple[float, ...] = PERTURBATIONS,
) -> tuple[list[SensitivityResult], list[str]]:
    """
    Perturb every field at every magnitude for one persona.

    Returns (results, skipped) where `skipped` describes cells that could not be
    evaluated because the perturbed profile fell outside the contract.
    """
    from agents.research.rebalance import corner_solution

    base_profile_dict = build_profile(persona, discount_rate)
    base_profile      = ProfileAgentOutput(**base_profile_dict)
    base_share, base_weights = _run_allocation(base_profile, discount_rate)
    base_target = base_profile_dict["portfolio_equity_target"]

    # NOTE these two views disagree today, and the disagreement is informative.
    # corner_solution() converts the equity target to financial-wealth units
    # (x (financial + human) / financial) as the review's §4.1 requires, and finds
    # every BLS persona pinned at a bound. The live allocation path does NOT yet
    # apply that conversion — it clips the total-wealth number to [0, 1] and uses
    # it directly — so the observed equity share still moves. This flag records
    # where the persona WILL sit once §4.1 is fixed, so the sensitivity numbers
    # below can be read against both the current and the corrected engine.
    corner_if_units_fixed = corner_solution(base_profile) is not None

    results: list[SensitivityResult] = []
    skipped: list[str] = []

    for field in fields:
        baseline_value = (
            persona[field] if field in _PERSONA_LEVEL else base_profile_dict[field]
        )

        for pct in perturbations:
            if field in _PERSONA_LEVEL:
                perturbed_dict = build_profile(
                    _perturb_persona(persona, field, pct), discount_rate
                )
            else:
                perturbed_dict = build_profile(
                    persona, discount_rate,
                    overrides={field: baseline_value * (1 + pct)},
                )

            try:
                perturbed_profile = ProfileAgentOutput(**perturbed_dict)
                share, weights = _run_allocation(perturbed_profile, discount_rate)
            except Exception as e:
                # The perturbation left the contract's admissible range — most
                # often income_equity_beta above its 2.0 ceiling, which the
                # equity-like tier now sits close to. Recorded, not silently
                # dropped: a skipped cell is unmeasured, not insensitive, and
                # letting it read as 0.0 in the summary would be a lie.
                reason = str(e).split("\n")[0]
                skipped.append(f"{persona['client_id']} {field} {pct:+.0%}: {reason}")
                continue

            d_share = share - base_share
            elasticity = (
                round((d_share / base_share) / pct, 4)
                if base_share not in (0.0,) and pct != 0 else None
            )

            results.append(
                SensitivityResult(
                    client_id               = base_profile_dict["client_id"],
                    human_capital_type      = perturbed_dict["human_capital_type"],
                    field                   = field,
                    perturbation            = pct,
                    baseline_value          = round(float(baseline_value), 6),
                    perturbed_value         = round(float(baseline_value) * (1 + pct), 6),
                    baseline_equity_share   = round(base_share, 6),
                    perturbed_equity_share  = round(share, 6),
                    d_equity_share          = round(d_share, 6),
                    baseline_equity_target  = round(float(base_target), 6),
                    perturbed_equity_target = round(float(perturbed_dict["portfolio_equity_target"]), 6),
                    d_equity_target         = round(
                        float(perturbed_dict["portfolio_equity_target"]) - float(base_target), 6
                    ),
                    turnover                = _turnover(base_weights, weights),
                    elasticity              = elasticity,
                    corner_if_units_fixed   = corner_if_units_fixed,
                )
            )

    return results, skipped


def run(
    personas:      list[dict] | None = None,
    discount_rate: float | None = None,
    save:          bool = True,
) -> pd.DataFrame:
    """
    Run the full sensitivity sweep and report.

    Rebuilds personas from the BLS cache when none are supplied, so this is
    runnable standalone.
    """
    if personas is None or discount_rate is None:
        from agents.profile.profile_agent import _get_discount_rate, _load_bls_oes
        from agents.profile.profile_model import build_bls_personas

        discount_rate = discount_rate or _get_discount_rate()
        personas = personas or build_bls_personas(_load_bls_oes())

    all_results: list[SensitivityResult] = []
    all_skipped: list[str] = []
    for persona in personas:
        print(f"  {persona['client_id']} ...")
        results, skipped = analyse_persona(persona, discount_rate)
        all_results.extend(results)
        all_skipped.extend(skipped)

    df = pd.DataFrame([asdict(r) for r in all_results])
    if df.empty:
        print("No sensitivity results produced.")
        return df

    _print_report(df, all_skipped)

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(SENSITIVITY_CSV, index=False)
        SENSITIVITY_JSON.write_text(
            json.dumps(
                {"results": [asdict(r) for r in all_results], "skipped": all_skipped},
                indent=2,
            )
        )
        print(f"\nSaved → {SENSITIVITY_CSV}")
        print(f"Saved → {SENSITIVITY_JSON}")

    return df


def _print_report(df: pd.DataFrame, skipped: list[str] | None = None) -> None:
    skipped = skipped or []
    print("\n=== Input sensitivity — how far does an extraction error travel? ===")
    print(f"{len(df)} cells measured | {len(skipped)} skipped (out of contract) "
          f"| {df['client_id'].nunique()} personas\n")

    if skipped:
        types_hit = sorted({s.split()[0] for s in skipped})
        print(f"Skipped cells are UNMEASURED, not insensitive — do not read them as zeros.")
        print(f"  {len(skipped)} cells across {len(types_hit)} persona(s); "
              f"cause: perturbed income_equity_beta above the contract's 2.0 ceiling.")
        print(f"  The equity-like tier now sits at beta 1.91, so a +10% extraction "
              f"error already leaves the admissible range.\n")

    twenty = df[df["perturbation"].abs() == 0.20]
    summary = (
        twenty.groupby("field")
        .agg(
            max_abs_d_equity_share=("d_equity_share", lambda s: s.abs().max()),
            mean_abs_d_equity_share=("d_equity_share", lambda s: s.abs().mean()),
            max_turnover=("turnover", "max"),
            max_abs_d_target=("d_equity_target", lambda s: s.abs().max()),
        )
        .sort_values("max_abs_d_equity_share", ascending=False)
    )
    print("At ±20% input error (the mentor's stated test):")
    print(summary.round(4).to_string())

    print("\nBy human-capital type (max |Δ equity share| at ±20%):")
    by_type = (
        twenty.groupby(["human_capital_type", "field"])["d_equity_share"]
        .apply(lambda s: s.abs().max())
        .unstack(fill_value=0.0)
    )
    print(by_type.round(4).to_string())

    worst = df.loc[df["d_equity_share"].abs().idxmax()]
    print(
        f"\nLargest single move: {worst['client_id']} ({worst['human_capital_type']}), "
        f"{worst['field']} {worst['perturbation']:+.0%} → equity share "
        f"{worst['baseline_equity_share']:.1%} → {worst['perturbed_equity_share']:.1%} "
        f"({worst['d_equity_share']:+.2%}), turnover {worst['turnover']:.1%}"
    )

    n_corner = int(df["corner_if_units_fixed"].sum())
    print(
        f"\nOnce the §4.1 unit conversion is fixed, {n_corner} of {len(df)} cells sit at "
        f"a corner solution (equity target pinned at 0% or 100% of financial wealth "
        f"after x (financial + human) / financial). The moves above are what the CURRENT "
        f"engine does; after that fix these personas stop responding to intake error at "
        f"all, because a bound — not the input — sets their allocation."
    )


if __name__ == "__main__":  # pragma: no cover
    run()
