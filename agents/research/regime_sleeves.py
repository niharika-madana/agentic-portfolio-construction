"""
regime_sleeves.py — regime → asset-class sleeve tilts, measured from returns.

Action item (July 24 minutes, decision 5): "map each regime to predefined
asset-class tilts (e.g., increase bond weight in 'Inflation Shock')."

The word in the minutes is *predefined*. This module derives them instead. A
declared table of tilts would be exactly the pattern the review objected to in
the stress drawdowns — a number that is true by construction and unfalsifiable.
Here every tilt comes from realised ETF returns inside each regime window, and
every tilt is reported next to the standard error of the mean it rests on, so a
reader can see which ones the data actually supports.

Scope: this produces the mapping only. Consuming it inside the optimiser is
review §4.2 and lives in agents/allocation, which this module does not touch.
`regime_sleeve_tilts()` returns a plain dict the allocation agent can read.

Method
------
For each (regime, sleeve) pair, using monthly CRSP total returns for the ETFs in
that sleeve, equal-weighted:

    tilt = bounded( sigma_sleeve_full_sample / sigma_sleeve_in_regime - 1 )

Same construction as the equity tilt in regime_returns.py, and for the same
reason: over 27-82 months per regime the mean is not estimable to anything like
the precision the tilt would need, while the volatility is. A sleeve that is
calmer than its own long-run behaviour during a regime earns a positive tilt.

Because the means are reported but not used, the module also flags whether the
regime's mean return for that sleeve differs from its full-sample mean by more
than one standard error. That flag is the honest annotation on each tilt: it
says whether the return evidence agrees with the volatility evidence, or whether
the tilt rests on volatility alone.

Reading the output
------------------
Tilts are relative, bounded, and mean nothing in isolation — they are intended
to be applied against a sleeve's baseline weight, then renormalised. A tilt of
+0.12 on Fixed Income during Inflation Shock does not mean "hold 12% bonds"; it
means "this sleeve was 12% calmer than usual in this regime, size it up
accordingly and let the optimiser enforce the actual constraints."
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "data" / "outputs"
SLEEVE_TILTS_JSON = OUTPUTS_DIR / "regime_sleeve_tilts.json"
SLEEVE_STATS_CSV  = OUTPUTS_DIR / "regime_sleeve_stats.csv"

MONTHS_PER_YEAR = 12

SLEEVES: dict[str, list[str]] = {
    "broad_equity":  ["SPY", "IWM", "EFA", "EEM"],
    "sector_equity": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE"],
    "fixed_income":  ["AGG", "TLT", "IEF", "SHY", "LQD", "TIP"],
    "credit":        ["HYG"],
    "real_assets":   ["GLD", "VNQ"],
    "cash":          ["BIL"],
}
"""
Sleeve definitions over the ETF universe in agents/allocation/adapters.py.

Broad and sector equity are kept apart deliberately — review §3 objects to
layering one on top of the other, and a mapping that lumped them together could
not express "prefer the broad core in this regime" at all. HYG is split out of
fixed income because high-yield behaves like equity in a crisis, which is
precisely when a bond tilt is supposed to help.
"""

NON_TILTABLE = {"cash"}
"""
Sleeves the volatility rule cannot speak to, held at zero tilt.

Cash has almost no volatility in any regime, so `sigma_baseline / sigma_regime`
is a ratio of two near-zero numbers — it returned 9.4 in Early Recovery and
produced a maximum positive tilt for cash in the calmest regime AND in the
financial crisis, which is not a signal, it is a divide-by-nearly-zero. A
risk-adjusted attractiveness measure has nothing to say about the numeraire.

Cash is sized by the optimiser's cash floor and by what the risky sleeves do not
take, not by a tilt. Excluding it here is a statement about what this method can
measure, not a claim that cash weight should never move.
"""

TILT_CAP = 0.15
"""Maximum |tilt| per sleeve. Matches regime_returns.TILT_CAP."""

TILT_SCALE = 0.50
"""Vol-ratio deviation mapping to a full-magnitude tilt. Matches regime_returns."""

MIN_SLEEVE_MONTHS = 12
"""Below this, the regime/sleeve pair gets a zero tilt rather than a noisy one."""


@dataclass
class SleeveRegimeStats:
    """Realised statistics for one sleeve inside one regime."""

    regime:        str
    sleeve:        str
    n_months:      int
    mean_annual:   float
    se_annual:     float
    vol_annual:    float
    baseline_vol:  float
    vol_ratio:     float
    tilt:          float
    tilt_capped:   bool
    mean_supports_tilt: bool
    """
    True when the regime mean differs from the sleeve's full-sample mean by more
    than one standard error, in the same direction as the tilt. False means the
    tilt rests on volatility evidence alone — which is not disqualifying, but a
    reader should know.
    """


def _sleeve_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted monthly return per sleeve, from a month × ticker frame."""
    out = {}
    for sleeve, tickers in SLEEVES.items():
        present = [t for t in tickers if t in returns.columns]
        if present:
            out[sleeve] = returns[present].mean(axis=1)
    return pd.DataFrame(out)


def _annualise(monthly: pd.Series) -> tuple[float, float, float]:
    """(annual mean, annual SE of the mean, annual volatility)."""
    mean_annual = (1.0 + monthly.mean()) ** MONTHS_PER_YEAR - 1.0
    vol_annual  = monthly.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
    se_annual   = monthly.std(ddof=1) / np.sqrt(len(monthly)) * MONTHS_PER_YEAR
    return float(mean_annual), float(se_annual), float(vol_annual)


def _tilt_from_vol_ratio(vol_ratio: float) -> tuple[float, bool]:
    deviation = vol_ratio - 1.0
    scaled    = deviation / TILT_SCALE
    capped    = abs(scaled) > 1.0
    return float(np.clip(scaled, -1.0, 1.0)) * TILT_CAP, capped


def compute_sleeve_stats(regime_by_month: pd.Series) -> list[SleeveRegimeStats]:
    """
    Realised sleeve statistics and derived tilts, per regime.

    Returns an empty list when the ETF return cache is unavailable — callers
    must treat that as "no mapping can be derived", not as a zero tilt.
    """
    from agents.research.ticker_betas import _load_returns

    returns = _load_returns()
    sleeves = _sleeve_returns(returns)
    if sleeves.empty:
        return []

    panel = sleeves.join(regime_by_month.rename("regime"), how="inner").dropna(
        subset=["regime"]
    )
    if panel.empty:
        return []

    results: list[SleeveRegimeStats] = []

    for sleeve in sleeves.columns:
        series = panel[sleeve].dropna()
        if len(series) < MIN_SLEEVE_MONTHS:
            continue
        base_mean, _, base_vol = _annualise(series)
        if base_vol <= 0:
            continue

        for regime, group in panel.groupby("regime"):
            window = group[sleeve].dropna()
            n = len(window)
            if n < MIN_SLEEVE_MONTHS:
                results.append(SleeveRegimeStats(
                    regime=regime, sleeve=sleeve, n_months=n,
                    mean_annual=float("nan"), se_annual=float("nan"),
                    vol_annual=float("nan"), baseline_vol=round(base_vol, 4),
                    vol_ratio=float("nan"), tilt=0.0, tilt_capped=False,
                    mean_supports_tilt=False,
                ))
                continue

            mean_a, se_a, vol_a = _annualise(window)
            if vol_a <= 0:
                continue

            vol_ratio    = base_vol / vol_a
            if sleeve in NON_TILTABLE:
                tilt, capped = 0.0, False
            else:
                tilt, capped = _tilt_from_vol_ratio(vol_ratio)

            # Does the return evidence point the same way as the vol evidence,
            # by more than one standard error?
            mean_gap = mean_a - base_mean
            supports = bool(
                se_a > 0
                and abs(mean_gap) > se_a
                and np.sign(mean_gap) == np.sign(tilt)
                and tilt != 0
            )

            results.append(SleeveRegimeStats(
                regime=regime, sleeve=sleeve, n_months=n,
                mean_annual=round(mean_a, 4), se_annual=round(se_a, 4),
                vol_annual=round(vol_a, 4), baseline_vol=round(base_vol, 4),
                vol_ratio=round(vol_ratio, 3), tilt=round(tilt, 4),
                tilt_capped=capped, mean_supports_tilt=supports,
            ))

    return results


def regime_sleeve_tilts(
    stats: list[SleeveRegimeStats] | None = None,
    regime_by_month: pd.Series | None = None,
) -> dict[str, dict[str, float]]:
    """
    The mapping itself: {regime_label: {sleeve: tilt}}.

    This is the object the Allocation Agent consumes (review §4.2). Sleeves
    absent from a regime's entry have no measured tilt and should be left at
    their baseline weight — not treated as zero-weight.
    """
    if stats is None:
        if regime_by_month is None:
            raise ValueError("Supply either computed stats or a regime series.")
        stats = compute_sleeve_stats(regime_by_month)

    mapping: dict[str, dict[str, float]] = {}
    for s in stats:
        if s.tilt == 0.0 and s.n_months < MIN_SLEEVE_MONTHS:
            continue
        mapping.setdefault(s.regime, {})[s.sleeve] = s.tilt
    return mapping


def stats_table(stats: list[SleeveRegimeStats]) -> pd.DataFrame:
    if not stats:
        return pd.DataFrame()
    return pd.DataFrame([asdict(s) for s in stats]).sort_values(
        ["regime", "tilt"], ascending=[True, False]
    ).reset_index(drop=True)


def _load_regime_series() -> pd.Series:
    """Month → regime label, from the persisted Research Agent sequence."""
    path = OUTPUTS_DIR / "regime_sequence.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run agents.research.research_agent.run_research_agent() first."
        )
    seq = json.loads(path.read_text())
    return pd.Series(
        {pd.Timestamp(k): v["regime_label"] for k, v in seq.items()}
    ).sort_index()


def run(save: bool = True) -> dict[str, dict[str, float]]:
    """Derive the mapping, print it for review, and persist."""
    regimes = _load_regime_series()
    stats   = compute_sleeve_stats(regimes)
    if not stats:
        print("No ETF return data available — no sleeve mapping derived.")
        return {}

    df = stats_table(stats)
    print("\n=== Regime → asset-class sleeve tilts (derived from realised returns) ===")
    pivot = df.pivot_table(index="sleeve", columns="regime", values="tilt")
    print(pivot.round(3).to_string())

    print("\nTilts where the return evidence agrees with the volatility evidence "
          "(|mean gap| > 1 SE, same direction):")
    supported = df[df["mean_supports_tilt"]]
    if supported.empty:
        print("  none — every tilt rests on volatility evidence alone")
    else:
        print(supported[["regime", "sleeve", "tilt", "mean_annual", "se_annual",
                         "vol_ratio", "n_months"]].to_string(index=False))

    capped = df[df["tilt_capped"]]
    print(f"\n{len(capped)} of {len(df)} tilts hit the ±{TILT_CAP:.0%} cap"
          + (":" if not capped.empty else "."))
    if not capped.empty:
        print(capped[["regime", "sleeve", "vol_ratio", "tilt"]].to_string(index=False))

    mapping = regime_sleeve_tilts(stats)

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        SLEEVE_TILTS_JSON.write_text(json.dumps(mapping, indent=2))
        df.to_csv(SLEEVE_STATS_CSV, index=False)
        print(f"\nSaved → {SLEEVE_TILTS_JSON}")
        print(f"Saved → {SLEEVE_STATS_CSV}")

    return mapping


if __name__ == "__main__":  # pragma: no cover
    run()
