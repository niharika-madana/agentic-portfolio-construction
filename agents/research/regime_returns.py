"""
regime_returns.py — regime-conditional market statistics and the equity tilt
derived from them.

Answers the oral-defence question "how do you know your regime classification is
a real analogue rather than a plausible-sounding one?" with a number: the five
regimes separate cleanly on realised market volatility, and the tilt is derived
from that separation rather than declared.

Why volatility and not expected return
--------------------------------------
Joining the smoothed regime sequence to CRSP value-weighted monthly returns
(1995-2024, 263 overlapping months) gives:

    Regime                     n    ann. mean   SE(mean)   ann. vol
    Early Recovery            31      20.1%       5.0%        8.1%
    Late-Cycle Expansion      61      16.7%       4.6%       10.5%
    Moderate Expansion        82      13.4%       5.6%       14.7%
    Financial Crisis & ZLB    62       9.5%       8.9%       20.1%
    Inflation Shock           27      -3.4%      12.0%       18.0%

The means are not statistically separable — standard errors of 5-12% straddle
almost the entire 23-point spread between the highest and lowest regime. Feeding
them into a Merton weight produces w = 9.2 for Early Recovery (920% equity),
which is estimation error, not signal.

The volatilities are separable: Financial Crisis & ZLB realises 2.5x the
volatility of Early Recovery, on 62 and 31 months respectively. So the tilt holds
the risk premium fixed at the full-sample estimate and lets only volatility vary
— the constant-Sharpe form of volatility targeting, where the equity weight
scales with sigma_base / sigma_regime.

The resulting raw ratio is then squashed into a bounded tilt (TILT_CAP) so a
single well-estimated volatility ratio cannot swing a client's book without
limit. Every constant below is named, and binding is reported rather than
hidden — per the review's point that a cap which always binds is the cap
allocating, not the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
_CRSP_PARQUET = PROJECT_ROOT / "data" / "storage" / "crsp_market_index.parquet"

MONTHS_PER_YEAR = 12

# ── Tilt construction constants ────────────────────────────────────────────
# Raw signal is (sigma_base / sigma_regime) - 1: positive when the regime is
# calmer than the full sample, negative when it is more turbulent.
TILT_CAP = 0.15
"""Maximum |tilt| as a fraction of the client's equity target (±15% relative)."""

TILT_SCALE = 0.50
"""
Vol-ratio deviation that maps to a full-magnitude tilt. A regime realising 2/3
of full-sample volatility (ratio 1.5, deviation 0.50) earns the full +15%.
Deviations beyond this saturate — the tilt is bounded by construction.
"""

MIN_REGIME_MONTHS = 12
"""
Minimum overlapping months before a regime's volatility is trusted. Below this
the regime falls back to a zero tilt rather than a noisy one — an unmeasured
regime must not silently receive a fabricated tilt (review §7).
"""


@dataclass(frozen=True)
class RegimeReturnStats:
    """Realised market statistics for one regime, annualised."""

    regime:      str
    n_months:    int
    mean_annual: float
    se_annual:   float
    vol_annual:  float
    vol_ratio:   float  # sigma_base / sigma_regime — >1 means calmer than baseline
    tilt:        float  # bounded equity tilt, as a fraction of the equity target
    tilt_capped: bool   # True when TILT_CAP bound the tilt


def load_crsp_returns() -> pd.DataFrame | None:
    """Return the CRSP value-weighted monthly market index, or None when absent."""
    if not _CRSP_PARQUET.exists():
        return None
    crsp = pd.read_parquet(_CRSP_PARQUET)
    if "vwretd" not in crsp.columns:
        return None
    return crsp


def _annualise(monthly: pd.Series) -> tuple[float, float, float]:
    """Return (annual mean, annual SE of the mean, annual volatility)."""
    mean_annual = (1.0 + monthly.mean()) ** MONTHS_PER_YEAR - 1.0
    vol_annual  = monthly.std() * np.sqrt(MONTHS_PER_YEAR)
    se_annual   = monthly.std() / np.sqrt(len(monthly)) * MONTHS_PER_YEAR
    return float(mean_annual), float(se_annual), float(vol_annual)


def _tilt_from_vol_ratio(vol_ratio: float) -> tuple[float, bool]:
    """
    Map a volatility ratio to a bounded equity tilt.

    Returns (tilt, capped). A calmer-than-baseline regime earns a positive tilt;
    a more turbulent one earns a negative tilt. Saturates at ±TILT_CAP.
    """
    deviation = vol_ratio - 1.0
    scaled    = deviation / TILT_SCALE
    capped    = abs(scaled) > 1.0
    scaled    = float(np.clip(scaled, -1.0, 1.0))
    return scaled * TILT_CAP, capped


def compute_regime_stats(regime_by_month: pd.Series) -> dict[str, RegimeReturnStats]:
    """
    Compute regime-conditional market statistics and equity tilts.

    Parameters
    ----------
    regime_by_month : pd.Series
        DatetimeIndex → regime label, one row per month (the smoothed sequence).

    Returns
    -------
    dict[str, RegimeReturnStats]
        Empty when CRSP data is unavailable — callers must treat an empty result
        as "no tilt can be derived" and fall back to a zero tilt explicitly,
        rather than substituting an invented number.
    """
    crsp = load_crsp_returns()
    if crsp is None:
        return {}

    joined = (
        pd.DataFrame({"ret": crsp["vwretd"]})
        .join(regime_by_month.rename("regime"), how="inner")
        .dropna()
    )
    if joined.empty:
        return {}

    _, _, base_vol = _annualise(joined["ret"])
    if base_vol <= 0:
        return {}

    stats: dict[str, RegimeReturnStats] = {}
    for regime, group in joined.groupby("regime"):
        n = len(group)
        mean_annual, se_annual, vol_annual = _annualise(group["ret"])

        if n < MIN_REGIME_MONTHS or vol_annual <= 0:
            # Too few observations to trust — explicit zero tilt, not a guess.
            stats[regime] = RegimeReturnStats(
                regime=regime, n_months=n, mean_annual=mean_annual,
                se_annual=se_annual, vol_annual=vol_annual,
                vol_ratio=float("nan"), tilt=0.0, tilt_capped=False,
            )
            continue

        vol_ratio    = base_vol / vol_annual
        tilt, capped = _tilt_from_vol_ratio(vol_ratio)
        stats[regime] = RegimeReturnStats(
            regime=regime, n_months=n, mean_annual=mean_annual,
            se_annual=se_annual, vol_annual=vol_annual,
            vol_ratio=vol_ratio, tilt=tilt, tilt_capped=capped,
        )

    return stats


def regime_tilt(stats: dict[str, RegimeReturnStats], regime: str) -> float:
    """Equity tilt for one regime; 0.0 when the regime was never measured."""
    entry = stats.get(regime)
    return entry.tilt if entry is not None else 0.0


def stats_table(stats: dict[str, RegimeReturnStats]) -> pd.DataFrame:
    """Render the regime statistics as a sorted frame for printing and the paper."""
    if not stats:
        return pd.DataFrame()
    rows = [
        {
            "regime":      s.regime,
            "n_months":    s.n_months,
            "mean_annual": round(s.mean_annual, 4),
            "se_annual":   round(s.se_annual, 4),
            "vol_annual":  round(s.vol_annual, 4),
            "vol_ratio":   round(s.vol_ratio, 3),
            "tilt":        round(s.tilt, 4),
            "tilt_capped": s.tilt_capped,
        }
        for s in stats.values()
    ]
    return pd.DataFrame(rows).sort_values("vol_annual").reset_index(drop=True)
