"""
features.py — 13-feature engineering pipeline.

Raw FRED series are non-stationary level processes. Each is transformed into a
stationary signal before classification:

  Z-scores       — yield curve, term spread, credit spread, CPI, breakevens, GDP
  Log + Z-score  — VIX and INDPRO (right-skew correction)
  Rate of change — fed funds, unemployment (turning-point signals)
  First diffs    — DGS5, DGS30 (unit-root removal)

Raw series are kept alongside for interpretability and JSON output. SIGNAL_COLS
is the 13-column matrix consumed by PELT and the clustering / XGBoost stages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_COLS = [
    "yield_curve_z", "term_spread_z", "credit_spread_z", "cpi_z",
    "fed_funds_chg", "unemployment_chg",
    "vix_z", "indpro_z", "gdp_z",
    "t5yie_z", "t10yie_z",
    "dgs5_chg", "dgs30_chg",
]


def z_score(series: pd.Series) -> pd.Series:
    """Standardise a series to zero mean / unit variance."""
    return (series - series.mean()) / series.std()


def build_features(macro_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the feature matrix from the 13-series macro frame.

    Returns (features_df, SIGNAL_COLS). features_df carries both the 13 signal
    columns and the raw macro columns (for downstream JSON output), with rows
    containing any NaN dropped.
    """
    features_df = pd.DataFrame(index=macro_df.index)

    # Z-scored levels
    features_df["yield_curve_z"]   = z_score(macro_df["yield_curve"])
    features_df["term_spread_z"]   = z_score(macro_df["term_spread"])
    features_df["credit_spread_z"] = z_score(macro_df["credit_spread"])
    features_df["cpi_z"]           = z_score(macro_df["cpi"])
    features_df["t5yie_z"]         = z_score(macro_df["t5yie"])
    features_df["t10yie_z"]        = z_score(macro_df["t10yie"])
    features_df["gdp_z"]           = z_score(macro_df["gdp"])

    # Log-transform then z-score (right-skewed series)
    features_df["vix_z"]    = z_score(np.log1p(macro_df["vix"].clip(lower=0)))
    features_df["indpro_z"] = z_score(np.log1p(macro_df["indpro"] + 100))

    # Rate of change (turning-point signals)
    features_df["fed_funds_chg"]    = macro_df["fed_funds"].diff()
    features_df["unemployment_chg"] = macro_df["unemployment"].diff()

    # First differences (non-stationary level series)
    features_df["dgs5_chg"]  = macro_df["dgs5"].diff()
    features_df["dgs30_chg"] = macro_df["dgs30"].diff()

    # Keep raw series alongside for interpretability and JSON output
    for col in macro_df.columns:
        features_df[col] = macro_df[col]

    features_df.dropna(inplace=True)
    return features_df, list(SIGNAL_COLS)
