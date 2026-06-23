"""
features.py — Research Agent
==============================
Feature engineering: transforms raw FRED series into stationary,
regime-sensitive signals for PELT and XGBoost.

Exports:
  - SIGNAL_COLS
  - z_score()
  - engineer_features()
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Signal columns used in PELT, clustering, and XGBoost
# ---------------------------------------------------------------------------

SIGNAL_COLS = [
    "yield_curve_z",   "term_spread_z",   "credit_spread_z", "cpi_z",
    "fed_funds_chg",   "unemployment_chg",
    "vix_z",           "indpro_z",        "gdp_z",
    "t5yie_z",         "t10yie_z",
    "dgs5_chg",        "dgs30_chg",
]


def z_score(series):
    """Standardise a series to zero mean, unit variance."""
    return (series - series.mean()) / series.std()


def engineer_features(macro_df):
    """
    Transform raw FRED series into stationary regime-sensitive signals.

    Transformations:
        Z-score  — yield_curve, term_spread, credit_spread, cpi,
                   t5yie, t10yie, gdp (already YoY/QoQ)
        Log + Z  — vix, indpro (right-skewed; log corrects before z-scoring)
        MoM diff — fed_funds, unemployment (turning-point signals)
        1st diff — dgs5, dgs30 (unit-root non-stationary levels)

    Raw series are kept alongside transformed columns for JSON output
    and interpretability.

    Returns:
        pd.DataFrame — transformed features + raw series, dropna applied
    """
    features_df = pd.DataFrame(index=macro_df.index)

    features_df["yield_curve_z"]   = z_score(macro_df["yield_curve"])
    features_df["term_spread_z"]   = z_score(macro_df["term_spread"])
    features_df["credit_spread_z"] = z_score(macro_df["credit_spread"])
    features_df["cpi_z"]           = z_score(macro_df["cpi"])
    features_df["t5yie_z"]         = z_score(macro_df["t5yie"])
    features_df["t10yie_z"]        = z_score(macro_df["t10yie"])
    features_df["gdp_z"]           = z_score(macro_df["gdp"])

    features_df["vix_z"]    = z_score(np.log1p(macro_df["vix"].clip(lower=0)))
    features_df["indpro_z"] = z_score(np.log1p(macro_df["indpro"] + 100))

    features_df["fed_funds_chg"]    = macro_df["fed_funds"].diff()
    features_df["unemployment_chg"] = macro_df["unemployment"].diff()

    features_df["dgs5_chg"]  = macro_df["dgs5"].diff()
    features_df["dgs30_chg"] = macro_df["dgs30"].diff()

    for col in macro_df.columns:
        features_df[col] = macro_df[col]

    features_df.dropna(inplace=True)

    print(f"Features engineered: {features_df.shape}")
    print(f"Signal columns: {len(SIGNAL_COLS)}")
    return features_df
