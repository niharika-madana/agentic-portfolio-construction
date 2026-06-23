"""
model_comparison.py — Research Agent
======================================
HMM and GMM alternative model comparison against XGBoost.
Produces match rates, regime duration stats, and CRSP return validation.

June 22 verdict: retain K-means + XGBoost pipeline.
See 22JUN_ResearchAgent_Design.md — Model Comparison section.

Exports:
  - fit_hmm_gmm()
  - map_clusters_to_regimes()
  - compute_comparison_metrics()
  - avg_regime_duration()
  - crsp_summary()
  - run_model_comparison()
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

from .features import SIGNAL_COLS


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_hmm_gmm(features_df, n_components=5):
    """
    Fit Gaussian HMM (diagonal covariance) and GMM (full covariance)
    on the same 13 signal_cols feature set.

    Diagonal covariance for HMM: full covariance requires 13×13=169
    parameters per state, infeasible with ~270 observations.

    Returns:
        hmm_model, gmm_model, hmm_raw_labels, gmm_raw_labels,
        hmm_proba, gmm_proba
    """
    X_full = features_df[SIGNAL_COLS].values

    hmm_model = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=1000,
        random_state=42,
    )
    hmm_model.fit(X_full)
    hmm_raw_labels = hmm_model.predict(X_full)
    hmm_proba      = hmm_model.predict_proba(X_full)

    print(f"HMM converged: {hmm_model.monitor_.converged}")
    print(f"HMM raw label distribution:\n{pd.Series(hmm_raw_labels).value_counts().sort_index()}")

    gmm_model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=10,
        random_state=42,
    )
    gmm_model.fit(X_full)
    gmm_raw_labels = gmm_model.predict(X_full)
    gmm_proba      = gmm_model.predict_proba(X_full)

    print(f"\nGMM converged: {gmm_model.converged_}")
    print(f"GMM raw label distribution:\n{pd.Series(gmm_raw_labels).value_counts().sort_index()}")

    return hmm_model, gmm_model, hmm_raw_labels, gmm_raw_labels, hmm_proba, gmm_proba


# ---------------------------------------------------------------------------
# Cluster → academic label mapping
# ---------------------------------------------------------------------------

def map_clusters_to_regimes(raw_labels, reference_labels):
    """
    For each unsupervised cluster ID, find the academic regime label
    with the highest overlap against the smoothed XGBoost labels.

    Returns:
        dict — {cluster_id: academic_label}
    """
    mapping = {}
    for cluster_id in sorted(set(raw_labels)):
        mask        = raw_labels == cluster_id
        most_common = pd.Series(reference_labels[mask]).value_counts().idxmax()
        mapping[cluster_id] = most_common
    return mapping


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def avg_regime_duration(labels):
    """Average number of consecutive months per regime label."""
    durations = []
    count = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            count += 1
        else:
            durations.append(count)
            count = 1
    durations.append(count)
    return round(np.mean(durations), 1)


def compute_comparison_metrics(features_df):
    """
    Compute and print the full model comparison:
        - Match rate vs smoothed XGBoost labels
        - Average regime duration (persistence)
        - Distinct regimes recovered (out of 5)

    Requires features_df to have hmm_label and gmm_label columns.
    """
    xgb_arr = features_df["regime_label_smoothed"].values
    hmm_arr = features_df["hmm_label"].values
    gmm_arr = features_df["gmm_label"].values

    hmm_match = (hmm_arr == xgb_arr).mean()
    gmm_match = (gmm_arr == xgb_arr).mean()

    print("=== Match Rate vs Smoothed XGBoost Labels ===")
    print(f"  HMM match rate: {hmm_match:.1%}")
    print(f"  GMM match rate: {gmm_match:.1%}")

    print("\n=== Average Regime Duration (months) ===")
    print(f"  XGBoost (smoothed): {avg_regime_duration(xgb_arr)}")
    print(f"  HMM:                {avg_regime_duration(hmm_arr)}")
    print(f"  GMM:                {avg_regime_duration(gmm_arr)}")

    print("\n=== Distinct Regime Labels Recovered (out of 5) ===")
    print(f"  XGBoost: {features_df['regime_label_smoothed'].nunique()}")
    print(f"  HMM:     {features_df['hmm_label'].nunique()}")
    print(f"  GMM:     {features_df['gmm_label'].nunique()}")


# ---------------------------------------------------------------------------
# CRSP return validation
# ---------------------------------------------------------------------------

def crsp_summary(features_df, label_col, model_name, crsp_path="crsp_market_index.csv"):
    """
    Compute average monthly return and volatility per regime label
    against the CRSP value-weighted market index.
    """
    crsp = pd.read_csv(crsp_path, parse_dates=["date"])
    crsp = crsp.set_index("date")
    crsp.index = crsp.index.to_period("M").to_timestamp("s")

    df = features_df[[label_col]].join(crsp[["vwretd"]], how="left")
    summary = (
        df.groupby(label_col)["vwretd"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "Avg Return", "std": "Std Dev", "count": "Months"})
    )
    summary["Avg Return"] = (summary["Avg Return"] * 100).round(2)
    summary["Std Dev"]    = (summary["Std Dev"] * 100).round(2)
    summary.index.name    = "Regime"

    print(f"\n=== CRSP Validation — {model_name} ===")
    print(summary.to_string())
    return summary


def run_model_comparison(features_df, crsp_path="crsp_market_index.csv"):
    """
    Full model comparison pipeline:
        1. Fit HMM and GMM
        2. Map clusters to academic labels
        3. Attach labels to features_df
        4. Compute match rates, durations, distinct regimes
        5. CRSP return validation for all three models

    Returns:
        features_df (pd.DataFrame) — with hmm_label, gmm_label columns added
    """
    xgb_smoothed = features_df["regime_label_smoothed"].values

    _, _, hmm_raw, gmm_raw, hmm_proba, gmm_proba = fit_hmm_gmm(features_df)

    hmm_mapping = map_clusters_to_regimes(hmm_raw, xgb_smoothed)
    gmm_mapping = map_clusters_to_regimes(gmm_raw, xgb_smoothed)

    print(f"\nHMM cluster → regime mapping: {hmm_mapping}")
    print(f"GMM cluster → regime mapping: {gmm_mapping}")

    features_df["hmm_label"]      = pd.Series(hmm_raw, index=features_df.index).map(hmm_mapping)
    features_df["gmm_label"]      = pd.Series(gmm_raw, index=features_df.index).map(gmm_mapping)
    features_df["hmm_confidence"] = hmm_proba.max(axis=1).round(3)
    features_df["gmm_confidence"] = gmm_proba.max(axis=1).round(3)

    print(f"\nHMM label distribution:\n{features_df['hmm_label'].value_counts()}")
    print(f"\nGMM label distribution:\n{features_df['gmm_label'].value_counts()}")

    compute_comparison_metrics(features_df)

    crsp_summary(features_df, "regime_label_smoothed", "XGBoost (smoothed)", crsp_path)
    crsp_summary(features_df, "hmm_label",             "HMM",                crsp_path)
    crsp_summary(features_df, "gmm_label",             "GMM",                crsp_path)

    return features_df
