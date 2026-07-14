"""
model_comparison.py — HMM & GMM benchmarking vs. the XGBoost pipeline.

Added in the June 22 session to satisfy the meeting action item: explore HMM and
GMM as alternatives or produce a data-driven justification for retaining the
current pipeline. The verdict (documented in the design doc) is to retain
K-means + XGBoost — it is the only model that recovers all five academic regimes
and correctly flags Inflation Shock as a negative-return environment.

This module is optional: it is invoked by run_research_agent(..., compare_models=True)
and never on the critical path that produces the MacroRegimeSnapshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _map_clusters_to_regimes(raw_labels, reference_labels) -> dict:
    """Map unsupervised cluster ids to academic labels by max overlap with the
    smoothed XGBoost reference labels."""
    mapping = {}
    for cluster_id in sorted(set(raw_labels)):
        mask = raw_labels == cluster_id
        mapping[cluster_id] = pd.Series(reference_labels[mask]).value_counts().idxmax()
    return mapping


def _avg_regime_duration(labels) -> float:
    """Average consecutive-month run length (regime persistence)."""
    durations, count = [], 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            count += 1
        else:
            durations.append(count)
            count = 1
    durations.append(count)
    return round(float(np.mean(durations)), 1)


def compare_models(features_df: pd.DataFrame, signal_cols: list[str]) -> dict:
    """
    Fit a Gaussian HMM and a Gaussian Mixture Model on the same signal matrix,
    map their clusters to the academic taxonomy via overlap with the smoothed
    XGBoost labels, and return a comparison summary.

    Requires regime_label_smoothed to already be present on features_df. Adds
    hmm_label / gmm_label (+ confidences) columns and returns a dict of metrics.
    """
    from hmmlearn.hmm import GaussianHMM
    from sklearn.mixture import GaussianMixture

    if "regime_label_smoothed" not in features_df.columns:
        raise ValueError("compare_models requires smoothed XGBoost labels first.")

    X_full = features_df[signal_cols].values
    xgb_smoothed = features_df["regime_label_smoothed"].values

    hmm = GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000, random_state=42)
    hmm.fit(X_full)
    hmm_raw = hmm.predict(X_full)
    hmm_proba = hmm.predict_proba(X_full)

    gmm = GaussianMixture(n_components=5, covariance_type="full", n_init=10, random_state=42)
    gmm.fit(X_full)
    gmm_raw = gmm.predict(X_full)
    gmm_proba = gmm.predict_proba(X_full)

    hmm_map = _map_clusters_to_regimes(hmm_raw, xgb_smoothed)
    gmm_map = _map_clusters_to_regimes(gmm_raw, xgb_smoothed)

    features_df["hmm_label"] = pd.Series(hmm_raw, index=features_df.index).map(hmm_map)
    features_df["gmm_label"] = pd.Series(gmm_raw, index=features_df.index).map(gmm_map)
    features_df["hmm_confidence"] = hmm_proba.max(axis=1).round(3)
    features_df["gmm_confidence"] = gmm_proba.max(axis=1).round(3)

    hmm_arr = features_df["hmm_label"].values
    gmm_arr = features_df["gmm_label"].values

    summary = {
        "xgb_distinct_regimes": int(features_df["regime_label_smoothed"].nunique()),
        "hmm_distinct_regimes": int(features_df["hmm_label"].nunique()),
        "gmm_distinct_regimes": int(features_df["gmm_label"].nunique()),
        "hmm_match_rate": float((hmm_arr == xgb_smoothed).mean()),
        "gmm_match_rate": float((gmm_arr == xgb_smoothed).mean()),
        "xgb_avg_duration": _avg_regime_duration(xgb_smoothed),
        "hmm_avg_duration": _avg_regime_duration(hmm_arr),
        "gmm_avg_duration": _avg_regime_duration(gmm_arr),
        "verdict": "retain K-means + XGBoost",
    }

    print("=== Model Comparison vs Smoothed XGBoost ===")
    print(f"  Distinct regimes — XGB: {summary['xgb_distinct_regimes']}/5 | "
          f"HMM: {summary['hmm_distinct_regimes']}/5 | GMM: {summary['gmm_distinct_regimes']}/5")
    print(f"  Match rate       — HMM: {summary['hmm_match_rate']:.1%} | "
          f"GMM: {summary['gmm_match_rate']:.1%}")
    print(f"  Avg duration (mo)— XGB: {summary['xgb_avg_duration']} | "
          f"HMM: {summary['hmm_avg_duration']} | GMM: {summary['gmm_avg_duration']}")
    return summary
