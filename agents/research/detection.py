"""
detection.py — Research Agent
===============================
PELT change-point detection and K-means segment clustering.

Exports:
  - detect_change_points()
  - cluster_segments()
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .features import SIGNAL_COLS


def detect_change_points(features_df, pen=10):
    """
    PELT (Pruned Exact Linear Time) change-point detection, RBF kernel.

    pen=10 requires a large multivariate shift to trigger a break, filtering
    out minor transitions and retaining only economically significant ones.

    June 2026 result: 4 breaks → 5 segments
        2004-09, 2008-01, 2014-09, 2020-12

    Returns:
        list[pd.Timestamp] — break dates (excludes final data point)
    """
    signal_matrix = features_df[SIGNAL_COLS].values
    model = rpt.Pelt(model="rbf").fit(signal_matrix)
    breakpoints = model.predict(pen=pen)

    break_dates = [features_df.index[i - 1] for i in breakpoints[:-1]]

    print(f"Detected {len(break_dates)} structural breaks:")
    for d in break_dates:
        print(f"  {d.strftime('%Y-%m')}")

    return break_dates


def cluster_segments(features_df, break_dates):
    """
    Build segment fingerprints (mean, variance, delta per signal column)
    and cluster them with K-means, selecting K via max silhouette score.

    K is constrained to [2, N_segments − 1]. With 5 PELT segments,
    K=2 is expected — a coarse binary split (stress/low-growth vs.
    expansion/tightening). Fine-grained labelling is done by XGBoost.

    Returns:
        seg_df (pd.DataFrame) — segments with cluster assignments
        best_k (int)          — optimal K selected
    """
    break_indices = (
        [0]
        + [features_df.index.get_loc(d) for d in break_dates]
        + [len(features_df)]
    )
    segments = []

    for i in range(len(break_indices) - 1):
        start = break_indices[i]
        end   = break_indices[i + 1]
        seg   = features_df.iloc[start:end]

        seg_features = {}
        for col in SIGNAL_COLS:
            seg_features[f"{col}_mean"]  = seg[col].mean()
            seg_features[f"{col}_var"]   = seg[col].var()
            seg_features[f"{col}_delta"] = seg[col].iloc[-1] - seg[col].iloc[0]

        seg_features["start_date"] = features_df.index[start]
        seg_features["end_date"]   = features_df.index[end - 1]
        seg_features["n_months"]   = end - start
        segments.append(seg_features)

    seg_df = pd.DataFrame(segments)
    feature_cols = [c for c in seg_df.columns if c not in ["start_date", "end_date", "n_months"]]

    scaler = StandardScaler()
    X_seg  = scaler.fit_transform(seg_df[feature_cols])

    k_range    = range(2, min(5, len(seg_df)))
    sil_scores = []

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_seg)
        sil_scores.append(silhouette_score(X_seg, labels))

    best_k = list(k_range)[sil_scores.index(max(sil_scores))]
    print(f"Optimal K (max silhouette): {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    seg_df["cluster"] = kmeans.fit_predict(X_seg)

    print("\n=== Segments & Cluster Assignments ===")
    print(seg_df[["start_date", "end_date", "n_months", "cluster"]].to_string(index=False))

    return seg_df, best_k
