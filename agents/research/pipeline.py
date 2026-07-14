"""
pipeline.py — Research Agent ML pipeline: feature engineering, change-point
detection, regime clustering, XGBoost labelling, and smoothing.

Stage order:
  1. build_features()       13-signal stationary matrix from raw FRED series
  2. detect_change_points() PELT structural breaks in the signal matrix
  3. cluster_segments()     K-means sanity check on segment fingerprints (diagnostic only)
  4. train_and_predict()    XGBoost trained on 5 anchor windows → regime labels
  5. smooth_regimes()       6-month rolling majority vote → regime_label_smoothed
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Serialises cluster_segments() so parallel pipeline workers cannot interleave
# fingerprint construction and produce a partial / corrupt segment frame.
_CLUSTER_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Stage 1 — Feature engineering
# ---------------------------------------------------------------------------

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

    Transforms applied:
      Z-scores       — yield curve, term spread, credit spread, CPI, breakevens, GDP
      Log + Z-score  — VIX and INDPRO (right-skew correction)
      Rate of change — fed funds, unemployment (turning-point signals)
      First diffs    — DGS5, DGS30 (unit-root removal)
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


# ---------------------------------------------------------------------------
# Stage 2 — PELT change-point detection
# ---------------------------------------------------------------------------

def detect_change_points(
    features_df: pd.DataFrame,
    signal_cols: list[str],
    pen: float = 10.0,
    model: str = "rbf",
):
    """
    Detect structural breaks via PELT (Pruned Exact Linear Time).

    Returns (break_dates, breakpoints):
      break_dates  — list of pandas Timestamps marking the last month of each
                     completed segment (excludes the final series index)
      breakpoints  — raw ruptures index list (end-exclusive, includes len(df))

    Parameters
    ----------
    pen : float
        Regularisation penalty. Higher → fewer breakpoints (coarser
        segmentation); lower → more breakpoints (finer, may over-segment noise).
    model : str
        ruptures cost model / kernel. "rbf" (default) captures non-linear
        multivariate signal shifts; "l1" is robust to outliers; "l2" is fast but
        outlier-sensitive.
    """
    import ruptures as rpt

    signal_matrix = features_df[signal_cols].values
    pelt = rpt.Pelt(model=model).fit(signal_matrix)
    breakpoints = pelt.predict(pen=pen)

    break_dates = [features_df.index[i - 1] for i in breakpoints[:-1]]

    print(f"Detected {len(break_dates)} structural breaks:")
    for d in break_dates:
        print(f"  {d.strftime('%Y-%m')}")
    return break_dates, breakpoints


# ---------------------------------------------------------------------------
# Stage 3 — K-means segment clustering (diagnostic only)
# ---------------------------------------------------------------------------

def _validate_cluster_inputs(
    features_df: pd.DataFrame, signal_cols: list[str], break_dates: list
) -> list[int]:
    """
    Data-integrity gate for cluster_segments(). Raises ValueError before any
    clustering work begins if inputs cannot produce a valid segmentation.
    Returns the validated, end-exclusive break_indices list on success.
    """
    if features_df is None or features_df.empty:
        msg = "cluster_segments: input feature matrix is empty"
        logger.error(msg)
        raise ValueError(msg)

    missing = [c for c in signal_cols if c not in features_df.columns]
    if missing:
        msg = f"cluster_segments: signal columns absent from feature matrix: {missing}"
        logger.error(msg)
        raise ValueError(msg)

    nan_cols = [c for c in signal_cols if features_df[c].isna().any()]
    if nan_cols:
        msg = f"cluster_segments: NaN values present in signal columns {nan_cols}"
        logger.error(msg)
        raise ValueError(msg)

    out_of_range = [d for d in break_dates if d not in features_df.index]
    if out_of_range:
        msg = f"cluster_segments: break dates not found in feature index: {out_of_range}"
        logger.error(msg)
        raise ValueError(msg)

    break_indices = (
        [0]
        + [int(features_df.index.get_loc(d)) for d in break_dates]
        + [len(features_df)]
    )
    if any(b <= a for a, b in zip(break_indices, break_indices[1:])):
        msg = f"cluster_segments: break indices are non-monotonic: {break_indices}"
        logger.error(msg)
        raise ValueError(msg)

    n_segments = len(break_indices) - 1
    if n_segments < 2:
        msg = (
            f"cluster_segments: only {n_segments} segment(s) detected — "
            "at least 2 are required to run K-means"
        )
        logger.error(msg)
        raise ValueError(msg)

    return break_indices


def cluster_segments(
    features_df: pd.DataFrame, signal_cols: list[str], break_dates: list
) -> pd.DataFrame:
    """
    Build segment fingerprints (mean/var/delta per signal) and K-means cluster
    them. K is chosen by max silhouette, constrained to [2, n_segments − 1].

    Diagnostic only — the returned frame is NOT consumed by run_research_agent().
    XGBoost labels the full monthly feature matrix, not these segment fingerprints.
    Thread-safe via a module-level lock.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    with _CLUSTER_LOCK:
        break_indices = _validate_cluster_inputs(features_df, signal_cols, break_dates)

        segments = []
        for i in range(len(break_indices) - 1):
            start, end = break_indices[i], break_indices[i + 1]
            seg = features_df.iloc[start:end]
            seg_features = {}
            for col in signal_cols:
                seg_features[f"{col}_mean"] = seg[col].mean()
                seg_features[f"{col}_var"]  = seg[col].var()
                seg_features[f"{col}_delta"] = seg[col].iloc[-1] - seg[col].iloc[0]
            seg_features["start_date"] = features_df.index[start]
            seg_features["end_date"]   = features_df.index[end - 1]
            seg_features["n_months"]   = end - start
            segments.append(seg_features)

        seg_df = pd.DataFrame(segments)
        feature_cols = [c for c in seg_df.columns if c not in ("start_date", "end_date", "n_months")]
        X_seg = StandardScaler().fit_transform(seg_df[feature_cols])

        k_range = list(range(2, min(5, len(seg_df))))
        if not k_range:
            seg_df["cluster"] = 0
            print("Only two segments — clustering skipped (K-means needs K ≥ 2 < n).")
            return seg_df

        sil_scores = []
        for k in k_range:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_seg)
            sil_scores.append(silhouette_score(X_seg, labels))

        best_k = k_range[sil_scores.index(max(sil_scores))]
        seg_df["cluster"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_seg)
        print(f"Optimal K (max silhouette): {best_k}")
        return seg_df


# ---------------------------------------------------------------------------
# Stage 4 — XGBoost regime labelling
# ---------------------------------------------------------------------------

# Academic regime taxonomy grounded in the NBER business cycle and
# monetary-policy literature (Hamilton 1989; Clarida, Galí & Gertler 1999; Bernanke 2020).
REGIME_NAMES = {
    0: "Early Recovery",
    1: "Late-Cycle Expansion",
    2: "Financial Crisis & ZLB",
    3: "Moderate Expansion",
    4: "Inflation Shock",
}

# Anchor windows — one per PELT segment, from the most historically unambiguous
# months inside each segment boundary.
ANCHOR_WINDOWS = {
    0: ("2003-06", "2004-06"),   # Early Recovery: post dot-com, rates low, spreads narrowing
    1: ("2005-01", "2007-06"),   # Late-Cycle Expansion: pre-GFC boom, rising rates, low VIX
    2: ("2008-09", "2010-12"),   # Financial Crisis & ZLB: acute GFC + early ZIRP
    3: ("2015-01", "2019-06"),   # Moderate Expansion: post-QE normalisation, stable macro
    4: ("2022-01", "2023-06"),   # Inflation Shock: peak CPI, aggressive Fed tightening
}


def train_and_predict(features_df: pd.DataFrame, signal_cols: list[str]):
    """
    Train XGBoost on the anchor windows and label the full monthly matrix.

    Mutates features_df in place, adding:
      regime_confidence — max class probability per month
      regime_cluster    — predicted regime id
      regime_label      — academic label

    Returns (model, feature_importance Series). Anchor windows outside the data
    range are skipped so partial histories still train.
    """
    from xgboost import XGBClassifier

    train_frames = []
    for label, (start, end) in ANCHOR_WINDOWS.items():
        window = features_df.loc[start:end, signal_cols].copy()
        if window.empty:
            print(f"WARNING: anchor window {start}–{end} for '{REGIME_NAMES[label]}' "
                  "is empty — skipping")
            continue
        window["label"] = label
        train_frames.append(window)

    if not train_frames:
        raise ValueError("No anchor windows fell inside the data range; cannot train.")

    train_df = pd.concat(train_frames).sort_index()
    X_train, y_train = train_df[signal_cols], train_df["label"]

    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss")
    model.fit(X_train, y_train)

    proba = model.predict_proba(features_df[signal_cols])
    features_df["regime_confidence"] = proba.max(axis=1).round(3)
    features_df["regime_cluster"]    = model.predict(features_df[signal_cols])
    features_df["regime_label"]      = features_df["regime_cluster"].map(REGIME_NAMES)

    feat_importance = pd.Series(
        model.feature_importances_, index=signal_cols
    ).sort_values(ascending=False)
    return model, feat_importance


# ---------------------------------------------------------------------------
# Stage 5 — Regime smoothing
# ---------------------------------------------------------------------------

def smooth_regimes(features_df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """
    Apply a centred rolling majority vote over `window` months to enforce regime
    persistence. Adds regime_smoothed (id) and regime_label_smoothed (label).
    """
    from scipy import stats

    features_df["regime_smoothed"] = (
        features_df["regime_cluster"]
        .rolling(window, center=True, min_periods=1)
        .apply(lambda x: stats.mode(x, keepdims=True)[0][0])
        .astype(int)
    )
    features_df["regime_label_smoothed"] = features_df["regime_smoothed"].map(REGIME_NAMES)
    return features_df
