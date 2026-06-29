"""
test_research.py — Research Agent unit tests.

Run from the repo root:
    pytest agents/research/test_research.py -v

Lightweight tests (pydantic + pandas/numpy) cover feature engineering, per-row
validation, derived-field logic, and the contract snapshot. The full
PELT→XGBoost pipeline is exercised only when ruptures + xgboost are installed;
otherwise that smoke test skips.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contracts import MacroRegimeSnapshot
from agents.research.features import SIGNAL_COLS, build_features
from agents.research.adapters import (
    RegimeRecord,
    add_derived_fields,
    build_regime_sequence,
    build_snapshot,
)


# ── Synthetic fixtures ─────────────────────────────────────────────────────

def _synthetic_macro(n: int = 36) -> pd.DataFrame:
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    cols = [
        "yield_curve", "term_spread", "fed_funds", "unemployment", "cpi",
        "credit_spread", "indpro", "vix", "gdp", "t5yie", "t10yie", "dgs5", "dgs30",
    ]
    return pd.DataFrame(
        {c: rng.normal(loc=2.0, scale=1.0, size=n) for c in cols}, index=idx
    )


def _synthetic_features(n: int = 8) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="MS")
    labels = ["Moderate Expansion"] * (n - 2) + ["Late-Cycle Expansion"] * 2
    return pd.DataFrame(
        {
            "regime_label_smoothed": labels,
            "regime_confidence": np.linspace(0.7, 0.9, n),
            "yield_curve": np.linspace(0.5, 0.7, n),
            "term_spread": np.linspace(0.4, 0.5, n),
            "fed_funds": np.linspace(3.0, 3.5, n),
            "unemployment": np.linspace(4.0, 4.2, n),
            "cpi": np.linspace(2.5, 2.7, n),
            "credit_spread": np.linspace(1.8, 1.9, n),
            "vix": np.linspace(13.0, 15.0, n),
            "indpro": np.linspace(1.0, 1.3, n),
        },
        index=idx,
    )


# ── Feature engineering ────────────────────────────────────────────────────

def test_build_features_returns_signal_cols():
    features_df, signal_cols = build_features(_synthetic_macro())
    assert signal_cols == SIGNAL_COLS
    assert len(signal_cols) == 13
    for col in signal_cols:
        assert col in features_df.columns
    # raw columns retained for JSON output
    assert "credit_spread" in features_df.columns
    assert not features_df[signal_cols].isna().any().any()


# ── Per-row validation ─────────────────────────────────────────────────────

def _valid_record() -> dict:
    return {
        "regime_label": "Moderate Expansion",
        "prior_regime": "None",
        "regime_shift_date": "2024-01-01",
        "regime_confidence": 0.85,
        "regime_volatility": 0.03,
        "yield_curve": 0.7, "term_spread": 0.4, "fed_funds": 3.5,
        "unemployment": 4.1, "cpi": 2.6, "credit_spread": 1.8,
        "vix": 14.9, "indpro": 1.2,
    }


def test_regime_record_accepts_valid():
    assert RegimeRecord(**_valid_record()).regime_label == "Moderate Expansion"


def test_regime_record_rejects_bad_label():
    bad = _valid_record()
    bad["regime_label"] = "AI Boom"
    with pytest.raises(Exception):
        RegimeRecord(**bad)


def test_regime_record_rejects_out_of_range_confidence():
    bad = _valid_record()
    bad["regime_confidence"] = 1.5
    with pytest.raises(Exception):
        RegimeRecord(**bad)


# ── Derived fields + sequence + snapshot ───────────────────────────────────

def test_add_derived_fields_and_sequence():
    df = add_derived_fields(_synthetic_features())
    assert "regime_volatility" in df.columns
    assert "prior_regime" in df.columns
    assert "regime_shift_date" in df.columns
    # first prior_regime is 'None'
    assert df["prior_regime"].iloc[0] == "None"

    sequence = build_regime_sequence(df)
    assert len(sequence) == len(df)
    assert all("regime_label" in rec for rec in sequence.values())


def test_build_snapshot_is_contract_and_sets_flags():
    df = add_derived_fields(_synthetic_features())
    snap = build_snapshot(df)
    assert isinstance(snap, MacroRegimeSnapshot)
    # last two rows are Late-Cycle Expansion preceded by Moderate Expansion →
    # the most recent month's prior equals its own label (no change at the end)
    assert snap.regime_change_detected == (snap.regime_label != snap.prior_regime)
    assert snap.is_low_confidence == (snap.regime_confidence < 0.60)


# ── cluster_segments race-condition / data-integrity guard ─────────────────

def test_cluster_segments_rejects_empty_matrix():
    """Empty input matrix raises ValueError before any clustering work."""
    pytest.importorskip("sklearn")
    from agents.research.regime_model import cluster_segments
    with pytest.raises(ValueError, match="empty"):
        cluster_segments(pd.DataFrame(), list(SIGNAL_COLS), [])


def test_cluster_segments_rejects_nan_signal():
    """A NaN in any signal column is rejected with the offending column logged."""
    pytest.importorskip("sklearn")
    from agents.research.regime_model import cluster_segments
    features_df, signal_cols = build_features(_synthetic_macro(n=36))
    break_date = features_df.index[len(features_df) // 2]
    features_df.loc[features_df.index[0], signal_cols[0]] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cluster_segments(features_df, signal_cols, [break_date])


def test_cluster_segments_rejects_out_of_range_break():
    """A break date absent from the feature index is rejected."""
    pytest.importorskip("sklearn")
    from agents.research.regime_model import cluster_segments
    features_df, signal_cols = build_features(_synthetic_macro(n=36))
    bogus = pd.Timestamp("1900-01-01")
    with pytest.raises(ValueError, match="not found in feature index"):
        cluster_segments(features_df, signal_cols, [bogus])


def test_cluster_segments_rejects_single_segment():
    """No break dates → a single segment → cannot run K-means → ValueError."""
    pytest.importorskip("sklearn")
    from agents.research.regime_model import cluster_segments
    features_df, signal_cols = build_features(_synthetic_macro(n=36))
    with pytest.raises(ValueError, match="segment"):
        cluster_segments(features_df, signal_cols, [])


# ── Full pipeline smoke test (heavy deps) ──────────────────────────────────

def test_full_pipeline_smoke():
    pytest.importorskip("ruptures")
    pytest.importorskip("xgboost")
    pytest.importorskip("scipy")
    from agents.research.detection import detect_change_points
    from agents.research.regime_model import train_and_predict, smooth_regimes

    features_df, signal_cols = build_features(_synthetic_macro(n=300))
    # PELT may find few/no breaks on noise — that is fine for a smoke test.
    detect_change_points(features_df, signal_cols, pen=10.0)
    try:
        train_and_predict(features_df, signal_cols)
    except ValueError:
        pytest.skip("synthetic data has no anchor-window overlap")
    smooth_regimes(features_df)
    assert "regime_label_smoothed" in features_df.columns
