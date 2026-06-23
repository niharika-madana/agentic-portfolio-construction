"""
test_research.py — Research Agent unit tests
=============================================
No live FRED calls — all tests use synthetic DataFrames.

Coverage:
    1–7.  RegimeRecord Pydantic validation
    8–12. Feature engineering (SIGNAL_COLS, z_score, no NaN)
   13–14. Regime smoothing
   15–16. build_regime_sequence (validation gate, non-empty output)
   17–18. Cluster → regime mapping
   19–20. avg_regime_duration
   21–22. to_macro_regime_snapshot (adapters.py)

Run with: pytest agents/research/test_research.py -v
"""

import numpy as np
import pandas as pd
import pytest

from .adapters import to_macro_regime_snapshot
from .features import SIGNAL_COLS, engineer_features, z_score
from .model_comparison import avg_regime_duration, map_clusters_to_regimes
from .regime_model import (
    REGIME_NAMES,
    VALID_REGIME_LABELS,
    RegimeRecord,
    build_regime_sequence,
    smooth_regimes,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_macro_df():
    """Minimal macro DataFrame covering all 13 raw FRED series."""
    dates = pd.date_range("2003-02-01", periods=60, freq="MS")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "yield_curve":   np.random.normal(0.5,  0.5,  60),
            "term_spread":   np.random.normal(1.0,  0.5,  60),
            "credit_spread": np.random.normal(2.0,  0.5,  60),
            "cpi":           np.random.normal(2.5,  1.0,  60),
            "fed_funds":     np.random.normal(2.0,  1.5,  60),
            "unemployment":  np.random.normal(5.0,  1.0,  60),
            "indpro":        np.random.normal(2.0,  3.0,  60),
            "vix":           np.random.normal(15.0, 5.0,  60).clip(min=1),
            "gdp":           np.random.normal(2.5,  1.5,  60),
            "t5yie":         np.random.normal(2.0,  0.5,  60),
            "t10yie":        np.random.normal(2.2,  0.5,  60),
            "dgs5":          np.random.normal(2.5,  1.0,  60),
            "dgs30":         np.random.normal(3.5,  0.5,  60),
        },
        index=dates,
    )


@pytest.fixture
def mock_features_df(mock_macro_df):
    return engineer_features(mock_macro_df)


@pytest.fixture
def mock_features_with_regimes(mock_features_df):
    """Add regime columns needed for smoothing and sequence tests."""
    df = mock_features_df.copy()
    np.random.seed(42)
    raw_clusters = np.random.randint(0, 5, len(df))
    df["regime_cluster"]    = raw_clusters
    df["regime_label"]      = pd.Series(raw_clusters).map(REGIME_NAMES).values
    df["regime_confidence"] = np.random.uniform(0.5, 1.0, len(df))
    return df


# ---------------------------------------------------------------------------
# 1–7. RegimeRecord Pydantic validation
# ---------------------------------------------------------------------------

VALID_RECORD = {
    "regime_label":      "Early Recovery",
    "prior_regime":      "None",
    "regime_shift_date": "2003-02-01",
    "regime_confidence": 0.85,
    "regime_volatility": 0.03,
    "yield_curve":       0.5,
    "term_spread":       1.0,
    "fed_funds":         1.0,
    "unemployment":      5.5,
    "cpi":               2.3,
    "credit_spread":     1.8,
    "vix":               14.5,
    "indpro":            1.2,
}


def test_valid_regime_record_passes():
    rec = RegimeRecord(**VALID_RECORD)
    assert rec.regime_label == "Early Recovery"


def test_invalid_regime_label_rejected():
    with pytest.raises(Exception):
        RegimeRecord(**dict(VALID_RECORD, regime_label="Dot-com Crash"))


def test_confidence_above_one_rejected():
    with pytest.raises(Exception):
        RegimeRecord(**dict(VALID_RECORD, regime_confidence=1.5))


def test_negative_volatility_rejected():
    with pytest.raises(Exception):
        RegimeRecord(**dict(VALID_RECORD, regime_volatility=-0.01))


def test_all_five_labels_are_valid():
    for label in VALID_REGIME_LABELS:
        rec = RegimeRecord(**dict(VALID_RECORD, regime_label=label))
        assert rec.regime_label == label


def test_prior_regime_none_is_valid():
    rec = RegimeRecord(**dict(VALID_RECORD, prior_regime="None"))
    assert rec.prior_regime == "None"


def test_invalid_prior_regime_rejected():
    with pytest.raises(Exception):
        RegimeRecord(**dict(VALID_RECORD, prior_regime="Dot-com Crash"))


# ---------------------------------------------------------------------------
# 8–12. Feature engineering
# ---------------------------------------------------------------------------

def test_signal_cols_count():
    assert len(SIGNAL_COLS) == 13


def test_all_signal_cols_present(mock_features_df):
    for col in SIGNAL_COLS:
        assert col in mock_features_df.columns, f"Missing: {col}"


def test_no_nan_in_signal_cols(mock_features_df):
    assert mock_features_df[SIGNAL_COLS].isna().sum().sum() == 0


def test_z_score_mean_near_zero(mock_macro_df):
    result = z_score(mock_macro_df["yield_curve"])
    assert abs(result.mean()) < 1e-10


def test_z_score_std_near_one(mock_macro_df):
    result = z_score(mock_macro_df["yield_curve"])
    assert abs(result.std() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# 13–14. Regime smoothing
# ---------------------------------------------------------------------------

def test_smoothing_creates_column(mock_features_with_regimes):
    df = smooth_regimes(mock_features_with_regimes)
    assert "regime_label_smoothed" in df.columns


def test_smoothed_labels_are_valid(mock_features_with_regimes):
    df = smooth_regimes(mock_features_with_regimes)
    assert df["regime_label_smoothed"].dropna().isin(VALID_REGIME_LABELS).all()


# ---------------------------------------------------------------------------
# 15–16. build_regime_sequence
# ---------------------------------------------------------------------------

def test_regime_sequence_only_valid_rows(mock_features_with_regimes):
    df = smooth_regimes(mock_features_with_regimes)
    sequence = build_regime_sequence(df)
    for date_str, record in sequence.items():
        assert record["regime_label"] in VALID_REGIME_LABELS
        assert 0.0 <= record["regime_confidence"] <= 1.0
        assert record["regime_volatility"] >= 0.0


def test_regime_sequence_not_empty(mock_features_with_regimes):
    df = smooth_regimes(mock_features_with_regimes)
    sequence = build_regime_sequence(df)
    assert len(sequence) > 0


# ---------------------------------------------------------------------------
# 17–18. Cluster → regime mapping
# ---------------------------------------------------------------------------

def test_map_clusters_assigns_most_common():
    raw_labels = np.array([0, 0, 0, 1, 1])
    reference  = np.array([
        "Early Recovery", "Early Recovery", "Inflation Shock",
        "Moderate Expansion", "Moderate Expansion",
    ])
    mapping = map_clusters_to_regimes(raw_labels, reference)
    assert mapping[0] == "Early Recovery"
    assert mapping[1] == "Moderate Expansion"


# ---------------------------------------------------------------------------
# 19–20. avg_regime_duration
# ---------------------------------------------------------------------------

def test_avg_regime_duration_single_run():
    assert avg_regime_duration(["A", "A", "A", "B", "B"]) == 2.5


def test_avg_regime_duration_all_same():
    assert avg_regime_duration(["A"] * 10) == 10.0


def test_avg_regime_duration_all_different():
    assert avg_regime_duration(["A", "B", "C", "D"]) == 1.0


# ---------------------------------------------------------------------------
# 21–22. to_macro_regime_snapshot (adapters.py)
# ---------------------------------------------------------------------------

SAMPLE_SEQUENCE = {
    "2025-11-01": {
        "regime_label":      "Late-Cycle Expansion",
        "prior_regime":      "Late-Cycle Expansion",
        "regime_shift_date": "2024-01-01",
        "regime_confidence": 0.847,
        "regime_volatility": 0.0312,
        "yield_curve":       0.71,
        "term_spread":       0.43,
        "fed_funds":         3.72,
        "unemployment":      4.1,
        "cpi":               2.65,
        "credit_spread":     1.84,
        "vix":               14.95,
        "indpro":            1.23,
    },
    "2025-12-01": {
        "regime_label":      "Late-Cycle Expansion",
        "prior_regime":      "Late-Cycle Expansion",
        "regime_shift_date": "2024-01-01",
        "regime_confidence": 0.862,
        "regime_volatility": 0.0298,
        "yield_curve":       0.68,
        "term_spread":       0.41,
        "fed_funds":         3.50,
        "unemployment":      4.2,
        "cpi":               2.53,
        "credit_spread":     1.79,
        "vix":               13.82,
        "indpro":            1.31,
    },
}


def test_to_macro_regime_snapshot_uses_latest():
    from contracts import MacroRegimeSnapshot
    from datetime import date
    snapshot = to_macro_regime_snapshot(SAMPLE_SEQUENCE)
    assert isinstance(snapshot, MacroRegimeSnapshot)
    assert snapshot.as_of == date(2025, 12, 1)
    assert snapshot.regime_label == "Late-Cycle Expansion"


def test_to_macro_regime_snapshot_sets_derived_flags():
    """is_low_confidence and regime_change_detected set by MacroRegimeSnapshot validator."""
    snapshot = to_macro_regime_snapshot(SAMPLE_SEQUENCE)
    assert snapshot.is_low_confidence is False        # 0.862 ≥ 0.60
    assert snapshot.regime_change_detected is False   # label == prior_regime


def test_to_macro_regime_snapshot_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        to_macro_regime_snapshot({})


def test_to_macro_regime_snapshot_specific_date():
    from datetime import date
    snapshot = to_macro_regime_snapshot(SAMPLE_SEQUENCE, as_of=date(2025, 11, 1))
    assert snapshot.as_of == date(2025, 11, 1)
