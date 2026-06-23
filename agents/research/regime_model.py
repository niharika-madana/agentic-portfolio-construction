"""
regime_model.py — Research Agent
==================================
XGBoost regime mapping, 6-month smoothing, Pydantic validation,
and regime sequence construction.

Exports:
  - REGIME_NAMES
  - ANCHOR_WINDOWS
  - VALID_REGIME_LABELS
  - RegimeRecord
  - map_regimes_xgboost()
  - smooth_regimes()
  - build_regime_sequence()
"""

import json

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier

from .features import SIGNAL_COLS

# ---------------------------------------------------------------------------
# Academic regime taxonomy
# Replaces ad-hoc event labels (Dot-com, COVID, AI Boom)
# ---------------------------------------------------------------------------

REGIME_NAMES = {
    0: "Early Recovery",
    1: "Late-Cycle Expansion",
    2: "Financial Crisis & ZLB",
    3: "Moderate Expansion",
    4: "Inflation Shock",
}

VALID_REGIME_LABELS = set(REGIME_NAMES.values())

# Anchor windows — one per PELT segment, historically unambiguous months
ANCHOR_WINDOWS = {
    0: ("2003-06", "2004-06"),   # Early Recovery: post dot-com, rates low, spreads narrowing
    1: ("2005-01", "2007-06"),   # Late-Cycle Expansion: pre-GFC boom, rising rates, low VIX
    2: ("2008-09", "2010-12"),   # Financial Crisis & ZLB: acute GFC + early ZIRP
    3: ("2015-01", "2019-06"),   # Moderate Expansion: post-QE normalisation, stable macro
    4: ("2022-01", "2023-06"),   # Inflation Shock: peak CPI, aggressive Fed tightening
}

LOW_CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_WINDOW = 6


# ---------------------------------------------------------------------------
# Pydantic validation schema — internal to Research Agent
# ---------------------------------------------------------------------------

class RegimeRecord(BaseModel):
    """
    Validated row in the regime sequence.
    Every row must pass before being written to regime_sequence.json.
    This is the Research Agent's internal schema — MacroRegimeSnapshot
    in contracts.py is the cross-agent contract type.
    """
    regime_label:      str
    prior_regime:      str
    regime_shift_date: str
    regime_confidence: float = Field(ge=0.0, le=1.0)
    regime_volatility: float = Field(ge=0.0)
    yield_curve:       float
    term_spread:       float
    fed_funds:         float
    unemployment:      float
    cpi:               float
    credit_spread:     float
    vix:               float
    indpro:            float

    @field_validator("regime_label")
    @classmethod
    def label_must_be_valid(cls, v):
        if v not in VALID_REGIME_LABELS:
            raise ValueError(f"Invalid regime label: '{v}'")
        return v

    @field_validator("prior_regime")
    @classmethod
    def prior_must_be_valid(cls, v):
        if v != "None" and v not in VALID_REGIME_LABELS:
            raise ValueError(f"Invalid prior_regime: '{v}'")
        return v


# ---------------------------------------------------------------------------
# XGBoost regime mapping
# ---------------------------------------------------------------------------

def map_regimes_xgboost(features_df):
    """
    Train XGBoost on 5 anchor windows and predict regime for all months.

    Time-series CV (5 folds) confirms no future data leakage. Low accuracy
    in early folds is expected by design — not a generalisation benchmark.

    Returns:
        features_df (pd.DataFrame) — with regime_cluster, regime_label,
                                     regime_confidence columns added
        xgb (XGBClassifier)        — fitted model
    """
    train_frames = []
    for label, (start, end) in ANCHOR_WINDOWS.items():
        window = features_df.loc[start:end, SIGNAL_COLS].copy()
        window["label"] = label
        train_frames.append(window)

    train_df = pd.concat(train_frames).sort_index()
    X_train  = train_df[SIGNAL_COLS]
    y_train  = train_df["label"]

    print(f"Training samples: {len(train_df)}")
    print(y_train.value_counts().sort_index())

    tscv    = TimeSeriesSplit(n_splits=5)
    xgb_cv  = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss")
    cv_scores = cross_val_score(xgb_cv, X_train, y_train, cv=tscv, scoring="accuracy")

    print("\n=== Time-Series CV Accuracy ===")
    for i, score in enumerate(cv_scores):
        print(f"  Fold {i + 1}: {score:.3f}")
    print(f"  Mean: {cv_scores.mean():.3f}  |  Std: {cv_scores.std():.3f}")

    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss")
    xgb.fit(X_train, y_train)

    proba = xgb.predict_proba(features_df[SIGNAL_COLS])
    features_df["regime_confidence"] = proba.max(axis=1).round(3)
    features_df["regime_cluster"]    = xgb.predict(features_df[SIGNAL_COLS])
    features_df["regime_label"]      = features_df["regime_cluster"].map(REGIME_NAMES)

    print("\n=== Predicted Regime Distribution ===")
    print(features_df["regime_label"].value_counts())

    return features_df, xgb


# ---------------------------------------------------------------------------
# Regime smoothing
# ---------------------------------------------------------------------------

def smooth_regimes(features_df):
    """
    6-month rolling majority vote to enforce regime persistence.

    A regime must dominate a 6-month window before being accepted,
    preventing the Allocation Agent from rebalancing on transient noise.

    Returns:
        features_df (pd.DataFrame) — with regime_smoothed and
                                     regime_label_smoothed columns added
    """
    features_df["regime_smoothed"] = (
        features_df["regime_cluster"]
        .rolling(SMOOTHING_WINDOW, center=True, min_periods=1)
        .apply(lambda x: stats.mode(x, keepdims=True)[0][0])
        .astype(int)
    )
    features_df["regime_label_smoothed"] = features_df["regime_smoothed"].map(REGIME_NAMES)

    print("=== Smoothed Regime Distribution ===")
    print(features_df["regime_label_smoothed"].value_counts())

    return features_df


# ---------------------------------------------------------------------------
# Regime sequence construction and Pydantic validation
# ---------------------------------------------------------------------------

def build_regime_sequence(features_df):
    """
    Assemble the full regime sequence JSON, validating every row
    against RegimeRecord before inclusion.

    Rows failing validation are logged and excluded — they cannot
    enter the output JSON.

    Returns:
        regime_sequence (dict) — date-keyed validated regime records
    """
    features_df["regime_volatility"] = (
        features_df["credit_spread"]
        .rolling(6, min_periods=1).std().fillna(0).round(4)
    )
    features_df["prior_regime"] = (
        features_df["regime_label_smoothed"].shift(1).fillna("None")
    )

    shift_dates   = []
    current_start = features_df.index[0]
    for i, (date, row) in enumerate(features_df.iterrows()):
        if i == 0:
            current_start = date
        elif row["regime_label_smoothed"] != features_df["regime_label_smoothed"].iloc[i - 1]:
            current_start = date
        shift_dates.append(current_start)
    features_df["regime_shift_date"] = shift_dates

    regime_sequence = {}
    passed = failed = 0

    for date, row in features_df.iterrows():
        record_dict = {
            "regime_label":      row["regime_label_smoothed"],
            "prior_regime":      row["prior_regime"],
            "regime_shift_date": str(row["regime_shift_date"].date()),
            "regime_confidence": round(float(row["regime_confidence"]), 3),
            "regime_volatility": round(float(row["regime_volatility"]), 4),
            "yield_curve":       round(float(row["yield_curve"]),    4),
            "term_spread":       round(float(row["term_spread"]),    4),
            "fed_funds":         round(float(row["fed_funds"]),      4),
            "unemployment":      round(float(row["unemployment"]),   4),
            "cpi":               round(float(row["cpi"]),            4),
            "credit_spread":     round(float(row["credit_spread"]),  4),
            "vix":               round(float(row["vix"]),            4),
            "indpro":            round(float(row["indpro"]),         4),
        }
        try:
            RegimeRecord(**record_dict)
            regime_sequence[str(date.date())] = record_dict
            passed += 1
        except Exception as e:
            print(f"  VALIDATION FAILED [{date.date()}]: {e}")
            failed += 1

    print(f"\nValidation complete — Passed: {passed} | Failed: {failed}")
    return regime_sequence
