"""
research_agent.py — Research Agent entry point (Agent 2 of 5).

run_research_agent() runs the four-stage deterministic pipeline on 13 FRED macro
series and returns a validated MacroRegimeSnapshot for the most recent month:

  FRED macro (cache) → features → PELT breaks → K-means sanity clustering
  → XGBoost regime mapping → 6-month majority-vote smoothing → Pydantic

Side-effect outputs (design doc §Implementation Notes):
  agents/research/fred_macro_regimes.csv      full smoothed feature matrix
  agents/research/regime_sequence.json        validated date-keyed sequence
  agents/research/macro_regime_snapshot.json  most recent MacroRegimeSnapshot
  data/storage/fred_macro_regimes.parquet     feature matrix for downstream agents

The returned snapshot feeds straight into orchestrator.run_all(personas, macro).
"""

from __future__ import annotations

import json
from pathlib import Path

from contracts import MacroRegimeSnapshot

from agents.research.adapters import (
    add_derived_fields,
    build_regime_sequence,
    build_snapshot,
)
from agents.research.detection import detect_change_points
from agents.research.features import build_features
from agents.research.loaders import load_crsp, load_macro_data
from agents.research.regime_model import (
    cluster_segments,
    smooth_regimes,
    train_and_predict,
)

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
STORAGE_DIR = PROJECT_ROOT / "data" / "storage"

CSV_OUTPUT = _THIS_DIR / "fred_macro_regimes.csv"
SEQUENCE_OUTPUT = _THIS_DIR / "regime_sequence.json"
SNAPSHOT_OUTPUT = _THIS_DIR / "macro_regime_snapshot.json"
PARQUET_OUTPUT = STORAGE_DIR / "fred_macro_regimes.parquet"

# ── PELT change-point parameters (design doc §Change-Point Detection) ───────
# Exposed as named constants so callers can tune break-point sensitivity.
PELT_PEN = 10.0      # penalty — higher → fewer breaks (coarser segmentation)
PELT_MODEL = "rbf"   # kernel — "rbf" non-linear; "l1" robust; "l2" fast/sensitive


def _validate_crsp(features_df) -> None:
    """Optional CRSP return validation — prints avg monthly return per regime."""
    crsp = load_crsp()
    if crsp is None or "vwretd" not in getattr(crsp, "columns", []):
        return
    validation_df = features_df[["regime_label_smoothed"]].join(crsp[["vwretd"]], how="left")
    summary = (
        validation_df.groupby("regime_label_smoothed")["vwretd"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "Avg Monthly Return", "std": "Std Dev", "count": "Months"})
    )
    print("\n=== Avg Monthly Market Return per Regime (CRSP) ===")
    print(summary.to_string())


def _save_outputs(features_df, regime_sequence: dict, snapshot: MacroRegimeSnapshot) -> None:
    _THIS_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(CSV_OUTPUT)
    print(f"Saved → {CSV_OUTPUT}")

    try:
        features_df.to_parquet(PARQUET_OUTPUT)
        print(f"Saved → {PARQUET_OUTPUT}")
    except Exception as e:  # pragma: no cover - optional dependency path
        print(f"Parquet export skipped ({e}).")

    with open(SEQUENCE_OUTPUT, "w") as f:
        json.dump(regime_sequence, f, indent=2)
    print(f"Saved → {SEQUENCE_OUTPUT}")

    with open(SNAPSHOT_OUTPUT, "w") as f:
        f.write(snapshot.model_dump_json(indent=2))
    print(f"Saved → {SNAPSHOT_OUTPUT}")


def run_research_agent(
    fred_api_key: str | None = None,
    pen: float = PELT_PEN,
    pelt_model: str = PELT_MODEL,
    save: bool = True,
    validate_crsp: bool = True,
    compare_models: bool = False,
) -> MacroRegimeSnapshot:
    """
    Run the full Research Agent pipeline.

    Parameters
    ----------
    fred_api_key : str | None
        Only needed on a cold cache. When data/storage/fred_macro.parquet exists,
        no API call is made.
    pen : float
        PELT penalty (higher → fewer structural breaks). Defaults to PELT_PEN.
    pelt_model : str
        PELT cost model / kernel ("rbf" non-linear, "l1" robust, "l2" fast).
        Defaults to PELT_MODEL. Together with `pen` this controls how large a
        multivariate shift must be to register a structural break.
    save : bool
        Persist CSV / JSON / parquet outputs when True.
    validate_crsp : bool
        Run the optional CRSP return validation when True.
    compare_models : bool
        Run the optional HMM/GMM benchmark when True. Off the production path —
        kept for the paper's Future Work section only (design doc §Model Comparison).

    Returns
    -------
    MacroRegimeSnapshot
        Validated snapshot for the most recent month.
    """
    macro_df = load_macro_data(fred_api_key)
    features_df, signal_cols = build_features(macro_df)

    break_dates, _ = detect_change_points(features_df, signal_cols, pen=pen, model=pelt_model)
    # Diagnostic sanity check only — also gates on empty/NaN/out-of-range inputs.
    # Its return value is intentionally not consumed; XGBoost labels the full
    # monthly matrix, not these segment fingerprints (design doc §Clustering).
    cluster_segments(features_df, signal_cols, break_dates)

    _, feat_importance = train_and_predict(features_df, signal_cols)
    print("\n=== Feature Importance ===")
    print(feat_importance.to_string())

    smooth_regimes(features_df)
    add_derived_fields(features_df)

    regime_sequence = build_regime_sequence(features_df)
    snapshot = build_snapshot(features_df)

    print("\n=== MacroRegimeSnapshot (most recent month) ===")
    print(snapshot.model_dump_json(indent=2))

    if compare_models:
        from agents.research.model_comparison import compare_models as _cmp
        _cmp(features_df, signal_cols)

    if validate_crsp:
        _validate_crsp(features_df)

    if save:
        _save_outputs(features_df, regime_sequence, snapshot)

    return snapshot


if __name__ == "__main__":  # pragma: no cover
    snap = run_research_agent()
    print(f"\nRegime: {snap.regime_label} | confidence: {snap.regime_confidence:.3f}")
