"""
research_agent.py — Research Agent
=====================================
Entry point: run_research_agent() → MacroRegimeSnapshot

Orchestrates the full Research Agent pipeline:
    1. Pull 13 FRED macro series
    2. Engineer features (z-scores, log-transforms, first diffs, YoY)
    3. PELT change-point detection
    4. Segment fingerprinting + K-means clustering
    5. XGBoost regime mapping (5 anchor windows)
    6. 6-month rolling majority vote (smoothing)
    7. HMM & GMM comparison (optional — for paper validation)
    8. Pydantic validation + regime sequence construction
    9. Save outputs to agents/research/
   10. Convert latest entry to MacroRegimeSnapshot via adapters.py

Returns MacroRegimeSnapshot — typed, validated Pydantic object
ready to be passed directly to orchestrator.run_pipeline().
"""

import json
import os

from .adapters import to_macro_regime_snapshot
from .detection import cluster_segments, detect_change_points
from .features import engineer_features
from .loaders import pull_fred_data
from .model_comparison import run_model_comparison
from .regime_model import build_regime_sequence, map_regimes_xgboost, smooth_regimes

OUTPUT_DIR = "agents/research"


def run_research_agent(
    fred_api_key,
    pelt_penalty=10,
    run_comparison=True,
    crsp_path="crsp_market_index.csv",
    output_dir=OUTPUT_DIR,
):
    """
    Run the full Research Agent pipeline.

    Args:
        fred_api_key:   FRED API key (free at https://fred.stlouisfed.org/docs/api/)
        pelt_penalty:   PELT regularisation penalty (default 10; higher = fewer breaks)
        run_comparison: Whether to run HMM/GMM model comparison (default True)
        crsp_path:      Path to crsp_market_index.csv (for CRSP validation)
        output_dir:     Directory to save output files

    Returns:
        MacroRegimeSnapshot — latest validated snapshot, pass directly to
                              orchestrator.run_pipeline(profile, macro)
    """
    # Step 1: Pull FRED data
    macro_df = pull_fred_data(fred_api_key)

    # Step 2: Feature engineering
    features_df = engineer_features(macro_df)

    # Step 3: PELT change-point detection
    break_dates = detect_change_points(features_df, pen=pelt_penalty)

    # Step 4: Segment clustering
    seg_df, best_k = cluster_segments(features_df, break_dates)

    # Step 5: XGBoost regime mapping
    features_df, xgb = map_regimes_xgboost(features_df)

    # Step 6: Regime smoothing
    features_df = smooth_regimes(features_df)

    # Step 7: HMM & GMM comparison (optional — paper validation only)
    if run_comparison:
        print("\n=== Running HMM & GMM Model Comparison ===")
        features_df = run_model_comparison(features_df, crsp_path=crsp_path)

    # Step 8: Pydantic validation + regime sequence construction
    regime_sequence = build_regime_sequence(features_df)

    # Step 9: Save outputs
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "fred_macro_regimes.csv")
    features_df.to_csv(csv_path)
    print(f"Saved → {csv_path}")

    json_path = os.path.join(output_dir, "regime_sequence.json")
    with open(json_path, "w") as f:
        json.dump(regime_sequence, f, indent=2)
    print(f"Saved → {json_path}")

    last_3 = dict(list(regime_sequence.items())[-3:])
    print("\n=== Last 3 Months in Regime Sequence ===")
    print(json.dumps(last_3, indent=2))

    # Step 10: Convert to MacroRegimeSnapshot (latest date)
    snapshot = to_macro_regime_snapshot(regime_sequence)

    print(f"\nMacroRegimeSnapshot — {snapshot.as_of}")
    print(f"  Regime:     {snapshot.regime_label} (confidence: {snapshot.regime_confidence:.1%})")
    print(f"  Regime since: {snapshot.regime_shift_date}")
    print(f"  Low confidence: {snapshot.is_low_confidence}")
    print(f"  Regime change:  {snapshot.regime_change_detected}")

    return snapshot
