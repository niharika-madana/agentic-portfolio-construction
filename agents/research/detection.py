"""
detection.py — PELT change-point detection.

Pruned Exact Linear Time (PELT, `ruptures`) with an RBF kernel finds structural
breaks in the 13-signal matrix without any date assumptions. With 13 features a
larger multivariate shift is required to trigger a break, so only the most
economically significant structural shifts survive (≈4 breaks → 5 segments).
"""

from __future__ import annotations

import pandas as pd


def detect_change_points(
    features_df: pd.DataFrame, signal_cols: list[str], pen: float = 10.0
):
    """
    Detect structural breaks via PELT (RBF kernel).

    Returns (break_dates, breakpoints):
      break_dates  — list of pandas Timestamps marking the last month of each
                     completed segment (excludes the final series index)
      breakpoints  — raw ruptures index list (end-exclusive, includes len(df))

    Higher `pen` → fewer breaks.
    """
    import ruptures as rpt

    signal_matrix = features_df[signal_cols].values
    model = rpt.Pelt(model="rbf").fit(signal_matrix)
    breakpoints = model.predict(pen=pen)

    break_dates = [features_df.index[i - 1] for i in breakpoints[:-1]]

    print(f"Detected {len(break_dates)} structural breaks:")
    for d in break_dates:
        print(f"  {d.strftime('%Y-%m')}")
    return break_dates, breakpoints
