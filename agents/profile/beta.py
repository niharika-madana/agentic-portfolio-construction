"""
beta.py — DEPRECATED.

The original OLS approach regressed synthetic Gaussian noise (σ × N(0,1))
against Fama-French sector returns and produced β ≈ 0 for every client
regardless of career type. It has been replaced by the calibrated β/ρ table.

Use agents.profile.hc_beta_table.lookup_hc_beta() instead.
"""

raise ImportError(
    "agents.profile.beta is deprecated. Import lookup_hc_beta from "
    "agents.profile.hc_beta_table instead."
)
