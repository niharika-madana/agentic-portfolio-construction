"""
beta.py — DEPRECATED
=====================
This module has been replaced by hc_beta_table.py.

The OLS regression here regressed synthetic Gaussian noise (sigma * N(0,1))
against Fama-French sector returns, producing beta ≈ 0 for every client
regardless of career type. The regression was not estimating income sensitivity
to markets — it was fitting noise.

Replacement:
    from .hc_beta_table import lookup_hc_beta
    beta, correlation = lookup_hc_beta(hc_type)

The calibrated table uses values from Ibbotson, Milevsky, Chen, Zhu (2007)
and Davis & Willen (2000). See hc_beta_table.py for full citations.
"""

raise ImportError(
    "beta.py is deprecated. Use hc_beta_table.lookup_hc_beta() instead."
)
