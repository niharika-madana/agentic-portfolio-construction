"""
Tests for agents/shared/core/allocation.py — pure math, no network calls.
All tests use simulated returns so they run offline.
"""

import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.shared.core.allocation import (
    build_covariance_matrix,
    compute_equilibrium_returns,
    build_ff_views,
    black_litterman,
    optimize_weights,
    compute_factor_exposures,
    DELTA, TAU, MONTHS_PER_YEAR,
)
from agents.shared.core.constraints import SINGLE_NAME_LIMIT, SECTOR_LIMIT
from contracts import ConstraintType, AllocationConstraint


@pytest.fixture
def sim_data():
    """15-ticker simulated returns — optimizer is feasible offline."""
    np.random.seed(99)
    T, N = 60, 15
    tickers = [f"T{i:02d}" for i in range(N)]
    sectors = {
        t: ["Information Technology", "Financials", "Health Care",
            "Energy", "Industrials"][i % 5]
        for i, t in enumerate(tickers)
    }
    mkt_w  = np.ones(N) / N
    excess = np.random.multivariate_normal(
        mean=np.linspace(0.004, 0.012, N),
        cov =np.eye(N) * 0.003 + np.full((N, N), 0.001),
        size=T,
    )
    ff_arr = np.random.randn(T, 4) * 0.02
    return tickers, sectors, mkt_w, excess, ff_arr


def test_covariance_is_symmetric(sim_data):
    _, _, _, excess, _ = sim_data
    cov = build_covariance_matrix(excess)
    assert np.allclose(cov, cov.T)


def test_covariance_is_psd(sim_data):
    _, _, _, excess, _ = sim_data
    cov     = build_covariance_matrix(excess)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-10)


def test_covariance_annualized(sim_data):
    _, _, _, excess, _ = sim_data
    cov_m = np.cov(excess, rowvar=False)
    cov_a = build_covariance_matrix(excess)
    assert np.allclose(cov_a, cov_m * MONTHS_PER_YEAR)


def test_equilibrium_returns_shape(sim_data):
    _, _, mkt_w, excess, _ = sim_data
    cov = build_covariance_matrix(excess)
    pi  = compute_equilibrium_returns(cov, mkt_w)
    assert pi.shape == mkt_w.shape


def test_equilibrium_returns_formula(sim_data):
    _, _, mkt_w, excess, _ = sim_data
    cov = build_covariance_matrix(excess)
    pi  = compute_equilibrium_returns(cov, mkt_w)
    assert np.allclose(pi, DELTA * cov @ mkt_w)


def test_ff_views_shapes(sim_data):
    _, _, _, excess, ff_arr = sim_data
    N = excess.shape[1]
    P, Q, Omega = build_ff_views(excess, ff_arr)
    assert P.shape     == (N, N)
    assert Q.shape     == (N,)
    assert Omega.shape == (N, N)


def test_ff_views_P_is_identity(sim_data):
    _, _, _, excess, ff_arr = sim_data
    N = excess.shape[1]
    P, _, _ = build_ff_views(excess, ff_arr)
    assert np.allclose(P, np.eye(N))


def test_ff_views_omega_diagonal(sim_data):
    _, _, _, excess, ff_arr = sim_data
    _, _, Omega = build_ff_views(excess, ff_arr)
    assert np.allclose(Omega, np.diag(np.diag(Omega)))


def test_bl_mu_shape(sim_data):
    _, _, mkt_w, excess, ff_arr = sim_data
    cov = build_covariance_matrix(excess)
    pi  = compute_equilibrium_returns(cov, mkt_w)
    P, Q, Omega = build_ff_views(excess, ff_arr)
    mu_BL, Sigma_BL = black_litterman(cov, pi, P, Q, Omega)
    assert mu_BL.shape    == pi.shape
    assert Sigma_BL.shape == cov.shape


def test_bl_sigma_is_symmetric(sim_data):
    _, _, mkt_w, excess, ff_arr = sim_data
    cov = build_covariance_matrix(excess)
    pi  = compute_equilibrium_returns(cov, mkt_w)
    P, Q, Omega = build_ff_views(excess, ff_arr)
    _, Sigma_BL = black_litterman(cov, pi, P, Q, Omega)
    assert np.allclose(Sigma_BL, Sigma_BL.T, atol=1e-10)


def test_bl_with_zero_views_equals_prior(sim_data):
    _, _, mkt_w, excess, ff_arr = sim_data
    cov = build_covariance_matrix(excess)
    pi  = compute_equilibrium_returns(cov, mkt_w)
    P, Q, Omega = build_ff_views(excess, ff_arr)
    mu_BL, _ = black_litterman(cov, pi, P, np.zeros_like(Q), Omega * 1e10)
    assert np.allclose(mu_BL, pi, atol=1e-4)


@pytest.fixture
def bl_weights(sim_data):
    tickers, sectors, mkt_w, excess, ff_arr = sim_data
    cov = build_covariance_matrix(excess)
    pi  = compute_equilibrium_returns(cov, mkt_w)
    P, Q, Omega = build_ff_views(excess, ff_arr)
    mu_BL, Sigma_BL = black_litterman(cov, pi, P, Q, Omega)
    w = optimize_weights(mu_BL, Sigma_BL, tickers, sectors, [])
    return w, tickers, sectors


def test_weights_sum_to_one(bl_weights):
    w, _, _ = bl_weights
    assert abs(w.sum() - 1.0) < 1e-6


def test_weights_non_negative(bl_weights):
    w, _, _ = bl_weights
    assert np.all(w >= -1e-9)


def test_single_name_limit_respected(bl_weights):
    w, _, _ = bl_weights
    assert w.max() <= SINGLE_NAME_LIMIT + 1e-6


def test_sector_limit_respected(bl_weights):
    from collections import defaultdict
    w, tickers, sectors = bl_weights
    sec_w: dict[str, float] = defaultdict(float)
    for t, wi in zip(tickers, w):
        sec_w[sectors[t]] += wi
    assert all(sw <= SECTOR_LIMIT + 1e-6 for sw in sec_w.values())


def test_flag_constraint_tightens_bound(sim_data):
    tickers, sectors, mkt_w, excess, ff_arr = sim_data
    cov = build_covariance_matrix(excess)
    pi  = compute_equilibrium_returns(cov, mkt_w)
    P, Q, Omega = build_ff_views(excess, ff_arr)
    mu_BL, Sigma_BL = black_litterman(cov, pi, P, Q, Omega)

    target = tickers[0]
    flag = [AllocationConstraint(
        constraint_type=ConstraintType.SINGLE_NAME,
        target=target, current_value=0.09, limit=0.07,
    )]
    w   = optimize_weights(mu_BL, Sigma_BL, tickers, sectors, flag)
    idx = tickers.index(target)
    assert w[idx] <= 0.07 * 0.99 + 1e-6


def test_factor_exposures_finite(sim_data):
    _, _, mkt_w, excess, ff_arr = sim_data
    w  = np.ones(excess.shape[1]) / excess.shape[1]
    fe = compute_factor_exposures(w, excess, ff_arr)
    assert all(np.isfinite(v) for v in [fe.market_beta, fe.smb, fe.hml, fe.mom])
