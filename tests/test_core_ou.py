"""Quality gate 1-2: OU parameter recovery and non-mean-reverting flags."""

import math

import numpy as np
import pytest

from ou_model.core.ou import OUParams, compare_ou_methods, fit_ou, simulate_ou

DT = 1.0 / 252
SEEDS = range(5)
# (kappa, mu, sigma, n): long daily paths (n / 252 years).
CASES = [(5.0, 0.5, 0.3, 20_000), (50.0, -1.0, 1.0, 10_000)]
# Stated tolerances, per seed: kappa within 25% relative, mu within 0.05 absolute,
# sigma within 3% relative. Across the 5 seeds the mean kappa must be within 10%.
KAPPA_REL, MU_ABS, SIGMA_REL, KAPPA_MEAN_REL = 0.25, 0.05, 0.03, 0.10


def test_simulate_ou_exact_moments():
    """Exact discretisation: stationary variance sigma^2 / (2 kappa), lag-1 autocorr exp(-kappa dt)."""
    kappa, mu, sigma = 10.0, 2.0, 0.5
    x = simulate_ou(kappa, mu, sigma, mu, 200_000, DT, seed=1)
    assert len(x) == 200_000 and x[0] == mu
    assert x.mean() == pytest.approx(mu, abs=0.01)
    assert x.var() == pytest.approx(sigma**2 / (2 * kappa), rel=0.05)
    assert np.corrcoef(x[1:], x[:-1])[0, 1] == pytest.approx(math.exp(-kappa * DT), abs=0.002)


@pytest.mark.parametrize("method", ["ols", "mle"])
@pytest.mark.parametrize("kappa,mu,sigma,n", CASES)
def test_fit_ou_recovers_parameters(method, kappa, mu, sigma, n):
    kappas = []
    for seed in SEEDS:
        x = simulate_ou(kappa, mu, sigma, mu, n, DT, seed=seed)
        p = fit_ou(x, DT, method=method)
        assert p.mean_reverting and p.method == method and p.n_obs == n
        assert abs(p.kappa / kappa - 1) < KAPPA_REL
        assert abs(p.mu - mu) < MU_ABS
        assert abs(p.sigma / sigma - 1) < SIGMA_REL
        assert p.half_life == pytest.approx(math.log(2) / p.kappa)
        assert p.b == pytest.approx(math.exp(-p.kappa * DT))
        kappas.append(p.kappa)
    assert abs(np.mean(kappas) / kappa - 1) < KAPPA_MEAN_REL


def test_ols_formulas_match_contract():
    x = simulate_ou(8.0, 0.0, 0.2, 0.0, 5000, DT, seed=3)
    p = fit_ou(x, DT, "ols")
    b, a = np.polyfit(x[:-1], x[1:], 1)
    resid = x[1:] - a - b * x[:-1]
    assert p.b == pytest.approx(b, rel=1e-10)
    assert p.kappa == pytest.approx(-math.log(b) / DT, rel=1e-10)
    assert p.mu == pytest.approx(a / (1 - b), rel=1e-8)
    expected_sigma = resid.std(ddof=2) * math.sqrt(-2 * math.log(b) / (DT * (1 - b * b)))
    assert p.sigma == pytest.approx(expected_sigma, rel=1e-10)


def test_mle_close_to_ols_and_compare():
    """Exact MLE and AR(1) OLS coincide up to the residual-variance dof (1e-3 relative)."""
    x = simulate_ou(5.0, 0.5, 0.3, 0.5, 5000, DT, seed=11)
    fits = compare_ou_methods(x, DT)
    assert set(fits) == {"ols", "mle"}
    assert fits["mle"].kappa == pytest.approx(fits["ols"].kappa, rel=1e-3)
    assert fits["mle"].mu == pytest.approx(fits["ols"].mu, abs=1e-4)
    assert fits["mle"].sigma == pytest.approx(fits["ols"].sigma, rel=1e-3)


def _assert_flagged(p: OUParams):
    assert p.mean_reverting is False
    assert p.kappa == 0.0 and p.half_life == math.inf
    for v in (p.kappa, p.mu, p.sigma):
        assert math.isfinite(v)
    d = p.to_dict()
    assert d["half_life"] is None  # inf -> null in JSON


@pytest.mark.parametrize("method", ["ols", "mle"])
def test_explosive_flagged(method):
    rng = np.random.default_rng(0)
    x = np.zeros(1000)
    for i in range(1, 1000):
        x[i] = 1.01 * x[i - 1] + rng.standard_normal()
    p = fit_ou(x, DT, method)
    assert p.b > 1
    _assert_flagged(p)


@pytest.mark.parametrize("method", ["ols", "mle"])
def test_negative_b_flagged(method):
    rng = np.random.default_rng(1)
    x = np.zeros(2000)
    for i in range(1, 2000):
        x[i] = -0.5 * x[i - 1] + rng.standard_normal()
    p = fit_ou(x, DT, method)
    assert p.b <= 0
    _assert_flagged(p)


@pytest.mark.parametrize("method", ["ols", "mle"])
def test_random_walk_flagged(method):
    """A random walk is flagged: either b >= 1, or the Dickey-Fuller test (significance 5%)
    fails to reject the unit root. At a 5% size we allow at most 2 of 20 seeds to slip."""
    slipped = 0
    for seed in range(20):
        x = np.cumsum(np.random.default_rng(seed).standard_normal(2000))
        p = fit_ou(x, DT, method, significance=0.05)
        assert all(math.isfinite(v) for v in (p.kappa, p.mu, p.sigma))
        slipped += p.mean_reverting
        if p.b >= 1:
            _assert_flagged(p)
    assert slipped <= 2


@pytest.mark.parametrize("x", [np.ones(500), np.array([1.0, 2.0]), np.array([]),
                               np.array([np.nan] * 10), np.array([1.0, np.nan, 2.0, np.inf, 3.0])])
def test_degenerate_inputs_never_raise(x):
    for method in ("ols", "mle"):
        _assert_flagged(fit_ou(x, DT, method))


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        fit_ou(np.arange(10.0), DT, "bogus")
