"""Ornstein-Uhlenbeck simulation and estimation (AR(1) OLS and exact Gaussian MLE).

Model: dX = kappa (mu - X) dt + sigma dW. Sampled every ``dt`` (in years), the exact
transition is ``X_t = mu + (X_{t-1} - mu) b + eps``, ``b = exp(-kappa dt)``,
``Var(eps) = sigma^2 (1 - b^2) / (2 kappa)``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import optimize
from statsmodels.tsa.adfvalues import mackinnonp

from .utils import json_safe

LN2 = math.log(2.0)
_MIN_POINTS = 3  # mathematical minimum for an AR(1) regression with a constant


@dataclass(frozen=True)
class OUParams:
    """Fitted OU parameters. ``half_life`` is in years (same base as ``dt``).

    ``mean_reverting`` is False whenever the AR(1) slope ``b`` is outside (0, 1); then
    ``kappa = 0`` and ``half_life = inf``. ``unit_root_pvalue`` is the Dickey-Fuller p-value
    (constant, no lags) for ``b = 1``; when ``fit_ou`` is given ``significance`` it is also
    required to be below it for ``mean_reverting``.
    """

    kappa: float
    mu: float
    sigma: float
    half_life: float
    b: float
    r2: float
    n_obs: int
    mean_reverting: bool
    method: str
    unit_root_pvalue: float = float("nan")

    def to_dict(self) -> dict:
        """JSON-safe dict (``inf``/``nan`` -> ``None``)."""
        return json_safe(asdict(self))


def simulate_ou(
    kappa: float,
    mu: float,
    sigma: float,
    x0: float,
    n: int,
    dt: float,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate ``n`` points of an OU path with the exact discretisation (first element ``x0``).

    ``kappa == 0`` gives a driftless Brownian motion with volatility ``sigma``.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=float)
    if n == 0:
        return out
    out[0] = x0
    if kappa > 0:
        b = math.exp(-kappa * dt)
        sd = sigma * math.sqrt((1.0 - b * b) / (2.0 * kappa))
    else:
        b, sd = 1.0, sigma * math.sqrt(dt)
    shocks = rng.standard_normal(n - 1) * sd
    for i in range(1, n):
        out[i] = mu + (out[i - 1] - mu) * b + shocks[i - 1]
    return out


def _clean(series: pd.Series | np.ndarray) -> np.ndarray:
    """Finite float values of ``series`` in order."""
    arr = np.asarray(series, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _ar1(x: np.ndarray) -> dict:
    """OLS of x_t on (1, x_{t-1}): a, b, residuals, r2, se(b)."""
    y, z = x[1:], x[:-1]
    zc = z - z.mean()
    sxx = float(zc @ zc)
    if sxx <= 0.0:
        return {"a": float(y.mean()) if y.size else 0.0, "b": 0.0, "resid": y - y.mean(),
                "r2": 0.0, "se_b": float("inf")}
    b = float(zc @ (y - y.mean())) / sxx
    a = float(y.mean() - b * z.mean())
    resid = y - a - b * z
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else 0.0
    dof = max(y.size - 2, 1)
    se_b = math.sqrt(float(resid @ resid) / dof / sxx)
    return {"a": a, "b": b, "resid": resid, "r2": r2, "se_b": se_b}


def _unit_root_pvalue(b: float, se_b: float) -> float:
    """Dickey-Fuller p-value (constant, no lags) for H0: b = 1."""
    if not math.isfinite(se_b) or se_b <= 0:
        return 1.0
    return float(mackinnonp((b - 1.0) / se_b, regression="c", N=1))


def _not_mean_reverting(x: np.ndarray, fit: dict, dt: float, method: str, pval: float) -> OUParams:
    """Flagged parameters: kappa 0, half-life inf, mu = sample mean, sigma = diffusion of increments."""
    mu = float(x.mean()) if x.size else 0.0
    inc = np.diff(x)
    sigma = float(inc.std(ddof=1) / math.sqrt(dt)) if inc.size > 1 else 0.0
    return OUParams(kappa=0.0, mu=mu, sigma=sigma, half_life=float("inf"), b=float(fit["b"]),
                    r2=float(fit["r2"]), n_obs=int(x.size), mean_reverting=False,
                    method=method, unit_root_pvalue=pval)


def _ols_params(x: np.ndarray, fit: dict, dt: float, mr: bool, pval: float) -> OUParams:
    """Map AR(1) OLS coefficients to OU parameters with the CONTRACTS.md formulas."""
    a, b = fit["a"], fit["b"]
    kappa = -math.log(b) / dt
    mu = a / (1.0 - b)
    eps_sd = float(np.std(fit["resid"], ddof=2)) if fit["resid"].size > 2 else 0.0
    sigma = eps_sd * math.sqrt(-2.0 * math.log(b) / (dt * (1.0 - b * b)))
    return OUParams(kappa=kappa, mu=mu, sigma=sigma, half_life=LN2 / kappa, b=b, r2=fit["r2"],
                    n_obs=int(x.size), mean_reverting=mr, method="ols", unit_root_pvalue=pval)


def ou_neg_loglik(theta: np.ndarray, x: np.ndarray, dt: float) -> float:
    """Negative exact OU transition log-likelihood, ``theta = (log kappa, mu, log sigma)``,
    conditional on the first observation."""
    kappa, mu, sigma = math.exp(theta[0]), theta[1], math.exp(theta[2])
    b = math.exp(-kappa * dt)
    var = sigma * sigma * (1.0 - b * b) / (2.0 * kappa)
    if not (var > 0 and math.isfinite(var)):
        return 1e300
    resid = x[1:] - mu - (x[:-1] - mu) * b
    n = resid.size
    return 0.5 * (n * math.log(2.0 * math.pi * var) + float(resid @ resid) / var)


def _mle_params(x: np.ndarray, fit: dict, dt: float, mr: bool, pval: float) -> OUParams:
    """Maximise the exact OU likelihood numerically, starting from the closed-form AR(1) solution."""
    start = _ols_params(x, fit, dt, mr, pval)
    theta0 = np.array([math.log(start.kappa), start.mu, math.log(max(start.sigma, 1e-12))])
    res = optimize.minimize(ou_neg_loglik, theta0, args=(x, dt), method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000})
    theta = res.x if res.success and np.all(np.isfinite(res.x)) else theta0
    kappa, mu, sigma = math.exp(theta[0]), float(theta[1]), math.exp(theta[2])
    return OUParams(kappa=kappa, mu=mu, sigma=sigma, half_life=LN2 / kappa,
                    b=math.exp(-kappa * dt), r2=fit["r2"], n_obs=int(x.size),
                    mean_reverting=mr, method="mle", unit_root_pvalue=pval)


def fit_ou(
    series: pd.Series | np.ndarray,
    dt: float,
    method: str = "ols",
    significance: float | None = None,
) -> OUParams:
    """Fit OU parameters to ``series`` sampled every ``dt`` years.

    ``method="ols"``: AR(1) regression ``x_t = a + b x_{t-1} + eps`` mapped with
    ``kappa = -ln b / dt``, ``mu = a / (1 - b)``,
    ``sigma = std(eps) sqrt(-2 ln b / (dt (1 - b^2)))``.
    ``method="mle"``: numerical maximisation of the exact Gaussian OU transition likelihood.

    Never raises on degenerate data and never returns NaN in kappa/mu/sigma: if ``b`` is
    outside (0, 1), or the series has fewer than 3 finite points, the result has
    ``mean_reverting=False``, ``kappa=0`` and ``half_life=inf``. If ``significance`` is
    given, a Dickey-Fuller p-value >= ``significance`` also sets ``mean_reverting=False``
    (kappa etc. are still reported).
    """
    if method not in ("ols", "mle"):
        raise ValueError(f"unknown OU method '{method}'")
    x = _clean(series)
    if x.size < _MIN_POINTS:
        return _not_mean_reverting(x, {"b": 0.0, "r2": 0.0}, dt, method, 1.0)
    fit = _ar1(x)
    b = fit["b"]
    pval = _unit_root_pvalue(b, fit["se_b"])
    if not (0.0 < b < 1.0) or not math.isfinite(b):
        return _not_mean_reverting(x, fit, dt, method, pval)
    mr = significance is None or pval < significance
    if method == "ols":
        return _ols_params(x, fit, dt, mr, pval)
    return _mle_params(x, fit, dt, mr, pval)


def compare_ou_methods(
    series: pd.Series | np.ndarray,
    dt: float,
    methods: tuple[str, ...] | list[str] = ("ols", "mle"),
    significance: float | None = None,
) -> dict[str, OUParams]:
    """Fit ``series`` with each method; returns ``{method: OUParams}``."""
    return {m: fit_ou(series, dt, method=m, significance=significance) for m in methods}
