"""Synthetic data generators for tests and demos (not used on real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ou import simulate_ou


def synthetic_cointegrated_pair(
    n: int,
    beta: float,
    kappa: float,
    sigma_spread: float,
    sigma_x: float,
    dt: float,
    start: str,
    x0: float,
    alpha: float = 0.0,
    seed: int | None = None,
    names: tuple[str, str] = ("Y", "X"),
) -> pd.DataFrame:
    """Two price legs with ``log Y = alpha + beta log X + OU``; ``log X`` is a random walk.

    ``kappa``/``sigma_spread`` are annualised OU parameters of the log-spread, ``sigma_x`` the
    annual volatility of X. Index: ``n`` business days from ``start``.
    """
    rng = np.random.default_rng(seed)
    log_x = np.log(x0) + np.cumsum(rng.standard_normal(n) * sigma_x * np.sqrt(dt))
    ou = simulate_ou(kappa, 0.0, sigma_spread, 0.0, n, dt, seed=None if seed is None else seed + 1)
    log_y = alpha + beta * log_x + ou
    idx = pd.bdate_range(start, periods=n, name="date")
    return pd.DataFrame({names[0]: np.exp(log_y), names[1]: np.exp(log_x)}, index=idx)
