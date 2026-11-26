"""Hedge ratios: static OLS, rolling OLS (causal) and an optional Kalman filter."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .utils import json_safe


@dataclass
class HedgeFit:
    """OLS fit ``y = alpha + beta x + e``."""

    alpha: float
    beta: float
    r2: float

    def to_dict(self) -> dict:
        """JSON-safe dict."""
        return json_safe(asdict(self))


def ols_hedge_ratio(y: pd.Series, x: pd.Series, add_constant: bool = True) -> HedgeFit:
    """Static OLS hedge ratio of ``y`` on ``x`` over their common non-NaN observations."""
    frame = pd.concat([pd.Series(y), pd.Series(x)], axis=1).dropna().to_numpy(dtype=float)
    yy, xx = frame[:, 0], frame[:, 1]
    if len(yy) < 2:
        return HedgeFit(alpha=float("nan"), beta=float("nan"), r2=float("nan"))
    design = np.column_stack([np.ones_like(xx), xx]) if add_constant else xx[:, None]
    coef = np.linalg.lstsq(design, yy, rcond=None)[0]
    alpha, beta = (float(coef[0]), float(coef[1])) if add_constant else (0.0, float(coef[0]))
    resid = yy - alpha - beta * xx
    sst = float(((yy - yy.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else float("nan")
    return HedgeFit(alpha=alpha, beta=beta, r2=r2)


def rolling_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    window: int,
    min_periods: int,
    add_constant: bool = True,
) -> pd.DataFrame:
    """Trailing-window OLS hedge ratio; the value at ``t`` uses observations ``<= t`` only.

    Returns a DataFrame indexed like ``y`` with columns ``alpha`` and ``beta``
    (NaN until ``min_periods`` joint observations are available).
    """
    frame = pd.concat([pd.Series(y, dtype=float).rename("y"), pd.Series(x, dtype=float).rename("x")],
                      axis=1)
    valid = frame.notna().all(axis=1)
    frame = frame.where(valid)
    roll = frame.rolling(window, min_periods=min_periods)
    if add_constant:
        means = roll.mean()
        cov = roll.cov().xs("x", level=1)  # rows: date, cols: y, x -> cov(y,x), var(x)
        beta = cov["y"] / cov["x"]
        alpha = means["y"] - beta * means["x"]
    else:
        sxy = (frame["x"] * frame["y"]).rolling(window, min_periods=min_periods).sum()
        sxx = (frame["x"] ** 2).rolling(window, min_periods=min_periods).sum()
        beta = sxy / sxx
        alpha = pd.Series(0.0, index=frame.index).where(beta.notna())
    out = pd.DataFrame({"alpha": alpha, "beta": beta}, index=frame.index)
    return out.replace([np.inf, -np.inf], np.nan)


def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float,
    obs_var: float,
    init_var: float,
) -> pd.DataFrame:
    """Kalman-filter random-walk hedge ratio ``y_t = alpha_t + beta_t x_t + e_t``.

    State ``(alpha, beta)`` follows a random walk with covariance ``delta / (1 - delta) I``;
    ``obs_var`` is the observation noise variance and ``init_var`` the prior state variance.
    The value at ``t`` is the filtered estimate after observing ``t`` (causal).
    """
    frame = pd.concat([pd.Series(y, dtype=float), pd.Series(x, dtype=float)], axis=1)
    yy, xx = frame.iloc[:, 0].to_numpy(), frame.iloc[:, 1].to_numpy()
    q = delta / (1.0 - delta) * np.eye(2)
    state, cov = np.zeros(2), init_var * np.eye(2)
    out = np.full((len(yy), 2), np.nan)
    started = False
    for t in range(len(yy)):
        if not (np.isfinite(yy[t]) and np.isfinite(xx[t])):
            if started:
                out[t] = state
            continue
        started = True
        cov = cov + q
        h = np.array([1.0, xx[t]])
        s = float(h @ cov @ h) + obs_var
        gain = cov @ h / s
        state = state + gain * (yy[t] - float(h @ state))
        cov = cov - np.outer(gain, h) @ cov
        out[t] = state
    return pd.DataFrame(out, index=frame.index, columns=["alpha", "beta"])
