"""Stationarity and cointegration tests: thin, crash-proof wrappers around statsmodels."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .hedge import ols_hedge_ratio
from .utils import json_safe

_CRIT_LABELS = ("1%", "5%", "10%")


@dataclass
class TestResult:
    """Outcome of a univariate/Engle-Granger test. ``extra`` holds test-specific details."""

    __test__ = False  # not a pytest test class

    name: str
    statistic: float
    pvalue: float
    critical_values: dict = field(default_factory=dict)
    n_obs: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-safe dict."""
        return json_safe(asdict(self))


@dataclass
class JohansenResult:
    """Johansen trace / max-eigenvalue statistics with 95% critical values.

    ``rank`` is the sequential trace-test rank at 95%; ``weights`` is the first
    cointegrating eigenvector normalised so the weight of leg 0 is 1.
    """

    trace_stats: list
    trace_crit_95: list
    max_eig_stats: list
    max_eig_crit_95: list
    rank: int
    weights: list
    legs: list = field(default_factory=list)
    n_obs: int = 0

    def to_dict(self) -> dict:
        """JSON-safe dict."""
        return json_safe(asdict(self))


def _failed(name: str, n_obs: int, err: Exception | str) -> TestResult:
    """Result for a test that could not be computed (degenerate input)."""
    return TestResult(name=name, statistic=float("nan"), pvalue=float("nan"),
                      critical_values={}, n_obs=n_obs, extra={"error": str(err)})


def adf_test(series: pd.Series | np.ndarray, regression: str, autolag: str | None) -> TestResult:
    """Augmented Dickey-Fuller test (H0: unit root). NaNs are dropped; never raises."""
    x = pd.Series(np.asarray(series, dtype=float)).dropna().to_numpy()
    try:
        stat, pval, lags, nobs, crit, *_ = adfuller(x, regression=regression, autolag=autolag,
                                                      result_object=False)
    except Exception as err:  # degenerate input (constant, too short)
        return _failed("adf", int(x.size), err)
    return TestResult(name="adf", statistic=float(stat), pvalue=float(pval),
                      critical_values={k: float(v) for k, v in crit.items()},
                      n_obs=int(nobs), extra={"used_lag": int(lags), "regression": regression})


def engle_granger(
    y: pd.Series,
    x: pd.Series | pd.DataFrame,
    trend: str,
    autolag: str | None = "aic",
) -> TestResult:
    """Engle-Granger cointegration test of ``y`` on ``x`` (one or more regressors).

    ``extra`` holds the static OLS hedge ratio (``hedge_ratio``: float for one regressor,
    list otherwise) and intercept (``alpha``) estimated on the same data.
    """
    frame = pd.concat([pd.Series(y).rename("__y"), pd.DataFrame(x)], axis=1).dropna()
    n = len(frame)
    if n == 0:
        return _failed("engle_granger", 0, "no overlapping observations")
    yy, xx = frame["__y"], frame.drop(columns="__y")
    try:
        stat, pval, crit = coint(yy.to_numpy(), xx.to_numpy(), trend=trend,
                                 autolag=autolag.lower() if autolag else None)
    except Exception as err:
        return _failed("engle_granger", n, err)
    if xx.shape[1] == 1:
        fit = ols_hedge_ratio(yy, xx.iloc[:, 0], add_constant=trend != "n")
        hedge, alpha = fit.beta, fit.alpha
    else:
        design = np.column_stack([np.ones(n), xx.to_numpy()])
        coef = np.linalg.lstsq(design, yy.to_numpy(), rcond=None)[0]
        hedge, alpha = coef[1:].tolist(), float(coef[0])
    return TestResult(name="engle_granger", statistic=float(stat), pvalue=float(pval),
                      critical_values=dict(zip(_CRIT_LABELS, map(float, crit))), n_obs=n,
                      extra={"hedge_ratio": hedge, "alpha": alpha, "trend": trend})


def johansen_test(prices: pd.DataFrame, det_order: int, k_ar_diff: int) -> JohansenResult:
    """Johansen cointegration test on the columns of ``prices`` (NaN rows dropped).

    Raises ``ValueError`` for fewer than 2 columns (Johansen is undefined there).
    """
    data = pd.DataFrame(prices).dropna()
    if data.shape[1] < 2:
        raise ValueError("johansen_test needs at least 2 series")
    res = coint_johansen(data.to_numpy(dtype=float), det_order, k_ar_diff)
    trace, trace_cv = res.lr1, res.cvt[:, 1]
    rank = 0
    for stat, crit in zip(trace, trace_cv):
        if stat > crit:
            rank += 1
        else:
            break
    vec = res.evec[:, 0]
    weights = vec / vec[0] if vec[0] != 0 else vec
    return JohansenResult(trace_stats=trace.tolist(), trace_crit_95=trace_cv.tolist(),
                          max_eig_stats=res.lr2.tolist(), max_eig_crit_95=res.cvm[:, 1].tolist(),
                          rank=int(rank), weights=weights.tolist(),
                          legs=[str(c) for c in data.columns], n_obs=int(len(data)))
