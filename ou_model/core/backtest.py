"""Walk-forward spread backtest.

Timing: the signal uses information up to close ``t`` and is filled at close
``t + execution_lag``. Shares are fixed at entry; P&L comes from leg prices; costs are
``cost_bps_per_leg / 1e4`` of traded notional per leg at entry and exit. Returns are on a
fixed capital base ``notional`` (non-compounding), so ``equity = 1 + cumsum(pnl)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import get
from .hedge import kalman_hedge_ratio, ols_hedge_ratio, rolling_hedge_ratio
from .ou import fit_ou
from .signals import generate_signals, rolling_zscore

BPS = 1e4
DAILY_COLUMNS = ["spread", "spread_mean", "spread_std", "zscore", "hedge_ratio", "target",
                 "position", "event", "pnl", "gross_pnl", "cost", "equity"]
TRADE_COLUMNS = ["trade_id", "entry_date", "exit_date", "direction", "entry_z", "exit_z",
                 "exit_reason", "holding_days", "gross_pnl", "cost", "pnl"]


@dataclass
class BacktestConfig:
    """All parameters the backtest needs; build with ``BacktestConfig.from_config(cfg)``."""

    execution_lag: int
    cost_bps_per_leg: float
    notional: float
    sizing: str
    reestimation_window: int
    reestimation_freq: int
    require_mean_reverting: bool
    max_holding_multiple: float | None
    zscore_window: int
    zscore_min_periods: int
    entry_z: float
    exit_z: float
    stop_z: float
    reentry_requires_exit: bool
    dt: float
    trading_days_per_year: int
    ou_method: str
    ou_min_obs: int

    @classmethod
    def from_config(cls, cfg: dict) -> "BacktestConfig":
        """Build from a merged config dict (``core.yaml`` layout)."""
        return cls(
            execution_lag=int(get(cfg, "backtest.execution_lag")),
            cost_bps_per_leg=float(get(cfg, "backtest.cost_bps_per_leg")),
            notional=float(get(cfg, "backtest.notional")),
            sizing=str(get(cfg, "backtest.sizing")),
            reestimation_window=int(get(cfg, "backtest.reestimation_window")),
            reestimation_freq=int(get(cfg, "backtest.reestimation_freq")),
            require_mean_reverting=bool(get(cfg, "backtest.require_mean_reverting")),
            max_holding_multiple=(None if get(cfg, "backtest.max_holding_multiple") is None
                                  else float(get(cfg, "backtest.max_holding_multiple"))),
            zscore_window=int(get(cfg, "signals.zscore_window")),
            zscore_min_periods=int(get(cfg, "signals.zscore_min_periods")),
            entry_z=float(get(cfg, "signals.entry_z")),
            exit_z=float(get(cfg, "signals.exit_z")),
            stop_z=float(get(cfg, "signals.stop_z")),
            reentry_requires_exit=bool(get(cfg, "signals.reentry_requires_exit")),
            dt=float(get(cfg, "calendar.dt")),
            trading_days_per_year=int(get(cfg, "calendar.trading_days_per_year")),
            ou_method=str(get(cfg, "ou.primary_method")),
            ou_min_obs=int(get(cfg, "ou.min_obs")),
        )


@dataclass
class BacktestResult:
    """Backtest output: ``daily`` (section 2 columns + ``cost``), ``trades``, ``ou_history``."""

    daily: pd.DataFrame
    trades: pd.DataFrame
    ou_history: pd.DataFrame = field(default_factory=pd.DataFrame)


# --------------------------------------------------------------------------- spread building

def _transform(prices: pd.Series, transform: str) -> pd.Series:
    """Apply the spread transform (``log`` masks non-positive prices to NaN, ``level`` is identity)."""
    s = pd.Series(prices, dtype=float)
    if transform == "log":
        return np.log(s.where(s > 0))
    if transform == "level":
        return s
    raise ValueError(f"unknown transform '{transform}'")


def build_pair_spread(
    y: pd.Series,
    x: pd.Series,
    transform: str,
    hedge_cfg: dict,
    fit_end: str | pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Hedged 2-leg spread ``T(y)_t - beta_t T(x)_t - alpha_t``.

    ``hedge_cfg`` is the ``hedge`` config section; ``method`` is ``rolling_ols`` (causal),
    ``static_ols`` (one fit on data ``<= fit_end`` - pass the in-sample end to avoid look-ahead;
    ``None`` fits on all data), ``fixed`` (``hedge_cfg["beta"]``, optional ``alpha``) or
    ``kalman`` (needs ``kalman_delta``, ``kalman_obs_var``, ``kalman_init_var``).

    Returns ``(spread, leg_weights, hedge_history)``. ``leg_weights`` are signed **shares** of
    each leg per +1 unit of spread at ``t``: ``(1, -beta)`` for ``level``,
    ``(1/P_y, -beta/P_x)`` for ``log`` (i.e. $1 of y against $beta of x).
    """
    ty, tx = _transform(y, transform), _transform(x, transform)
    method = hedge_cfg["method"]
    add_const = bool(hedge_cfg.get("add_constant", True))
    if method == "rolling_ols":
        hist = rolling_hedge_ratio(ty, tx, int(hedge_cfg["window"]), int(hedge_cfg["min_periods"]),
                                   add_constant=add_const)
    elif method == "static_ols":
        mask = ty.index <= pd.Timestamp(fit_end) if fit_end is not None else slice(None)
        fit = ols_hedge_ratio(ty[mask], tx[mask], add_constant=add_const)
        hist = pd.DataFrame({"alpha": fit.alpha, "beta": fit.beta}, index=ty.index)
    elif method == "fixed":
        hist = pd.DataFrame({"alpha": float(hedge_cfg.get("alpha", 0.0)),
                             "beta": float(hedge_cfg["beta"])}, index=ty.index)
    elif method == "kalman":
        hist = kalman_hedge_ratio(ty, tx, float(hedge_cfg["kalman_delta"]),
                                  float(hedge_cfg["kalman_obs_var"]),
                                  float(hedge_cfg["kalman_init_var"]))
    else:
        raise ValueError(f"unknown hedge method '{method}'")
    spread = (ty - hist["beta"] * tx - hist["alpha"]).rename("spread")
    names = [getattr(y, "name", None) or "y", getattr(x, "name", None) or "x"]
    if transform == "log":
        weights = pd.DataFrame({names[0]: 1.0 / y, names[1]: -hist["beta"] / x}, index=ty.index)
    else:
        weights = pd.DataFrame({names[0]: 1.0, names[1]: -hist["beta"]}, index=ty.index)
    weights = weights.where(hist["beta"].notna(), axis=0)
    return spread, weights, hist


# --------------------------------------------------------------------------- OU re-estimation

def rolling_ou_history(spread: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    """Re-fit OU every ``reestimation_freq`` obs on the trailing ``reestimation_window`` obs
    (data ``<= t``). Fits with fewer than ``ou_min_obs`` finite points are skipped.

    Returns one row per successful fit: ``date, kappa, mu, sigma, half_life, mean_reverting``.
    """
    values = pd.Series(spread, dtype=float).to_numpy()
    rows = []
    for i in range(0, len(values), cfg.reestimation_freq):
        window = values[max(0, i - cfg.reestimation_window + 1): i + 1]
        window = window[np.isfinite(window)]
        if window.size < cfg.ou_min_obs:
            continue
        p = fit_ou(window, cfg.dt, method=cfg.ou_method)
        rows.append({"date": spread.index[i], "kappa": p.kappa, "mu": p.mu, "sigma": p.sigma,
                     "half_life": p.half_life, "mean_reverting": bool(p.mean_reverting)})
    cols = ["date", "kappa", "mu", "sigma", "half_life", "mean_reverting"]
    return pd.DataFrame(rows, columns=cols)


def _ou_state(ou_hist: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Latest OU fit known at each date (forward-filled from fit dates; before the first fit:
    not mean-reverting, infinite half-life)."""
    if ou_hist.empty:
        return pd.DataFrame({"mean_reverting": False, "half_life": math.inf}, index=index)
    state = ou_hist.set_index("date")[["mean_reverting", "half_life"]].reindex(index).ffill()
    state["mean_reverting"] = state["mean_reverting"].astype("boolean").fillna(False).astype(bool)
    state["half_life"] = state["half_life"].astype(float).fillna(math.inf)
    return state


def compute_signals(
    spread: pd.Series,
    cfg: BacktestConfig,
    allow_entry: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Z-score, rolling-OU gating and the target signal (all causal).

    Returns ``(zscore_frame, signals, ou_history)``.
    """
    zdf = rolling_zscore(spread, cfg.zscore_window, cfg.zscore_min_periods)
    ou_hist = rolling_ou_history(spread, cfg)
    state = _ou_state(ou_hist, spread.index)
    allow = pd.Series(True, index=spread.index)
    if allow_entry is not None:
        allow &= pd.Series(allow_entry).reindex(spread.index).fillna(False).astype(bool)
    if cfg.require_mean_reverting:
        allow &= state["mean_reverting"]
    max_hold = None
    if cfg.max_holding_multiple is not None:
        max_hold = state["half_life"] * cfg.trading_days_per_year * cfg.max_holding_multiple
    sig = generate_signals(zdf["zscore"], cfg.entry_z, cfg.exit_z, cfg.stop_z,
                           reentry_requires_exit=cfg.reentry_requires_exit,
                           allow_entry=allow, max_holding=max_hold)
    return zdf, sig, ou_hist


# --------------------------------------------------------------------------- execution

def size_position(direction: int, weights: np.ndarray, prices: np.ndarray, notional: float,
                  sizing: str) -> np.ndarray:
    """Shares per leg for a ``direction`` (+1/-1) spread position.

    ``dollar_neutral``: long-side dollars = short-side dollars = notional / 2, split within a
    side in proportion to ``|w * P|``; if only one side exists (single leg, or all weights the
    same sign), that side gets the whole notional. ``units``: shares proportional to the weights,
    scaled to gross notional.
    """
    exposure = direction * weights * prices
    gross = np.abs(exposure)
    if not np.all(np.isfinite(exposure)) or gross.sum() <= 0:
        return np.zeros_like(weights, dtype=float)
    if sizing == "units":
        return direction * weights * notional / gross.sum()
    if sizing != "dollar_neutral":
        raise ValueError(f"unknown sizing '{sizing}'")
    long, short = exposure > 0, exposure < 0
    dollars = np.zeros_like(exposure)
    if long.any() and short.any():
        dollars[long] = notional / 2 * gross[long] / gross[long].sum()
        dollars[short] = -notional / 2 * gross[short] / gross[short].sum()
    else:
        dollars = np.sign(exposure) * notional * gross / gross.sum()
    return dollars / prices


def execute_target(
    legs: pd.DataFrame,
    leg_weights: pd.DataFrame,
    target: pd.Series,
    cfg: BacktestConfig,
    cost_bps_per_leg: float | None = None,
    events: pd.Series | None = None,
    zscore: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn a signal-date ``target`` into fills, positions, P&L and round-trip trades.

    ``position_t = target_{t - execution_lag}``; shares are sized at the fill close from the
    weights known at the signal date and held unchanged until the position changes.
    ``gross_pnl_t = sum_i shares_i (P_i,t - P_i,t-1) / notional``.
    ``events``/``zscore`` (signal-dated) only label trades.

    Returns ``(daily, trades)``: ``daily`` has ``position, gross_pnl, cost, pnl``.
    """
    lag = cfg.execution_lag
    cost_rate = (cfg.cost_bps_per_leg if cost_bps_per_leg is None else cost_bps_per_leg) / BPS
    idx = legs.index
    prices = legs.astype(float).ffill().to_numpy()
    weights = leg_weights.reindex(index=idx, columns=legs.columns).astype(float).to_numpy()
    tgt = pd.Series(target).reindex(idx).fillna(0).astype(np.int64)
    position = tgt.shift(lag).fillna(0).astype(np.int64).to_numpy()
    ev = (pd.Series(events).reindex(idx).fillna("").to_numpy(dtype=object)
          if events is not None else np.full(len(idx), "", dtype=object))
    zz = (pd.Series(zscore, dtype=float).reindex(idx).to_numpy()
          if zscore is not None else np.full(len(idx), np.nan))

    n, k = prices.shape
    gross, cost = np.zeros(n), np.zeros(n)
    shares = np.zeros(k)
    trades: list[dict] = []
    open_trade: dict | None = None
    prev_pos = 0
    for i in range(n):
        if i > 0:
            move = np.nan_to_num(prices[i] - prices[i - 1])
            gross[i] = float(shares @ move) / cfg.notional
            if open_trade is not None:
                open_trade["gross_pnl"] += gross[i]
        if position[i] == prev_pos:
            continue
        new = (size_position(int(position[i]), weights[i - lag], prices[i], cfg.notional,
                             cfg.sizing) if position[i] != 0 else np.zeros(k))
        cost[i] = cost_rate * float(np.abs(new - shares) @ np.nan_to_num(prices[i])) / cfg.notional
        exit_cost = cost_rate * float(np.abs(shares) @ np.nan_to_num(prices[i])) / cfg.notional
        if open_trade is not None:
            reason = ev[i - lag] if ev[i - lag] in ("exit", "stop", "time") else "exit"
            trades.append(_close(open_trade, idx[i], i, zz[i - lag], reason, exit_cost))
            open_trade = None
        if position[i] != 0:
            open_trade = {"entry_date": idx[i], "entry_i": i, "direction": int(position[i]),
                          "entry_z": zz[i - lag], "gross_pnl": 0.0,
                          "cost": cost[i] - (exit_cost if prev_pos != 0 else 0.0)}
        shares, prev_pos = new, int(position[i])
    if open_trade is not None:
        trades.append(_close(open_trade, idx[-1], n - 1, zz[-1], "end_of_data", 0.0))

    daily = pd.DataFrame({"position": position, "gross_pnl": gross, "cost": cost,
                          "pnl": gross - cost}, index=idx)
    trade_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    trade_df["trade_id"] = np.arange(1, len(trade_df) + 1, dtype=np.int64)
    return daily, trade_df


def _close(trade: dict, date: pd.Timestamp, i: int, exit_z: float, reason: str,
           exit_cost: float) -> dict:
    """Finalise an open trade record."""
    total_cost = trade["cost"] + exit_cost
    return {"trade_id": 0, "entry_date": trade["entry_date"], "exit_date": date,
            "direction": trade["direction"], "entry_z": float(trade["entry_z"]),
            "exit_z": float(exit_z), "exit_reason": reason,
            "holding_days": int(i - trade["entry_i"]), "gross_pnl": float(trade["gross_pnl"]),
            "cost": float(total_cost), "pnl": float(trade["gross_pnl"] - total_cost)}


def run_backtest(
    legs: pd.DataFrame,
    spread: pd.Series,
    leg_weights: pd.DataFrame,
    cfg: BacktestConfig,
    cost_bps_per_leg: float | None = None,
    trade_start: str | pd.Timestamp | None = None,
) -> BacktestResult:
    """Walk-forward backtest of a spread (see module docstring for timing and costs).

    ``legs``: tradable leg prices; ``spread``: series the signal is computed on;
    ``leg_weights``: signed shares per +1 spread unit known at ``t`` (same columns as ``legs``).
    New entries are allowed only where the latest rolling OU fit is mean-reverting (if
    ``require_mean_reverting``), weights and prices are finite at the signal date, and the
    signal date is ``>= trade_start`` (data before it is warm-up only).
    """
    legs = pd.DataFrame(legs).astype(float)
    idx = legs.index
    spread = pd.Series(spread, dtype=float).reindex(idx)
    leg_weights = pd.DataFrame(leg_weights).reindex(index=idx, columns=legs.columns)
    tradable = leg_weights.notna().all(axis=1) & legs.ffill().notna().all(axis=1)
    if trade_start is not None:
        tradable &= idx >= pd.Timestamp(trade_start)
    zdf, sig, ou_hist = compute_signals(spread, cfg, allow_entry=tradable)
    exe, trades = execute_target(legs, leg_weights, sig["target"], cfg, cost_bps_per_leg,
                                 events=sig["event"], zscore=zdf["zscore"])
    event_exec = sig["event"].shift(cfg.execution_lag).fillna("")
    daily = pd.DataFrame({
        "spread": spread, "spread_mean": zdf["spread_mean"], "spread_std": zdf["spread_std"],
        "zscore": zdf["zscore"], "hedge_ratio": np.nan, "target": sig["target"].astype(np.int64),
        "position": exe["position"].astype(np.int64), "event": event_exec.astype(object),
        "pnl": exe["pnl"], "gross_pnl": exe["gross_pnl"], "cost": exe["cost"],
    }, index=idx)
    daily["equity"] = 1.0 + daily["pnl"].cumsum()
    return BacktestResult(daily=daily[DAILY_COLUMNS], trades=trades, ou_history=ou_hist)
