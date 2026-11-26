"""Rolling z-score and the entry/exit/stop state machine."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

EVENTS = ("", "entry", "exit", "stop", "time")


def rolling_zscore(spread: pd.Series, window: int, min_periods: int) -> pd.DataFrame:
    """Trailing z-score of ``spread`` using observations ``<= t`` only.

    Returns columns ``spread_mean``, ``spread_std`` (sample std) and ``zscore``
    (NaN when the std is zero or undefined).
    """
    s = pd.Series(spread, dtype=float)
    roll = s.rolling(window, min_periods=min_periods)
    mean, std = roll.mean(), roll.std()
    std = std.where(std > 0)
    return pd.DataFrame({"spread_mean": mean, "spread_std": std, "zscore": (s - mean) / std},
                        index=s.index)


def _flat_step(z: float, locked: bool, can_enter: bool, entry_z: float, exit_z: float,
               stop_z: float) -> tuple[int, bool, str]:
    """Transition from a flat state: returns (target, locked, event)."""
    if locked and abs(z) < exit_z:
        locked = False
    if locked or not can_enter:
        return 0, locked, ""
    if -stop_z < z < -entry_z:
        return 1, locked, "entry"
    if entry_z < z < stop_z:
        return -1, locked, "entry"
    return 0, locked, ""


def _open_step(pos: int, z: float, held: int, max_hold: float, exit_z: float,
               stop_z: float, reentry_requires_exit: bool) -> tuple[int, bool, str]:
    """Transition from an open position ``pos``: returns (target, locked, event)."""
    signed = z * pos  # negative while the trade is still away from the mean
    if signed < -stop_z:
        return 0, reentry_requires_exit, "stop"
    if signed > -exit_z:
        return 0, False, "exit"
    if held >= max_hold:
        return 0, reentry_requires_exit, "time"
    return pos, False, ""


def generate_signals(
    zscore: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    reentry_requires_exit: bool = True,
    allow_entry: pd.Series | None = None,
    max_holding: pd.Series | None = None,
) -> pd.DataFrame:
    """State-machine trading signal on a z-score (all decisions use data ``<= t``).

    * flat: go long (+1) when ``-stop_z < z < -entry_z``, short (-1) when ``entry_z < z < stop_z``,
      only where ``allow_entry`` is True.
    * long: ``stop`` when ``z < -stop_z``; ``exit`` when ``z > -exit_z`` (|z| < exit_z or overshoot).
      Short is symmetric.
    * optional ``max_holding`` (bars, per date): ``time`` exit once held that many bars.
    * after a stop/time exit with ``reentry_requires_exit``, no new entry until ``|z| < exit_z``.
    * NaN z: hold the current state, no event.

    Returns columns ``target`` (int in {-1, 0, 1}) and ``event`` ("", "entry", "exit", "stop", "time"),
    both dated at the signal close ``t``.
    """
    z = pd.Series(zscore, dtype=float)
    idx = z.index
    allow = (np.ones(len(z), dtype=bool) if allow_entry is None
             else pd.Series(allow_entry).reindex(idx).fillna(False).to_numpy(dtype=bool))
    hold_cap = (np.full(len(z), math.inf) if max_holding is None
                else pd.Series(max_holding, dtype=float).reindex(idx).fillna(math.inf).to_numpy())
    targets = np.zeros(len(z), dtype=np.int64)
    events = [""] * len(z)
    pos, locked, held = 0, False, 0
    for i, zi in enumerate(z.to_numpy()):
        if np.isfinite(zi):
            if pos == 0:
                pos, locked, ev = _flat_step(zi, locked, allow[i], entry_z, exit_z, stop_z)
                held = 0
            else:
                held += 1
                pos, locked, ev = _open_step(pos, zi, held, hold_cap[i], exit_z, stop_z,
                                             reentry_requires_exit)
            events[i] = ev
        elif pos != 0:
            held += 1
        targets[i] = pos
    return pd.DataFrame({"target": targets, "event": pd.Series(events, index=idx, dtype=object)},
                        index=idx)
