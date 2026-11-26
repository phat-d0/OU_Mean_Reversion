"""End-to-end evaluation of one pair: IS tests and fits, walk-forward backtests, results."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pandas as pd

from .backtest import BacktestConfig, BacktestResult, _transform, run_backtest
from .config import get
from .metrics import compute_metrics
from .ou import OUParams, compare_ou_methods, fit_ou
from .results import write_pair_results
from .stationarity import adf_test, engle_granger, johansen_test
from .utils import json_safe

FLAG_KEYS = ("roll_gaps", "structural_breaks", "non_positive_prices", "notes")


@dataclasses.dataclass(frozen=True)
class Split:
    """In-sample / out-of-sample date boundaries."""

    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp | None

    @classmethod
    def from_config(cls, cfg: dict) -> "Split":
        """Read ``cfg["split"]``."""
        end = get(cfg, "split.out_of_sample_end")
        return cls(pd.Timestamp(get(cfg, "split.in_sample_start")),
                   pd.Timestamp(get(cfg, "split.in_sample_end")),
                   pd.Timestamp(get(cfg, "split.out_of_sample_start")),
                   None if end is None else pd.Timestamp(end))

    def is_mask(self, index: pd.Index) -> pd.Series:
        """True on in-sample dates."""
        return pd.Series((index >= self.is_start) & (index <= self.is_end), index=index)

    def oos_mask(self, index: pd.Index) -> pd.Series:
        """True on out-of-sample dates."""
        upper = index <= self.oos_end if self.oos_end is not None else True
        return pd.Series((index >= self.oos_start) & upper, index=index)

    def label(self, dates: pd.Series | pd.Index) -> list[str]:
        """'IS' for dates <= in-sample end, else 'OOS'."""
        return ["IS" if pd.Timestamp(d) <= self.is_end else "OOS" for d in dates]


def _flags(meta: dict) -> dict:
    """Data-quality flags with every contract key present as a list."""
    given = dict(meta.get("flags") or {})
    out = {k: list(given.pop(k, []) or []) for k in FLAG_KEYS}
    out["notes"] += list(meta.get("notes") or [])
    out.update(given)
    return out


def _ou_block(spread: pd.Series, cfg: dict, notes: list[str]) -> dict[str, OUParams]:
    """IS OU fits for every configured method; too few observations -> flagged, not fitted."""
    dt, sig = float(get(cfg, "calendar.dt")), float(get(cfg, "stationarity.significance"))
    fits = compare_ou_methods(spread, dt, methods=list(get(cfg, "ou.methods")), significance=sig)
    if spread.size < int(get(cfg, "ou.min_obs")):
        notes.append(f"in-sample spread has {spread.size} obs < ou.min_obs; OU not used")
        fits = {m: dataclasses.replace(p, mean_reverting=False) for m, p in fits.items()}
    return fits


def _tests_in_sample(test_legs: pd.DataFrame, spread: pd.Series, cfg: dict) -> dict:
    """ADF on the spread, Engle-Granger (2+ legs) and Johansen (2+ legs) on in-sample data."""
    st = cfg["stationarity"]
    out: dict = {"adf": adf_test(spread, st["adf_regression"], st["adf_autolag"]),
                 "engle_granger": None, "johansen": None}
    if test_legs.shape[1] >= 2:
        y, x = test_legs.iloc[:, 0], test_legs.iloc[:, 1:]
        x = x.iloc[:, 0] if x.shape[1] == 1 else x
        out["engle_granger"] = engle_granger(y, x, st["eg_trend"], autolag=st["adf_autolag"])
        try:
            out["johansen"] = johansen_test(test_legs, int(st["johansen_det_order"]),
                                            int(st["johansen_k_ar_diff"]))
        except Exception as err:  # degenerate data: record, do not crash the pipeline
            out["johansen"] = {"error": str(err)}
    return out


def _cointegrated(tests: dict, n_legs: int, significance: float) -> bool:
    """Verdict: ADF (1 leg), Engle-Granger (2 legs) or Johansen rank >= 1 (3+ legs)."""
    def below(res: object) -> bool:
        p = getattr(res, "pvalue", None)
        return p is not None and math.isfinite(p) and p < significance
    if n_legs <= 1:
        return below(tests["adf"])
    if n_legs == 2:
        return below(tests["engle_granger"])
    return bool(getattr(tests["johansen"], "rank", 0) >= 1)


def _mean_reverting(primary: OUParams, adf: object, significance: float) -> bool:
    """Verdict: the primary OU fit is mean-reverting (b in (0, 1), lag-0 DF p < significance)
    AND the AIC-lag ADF on the in-sample spread rejects a unit root at ``significance``."""
    p = getattr(adf, "pvalue", None)
    adf_ok = p is not None and math.isfinite(p) and p < significance
    return bool(primary.mean_reverting and adf_ok)


def _split_metrics(res: BacktestResult, split: Split, hl_days: float | None, tdpy: int) -> dict:
    """Metrics on IS and OOS dates (trades assigned by entry date)."""
    daily, trades = res.daily, res.trades
    t_is = trades["entry_date"] <= split.is_end if len(trades) else pd.Series(dtype=bool)
    return {
        "in_sample": compute_metrics(daily[split.is_mask(daily.index)],
                                     trades[t_is] if len(trades) else trades, hl_days, tdpy),
        "out_of_sample": compute_metrics(daily[split.oos_mask(daily.index)],
                                         trades[~t_is] if len(trades) else trades, hl_days, tdpy),
    }


def _trim(res: BacktestResult, start: pd.Timestamp) -> BacktestResult:
    """Drop warm-up rows before ``start`` (no positions exist there) and rebuild equity."""
    daily = res.daily[res.daily.index >= start].copy()
    daily["equity"] = 1.0 + daily["pnl"].cumsum()
    return BacktestResult(daily=daily, trades=res.trades, ou_history=res.ou_history)


def evaluate_pair(
    name: str,
    asset_class: str,
    legs: pd.DataFrame,
    spread: pd.Series,
    leg_weights: pd.DataFrame,
    cfg: dict,
    meta: dict | None = None,
    hedge_history: pd.DataFrame | None = None,
    results_root: str | Path | None = None,
) -> Path:
    """Evaluate one pair and write ``results/<asset_class>/<name>/`` (CONTRACTS.md section 2).

    ``legs``: tradable leg prices (columns = leg names, legs[0] = y); ``spread``: the traded
    spread; ``leg_weights``: signed shares per +1 spread unit; ``cfg``: effective pair config
    (``pair_config``). ``meta`` may carry ``flags`` (roll_gaps, structural_breaks,
    non_positive_prices, notes), ``notes``, ``construction`` and ``transform`` (defaults from
    ``cfg["pair"]``). ``hedge_history`` (``alpha``/``beta`` per date) fills ``hedge_ratio``.

    Tests and OU fits use in-sample data only; backtests run walk-forward over all data with
    entries allowed from ``split.in_sample_start``, at the base cost and every
    ``cost_sensitivity_bps``. Data before ``in_sample_start`` is warm-up and is not written.
    """
    meta = dict(meta or {})
    pair = dict(cfg.get("pair") or {})
    split = Split.from_config(cfg)
    legs = pd.DataFrame(legs).astype(float).sort_index()
    if split.oos_end is not None:
        legs = legs[legs.index <= split.oos_end]
    idx = legs.index
    spread = pd.Series(spread, dtype=float).reindex(idx)
    leg_weights = pd.DataFrame(leg_weights).reindex(index=idx, columns=legs.columns)
    construction = meta.get("construction", pair.get("construction"))
    transform = meta.get("transform", pair.get("transform", "level"))
    flags = _flags(meta)
    tdpy, dt = int(get(cfg, "calendar.trading_days_per_year")), float(get(cfg, "calendar.dt"))
    sig = float(get(cfg, "stationarity.significance"))

    is_mask, oos_mask = split.is_mask(idx), split.oos_mask(idx)
    spread_is, spread_oos = spread[is_mask].dropna(), spread[oos_mask].dropna()
    test_legs = legs[is_mask].apply(lambda s: _transform(s, transform))
    tests_is = _tests_in_sample(test_legs, spread_is, cfg)
    ou_is = _ou_block(spread_is, cfg, flags["notes"])
    primary = ou_is[get(cfg, "ou.primary_method")]
    mean_reverting = _mean_reverting(primary, tests_is["adf"], sig)
    hl_days = primary.half_life * tdpy if mean_reverting else None

    bcfg = BacktestConfig.from_config(cfg)
    costs = [bcfg.cost_bps_per_leg] + [float(c) for c in get(cfg, "backtest.cost_sensitivity_bps")]
    runs = {c: _trim(run_backtest(legs, spread, leg_weights, bcfg, c, trade_start=split.is_start),
                     split.is_start) for c in dict.fromkeys(costs)}
    base = runs[bcfg.cost_bps_per_leg]

    daily = base.daily.copy()
    if hedge_history is not None and "beta" in hedge_history:
        daily["hedge_ratio"] = pd.Series(hedge_history["beta"], dtype=float).reindex(daily.index)
    daily["sample"] = split.label(daily.index)
    trades = base.trades.copy()
    trades["sample"] = split.label(trades["entry_date"])

    eg = tests_is["engle_granger"]
    summary = {
        "asset_class": asset_class, "pair": name, "legs": [str(c) for c in legs.columns],
        "construction": construction, "transform": transform,
        "data": {"start": daily.index.min() if len(daily) else None,
                 "end": daily.index.max() if len(daily) else None,
                 "n_obs": int(len(daily)), "flags": flags},
        "split": {"in_sample": [split.is_start, split.is_end],
                  "out_of_sample": [split.oos_start, split.oos_end or (idx.max() if len(idx) else None)]},
        "hedge_ratio_in_sample": (eg.extra.get("hedge_ratio") if eg is not None and legs.shape[1] == 2
                                  else None),
        "ou_params": {"in_sample": {m: p.to_dict() for m, p in ou_is.items()},
                      "out_of_sample_diagnostic": {"ols": fit_ou(spread_oos, dt, "ols",
                                                                 significance=sig).to_dict()}},
        "tests": {"in_sample": tests_is,
                  "out_of_sample_diagnostic": {"adf": adf_test(
                      spread_oos, cfg["stationarity"]["adf_regression"],
                      cfg["stationarity"]["adf_autolag"])}},
        "verdict": {"cointegrated_in_sample": _cointegrated(tests_is, legs.shape[1], sig),
                    "mean_reverting_in_sample": mean_reverting},
        "metrics": _split_metrics(base, split, hl_days, tdpy),
        "cost_sensitivity": {str(float(c)): _split_metrics(runs[c], split, hl_days, tdpy)
                             for c in get(cfg, "backtest.cost_sensitivity_bps")},
        "config": {k: v for k, v in cfg.items() if k != "asset"},
    }
    root = Path(results_root if results_root is not None else get(cfg, "results.root"))
    return write_pair_results(root / asset_class / name, daily, trades, json_safe(summary))
