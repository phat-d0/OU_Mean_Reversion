# Interface contracts

This file is the single source of truth between agents. The Lead reviews every
deliverable against it. If an agent needs to change a contract, it proposes the
change to the Lead; only the Quant agent edits sections 1–2 (and only in Phase 2).

Research use only: no broker connections, no live orders.

---

## 0. Ownership

| Path | Owner | Others may |
|---|---|---|
| `ou_model/core/`, `tests/test_core_*.py` | Quant | import only |
| `ou_model/assets/equities/`, `tests/test_equities_*.py` | Equities | — |
| `ou_model/assets/commodities/`, `tests/test_commodities_*.py` | Commodities | — |
| `ou_model/assets/fx/`, `tests/test_fx_*.py` | FX | — |
| `ou_model/ui/`, `tests/test_ui_*.py` | UI | — |
| `config/`, `CONTRACTS.md`, `run_all.py`, `Makefile`, `REPORT.md` | Lead | propose changes |
| `results/<asset_class>/` | the asset agent for that class | UI reads only |

Asset agents may add keys to their own `config/<asset_class>.yaml` and must tell the Lead.

---

## 1. Engine API (`ou_model/core`)

Every public function has type hints and a docstring. No module-level numeric
constants for model parameters: they come from config (`config/core.yaml`).
Names below are fixed; the Quant agent may add optional keyword arguments.

### `core/config.py`
- `load_config(asset_class: str | None = None, config_dir: str | Path = "config") -> dict` —
  loads `core.yaml`, deep-merges `<asset_class>.yaml` (`overrides` key) on top.
- `pair_config(cfg: dict, pair: dict) -> dict` — further merges a pair's `overrides`
  (and `in_sample_start_override`) to give the effective config for one pair.
  `load_config` also stores the full asset YAML under `cfg["asset"]` and the name under `cfg["asset_class"]`;
  `pair_config` stores the pair definition under `cfg["pair"]`.

### `core/ou.py`
- `@dataclass(frozen=True) OUParams`: `kappa, mu, sigma, half_life, b, r2, n_obs, mean_reverting: bool, method: str, unit_root_pvalue: float`,
  plus `to_dict() -> dict` (JSON-safe: `inf`/`nan` → `null`).
  If `b ≥ 1` or `b ≤ 0`: `mean_reverting=False`, `kappa=0.0`, `half_life=inf`, no exception, no NaN in `kappa`/`mu`/`sigma`.
- `simulate_ou(kappa, mu, sigma, x0, n, dt, seed=None) -> np.ndarray` — exact discretisation, length `n` (first element `x0`).
- `fit_ou(series, dt, method="ols", significance=None) -> OUParams` — if `significance` is given, a Dickey-Fuller
  p-value (`unit_root_pvalue`, H0: b = 1) ≥ `significance` also sets `mean_reverting=False` (parameters still reported);
  the pipeline's in-sample fits pass `stationarity.significance`, the backtest's rolling gate does not (b-only). `ols`: AR(1) `x_t = a + b x_{t-1} + ε`;
  `κ = −ln b / Δt`, `μ = a / (1−b)`, `σ = std(ε)·√(−2 ln b / (Δt (1−b²)))`, `half_life = ln 2 / κ` (in units of Δt's time base; see below).
  `mle`: exact Gaussian likelihood of the OU transition density.
- `compare_ou_methods(series, dt, methods=("ols","mle"), significance=None) -> dict[str, OUParams]`.
- **Half-life units**: `OUParams.half_life` is in **years** (same base as `dt`); `metrics` reports `half_life_days = half_life * trading_days_per_year`.

### `core/stationarity.py`
- `@dataclass TestResult`: `name, statistic, pvalue, critical_values: dict, n_obs, extra: dict`, `to_dict()`.
- `adf_test(series, regression, autolag) -> TestResult`
- `engle_granger(y, x, trend, autolag="aic") -> TestResult` (`x` may be a DataFrame for 3+ legs; `extra` holds `hedge_ratio` and `alpha`).
- `@dataclass JohansenResult`: `trace_stats, trace_crit_95, max_eig_stats, max_eig_crit_95, rank, weights` (first eigenvector normalised so weight of leg 0 = 1), `to_dict()`.
- `johansen_test(prices: pd.DataFrame, det_order, k_ar_diff) -> JohansenResult`.

### `core/hedge.py`
- `@dataclass HedgeFit`: `alpha, beta, r2`.
- `ols_hedge_ratio(y, x, add_constant=True) -> HedgeFit`
- `rolling_hedge_ratio(y, x, window, min_periods, add_constant=True) -> pd.DataFrame` (`alpha`, `beta`), value at `t` uses data up to and including `t` only.
- `kalman_hedge_ratio(y, x, delta, obs_var, init_var) -> pd.DataFrame` (`alpha`, `beta`, causal filter). Used when
  `hedge.method: kalman`, which then needs `hedge.kalman_delta`, `hedge.kalman_obs_var`, `hedge.kalman_init_var` in config.

### `core/signals.py`
- `rolling_zscore(spread, window, min_periods) -> pd.DataFrame` (`spread_mean`, `spread_std`, `zscore`), past-and-present data only.
- `generate_signals(zscore, entry_z, exit_z, stop_z, reentry_requires_exit=True, allow_entry: pd.Series | None = None, max_holding: pd.Series | None = None) -> pd.DataFrame`
  with `target` ∈ {−1, 0, +1} (+1 = long spread, enter when `−stop_z < z < −entry_z`) and `event` ∈ {"", "entry", "exit", "stop", "time"}.
  Exit when z crosses back inside `exit_z` (or overshoots); `time` = held `max_holding` bars (from `max_holding_multiple`).

### `core/backtest.py`
- `@dataclass BacktestConfig` built by `BacktestConfig.from_config(cfg: dict)`.
- `run_backtest(legs: pd.DataFrame, spread: pd.Series, leg_weights: pd.DataFrame, cfg: BacktestConfig, cost_bps_per_leg: float | None = None, trade_start=None) -> BacktestResult`
  (no entries on signal dates before `trade_start`: earlier data is warm-up only).
  - `legs`: tradable leg **prices** (levels, incl. any ×42 scaling) — P&L is computed from these.
  - `spread`: the spread series the signal is computed on.
  - `leg_weights`: signed exposure of each leg per +1 unit of spread at date `t` (known at `t`), same columns as `legs`,
    expressed in **shares** (so `|w × price|` is dollars): `(1, −β)` for level spreads, `(1/P_y, −β/P_x)` for log spreads.
  - Timing: signal from information up to close `t`; position filled at close `t + execution_lag`. Shares fixed at entry, no intra-trade rebalancing. Costs = `cost_bps_per_leg / 1e4 ×` traded notional per leg, at entry and exit.
  - Sizing `dollar_neutral`: long-side dollars = short-side dollars = `notional / 2`, split within a side in proportion to `|leg_weight × price|`. Single-leg series: the whole `notional` in that leg. `units`: shares ∝ `leg_weights`, scaled to gross `notional`.
  - Rolling OU re-fit every `reestimation_freq` obs on the trailing `reestimation_window`; if `require_mean_reverting`, new entries are blocked while the latest fit has `mean_reverting=False`.
- `BacktestResult`: `daily: pd.DataFrame` (section 2 columns except `sample`, plus `cost`), `trades: pd.DataFrame`, `ou_history: pd.DataFrame` (date, kappa, mu, sigma, half_life, mean_reverting).
- `build_pair_spread(y, x, transform, hedge_cfg, fit_end=None) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]` → `(spread, leg_weights, hedge_history)` for 2-leg hedged spreads.
  `static_ols` fits on data ≤ `fit_end` (pass the in-sample end; `None` = all data, look-ahead). `log` masks prices ≤ 0 to NaN.
- Also public: `execute_target(legs, leg_weights, target, cfg, cost_bps_per_leg=None, ...)` (fills a given signal-date target),
  `compute_signals`, `rolling_ou_history`, `size_position`.

### `core/metrics.py`
- `compute_metrics(daily: pd.DataFrame, trades: pd.DataFrame, half_life_days: float | None, trading_days_per_year: int) -> dict` with keys:
  `ann_return, ann_vol, sharpe, max_drawdown, hit_rate, n_trades, avg_holding_days, half_life_days, holding_to_half_life`,
  plus `total_return, total_gross_return, total_cost, n_obs`. `ann_return` = mean daily pnl × N (returns are on fixed notional),
  `max_drawdown` ≤ 0 as a fraction of peak equity.
  All JSON-safe (`null` when undefined, e.g. zero trades).

### `core/pipeline.py`
- `evaluate_pair(name, asset_class, legs, spread, leg_weights, cfg, meta, hedge_history=None, results_root=None) -> Path`:
  runs IS-only tests and OU fits, the walk-forward backtest at the base cost and at every `cost_sensitivity_bps`,
  splits metrics IS/OOS by date, and writes section 2 via `core/results.py`. `meta` carries data flags
  (`{"flags": {"roll_gaps", "structural_breaks", "non_positive_prices", "notes"}}`, optional `construction`, `transform`;
  defaults come from `cfg["pair"]`). `hedge_history` (`beta` column) fills `hedge_ratio`. Rows before `in_sample_start` are warm-up and not written.
  Verdict `mean_reverting_in_sample`: primary OU fit mean-reverting (b ∈ (0, 1) and lag-0 DF p < significance) **and** in-sample ADF (`adf_autolag`) p < significance; when False, `half_life_days` is null in metrics.
  Verdict `cointegrated_in_sample`: ADF on the spread (1 leg), Engle-Granger (2 legs), Johansen rank ≥ 1 (3+ legs), at `stationarity.significance`.
- `core/results.py`: `write_pair_results(out_dir, spread_df, trades_df, summary) -> Path` and `validate_pair_results(path) -> list[str]` (empty list = valid).
- `core/synthetic.py`: `synthetic_cointegrated_pair(...)` for tests/demos.

---

## 2. Output contract (`results/<asset_class>/<pair>/`)

### `spread.parquet` — one row per date
| column | type | meaning |
|---|---|---|
| `date` | datetime64 | trading date |
| `spread` | float | spread value |
| `spread_mean` | float | rolling mean used for z |
| `spread_std` | float | rolling std used for z |
| `zscore` | float | rolling z-score |
| `hedge_ratio` | float | beta for 2-leg hedged spreads, else NaN |
| `target` | int | signal at close t (before execution lag) |
| `position` | int | position actually held at close t (−1/0/+1) |
| `event` | str | "", "entry", "exit", "stop", "time" (on execution date) |
| `pnl` | float | net daily return on `notional`, base cost |
| `gross_pnl` | float | daily return before costs |
| `equity` | float | cumulative net return curve, `1 + cumsum(pnl)` (non-compounding, fixed notional), starts at 1.0 |
| `sample` | str | "IS" or "OOS" |

### `trades.parquet` — one row per round trip
`trade_id, entry_date, exit_date, direction (+1/−1), entry_z, exit_z, exit_reason ("exit"|"stop"|"time"|"end_of_data"), holding_days, gross_pnl, cost, pnl, sample ("IS"|"OOS" by entry date)`.

### `summary.json`
```json
{
  "asset_class": "equities", "pair": "ko_pep", "legs": ["KO", "PEP"],
  "construction": "hedged_spread", "transform": "log",
  "data": {"start": "...", "end": "...", "n_obs": 0, "flags": {"roll_gaps": [], "structural_breaks": [], "non_positive_prices": [], "notes": []}},
  "split": {"in_sample": ["2015-01-01", "2021-12-31"], "out_of_sample": ["2022-01-01", "..."]},
  "hedge_ratio_in_sample": 0.0,
  "ou_params": {"in_sample": {"ols": {}, "mle": {}}, "out_of_sample_diagnostic": {"ols": {}}},
  "tests": {"in_sample": {"adf": {}, "engle_granger": {}, "johansen": {}},
            "out_of_sample_diagnostic": {"adf": {}}},
  "verdict": {"cointegrated_in_sample": true, "mean_reverting_in_sample": true},
  "metrics": {"in_sample": {}, "out_of_sample": {}},
  "cost_sensitivity": {"0.0": {"in_sample": {}, "out_of_sample": {}}, "5.0": {}, "10.0": {}},
  "config": {}
}
```
- `engle_granger` is `null` for single-series pairs; `johansen` is `null` for fewer than 2 legs (it is required for 3+).
- `*_diagnostic` entries describe the OOS period but are **never** used to choose a pair, threshold or parameter.
- Pairs that fail cointegration are still written, with `verdict` false.

---

## 3. Asset module contract

Each `ou_model/assets/<asset_class>/` exposes:
```python
def run(config_dir: str | Path = "config", results_root: str | Path = "results") -> list[Path]:
    """Download/clean data, build every configured pair, call core.pipeline.evaluate_pair, return result dirs."""
```
`run_all.py` calls `equities.run()`, `commodities.run()`, `fx.run()` in turn.

## 4. UI contract
`ou_model/ui/app.py` reads `results/**/summary.json`, `spread.parquet`, `trades.parquet` and `config/ui.yaml`.
It never downloads data and never imports `ou_model.core`, except `core.ou.simulate_ou` in the optional simulator tab.
