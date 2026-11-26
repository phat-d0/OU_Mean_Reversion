"""Pure data-loading and table-building helpers for the dashboard.

Everything here reads only ``results/`` and ``config/ui.yaml``; nothing renders,
downloads or imports ``ou_model.core``. Functions are defensive: missing files,
null values and absent keys return empty frames or ``None`` rather than raising.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SUMMARY_FILE = "summary.json"
SPREAD_FILE = "spread.parquet"
TRADES_FILE = "trades.parquet"
SAMPLE_LABELS = {"in_sample": "IS", "out_of_sample": "OOS"}
EVENT_TYPES = ("entry", "exit", "stop", "time")
POINT_FLAGS = ("roll_gaps", "structural_breaks", "non_positive_prices", "bad_ticks")


# --------------------------------------------------------------------------- config / discovery
def load_ui_config(path: str | Path = "config/ui.yaml") -> dict[str, Any]:
    """Load the dashboard config; an absent file gives an empty dict."""
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh) or {}


def discover_pairs(results_root: str | Path) -> dict[str, list[str]]:
    """Map asset class -> sorted pair names, from directories that contain a summary.json."""
    root = Path(results_root)
    out: dict[str, list[str]] = {}
    if not root.is_dir():
        return out
    for summary in sorted(root.glob(f"*/*/{SUMMARY_FILE}")):
        pair_dir = summary.parent
        out.setdefault(pair_dir.parent.name, []).append(pair_dir.name)
    return out


def pair_dir(results_root: str | Path, asset_class: str, pair: str) -> Path:
    """Directory holding one pair's outputs."""
    return Path(results_root) / asset_class / pair


# --------------------------------------------------------------------------- file loaders
def load_summary(directory: str | Path) -> dict[str, Any]:
    """Read summary.json; missing or unreadable gives ``{}``."""
    p = Path(directory) / SUMMARY_FILE
    try:
        with p.open() as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def load_spread(directory: str | Path) -> pd.DataFrame:
    """Read spread.parquet with a datetime ``date`` column, sorted; missing gives an empty frame."""
    p = Path(directory) / SPREAD_FILE
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    if "event" in df.columns:
        df["event"] = df["event"].fillna("").astype(str)
    return df


def load_trades(directory: str | Path) -> pd.DataFrame:
    """Read trades.parquet; missing gives an empty frame."""
    p = Path(directory) / TRADES_FILE
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    for col in ("entry_date", "exit_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


# --------------------------------------------------------------------------- safe access / formatting
def dig(obj: Any, *keys: str, default: Any = None) -> Any:
    """Nested ``dict`` lookup that returns ``default`` on any missing key or non-dict step."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return default if cur is None else cur


def is_missing(value: Any) -> bool:
    """True for ``None``, NaN and ±inf."""
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    return False


def fmt(value: Any, pattern: str, missing: str) -> str:
    """Format a scalar with ``pattern``; null/NaN/inf/unformattable gives ``missing``."""
    if is_missing(value):
        return missing
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return pattern.format(value)
    except (ValueError, TypeError):
        return str(value)


# --------------------------------------------------------------------------- signal settings / bands
def signal_thresholds(summary: dict[str, Any]) -> dict[str, float | None]:
    """entry/exit/stop z thresholds from ``summary.config.signals`` (``None`` if absent)."""
    sig = dig(summary, "config", "signals", default={})
    return {k: sig.get(f"{k}_z") for k in ("entry", "exit", "stop")}


def entry_bands(spread: pd.DataFrame, entry_z: float | None) -> pd.DataFrame:
    """``date, mean, upper, lower`` with bands ``spread_mean ± entry_z × spread_std``."""
    cols = {"date", "spread_mean", "spread_std"}
    if spread.empty or not cols.issubset(spread.columns):
        return pd.DataFrame(columns=["date", "mean", "upper", "lower"])
    z = float("nan") if entry_z is None else float(entry_z)
    return pd.DataFrame(
        {
            "date": spread["date"],
            "mean": spread["spread_mean"],
            "upper": spread["spread_mean"] + z * spread["spread_std"],
            "lower": spread["spread_mean"] - z * spread["spread_std"],
        }
    )


def oos_start(summary: dict[str, Any]) -> pd.Timestamp | None:
    """First out-of-sample date from ``split.out_of_sample[0]``."""
    oos = dig(summary, "split", "out_of_sample", default=[])
    if isinstance(oos, list) and oos and oos[0]:
        return pd.Timestamp(oos[0])
    return None


def event_points(spread: pd.DataFrame, value_col: str = "zscore") -> dict[str, pd.DataFrame]:
    """Rows of ``spread`` per event type (entry/exit/stop/time), with ``date`` and ``value_col``."""
    if spread.empty or "event" not in spread.columns or value_col not in spread.columns:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for ev in EVENT_TYPES:
        rows = spread.loc[spread["event"] == ev, ["date", value_col]]
        if not rows.empty:
            out[ev] = rows
    return out


# --------------------------------------------------------------------------- flags
def flag_records(summary: dict[str, Any], kind: str) -> pd.DataFrame:
    """Flag records of ``kind`` as a frame with a datetime ``date`` column.

    Accepts records (dicts with ``date``) or bare date strings; anything malformed is skipped.
    """
    raw = dig(summary, "data", "flags", kind, default=[])
    rows: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for rec in raw:
            if isinstance(rec, dict) and rec.get("date"):
                rows.append(dict(rec))
            elif isinstance(rec, str):
                rows.append({"date": rec})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def flag_notes(summary: dict[str, Any]) -> list[str]:
    """``data.flags.notes`` as a list of strings."""
    notes = dig(summary, "data", "flags", "notes", default=[])
    return [str(n) for n in notes] if isinstance(notes, list) else [str(notes)]


def flag_counts(summary: dict[str, Any]) -> dict[str, int]:
    """Number of records per point-flag kind present in the summary."""
    return {k: len(flag_records(summary, k)) for k in POINT_FLAGS}


# --------------------------------------------------------------------------- tables
OU_FIELDS = ("kappa", "mu", "sigma", "half_life", "b", "r2", "n_obs", "mean_reverting", "unit_root_pvalue")


def ou_params_table(summary: dict[str, Any], ui_cfg: dict[str, Any]) -> pd.DataFrame:
    """OU parameters: rows = fields (+ half-life in days), columns = IS OLS, IS MLE, OOS OLS (diagnostic)."""
    missing, nf = format_settings(ui_cfg)
    tdy = ui_cfg.get("trading_days_per_year")
    cols = {
        "IS · OLS": dig(summary, "ou_params", "in_sample", "ols", default={}),
        "IS · MLE": dig(summary, "ou_params", "in_sample", "mle", default={}),
        "OOS · OLS (diagnostic)": dig(summary, "ou_params", "out_of_sample_diagnostic", "ols", default={}),
    }
    data: dict[str, list[str]] = {}
    for name, p in cols.items():
        vals = [fmt(p.get(f), nf["int"] if f == "n_obs" else nf["default"], missing) for f in OU_FIELDS]
        hl = p.get("half_life")
        hl_days = None if is_missing(hl) or tdy is None else hl * tdy
        vals.append(fmt(hl_days, nf["default"], missing))
        data[name] = vals
    index = [*OU_FIELDS[:3], "half_life (years)", *OU_FIELDS[4:], "half_life (days)"]
    return pd.DataFrame(data, index=index)


def tests_table(summary: dict[str, Any], ui_cfg: dict[str, Any]) -> pd.DataFrame:
    """Stationarity/cointegration tests: statistic and p-value (Johansen: rank and trace stat)."""
    missing, nf = format_settings(ui_cfg)
    rows: list[dict[str, str]] = []
    for sample_key, label in (("in_sample", "IS"), ("out_of_sample_diagnostic", "OOS (diagnostic)")):
        tests = dig(summary, "tests", sample_key, default={})
        for name in ("adf", "engle_granger", "johansen"):
            if name not in tests:
                continue
            t = tests.get(name)
            if name == "johansen":
                t = t or {}
                stat = (t.get("trace_stats") or [None])[0]
                rows.append({"sample": label, "test": "johansen (trace r=0)",
                             "statistic": fmt(stat, nf["default"], missing),
                             "p-value / rank": f"rank {fmt(t.get('rank'), nf['int'], missing)}"})
            else:
                t = t or {}
                rows.append({"sample": label, "test": name,
                             "statistic": fmt(t.get("statistic"), nf["default"], missing),
                             "p-value / rank": fmt(t.get("pvalue"), nf["pvalue"], missing)})
    return pd.DataFrame(rows, columns=["sample", "test", "statistic", "p-value / rank"])


def verdict_table(summary: dict[str, Any], ui_cfg: dict[str, Any]) -> pd.DataFrame:
    """Verdict flags plus the in-sample hedge ratio."""
    missing, nf = format_settings(ui_cfg)
    v = dig(summary, "verdict", default={})
    rows = {
        "cointegrated (IS)": fmt(v.get("cointegrated_in_sample"), "{}", missing),
        "mean-reverting (IS)": fmt(v.get("mean_reverting_in_sample"), "{}", missing),
        "hedge ratio (IS)": fmt(summary.get("hedge_ratio_in_sample"), nf["default"], missing),
        "construction": fmt(summary.get("construction"), "{}", missing),
        "transform": fmt(summary.get("transform"), "{}", missing),
        "legs": ", ".join(map(str, summary.get("legs") or [])) or missing,
    }
    return pd.DataFrame({"value": rows})


METRIC_FORMATS = {
    "ann_return": "pct", "ann_vol": "pct", "sharpe": "default", "max_drawdown": "pct",
    "hit_rate": "pct", "n_trades": "int", "avg_holding_days": "default", "half_life_days": "default",
    "holding_to_half_life": "default", "total_return": "pct", "total_gross_return": "pct",
    "total_cost": "pct", "n_obs": "int",
}


def metrics_table(metrics: dict[str, Any], ui_cfg: dict[str, Any]) -> pd.DataFrame:
    """IS vs OOS metrics as formatted strings; rows = metric names."""
    missing, nf = format_settings(ui_cfg)
    data = {}
    for key, label in SAMPLE_LABELS.items():
        m = metrics.get(key) or {}
        data[label] = [fmt(m.get(k), nf[METRIC_FORMATS[k]], missing) for k in METRIC_FORMATS]
    return pd.DataFrame(data, index=list(METRIC_FORMATS))


SENS_METRICS = ("sharpe", "ann_return", "total_return", "total_cost", "n_trades")


def cost_sensitivity_table(summary: dict[str, Any], ui_cfg: dict[str, Any]) -> pd.DataFrame:
    """Rows = cost levels (bps per leg, ascending), columns = (sample, metric)."""
    missing, nf = format_settings(ui_cfg)
    sens = dig(summary, "cost_sensitivity", default={})
    rows = []
    for bps_key in sorted(sens, key=_as_float):
        row: dict[str, str] = {"cost (bps/leg)": bps_key}
        for key, label in SAMPLE_LABELS.items():
            m = (sens.get(bps_key) or {}).get(key) or {}
            for metric in SENS_METRICS:
                row[f"{label} {metric}"] = fmt(m.get(metric), nf[METRIC_FORMATS[metric]], missing)
        rows.append(row)
    return pd.DataFrame(rows)


def cross_asset_frame(results_root: str | Path) -> pd.DataFrame:
    """One numeric row per pair, sorted by OOS Sharpe descending (missing Sharpe last)."""
    rows = []
    for asset_class, pairs in discover_pairs(results_root).items():
        for pair in pairs:
            s = load_summary(pair_dir(results_root, asset_class, pair))
            is_m = dig(s, "metrics", "in_sample", default={})
            oos_m = dig(s, "metrics", "out_of_sample", default={})
            rows.append(
                {
                    "asset_class": asset_class,
                    "pair": pair,
                    "label": f"{asset_class}/{pair}",
                    "oos_sharpe": _num(oos_m.get("sharpe")),
                    "is_sharpe": _num(is_m.get("sharpe")),
                    "half_life_days": _num(is_m.get("half_life_days")),
                    "oos_holding_to_half_life": _num(oos_m.get("holding_to_half_life")),
                    "oos_n_trades": _num(oos_m.get("n_trades")),
                    "cointegrated_is": dig(s, "verdict", "cointegrated_in_sample"),
                    "mean_reverting_is": dig(s, "verdict", "mean_reverting_in_sample"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("oos_sharpe", ascending=False, na_position="last").reset_index(drop=True)


def format_cross_asset(df: pd.DataFrame, ui_cfg: dict[str, Any]) -> pd.DataFrame:
    """String-formatted version of :func:`cross_asset_frame` for display."""
    missing, nf = format_settings(ui_cfg)
    if df.empty:
        return df
    out = pd.DataFrame({"rank": range(1, len(df) + 1), "asset class": df["asset_class"], "pair": df["pair"]})
    out["OOS Sharpe"] = [fmt(v, nf["default"], missing) for v in df["oos_sharpe"]]
    out["IS Sharpe"] = [fmt(v, nf["default"], missing) for v in df["is_sharpe"]]
    out["half-life (days)"] = [fmt(v, nf["default"], missing) for v in df["half_life_days"]]
    out["OOS holding / half-life"] = [fmt(v, nf["default"], missing) for v in df["oos_holding_to_half_life"]]
    out["OOS trades"] = [fmt(v, nf["int"], missing) for v in df["oos_n_trades"]]
    out["cointegrated (IS)"] = [fmt(v, "{}", missing) for v in df["cointegrated_is"]]
    out["mean-reverting (IS)"] = [fmt(v, "{}", missing) for v in df["mean_reverting_is"]]
    return out


# --------------------------------------------------------------------------- simulator helper
def implied_half_life_days(kappa: float, trading_days_per_year: float) -> float | None:
    """``ln 2 / κ`` in trading days (κ per year); ``None`` when κ ≤ 0."""
    if kappa is None or kappa <= 0:
        return None
    return math.log(2.0) / kappa * trading_days_per_year


# --------------------------------------------------------------------------- internals
def format_settings(ui_cfg: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """(missing-value placeholder, number formats) from ui config, with defaults."""
    defaults = {"default": "{:.4f}", "pct": "{:.2%}", "int": "{:,.0f}", "pvalue": "{:.4f}"}
    return ui_cfg.get("missing_value", "—"), {**defaults, **(ui_cfg.get("number_format") or {})}


def _num(v: Any) -> float:
    return float("nan") if is_missing(v) or isinstance(v, bool) else float(v)


def _as_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return math.inf
