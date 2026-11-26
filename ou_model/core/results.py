"""Writing and validating ``results/<asset_class>/<pair>/`` (CONTRACTS.md section 2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRIC_KEYS
from .utils import json_safe

SPREAD_COLUMNS = ["date", "spread", "spread_mean", "spread_std", "zscore", "hedge_ratio",
                  "target", "position", "event", "pnl", "gross_pnl", "equity", "sample"]
TRADE_COLUMNS = ["trade_id", "entry_date", "exit_date", "direction", "entry_z", "exit_z",
                 "exit_reason", "holding_days", "gross_pnl", "cost", "pnl", "sample"]
SUMMARY_KEYS = ["asset_class", "pair", "legs", "construction", "transform", "data", "split",
                "hedge_ratio_in_sample", "ou_params", "tests", "verdict", "metrics",
                "cost_sensitivity", "config"]
EVENT_VALUES = {"", "entry", "exit", "stop", "time"}
EXIT_REASONS = {"exit", "stop", "time", "end_of_data"}
SAMPLES = {"IS", "OOS"}
FLOAT_SPREAD = ["spread", "spread_mean", "spread_std", "zscore", "hedge_ratio", "pnl",
                "gross_pnl", "equity"]
EQUITY_TOL = 1e-9


def _spread_frame(spread_df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a spread frame to the section 2 schema and column order."""
    df = spread_df.copy()
    if "date" not in df.columns:
        df = df.rename_axis("date").reset_index()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    for col in FLOAT_SPREAD:
        df[col] = df[col].astype(float)
    for col in ("target", "position"):
        df[col] = df[col].astype(np.int64)
    for col in ("event", "sample"):
        df[col] = df[col].fillna("").astype(str).astype(object)
    return df[SPREAD_COLUMNS].reset_index(drop=True)


def _trade_frame(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a trades frame to the section 2 schema (empty frames keep typed columns)."""
    df = trades_df.copy()
    df = df.reindex(columns=TRADE_COLUMNS)
    for col in ("entry_date", "exit_date"):
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    for col in ("trade_id", "direction", "holding_days"):
        df[col] = df[col].astype(np.int64)
    for col in ("entry_z", "exit_z", "gross_pnl", "cost", "pnl"):
        df[col] = df[col].astype(float)
    for col in ("exit_reason", "sample"):
        df[col] = df[col].astype(str).astype(object)
    return df.reset_index(drop=True)


def write_pair_results(
    out_dir: str | Path,
    spread_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    summary: dict,
) -> Path:
    """Write ``spread.parquet``, ``trades.parquet`` and ``summary.json`` (strict JSON) to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _spread_frame(spread_df).to_parquet(out / "spread.parquet", index=False)
    _trade_frame(trades_df).to_parquet(out / "trades.parquet", index=False)
    text = json.dumps(json_safe(summary), indent=2, allow_nan=False)
    (out / "summary.json").write_text(text + "\n", encoding="utf-8")
    return out


def _reject_constant(token: str) -> float:
    """``json.loads`` hook: NaN/Infinity are not allowed in summary.json."""
    raise ValueError(f"non-standard JSON constant {token}")


def _check_spread(df: pd.DataFrame, lag: int | None) -> list[str]:
    """Schema and consistency checks for spread.parquet."""
    errs = [f"spread.parquet: missing column '{c}'" for c in SPREAD_COLUMNS if c not in df.columns]
    if errs:
        return errs
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        errs.append("spread.parquet: 'date' is not datetime64")
    elif not (df["date"].is_monotonic_increasing and df["date"].is_unique):
        errs.append("spread.parquet: dates not strictly increasing")
    for col in FLOAT_SPREAD:
        if not pd.api.types.is_float_dtype(df[col]):
            errs.append(f"spread.parquet: '{col}' is not float")
    for col in ("target", "position"):
        if not pd.api.types.is_integer_dtype(df[col]):
            errs.append(f"spread.parquet: '{col}' is not int")
        elif not df[col].isin([-1, 0, 1]).all():
            errs.append(f"spread.parquet: '{col}' outside {{-1, 0, 1}}")
    if not df["event"].isin(EVENT_VALUES).all():
        errs.append("spread.parquet: unexpected 'event' values")
    if not df["sample"].isin(SAMPLES).all():
        errs.append("spread.parquet: 'sample' must be IS/OOS")
    for col in ("pnl", "gross_pnl", "equity"):
        if df[col].isna().any():
            errs.append(f"spread.parquet: NaN in '{col}'")
    if len(df) and not errs:
        expected = 1.0 + df["pnl"].cumsum()
        if (expected - df["equity"]).abs().max() > EQUITY_TOL:
            errs.append("spread.parquet: equity != 1 + cumsum(pnl)")
        if (df["sample"] == "IS").any() and (df["sample"] == "OOS").any():
            if df.loc[df["sample"] == "IS", "date"].max() >= df.loc[df["sample"] == "OOS", "date"].min():
                errs.append("spread.parquet: IS dates overlap OOS dates")
        if lag is not None and lag >= 0:
            shifted = df["target"].shift(lag).fillna(0).astype(np.int64)
            if not (shifted == df["position"]).all():
                errs.append(f"spread.parquet: position != target shifted by execution_lag={lag}")
    return errs


def _check_trades(df: pd.DataFrame) -> list[str]:
    """Schema checks for trades.parquet."""
    errs = [f"trades.parquet: missing column '{c}'" for c in TRADE_COLUMNS if c not in df.columns]
    if errs or df.empty:
        return errs
    if not df["direction"].isin([-1, 1]).all():
        errs.append("trades.parquet: direction must be +1/-1")
    if not df["exit_reason"].isin(EXIT_REASONS).all():
        errs.append("trades.parquet: unexpected exit_reason")
    if not df["sample"].isin(SAMPLES).all():
        errs.append("trades.parquet: 'sample' must be IS/OOS")
    if (pd.to_datetime(df["exit_date"]) < pd.to_datetime(df["entry_date"])).any():
        errs.append("trades.parquet: exit_date before entry_date")
    if ((df["gross_pnl"] - df["cost"] - df["pnl"]).abs() > EQUITY_TOL).any():
        errs.append("trades.parquet: pnl != gross_pnl - cost")
    return errs


def _check_metrics(block: object, where: str) -> list[str]:
    """Every metrics dict must carry all METRIC_KEYS."""
    if not isinstance(block, dict):
        return [f"summary.json: {where} is not an object"]
    return [f"summary.json: {where} missing '{k}'" for k in METRIC_KEYS if k not in block]


def _check_summary(summary: dict) -> list[str]:
    """Key presence checks for summary.json."""
    errs = [f"summary.json: missing key '{k}'" for k in SUMMARY_KEYS if k not in summary]
    if errs:
        return errs
    for key in ("in_sample", "out_of_sample"):
        errs += _check_metrics(summary["metrics"].get(key), f"metrics.{key}")
    for cost, block in summary["cost_sensitivity"].items():
        for key in ("in_sample", "out_of_sample"):
            errs += _check_metrics((block or {}).get(key), f"cost_sensitivity.{cost}.{key}")
    for key in ("cointegrated_in_sample", "mean_reverting_in_sample"):
        if not isinstance(summary["verdict"].get(key), bool):
            errs.append(f"summary.json: verdict.{key} must be bool")
    if "ols" not in summary["ou_params"].get("in_sample", {}):
        errs.append("summary.json: ou_params.in_sample.ols missing")
    for key in ("adf", "engle_granger", "johansen"):
        if key not in summary["tests"].get("in_sample", {}):
            errs.append(f"summary.json: tests.in_sample.{key} missing")
    flags = summary["data"].get("flags", {})
    for key in ("roll_gaps", "structural_breaks", "non_positive_prices", "notes"):
        if not isinstance(flags.get(key), list):
            errs.append(f"summary.json: data.flags.{key} must be a list")
    return errs


def validate_pair_results(path: str | Path) -> list[str]:
    """Validate a pair result directory against CONTRACTS.md section 2. Empty list = valid."""
    path = Path(path)
    errs: list[str] = []
    for name in ("spread.parquet", "trades.parquet", "summary.json"):
        if not (path / name).is_file():
            errs.append(f"missing file {name}")
    if errs:
        return errs
    try:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"),
                             parse_constant=_reject_constant)
    except ValueError as err:
        return [f"summary.json: invalid JSON ({err})"]
    errs += _check_summary(summary)
    lag = None
    try:
        lag = int(summary["config"]["backtest"]["execution_lag"])
    except (KeyError, TypeError, ValueError):
        errs.append("summary.json: config.backtest.execution_lag missing")
    errs += _check_spread(pd.read_parquet(path / "spread.parquet"), lag)
    errs += _check_trades(pd.read_parquet(path / "trades.parquet"))
    return errs
