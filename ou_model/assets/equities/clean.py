"""Cleaning of equity price series: drop bad prints, inner-join on common dates."""
from __future__ import annotations

import pandas as pd


def drop_bad_prices(series: pd.Series, min_price: float) -> tuple[pd.Series, dict]:
    """Drop missing, non-finite and non-positive (or below ``min_price``) values.

    Returns the cleaned series and counts ``{"missing": n, "non_positive": n}`` plus the dates
    of non-positive prints under ``"non_positive_dates"``.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s = s.replace([float("inf"), float("-inf")], float("nan"))
    missing = s.isna()
    bad = s.notna() & (s < min_price)
    counts = {
        "missing": int(missing.sum()),
        "non_positive": int(bad.sum()),
        "non_positive_dates": [d.strftime("%Y-%m-%d") for d in s.index[bad]],
    }
    return s[~missing & ~bad], counts


def align_legs(series: dict[str, pd.Series], legs: list[str], min_price: float) -> tuple[pd.DataFrame, dict]:
    """Clean each leg and inner-join on common dates (no forward fill).

    Returns ``(prices, report)``: ``prices`` has one column per leg in ``legs`` order;
    ``report`` has per-leg counts, ``dates_dropped_by_join`` and ``non_positive_prices``
    (``"TICKER YYYY-MM-DD"`` strings) for the contract flags.
    """
    cleaned, per_leg, non_positive = {}, {}, []
    for leg in legs:
        s, counts = drop_bad_prices(series[leg], min_price)
        cleaned[leg] = s
        per_leg[leg] = {"raw_obs": int(len(series[leg])), "missing": counts["missing"],
                        "non_positive": counts["non_positive"]}
        non_positive += [f"{leg} {d}" for d in counts["non_positive_dates"]]
    prices = pd.concat([cleaned[l] for l in legs], axis=1, join="inner", keys=legs).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.index.name = "date"
    union = pd.Index([]).append([cleaned[l].index for l in legs]).unique()
    report = {"per_leg": per_leg, "dates_dropped_by_join": int(len(union) - len(prices)),
              "non_positive_prices": non_positive}
    return prices, report


def cleaning_notes(report: dict) -> list[str]:
    """Human-readable notes summarising a cleaning ``report``."""
    notes = [f"{leg}: {c['raw_obs']} raw obs, dropped {c['missing']} missing and "
             f"{c['non_positive']} non-positive/below-min prices" for leg, c in report["per_leg"].items()]
    notes.append(f"inner join on common dates dropped {report['dates_dropped_by_join']} dates (no forward fill)")
    return notes
