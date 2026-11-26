"""Small shared helpers for the core engine (JSON safety)."""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_safe(obj: Any) -> Any:
    """Recursively convert ``obj`` into strict-JSON-serialisable Python objects.

    numpy scalars/arrays become Python scalars/lists, ``nan``/``inf`` become ``None``,
    timestamps become ISO date strings, tuples/sets become lists and dict keys become strings.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date, np.datetime64)):
        ts = pd.Timestamp(obj)
        return None if pd.isna(ts) else ts.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [json_safe(v) for v in obj.tolist()]
    if hasattr(obj, "to_dict"):
        return json_safe(obj.to_dict())
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)
