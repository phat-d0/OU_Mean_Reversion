"""Z-score causality and the signal state machine."""

import numpy as np
import pandas as pd

from ou_model.core.signals import generate_signals, rolling_zscore

ENTRY, EXIT, STOP = 2.0, 0.5, 4.0


def _sig(z, **kw):
    return generate_signals(pd.Series(z, dtype=float), ENTRY, EXIT, STOP, **kw)


def test_rolling_zscore_causal():
    s = pd.Series(np.random.default_rng(0).standard_normal(300).cumsum())
    z = rolling_zscore(s, 30, 20)
    assert z["zscore"].iloc[:19].isna().all()
    w = s.iloc[100 - 29:101]
    assert np.isclose(z["zscore"].iloc[100], (s.iloc[100] - w.mean()) / w.std(), rtol=1e-9)
    s2 = s.copy()
    s2.iloc[150:] += 100.0
    pd.testing.assert_frame_equal(z.iloc[:150], rolling_zscore(s2, 30, 20).iloc[:150])


def test_entry_exit_long_and_short():
    out = _sig([0, -2.5, -1.0, -0.2, 0, 2.5, 1.0, 0.4, 0])
    assert out["target"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0]
    assert out["event"].tolist() == ["", "entry", "", "exit", "", "entry", "", "exit", ""]


def test_overshoot_exits():
    out = _sig([0, -2.5, 1.5])
    assert out["target"].tolist() == [0, 1, 0] and out["event"].iloc[2] == "exit"


def test_stop_and_reentry_lockout():
    z = [0, -2.5, -4.5, -3.0, -2.5, -0.3, -2.5]
    out = _sig(z, reentry_requires_exit=True)
    assert out["target"].tolist() == [0, 1, 0, 0, 0, 0, 1]
    assert out["event"].tolist() == ["", "entry", "stop", "", "", "", "entry"]
    free = _sig(z, reentry_requires_exit=False)
    assert free["target"].tolist() == [0, 1, 0, 1, 1, 0, 1]


def test_no_entry_beyond_stop_and_allow_mask():
    assert _sig([0, -5.0])["target"].tolist() == [0, 0]
    allow = pd.Series([True, False, True])
    out = _sig([0, -2.5, -2.5], allow_entry=allow)
    assert out["target"].tolist() == [0, 0, 1]


def test_allow_mask_does_not_force_exit_and_nan_holds():
    out = _sig([-2.5, np.nan, -1.0, -0.1], allow_entry=pd.Series([True, False, False, False]))
    assert out["target"].tolist() == [1, 1, 1, 0]


def test_time_stop():
    out = _sig([-2.5, -2.5, -2.5, -2.5, -2.5, -0.1, -2.5], max_holding=pd.Series([2.0] * 7))
    assert out["target"].tolist() == [1, 1, 0, 0, 0, 0, 1]
    assert out["event"].iloc[2] == "time"
