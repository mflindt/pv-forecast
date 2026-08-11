"""Tests for the target-variable logic in pvforecast.features."""

import pandas as pd
import pytest

from pvforecast import features

WINDOW = "10D"
SHIFT_H = 48


def _flat_series(days: int = 60) -> pd.Series:
    idx = pd.date_range("2020-01-01 00:00", periods=days * 24, freq="h", tz="UTC")
    return pd.Series(1.0, index=idx, name="pv_mwh")


def test_rolling_capacity_ignores_future_values():
    """Nothing from t - shift_h onwards may reach cap_roll(t)."""
    pv = _flat_series()
    t = pv.index[24 * 40]
    base = features.rolling_capacity(pv, window=WINDOW, shift_h=SHIFT_H)

    leaked = pv.copy()
    leaked.loc[t - pd.Timedelta(hours=SHIFT_H - 1) :] = 1e6
    out = features.rolling_capacity(leaked, window=WINDOW, shift_h=SHIFT_H)

    assert out.loc[t] == base.loc[t]


def test_rolling_capacity_reacts_to_values_inside_the_window():
    """Counterpart to the leakage test: the window is not simply blind."""
    pv = _flat_series()
    t = pv.index[24 * 40]
    base = features.rolling_capacity(pv, window=WINDOW, shift_h=SHIFT_H)

    spiked = pv.copy()
    # The last five hours the window ending at t - shift_h is still allowed to see.
    spiked.loc[
        t - pd.Timedelta(hours=SHIFT_H + 4) : t - pd.Timedelta(hours=SHIFT_H)
    ] = 1e6
    out = features.rolling_capacity(spiked, window=WINDOW, shift_h=SHIFT_H)

    assert out.loc[t] > base.loc[t]


def test_rolling_capacity_shift_is_time_based():
    pv = _flat_series()
    t = pv.index[24 * 40]

    capacity = features.rolling_capacity(pv, window=WINDOW, shift_h=SHIFT_H)
    trailing = pv.rolling(WINDOW).quantile(0.995)

    assert capacity.loc[t] == trailing.loc[t - pd.Timedelta(hours=SHIFT_H)]


def test_rolling_capacity_warmup_is_nan_not_filled():
    pv = _flat_series()

    capacity = features.rolling_capacity(pv, window=WINDOW, shift_h=SHIFT_H)

    # The first shift_h hours cannot have a predecessor -> NaN, never back-filled.
    assert capacity.iloc[:SHIFT_H].isna().all()
    assert capacity.index.equals(pv.index)


def test_rolling_capacity_raises_on_unsorted_index():
    pv = _flat_series(days=2)

    with pytest.raises(ValueError, match="aufsteigend"):
        features.rolling_capacity(pv.iloc[::-1])


def test_rolling_capacity_raises_on_duplicate_index():
    pv = _flat_series(days=2)
    doubled = pd.concat([pv, pv.iloc[[-1]]])

    with pytest.raises(ValueError, match="doppelte"):
        features.rolling_capacity(doubled)


def test_rolling_capacity_raises_on_naive_index():
    pv = _flat_series(days=2)

    with pytest.raises(ValueError, match="zeitzonenbehaftet"):
        features.rolling_capacity(pv.tz_localize(None))


def test_capacity_factor_round_trip():
    pv = _flat_series()
    capacity = features.rolling_capacity(pv, window=WINDOW, shift_h=SHIFT_H)

    cf = features.capacity_factor(pv, capacity)
    restored = features.to_energy(cf, capacity)

    assert cf.name == "cf"
    valid = restored.notna()
    assert valid.any()
    pd.testing.assert_series_equal(
        restored[valid], pv[valid], check_names=False, check_freq=False
    )


def test_capacity_factor_raises_on_non_positive_capacity():
    pv = _flat_series(days=2)
    capacity = pd.Series(0.0, index=pv.index)

    with pytest.raises(ValueError, match="<= 0"):
        features.capacity_factor(pv, capacity)


def test_data_start_leaves_the_warmup_behind():
    # 2015 is warm-up; the first usable model timestamp is the start of 2016.
    assert features.DATA_START == pd.Timestamp("2016-01-01", tz="UTC")
