"""Tests for the PV + weather join in pvforecast.data.join."""

import pandas as pd
import pytest

from pvforecast.data import join


def _pv(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"pv_mwh": range(len(idx))}, index=idx)


def _weather(idx: pd.DatetimeIndex) -> pd.DataFrame:
    w = pd.DataFrame(
        {"shortwave_radiation": range(len(idx)), "temperature_2m": range(len(idx))},
        index=idx,
    )
    w.index.name = "time"
    return w


def test_join_pv_weather_inner():
    pv_idx = pd.date_range("2020-01-01 00:00", periods=3, freq="h", tz="UTC")
    # Weather starts one hour earlier (extra head room), covers all PV hours.
    w_idx = pd.date_range("2019-12-31 23:00", periods=4, freq="h", tz="UTC")

    out = join.join_pv_weather(_pv(pv_idx), _weather(w_idx))

    assert len(out) == 3
    assert list(out.columns) == ["pv_mwh", "shortwave_radiation", "temperature_2m"]
    assert out.index.name == "time"
    assert out.index.equals(pv_idx)


def test_join_pv_weather_raises_on_missing_weather():
    pv_idx = pd.date_range("2020-01-01 00:00", periods=3, freq="h", tz="UTC")
    # Weather lacks the last PV hour.
    w_idx = pd.date_range("2020-01-01 00:00", periods=2, freq="h", tz="UTC")

    with pytest.raises(ValueError, match="Wetter fehlt"):
        join.join_pv_weather(_pv(pv_idx), _weather(w_idx))


def test_load_weather(tmp_path):
    idx = pd.date_range("2020-01-01 00:00", periods=3, freq="h", tz="UTC")
    csv = tmp_path / "weather.csv"
    _weather(idx).reset_index().to_csv(csv, index=False)

    out = join.load_weather(csv)

    assert out.index.name == "time"
    assert str(out.index.tz) == "UTC"
    assert list(out.columns) == ["shortwave_radiation", "temperature_2m"]
