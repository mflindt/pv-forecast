"""Tests for the PV + weather join in pvforecast.data.join."""

import numpy as np
import pandas as pd
import pytest

from pvforecast.config import PROCESSED_DIR
from pvforecast.data import join
from pvforecast.data.openmeteo import INSTANT_VARS, RADIATION_VARS

MODEL_INPUT = PROCESSED_DIR / "pv_weather_hourly.parquet"


def _pv(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"pv_mwh": range(len(idx))}, index=idx)


def _weather(idx: pd.DatetimeIndex) -> pd.DataFrame:
    w = pd.DataFrame(
        {"shortwave_radiation": range(len(idx)), "temperature_2m": range(len(idx))},
        index=idx,
    )
    w.index.name = "time"
    return w


def _weather_full(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Weather frame carrying every Open-Meteo variable, values = hour position."""
    w = pd.DataFrame(
        {var: range(len(idx)) for var in RADIATION_VARS + INSTANT_VARS}, index=idx
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


def test_align_radiation_labels_shifts_only_radiation():
    idx = pd.date_range("2020-01-01 00:00", periods=4, freq="h", tz="UTC")

    out = join.align_radiation_labels(_weather_full(idx))

    # One hour is consumed at the tail; nothing is filled in.
    assert len(out) == 3
    assert out.index.equals(idx[:-1])
    assert out.notna().all().all()
    # Radiation now carries the value of the following timestamp ...
    for var in RADIATION_VARS:
        assert out[var].tolist() == [1, 2, 3]
    # ... while the instantaneous values keep their own timestamp.
    for var in INSTANT_VARS:
        assert out[var].tolist() == [0, 1, 2]


def test_align_radiation_labels_raises_on_gap():
    idx = pd.DatetimeIndex(
        ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 03:00"]
    ).tz_localize("UTC")

    with pytest.raises(ValueError, match="Lücken"):
        join.align_radiation_labels(_weather_full(idx))


def test_align_radiation_labels_recovers_the_source_convention():
    """The alignment must undo Open-Meteo's end-of-interval labelling."""
    idx = pd.date_range("2020-06-01 00:00", periods=240, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    driver = np.clip(np.sin(np.arange(len(idx)) * np.pi / 24), 0, None)
    driver = driver * rng.uniform(0.3, 1.0, len(idx))

    pv = pd.DataFrame({"pv_mwh": driver}, index=idx)
    weather = _weather_full(idx).astype(float)
    # Open-Meteo would stamp this hour's mean radiation on the following timestamp.
    for var in RADIATION_VARS:
        weather[var] = np.roll(driver, 1) * 800

    aligned = join.align_radiation_labels(weather)
    joined = join.join_pv_weather(pv.iloc[:-1], aligned)

    at_alignment = joined["pv_mwh"].corr(joined["shortwave_radiation"])
    one_early = joined["pv_mwh"].corr(joined["shortwave_radiation"].shift(-1))
    one_late = joined["pv_mwh"].corr(joined["shortwave_radiation"].shift(1))

    assert at_alignment > one_early
    assert at_alignment > one_late
    assert at_alignment > 0.99


@pytest.mark.skipif(
    not MODEL_INPUT.exists(), reason="data/processed ist nicht eingecheckt"
)
def test_radiation_alignment_on_model_input():
    """The stored model input must sit at the correlation optimum, not next to it."""
    df = pd.read_parquet(MODEL_INPUT)
    pv, ghi = df["pv_mwh"], df["shortwave_radiation"]

    at_alignment = pv.corr(ghi)
    one_early = pv.corr(ghi.shift(-1))
    one_late = pv.corr(ghi.shift(1))

    assert at_alignment > one_early
    assert at_alignment > one_late
    assert at_alignment >= 0.92


def test_load_weather(tmp_path):
    idx = pd.date_range("2020-01-01 00:00", periods=3, freq="h", tz="UTC")
    csv = tmp_path / "weather.csv"
    _weather(idx).reset_index().to_csv(csv, index=False)

    out = join.load_weather(csv)

    assert out.index.name == "time"
    assert str(out.index.tz) == "UTC"
    assert list(out.columns) == ["shortwave_radiation", "temperature_2m"]
