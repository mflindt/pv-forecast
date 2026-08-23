"""Tests for the TSO day-ahead forecast loader in pvforecast.data.smard_forecast."""

import pandas as pd
import pytest

from pvforecast.data import smard_forecast


def _forecast(hours: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2024-06-01", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({smard_forecast.FORECAST_COLUMN: 1.0}, index=idx)


def test_day_ahead_filter_id_is_the_pv_series():
    # '3791' is wind offshore and '5097' is wind plus PV -- both look plausible
    # and are wrong. This constant is the one verified against SMARD's registry.
    assert smard_forecast.DAY_AHEAD_FILTER_ID == "125"


def test_align_to_target_reindexes_onto_the_target_hours():
    forecast = _forecast()
    target = forecast.index[:24]

    aligned = smard_forecast.align_to_target(forecast, target)

    assert aligned.index.equals(target)
    assert aligned.name == smard_forecast.FORECAST_COLUMN


def test_align_to_target_raises_on_a_missing_hour():
    forecast = _forecast()
    target = pd.date_range("2024-06-01", periods=72, freq="h", tz="UTC")

    with pytest.raises(ValueError, match="ÜNB-Prognose fehlt"):
        smard_forecast.align_to_target(forecast, target)


def _raw_csv(tmp_path, periods: int):
    """Raw quarter-hour CSV in the shape the SMARD loader writes it."""
    idx = pd.date_range("2024-06-01", periods=periods, freq="15min", tz="UTC")
    raw = pd.DataFrame(
        {
            "timestamp_ms": idx.tz_convert(None)
            .astype("datetime64[ms]")
            .astype("int64"),
            smard_forecast.FORECAST_COLUMN: 1.0,
        }
    )
    path = tmp_path / "forecast.csv"
    raw.to_csv(path, index=False)
    return path, idx


def test_build_hourly_forecast_uses_the_target_aggregation(tmp_path):
    """The forecast has to be aggregated exactly like the realised series."""
    path, idx = _raw_csv(tmp_path, periods=8)

    hourly = smard_forecast.build_hourly_forecast(path, idx[0], idx[-1])

    assert len(hourly) == 2
    # Four quarter-hours of 1.0 sum to 4.0 per hour.
    assert (hourly[smard_forecast.FORECAST_COLUMN] == 4.0).all()


def test_build_hourly_forecast_rejects_an_incomplete_hour(tmp_path):
    path, idx = _raw_csv(tmp_path, periods=6)

    with pytest.raises(ValueError, match="NaN-Werte"):
        smard_forecast.build_hourly_forecast(path, idx[0], idx[-1])
