"""Tests for the Open-Meteo param builder and hourly parser (offline)."""

import pandas as pd

from pvforecast.data import openmeteo


def test_build_params():
    params = openmeteo.build_params(51.2, 10.4, "2023-06-20", "2023-06-21")

    assert params["latitude"] == 51.2
    assert params["longitude"] == 10.4
    assert params["start_date"] == "2023-06-20"
    assert params["end_date"] == "2023-06-21"
    assert params["timezone"] == "UTC"
    assert params["models"] == "era5"
    assert params["hourly"].split(",") == openmeteo.HOURLY_VARS


def test_parse_hourly():
    # Inline fixture mimicking the Open-Meteo JSON: parallel hourly arrays.
    payload = {
        "hourly": {
            "time": [
                "2023-06-20T00:00",
                "2023-06-20T01:00",
                "2023-06-20T02:00",
            ],
            "shortwave_radiation": [0.0, 0.0, 5.0],
            "direct_radiation": [0.0, 0.0, 2.0],
            "diffuse_radiation": [0.0, 0.0, 3.0],
            "temperature_2m": [12.0, 11.5, 11.0],
            "cloud_cover": [10, 20, 30],
            "relative_humidity_2m": [80, 82, 85],
            "wind_speed_10m": [3.0, 2.5, 2.0],
        }
    }

    df = openmeteo.parse_hourly(payload)

    assert list(df.columns) == ["time"] + openmeteo.HOURLY_VARS
    assert len(df) == 3
    # tz-aware UTC; check the property, not the exact dtype string (pandas 3.0 = us).
    assert str(df["time"].dt.tz) == "UTC"
    assert isinstance(df["time"].dtype, pd.DatetimeTZDtype)
