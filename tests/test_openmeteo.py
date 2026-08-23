"""Tests for the Open-Meteo param builder and hourly parser (offline)."""

import pandas as pd
import pytest

from pvforecast.data import openmeteo

SITES = (("a", 51.2, 10.4, 287.0), ("b", 48.6, 9.0, 570.0))


def _block(latitude: float, longitude: float) -> dict:
    """Inline fixture mimicking one Open-Meteo location block."""
    return {
        "latitude": latitude,
        "longitude": longitude,
        "elevation": 287.0,
        "hourly": {
            "time": ["2023-06-20T00:00", "2023-06-20T01:00", "2023-06-20T02:00"],
            "shortwave_radiation": [0.0, 0.0, 5.0],
            "direct_radiation": [0.0, 0.0, 2.0],
            "diffuse_radiation": [0.0, 0.0, 3.0],
            "temperature_2m": [12.0, 11.5, 11.0],
            "cloud_cover": [10, 20, 30],
            "relative_humidity_2m": [80, 82, 85],
            "wind_speed_10m": [3.0, 2.5, 2.0],
        },
    }


def test_build_params_lists_every_site():
    params = openmeteo.build_params(SITES, "2023-06-20", "2023-06-21")

    assert params["latitude"] == "51.2,48.6"
    assert params["longitude"] == "10.4,9.0"
    assert params["start_date"] == "2023-06-20"
    assert params["end_date"] == "2023-06-21"
    assert params["timezone"] == "UTC"
    assert params["models"] == "era5"
    assert params["hourly"].split(",") == openmeteo.HOURLY_VARS


def test_parse_hourly_labels_each_site():
    payload = [_block(51.25, 10.5), _block(48.5, 9.0)]

    df = openmeteo.parse_hourly(payload, SITES)

    assert list(df.columns) == ["time", "site"] + openmeteo.HOURLY_VARS
    assert len(df) == 6
    assert df["site"].tolist() == ["a"] * 3 + ["b"] * 3
    # tz-aware UTC; check the property, not the exact dtype string (pandas 3.0 = us).
    assert str(df["time"].dt.tz) == "UTC"
    assert isinstance(df["time"].dtype, pd.DatetimeTZDtype)


def test_parse_hourly_raises_on_a_block_count_mismatch():
    with pytest.raises(ValueError, match="Antwortblöcke"):
        openmeteo.parse_hourly([_block(51.25, 10.5)], SITES)


def test_parse_hourly_raises_when_a_cell_is_far_from_the_request():
    """Silently snapping to a distant cell would move the site unnoticed."""
    payload = [_block(51.25, 10.5), _block(45.0, 9.0)]

    with pytest.raises(ValueError, match="vom Punkt entfernt"):
        openmeteo.parse_hourly(payload, SITES)


def test_sites_span_the_country():
    latitudes = [lat for _, lat, _, _ in openmeteo.SITES]

    assert len(openmeteo.SITES) >= 5
    assert max(latitudes) - min(latitudes) > 3.0
