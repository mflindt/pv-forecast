"""Tests for the Energy-Charts param builder and capacity parser (offline)."""

import pandas as pd
import pytest

from pvforecast.config import RAW_DIR
from pvforecast.data import capacity

RAW_CAPACITY = RAW_DIR / "capacity_energycharts_solar_2002-2026.csv"


def _payload(time, solar_ac, solar_dc) -> dict:
    """Inline fixture mimicking the Energy-Charts JSON: parallel monthly arrays."""
    return {
        "time": time,
        "production_types": [
            {"name": "Biomass", "data": [8.0] * len(time)},
            {"name": "Solar DC", "data": solar_dc},
            {"name": "Solar AC", "data": solar_ac},
        ],
        "last_update": "2026-08-01T00:00:00",
    }


def test_build_params():
    params = capacity.build_params("de", "monthly")

    assert params["country"] == "de"
    assert params["time_step"] == "monthly"


def test_parse_installed_power():
    payload = _payload(
        ["01.2015", "02.2015", "03.2015"], [35.8, 36.0, 36.3], [38.5, 38.8, 39.1]
    )

    df = capacity.parse_installed_power(payload)

    assert list(df.columns) == ["time", "solar_ac_gw", "solar_dc_gw"]
    assert len(df) == 3
    # Months are anchored on the first day, tz-aware UTC.
    assert str(df["time"].dt.tz) == "UTC"
    assert df["time"].iloc[0] == pd.Timestamp("2015-01-01", tz="UTC")
    assert df["solar_ac_gw"].iloc[0] == 35.8
    assert df["solar_dc_gw"].iloc[0] == 38.5


def test_parse_installed_power_cuts_null_tail():
    # The API always reports whole calendar years; future months come back as null.
    payload = _payload(
        ["01.2015", "02.2015", "03.2015"], [35.8, 36.0, None], [38.5, 38.8, None]
    )

    df = capacity.parse_installed_power(payload)

    assert len(df) == 2
    assert df["time"].max() == pd.Timestamp("2015-02-01", tz="UTC")


def test_parse_installed_power_raises_on_interior_gap():
    payload = _payload(
        ["01.2015", "02.2015", "03.2015"], [35.8, None, 36.3], [38.5, 38.8, 39.1]
    )

    with pytest.raises(ValueError, match="NaN"):
        capacity.parse_installed_power(payload)


def test_parse_installed_power_raises_on_missing_production_type():
    payload = _payload(["01.2015"], [35.8], [38.5])
    payload["production_types"] = [payload["production_types"][0]]

    with pytest.raises(ValueError, match="Solar"):
        capacity.parse_installed_power(payload)


def test_load_installed_power(tmp_path):
    payload = _payload(["01.2015", "02.2015"], [35.8, 36.0], [38.5, 38.8])
    csv = tmp_path / "capacity.csv"
    capacity.parse_installed_power(payload).to_csv(csv, index=False)

    out = capacity.load_installed_power(csv)

    assert out.index.name == "time"
    assert str(out.index.tz) == "UTC"
    assert list(out.columns) == ["solar_ac_gw", "solar_dc_gw"]


@pytest.mark.skipif(
    not RAW_CAPACITY.exists(), reason="Kapazitäts-Rohdaten nicht abgerufen"
)
def test_raw_capacity_covers_project_period():
    """The stored series must span 2015-2025 and show the known PV build-out."""
    df = capacity.load_installed_power(RAW_CAPACITY)

    start = pd.Timestamp("2015-01-01", tz="UTC")
    end = pd.Timestamp("2025-12-01", tz="UTC")
    assert start in df.index
    assert end in df.index
    assert df.loc[start:end].notna().all().all()

    # Plausibility, not exact values: Energy-Charts revises past months.
    assert 30 < df.loc[start, "solar_ac_gw"] < 42
    assert 100 < df.loc[end, "solar_ac_gw"] < 115
    # DC always exceeds AC, and the series is monotonically built out.
    assert (df["solar_dc_gw"] >= df["solar_ac_gw"]).all()
    assert df.loc[start:end, "solar_ac_gw"].is_monotonic_increasing
