"""Tests for the SMARD URL builders."""

from pvforecast.data import smard


def test_build_index_url():
    url = smard.build_index_url("4068", "DE", "hour")

    assert url == "https://www.smard.de/app/chart_data/4068/DE/index_hour.json"


def test_build_series_url():
    url = smard.build_series_url("4068", "DE", "hour", "1730000000000")

    assert url == (
        "https://www.smard.de/app/chart_data/4068/DE/4068_DE_hour_1730000000000.json"
    )
