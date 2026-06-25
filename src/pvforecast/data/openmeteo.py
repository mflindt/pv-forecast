"""
Core logic for the Open-Meteo historical weather loader.

See: https://open-meteo.com/en/docs/historical-weather-api
"""

import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Open-Meteo Historical Weather API (ERA5 reanalysis archive)
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT_S = 60

HOURLY_VARS = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "temperature_2m",
    "cloud_cover",
    "relative_humidity_2m",
    "wind_speed_10m",
]


def request_data(url: str, params: dict) -> dict:
    """Request data from the Open-Meteo archive and return parsed JSON."""
    logger.debug(f"GET {url} params={params}")
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def build_params(
    latitude: float, longitude: float, start_date: str, end_date: str
) -> dict:
    """Build the archive query: UTC timestamps, ERA5 model, the 7 hourly vars."""
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "models": "era5",
    }


def parse_hourly(payload: dict) -> pd.DataFrame:
    """Turn the parallel hourly arrays into a tidy DataFrame with a UTC time column."""
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def fetch_weather(
    latitude: float, longitude: float, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch the hourly ERA5 weather series for one point as a DataFrame."""
    params = build_params(latitude, longitude, start_date, end_date)
    payload = request_data(BASE_URL, params)
    df = parse_hourly(payload)

    logger.info(
        f"Genutzte Gitterzelle: {payload['latitude']}/{payload['longitude']} "
        f"| Elevation: {payload['elevation']} m"
    )
    logger.info(f"{len(df)} Stundenwerte geladen")
    return df
