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
REQUEST_TIMEOUT_S = 120

# Five ERA5 cells spanning Germany. Their mean tracks the national feed-in far better
# than any single cell (R^2 0.94 against 0.84); see docs/arbeitsplan.md.
# Name, latitude, longitude, grid-cell elevation in m as reported by the API.
SITES = (
    ("nord", 53.0, 9.5, 46.0),
    ("ost", 52.3, 13.0, 79.0),
    ("west", 51.0, 7.5, 186.0),
    ("suedwest", 48.6, 9.0, 570.0),
    ("suedost", 48.8, 11.8, 363.0),
)

# Grid cells may sit up to half a cell away from the requested point.
MAX_SNAP_DEG = 0.3

RADIATION_VARS = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
]

# Instantaneous values: these refer to the timestamp itself and stay untouched.
INSTANT_VARS = [
    "temperature_2m",
    "cloud_cover",
    "relative_humidity_2m",
    "wind_speed_10m",
]

HOURLY_VARS = RADIATION_VARS + INSTANT_VARS


def request_data(url: str, params: dict) -> dict | list:
    """Request data from the Open-Meteo archive and return parsed JSON."""
    logger.debug(f"GET {url} params={params}")
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def build_params(sites: tuple, start_date: str, end_date: str) -> dict:
    """Build the archive query: UTC timestamps, ERA5 model, the 7 hourly vars."""
    return {
        "latitude": ",".join(f"{lat}" for _, lat, _, _ in sites),
        "longitude": ",".join(f"{lon}" for _, _, lon, _ in sites),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "models": "era5",
    }


def parse_hourly(payload: dict | list, sites: tuple) -> pd.DataFrame:
    """Turn the per-site hourly arrays into one long DataFrame with a site column."""
    # A single-location request answers with a dict, several with a list.
    blocks = payload if isinstance(payload, list) else [payload]
    if len(blocks) != len(sites):
        raise ValueError(f"{len(blocks)} Antwortblöcke für {len(sites)} Standorte")

    frames = []
    for block, (name, lat, lon, _) in zip(blocks, sites, strict=True):
        snap = max(abs(block["latitude"] - lat), abs(block["longitude"] - lon))
        if snap > MAX_SNAP_DEG:
            cell = f"{block['latitude']}/{block['longitude']}"
            raise ValueError(
                f"Standort {name}: Zelle {cell} liegt {snap:.2f}° vom Punkt entfernt"
            )

        df = pd.DataFrame(block["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.insert(1, "site", name)
        frames.append(df)

        logger.info(
            f"{name}: Gitterzelle {block['latitude']}/{block['longitude']} "
            f"| Elevation {block['elevation']} m | {len(df)} Stunden"
        )

    return pd.concat(frames, ignore_index=True)


def fetch_weather(sites: tuple, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch the hourly ERA5 weather series for all sites as one long DataFrame."""
    payload = request_data(BASE_URL, build_params(sites, start_date, end_date))
    df = parse_hourly(payload, sites)

    logger.info(f"{len(df)} Zeilen für {len(sites)} Standorte geladen")
    return df
