"""
Core logic for the Fraunhofer ISE Energy-Charts installed-capacity loader.

The SMARD API exposes realised generation but returns 404 for the installed PV
capacity (filter 3795, all resolutions), so the nameplate series comes from
Energy-Charts instead.

See: https://api.energy-charts.info/
"""

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.energy-charts.info/installed_power"
REQUEST_TIMEOUT_S = 60

# Production types as named by the API, mapped to our column convention.
# Energy-Charts reports installed power in GW.
CAPACITY_VARS = {
    "Solar AC": "solar_ac_gw",
    "Solar DC": "solar_dc_gw",
}


def request_data(url: str, params: dict) -> dict:
    """Request data from the Energy-Charts API and return parsed JSON."""
    logger.debug(f"GET {url} params={params}")
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def build_params(country: str, time_step: str) -> dict:
    """Build the installed-power query for one country and resolution."""
    return {
        "country": country,
        "time_step": time_step,
    }


def parse_installed_power(payload: dict) -> pd.DataFrame:
    """Turn the production-type arrays into a tidy DataFrame with a UTC time column."""
    series = {p["name"]: p["data"] for p in payload["production_types"]}
    missing = [name for name in CAPACITY_VARS if name not in series]
    if missing:
        raise ValueError(f"Energy-Charts liefert nicht: {missing}")

    df = pd.DataFrame({col: series[name] for name, col in CAPACITY_VARS.items()})
    df.insert(0, "time", pd.to_datetime(payload["time"], format="%m.%Y", utc=True))

    complete = df.dropna()
    if complete.empty:
        raise ValueError("Energy-Charts lieferte keine gültigen Kapazitätswerte")
    df = df.loc[: complete.index.max()]

    holes = int(df.isna().sum().sum())
    if holes:
        raise ValueError(f"{holes} NaN-Werte innerhalb der Kapazitätsreihe")

    return df


def fetch_installed_power(country: str, time_step: str) -> pd.DataFrame:
    """Fetch the installed PV capacity series for one country as a DataFrame."""
    params = build_params(country, time_step)
    payload = request_data(BASE_URL, params)
    df = parse_installed_power(payload)

    # last_update arrives as epoch seconds.
    last_update = pd.to_datetime(payload["last_update"], unit="s", utc=True)
    logger.info(f"Stand der Energy-Charts-Daten: {last_update:%Y-%m-%d}")
    logger.info(
        f"{len(df)} Monatswerte geladen: "
        f"{df['time'].min():%Y-%m} bis {df['time'].max():%Y-%m}"
    )
    return df


def load_installed_power(path: Path) -> pd.DataFrame:
    """Read the raw capacity CSV and build a UTC DatetimeIndex named 'time'."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    logger.info(f"{len(df)} Kapazitäts-Monatswerte geladen: {path.name}")
    return df
