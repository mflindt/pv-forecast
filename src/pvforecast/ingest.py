"""Download raw data into data/raw."""

import logging
import time

import pandas as pd
import requests

from pvforecast.config import (
    CAPACITY_RAW,
    FORECAST_RAW,
    HOURLY_VARS,
    PV_RAW,
    PV_RAW_HOURLY,
    RAW_DIR,
    SITES,
    WEATHER_RAW,
    setup_logging,
)

logger = logging.getLogger(__name__)

SMARD_URL = "https://www.smard.de/app/chart_data"
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
ENERGYCHARTS_URL = "https://api.energy-charts.info/installed_power"

# filter_id 4068 = photovoltaic generation, 125 = TSO day-ahead PV forecast.
PV_FILTER_ID = "4068"
FORECAST_FILTER_ID = "125"

PV_COLUMN = "pv_mwh"
FORECAST_COLUMN = "pv_fcst_mwh"

WEATHER_START = "2014-12-30"
WEATHER_END = "2026-01-01"

# Grid cells may sit up to half a cell away from the requested point.
MAX_SNAP_DEG = 0.3

# Production types as named by Energy-Charts, which reports in GW.
CAPACITY_VARS = {"Solar AC": "solar_ac_gw", "Solar DC": "solar_dc_gw"}


def request_json(url: str, params: dict | None = None, timeout: int = 60):
    """GET a URL and return the parsed JSON body."""
    logger.debug(f"GET {url} params={params}")
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_smard(filter_id: str, resolution: str, column: str) -> pd.DataFrame:
    """Fetch all weekly chunks of one SMARD series into a single DataFrame."""
    index_url = f"{SMARD_URL}/{filter_id}/DE/index_{resolution}.json"
    timestamps = request_json(index_url, timeout=30)["timestamps"]

    series: list[list] = []
    failed = 0
    for i, timestamp in enumerate(timestamps):
        url = f"{SMARD_URL}/{filter_id}/DE/{filter_id}_DE_{resolution}_{timestamp}.json"
        try:
            series.extend(request_json(url, timeout=30)["series"])
        except requests.RequestException as error:
            logger.warning(f"Chunk {timestamp} fehlgeschlagen: {error}")
            failed += 1
        else:
            logger.debug(f"Chunk {i + 1}/{len(timestamps)} geladen")
        time.sleep(0.2)

    if failed:
        logger.warning(f"{failed} von {len(timestamps)} Chunks fehlten")
    logger.info(f"{len(series)} Datenpunkte aus {len(timestamps)} Chunks geladen")
    return pd.DataFrame(series, columns=["timestamp_ms", column])


def parse_weather(payload: dict | list, sites: tuple = SITES) -> pd.DataFrame:
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
            f"{name}: Zelle {block['latitude']}/{block['longitude']} | "
            f"Elevation {block['elevation']} m | {len(df)} Stunden"
        )

    return pd.concat(frames, ignore_index=True)


def fetch_weather(sites: tuple = SITES) -> pd.DataFrame:
    """Fetch the hourly ERA5 weather series for all sites."""
    params = {
        "latitude": ",".join(f"{lat}" for _, lat, _, _ in sites),
        "longitude": ",".join(f"{lon}" for _, _, lon, _ in sites),
        "start_date": WEATHER_START,
        "end_date": WEATHER_END,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "models": "era5",
    }
    df = parse_weather(request_json(OPENMETEO_URL, params, timeout=120), sites)
    logger.info(f"{len(df)} Wetterzeilen für {len(sites)} Standorte geladen")
    return df


def parse_capacity(payload: dict) -> pd.DataFrame:
    """Turn the production-type arrays into a tidy frame with a UTC time column."""
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


def fetch_capacity() -> pd.DataFrame:
    """Fetch the monthly installed PV capacity for Germany."""
    payload = request_json(
        ENERGYCHARTS_URL, {"country": "de", "time_step": "monthly"}, timeout=60
    )
    df = parse_capacity(payload)
    logger.info(
        f"{len(df)} Monatswerte geladen: "
        f"{df['time'].min():%Y-%m} bis {df['time'].max():%Y-%m}"
    )
    return df


def main() -> None:
    """Download every raw series into data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for resolution, path in (("quarterhour", PV_RAW), ("hour", PV_RAW_HOURLY)):
        # The hourly series is fetched for validation against our own aggregation.
        fetch_smard(PV_FILTER_ID, resolution, PV_COLUMN).to_csv(path, index=False)
        logger.info(f"Gespeichert: {path.name}")

    fetch_smard(FORECAST_FILTER_ID, "quarterhour", FORECAST_COLUMN).to_csv(
        FORECAST_RAW, index=False
    )
    logger.info(f"Gespeichert: {FORECAST_RAW.name}")

    fetch_weather().to_csv(WEATHER_RAW, index=False)
    logger.info(f"Gespeichert: {WEATHER_RAW.name}")

    fetch_capacity().to_csv(CAPACITY_RAW, index=False)
    logger.info(f"Gespeichert: {CAPACITY_RAW.name}")


if __name__ == "__main__":
    setup_logging()
    main()
