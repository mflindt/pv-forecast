"""
Core logic for the SMARD API data loader.

See: https://github.com/bundesAPI/smard-api
"""

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.smard.de/app/chart_data"
REQUEST_DELAY_S = 0.2


def request_data(url: str) -> dict:
    """Request data from URL and return parsed JSON."""
    logger.debug(f"GET {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def build_index_url(filter_id: str, region: str, resolution: str) -> str:
    """URL for the index of available weekly timestamps."""
    return f"{BASE_URL}/{filter_id}/{region}/index_{resolution}.json"


def build_series_url(
    filter_id: str, region: str, resolution: str, timestamp: str
) -> str:
    """URL for one weekly data chunk starting at 'timestamp'."""
    return (
        f"{BASE_URL}/{filter_id}/{region}"
        f"/{filter_id}_{region}_{resolution}_{timestamp}.json"
    )


def fetch_series(filter_id: str, region: str, resolution: str) -> pd.DataFrame:
    """Main function that constructs the URL, iterates through all timestamps, and outputs the resulting list as a dataframe"""
    timestamps = request_data(build_index_url(filter_id, region, resolution))[
        "timestamps"
    ]

    series: list[list] = []
    failed: list[int] = []

    for i, timestamp in enumerate(timestamps):
        try:
            chunk = request_data(
                build_series_url(filter_id, region, resolution, timestamp)
            )
            series.extend(chunk["series"])
            logger.debug(f"Chunk {i + 1}/{len(timestamps)} geladen")
        except requests.RequestException as e:
            logger.warning(f"Chunk {timestamp} fehlgeschlagen: {e}")
            failed.append(timestamp)
        time.sleep(REQUEST_DELAY_S)

    if failed:
        logger.warning(f"{len(failed)} von {len(timestamps)} Chunks fehlten")

    logger.info(f"{len(series)} Datenpunkte aus {len(timestamps)} Chunks geladen")
    return pd.DataFrame(series, columns=["timestamp_ms", "pv_mw"])
