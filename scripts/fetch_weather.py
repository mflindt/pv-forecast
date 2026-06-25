"""Entry point for the Open-Meteo weather loader."""

import logging

from pvforecast.config import RAW_DIR
from pvforecast.data.openmeteo import fetch_weather
from pvforecast.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# Single representative point: geographic centre of Germany
LATITUDE = 51.2
LONGITUDE = 10.4

# 2015–2025 coverage; UTC edges (local midnight = 23:00 UTC the day before)
START_DATE = "2014-12-30"
END_DATE = "2025-12-31"


def main():
    """Fetch the raw ERA5 hourly weather series from Open-Meteo into data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_weather(LATITUDE, LONGITUDE, START_DATE, END_DATE)
    out_path = RAW_DIR / "weather_openmeteo_era5_2015-2026.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"{len(df)} Zeilen gespeichert: {out_path}")


if __name__ == "__main__":
    setup_logging("weather")
    main()
