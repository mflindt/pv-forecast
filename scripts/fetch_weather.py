"""Entry point for the Open-Meteo weather loader."""

import logging

from pvforecast.config import RAW_DIR
from pvforecast.data.openmeteo import SITES, fetch_weather
from pvforecast.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# 2015-2025 coverage
START_DATE = "2014-12-30"
END_DATE = "2026-01-01"


def main():
    """Fetch the raw ERA5 hourly weather series for all sites into data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_weather(SITES, START_DATE, END_DATE)

    out_path = RAW_DIR / "weather_openmeteo_era5_2015-2026.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"{len(df)} Zeilen gespeichert: {out_path}")


if __name__ == "__main__":
    setup_logging("weather")
    main()
