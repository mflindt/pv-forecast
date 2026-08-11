"""Entry point for building the joined PV + weather model input."""

import logging

import pandas as pd

from pvforecast.config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR
from pvforecast.data.join import align_radiation_labels, join_pv_weather, load_weather
from pvforecast.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Join the clean PV series with the weather series into data/processed."""
    pv = pd.read_parquet(INTERIM_DIR / "pv_hourly.parquet")
    weather = load_weather(RAW_DIR / "weather_openmeteo_era5_2015-2026.csv")
    weather = align_radiation_labels(weather)

    df = join_pv_weather(pv, weather)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "pv_weather_hourly.parquet"
    df.to_parquet(out_path)
    logger.info(f"{len(df)} Stunden gespeichert: {out_path}")


if __name__ == "__main__":
    setup_logging("join")
    main()
