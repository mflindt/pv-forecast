"""Entry point for the SMARD TSO day-ahead PV forecast loader."""

import logging

from pvforecast.config import RAW_DIR
from pvforecast.data.smard import fetch_series
from pvforecast.data.smard_forecast import DAY_AHEAD_FILTER_ID, FORECAST_COLUMN
from pvforecast.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# Quarter-hour is the source resolution, as for the realised series.
RESOLUTION = "quarterhour"


def main():
    """Fetch the raw TSO day-ahead PV forecast from SMARD into data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_series(DAY_AHEAD_FILTER_ID, "DE", RESOLUTION, FORECAST_COLUMN)

    out_path = RAW_DIR / f"smard_pv_forecast_dayahead_{RESOLUTION}_2015-2026.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"{len(df)} Zeilen gespeichert: {out_path}")


if __name__ == "__main__":
    setup_logging("forecast")
    main()
