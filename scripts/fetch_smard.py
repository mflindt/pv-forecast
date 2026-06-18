"""Entry point for the SMARD data loader."""

import logging

from pvforecast.config import RAW_DIR
from pvforecast.data.smard import fetch_series
from pvforecast.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# filter_id '4068' = power generation: photovoltaics (see bundesAPI/smard-api)
PV_FILTER_ID = "4068"

# Load quarter-hour data (source) and SMARD hourly data (for validation only)
RESOLUTIONS = ("quarterhour", "hour")


def main():
    """Fetch the raw PV series from SMARD (quarter-hour + hour) into data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for resolution in RESOLUTIONS:
        df = fetch_series(PV_FILTER_ID, "DE", resolution)
        out_path = RAW_DIR / f"smard_pv_realized_{resolution}_2015-2026.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"{len(df)} Zeilen gespeichert: {out_path}")


if __name__ == "__main__":
    setup_logging("fetch")
    main()
