"""Entry point for the SMARD data loader."""

import logging

from pvforecast.config import RAW_DIR
from pvforecast.data.smard import fetch_series
from pvforecast.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Fetch the raw hourly PV series from SMARD and save it to data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # filter_id '4068' = power generation: photovoltaics (see bundesAPI/smard-api)
    df = fetch_series("4068", "DE", "hour")
    out_path = RAW_DIR / "smard_pv_realized_hour_2015-2026.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"{len(df)} Zeilen gespeichert: {out_path}")


if __name__ == "__main__":
    setup_logging("fetch")
    main()
