"""Entry point for building the clean hourly dataset."""

import logging

import pandas as pd

from pvforecast.config import PROCESSED_DIR, RAW_DIR
from pvforecast.data.clean import build_hourly_series
from pvforecast.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# 2015–2025 only; 2026 excluded by design. Year edges defined in UTC (local midnight = 23:00 UTC)
START = pd.Timestamp("2014-12-31 23:00", tz="UTC")
END = pd.Timestamp("2025-12-31 23:00", tz="UTC")


def main():
    """Build the clean hourly PV series from raw data and save it to data/processed."""
    raw_path = RAW_DIR / "smard_pv_realized_quarterhour_2015-2026.csv"
    df = build_hourly_series(raw_path, START, END)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "pv_hourly.parquet"
    df.to_parquet(out_path)
    logger.info(f"{len(df)} Stunden gespeichert: {out_path}")


if __name__ == "__main__":
    setup_logging("dataset")
    main()
