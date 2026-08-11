"""Entry point for the Energy-Charts installed-capacity loader."""

import logging

import pandas as pd

from pvforecast.config import RAW_DIR
from pvforecast.data.capacity import fetch_installed_power
from pvforecast.logging_setup import setup_logging

logger = logging.getLogger(__name__)

COUNTRY = "de"
TIME_STEP = "monthly"

# Anchors for the plausibility check against the project period (2015-2025)
CHECK_MONTHS = (
    pd.Timestamp("2015-01-01", tz="UTC"),
    pd.Timestamp("2025-12-01", tz="UTC"),
)


def main():
    """Fetch the monthly installed PV capacity from Energy-Charts into data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_installed_power(COUNTRY, TIME_STEP)

    # Raw stays raw: the full API range is stored, consumers trim to their period.
    out_path = RAW_DIR / "capacity_energycharts_solar_2002-2026.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"{len(df)} Zeilen gespeichert: {out_path}")

    covered = df.set_index("time")
    for month in CHECK_MONTHS:
        if month not in covered.index:
            raise ValueError(f"Kapazitätsreihe deckt {month:%Y-%m} nicht ab")
        logger.info(
            f"{month:%Y-%m}: {covered.loc[month, 'solar_ac_gw']:.1f} GW AC / "
            f"{covered.loc[month, 'solar_dc_gw']:.1f} GW DC"
        )


if __name__ == "__main__":
    setup_logging("capacity")
    main()
