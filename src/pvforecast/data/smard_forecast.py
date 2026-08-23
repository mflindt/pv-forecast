"""
Loader for the TSO day-ahead PV forecast published on SMARD.

Published by 18:00 on D-1, hence after our gate: an external reference, never a
feature. Filter ids come from SMARD's module registry, which the community API docs
do not cover: https://www.smard.de/app/chart_configuration/market_data_configuration.json
"""

import logging
from pathlib import Path

import pandas as pd

from pvforecast.data.clean import (
    aggregate_to_hourly,
    ensure_regular_grid,
    load_raw_quarterhour,
    trim_to_period,
)

logger = logging.getLogger(__name__)

# Module 2000125. '3791' is wind offshore, '5097' is wind plus PV.
DAY_AHEAD_FILTER_ID = "125"

# Same series intraday. Not used, kept for provenance.
INTRADAY_FILTER_ID = "5126"

FORECAST_COLUMN = "pv_fcst_mwh"


def build_hourly_forecast(
    path: Path, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Build the clean hourly TSO forecast from the raw quarter-hour CSV."""
    df = load_raw_quarterhour(path)
    df = aggregate_to_hourly(df, FORECAST_COLUMN)
    df = trim_to_period(df, start, end)
    df = ensure_regular_grid(df, FORECAST_COLUMN)
    return df


def align_to_target(
    forecast: pd.DataFrame, target_index: pd.DatetimeIndex
) -> pd.Series:
    """Reindex the forecast onto the target hours; a missing hour is an error."""
    missing = target_index.difference(forecast.index)
    if not missing.empty:
        raise ValueError(
            f"ÜNB-Prognose fehlt für {len(missing)} Zielstunden, "
            f"z. B. {missing[0]:%Y-%m-%d %H:%M}"
        )

    aligned = forecast.loc[target_index, FORECAST_COLUMN]
    logger.info(f"ÜNB-Prognose auf {len(aligned)} Zielstunden ausgerichtet")
    return aligned
