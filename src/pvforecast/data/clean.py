"""
Cleaning and resampling of the raw SMARD quarter-hour series.

Turns a raw quarter-hour energy series into a clean, gap-free hourly series. The
column is a parameter so that the realised feed-in and the TSO forecast go through
exactly the same aggregation -- comparing them requires nothing less.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


PV_COLUMN = "pv_mwh"


def load_raw_quarterhour(path: Path) -> pd.DataFrame:
    """Read the raw quarter-hour CSV and build a UTC DatetimeIndex."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("time").drop(columns="timestamp_ms")
    logger.info(f"{len(df)} Viertelstundenwerte geladen: {path.name}")
    return df


def aggregate_to_hourly(df: pd.DataFrame, column: str = PV_COLUMN) -> pd.DataFrame:
    """Aggregate quarter-hour energy to hourly sums.

    Requires all 4 quarter-hours per hour (min_count=4); incomplete hours become NaN.
    """
    hourly = df[column].resample("h").sum(min_count=4).to_frame()
    incomplete = int(hourly[column].isna().sum())
    if incomplete:
        logger.warning(f"{incomplete} unvollständige Stunden als NaN markiert")
    logger.info(f"{len(df)} Viertelstunden zu {len(hourly)} Stunden aggregiert")
    return hourly


def trim_to_period(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Cut the series to the closed interval [start, end]."""
    trimmed = df.loc[start:end]
    logger.info(f"Zeitraum {start} bis {end}: {len(trimmed)} Stunden")
    return trimmed


def ensure_regular_grid(df: pd.DataFrame, column: str = PV_COLUMN) -> pd.DataFrame:
    """Ensure a complete, NaN-free hourly grid and fix index frequency.

    Missing hours and NaNs are treated as errors; the pipeline must not contain gaps.
    """
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing = full.difference(df.index)
    if not missing.empty:
        raise ValueError(f"Lücken im Stundenraster: {missing.tolist()}")

    df = df.reindex(full)
    df.index.name = "time"

    holes = int(df[column].isna().sum())
    if holes:
        raise ValueError(f"{holes} NaN-Werte in der Stundenreihe")

    logger.info(f"Lückenloses Stundenraster: {len(df)} Stunden, freq={df.index.freq}")
    return df


def build_hourly_series(
    path: Path, start: pd.Timestamp, end: pd.Timestamp, column: str = PV_COLUMN
) -> pd.DataFrame:
    """Build the clean hourly series from the raw quarter-hour CSV."""
    df = load_raw_quarterhour(path)
    df = aggregate_to_hourly(df, column)
    df = trim_to_period(df, start, end)
    df = ensure_regular_grid(df, column)
    return df
