"""
Join of the clean hourly PV series with the Open-Meteo weather series.

Aligns both series on their UTC hourly timestamp into one model-ready table.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_weather(path: Path) -> pd.DataFrame:
    """Read the raw weather CSV and build a UTC DatetimeIndex named 'time'."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    logger.info(f"{len(df)} Wetterstunden geladen: {path.name}")
    return df


def join_pv_weather(pv: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Inner-join PV and weather on the hourly UTC index.

    Weather must cover every PV hour; a missing hour is an error (gap-free model input).
    """
    missing = pv.index.difference(weather.index)
    if not missing.empty:
        raise ValueError(
            f"Wetter fehlt für {len(missing)} PV-Stunden: {missing.tolist()}"
        )

    joined = pv.join(weather, how="inner")
    joined.index.name = "time"

    holes = int(joined.isna().sum().sum())
    if holes:
        per_col = joined.isna().sum()
        logger.warning(f"{holes} NaN-Werte im Join:\n{per_col[per_col > 0]}")

    logger.info(
        f"Join: {len(pv)} PV-Stunden x {len(weather)} Wetterstunden "
        f"-> {len(joined)} Zeilen, {joined.shape[1]} Spalten"
    )
    return joined
