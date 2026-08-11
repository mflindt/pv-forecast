"""
Join of the clean hourly PV series with the Open-Meteo weather series.

Aligns both series on their UTC hourly timestamp into one model-ready table.
"""

import logging
from pathlib import Path

import pandas as pd

from pvforecast.data.openmeteo import RADIATION_VARS

logger = logging.getLogger(__name__)


def load_weather(path: Path) -> pd.DataFrame:
    """Read the raw weather CSV and build a UTC DatetimeIndex named 'time'."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    logger.info(f"{len(df)} Wetterstunden geladen: {path.name}")
    return df


def align_radiation_labels(weather: pd.DataFrame) -> pd.DataFrame:
    """Shift the radiation columns one hour back onto SMARD's label convention."""
    full = pd.date_range(weather.index.min(), weather.index.max(), freq="h")
    missing = full.difference(weather.index)
    if not missing.empty:
        raise ValueError(f"Lücken im Wetter-Stundenraster: {len(missing)} Stunden")

    aligned = weather.copy()
    aligned[RADIATION_VARS] = weather[RADIATION_VARS].shift(-1)
    aligned = aligned.iloc[:-1]

    logger.info(
        f"Strahlung um 1 h nach vorn ausgerichtet ({', '.join(RADIATION_VARS)}); "
        f"letzte Wetterstunde entfällt -> {len(aligned)} Stunden"
    )
    return aligned


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

    # Restore the hourly frequency the join drops
    joined = joined.asfreq("h")
    if len(joined) != len(pv):
        raise ValueError(f"Join ergab {len(joined)} statt {len(pv)} Stunden")

    holes = int(joined.isna().sum().sum())
    if holes:
        per_col = joined.isna().sum()
        logger.warning(f"{holes} NaN-Werte im Join:\n{per_col[per_col > 0]}")

    logger.info(
        f"Join: {len(pv)} PV-Stunden x {len(weather)} Wetterstunden "
        f"-> {len(joined)} Zeilen, {joined.shape[1]} Spalten"
    )
    return joined
