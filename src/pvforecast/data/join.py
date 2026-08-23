"""
Join of the clean hourly PV series with the Open-Meteo weather series.

The weather arrives per site and is reduced to one national mean before the join.
Equal weights: a capacity weighting would need regional capacity shares we do not
have, and the mean already carries the spatial signal (see docs/arbeitsplan.md).
"""

import logging
from pathlib import Path

import pandas as pd

from pvforecast.data.openmeteo import HOURLY_VARS, RADIATION_VARS

logger = logging.getLogger(__name__)


def load_weather(path: Path) -> pd.DataFrame:
    """Read the raw long-format weather CSV and build a UTC 'time' column."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    if "site" not in df.columns:
        raise ValueError("Wetterdatei ohne 'site'-Spalte")

    sites = df["site"].nunique()
    logger.info(f"{len(df)} Wetterzeilen, {sites} Standorte geladen: {path.name}")
    return df


def align_radiation_labels(weather: pd.DataFrame) -> pd.DataFrame:
    """Shift the radiation columns one hour back onto SMARD's label convention.

    Open-Meteo labels a radiation mean at the end of its interval, SMARD at the
    start. Applied per site, because the shift must not cross a site boundary.
    """
    aligned = []
    for name, block in weather.groupby("site", sort=False):
        block = block.sort_values("time")
        full = pd.date_range(block["time"].min(), block["time"].max(), freq="h")
        missing = full.difference(pd.DatetimeIndex(block["time"]))
        if not missing.empty:
            raise ValueError(f"Standort {name}: {len(missing)} Stunden fehlen")

        shifted = block.copy()
        shifted[RADIATION_VARS] = block[RADIATION_VARS].shift(-1)
        aligned.append(shifted.iloc[:-1])

    out = pd.concat(aligned, ignore_index=True)
    logger.info(
        f"Strahlung je Standort um 1 h nach vorn ausgerichtet "
        f"({', '.join(RADIATION_VARS)}); letzte Stunde entfällt -> {len(out)} Zeilen"
    )
    return out


def spatial_mean(weather: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-site weather to one national hourly mean."""
    sites = weather["site"].nunique()
    counts = weather.groupby("time").size()
    incomplete = counts[counts != sites]
    if not incomplete.empty:
        raise ValueError(
            f"{len(incomplete)} Stunden ohne alle {sites} Standorte, "
            f"z. B. {incomplete.index[0]:%Y-%m-%d %H:%M}"
        )

    out = weather.groupby("time")[HOURLY_VARS].mean()
    out.index.name = "time"

    logger.info(f"{sites} Standorte zu {len(out)} Stundenmitteln gemittelt")
    return out


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
