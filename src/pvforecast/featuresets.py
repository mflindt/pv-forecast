"""
Feature stages for the information ablation.

The stages are nested: S1 < S2 < S3. S3 is the full matrix built by features.py,
so the ablation drops columns rather than building new ones.
"""

import logging

from pvforecast.data.openmeteo import HOURLY_VARS

logger = logging.getLogger(__name__)

# S1: available without an external source; solar geometry follows from the timestamp.
HISTORY = [
    "cf_lag48h",
    "cf_lag168h",
    "sun_elevation",
    "sun_azimuth",
    "cos_zenith",
    "cs_ghi",
    "doy_sin",
    "doy_cos",
]

# S2 adds the raw NWP weather of the target hour.
WEATHER = list(HOURLY_VARS)

# S3 adds transformations of S2: no new information, only a different representation.
SOLAR_PHYSICS = ["kt", "diffuse_fraction"]

STAGES = {
    "S1": HISTORY,
    "S2": HISTORY + WEATHER,
    "S3": HISTORY + WEATHER + SOLAR_PHYSICS,
}

FULL_STAGE = "S3"


def columns(stage: str) -> list[str]:
    """Column names of one feature stage."""
    if stage not in STAGES:
        raise ValueError(
            f"Unbekannte Feature-Stufe: {stage!r} (bekannt: {list(STAGES)})"
        )
    return list(STAGES[stage])


def select(X, stage: str):
    """Reduce a feature matrix to one stage; a missing column is a pipeline error."""
    cols = columns(stage)
    missing = [col for col in cols if col not in X.columns]
    if missing:
        raise ValueError(f"Stufe {stage} braucht fehlende Spalten: {missing}")

    logger.debug(f"Feature-Stufe {stage}: {len(cols)} von {X.shape[1]} Spalten")
    return X[cols]
