"""
Target variable for the day-ahead PV forecast.

The feed-in series is normalised by an empirical, rolling capacity estimate rather
than by the installed nameplate power: feed-in per installed MW drifts downward over
the project period (see notebooks/05_kapazitaet_drift.ipynb), so a nameplate-based
capacity factor would not be comparable across years.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# 2015 serves as warm-up for the 365-day window; usable model data starts after it.
DATA_START = pd.Timestamp("2016-01-01", tz="UTC")


def rolling_capacity(
    pv: pd.Series,
    window: str = "365D",
    q: float = 0.995,
    shift_h: int = 48,
) -> pd.Series:
    """Empirical capacity: trailing quantile of the feed-in, shifted forward."""
    if not pv.index.is_monotonic_increasing:
        raise ValueError("Index muss aufsteigend sortiert sein")
    if pv.index.has_duplicates:
        raise ValueError("Index enthält doppelte Zeitstempel")
    if pv.index.tz is None:
        raise ValueError("Index muss zeitzonenbehaftet sein (UTC)")

    trailing = pv.rolling(window).quantile(q)
    # Shifting the index (not the rows) is what makes the leakage guarantee hold.
    shifted = trailing.shift(freq=pd.Timedelta(hours=shift_h))
    capacity = shifted.reindex(pv.index)
    capacity.name = "cap_roll_mwh"

    logger.info(
        f"Rollierende Kapazität: window={window}, q={q}, shift={shift_h} h "
        f"-> ab {capacity.first_valid_index()} verfügbar"
    )
    return capacity


def capacity_factor(pv: pd.Series, capacity: pd.Series) -> pd.Series:
    """Normalise the feed-in by the rolling capacity estimate."""
    if (capacity <= 0).any():
        raise ValueError("Kapazitätsschätzung enthält Werte <= 0")

    cf = pv / capacity
    cf.name = "cf"
    return cf


def to_energy(cf: pd.Series, capacity: pd.Series) -> pd.Series:
    """Back-transform a capacity factor into MWh."""
    energy = cf * capacity
    energy.name = "pv_mwh"
    return energy
