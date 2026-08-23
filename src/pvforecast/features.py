"""Target variable and feature matrix for the day-ahead PV forecast."""

import logging

import numpy as np
import pandas as pd
import pvlib

from pvforecast.data.openmeteo import HOURLY_VARS

logger = logging.getLogger(__name__)

# 2015 serves as warm-up for the 365-day window; usable model data starts after it.
DATA_START = pd.Timestamp("2016-01-01", tz="UTC")

# ERA5 grid cell the radiation comes from: the request point 51.2/10.4 snaps here.
SITE_LATITUDE = 51.25
SITE_LONGITUDE = 10.5
SITE_ALTITUDE_M = 287.0

# PV and radiation are hourly means labelled at the interval start, so the sun position
# representative for that interval is the one at its middle.
HOUR_MIDPOINT = pd.Timedelta(minutes=30)

# Day-ahead gate: 12:00 local time on D-1.
ISSUE_HOUR_UTC = 10

# t-24h would leak: from target hour 12:00 UTC on it reaches past the gate.
LAG_HOURS = (48, 168)

# Below this clear-sky irradiance the ratio is twilight noise; kt is set to 0 there.
MIN_CS_GHI_WM2 = 20.0

# kt runs above 1 for two reasons: at low sun the midpoint clear-sky underestimates the
# hourly mean, and the Linke climatology attenuates a little too strongly (kt averages
# 1.11 on clear days). The cap bounds that; ~2 % of the daylight hours sit on it.
KT_MAX = 1.5


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


def issue_time(target: pd.DatetimeIndex | pd.Timestamp) -> pd.DatetimeIndex:
    """Gate of the day-ahead forecast for the given target hours: 10:00 UTC on D-1."""
    lead = pd.Timedelta(days=1) - pd.Timedelta(hours=ISSUE_HOUR_UTC)
    return target.normalize() - lead


def solar_geometry(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Solar position and Ineichen clear-sky GHI at the middle of each labelled hour."""
    mid = index + HOUR_MIDPOINT
    site = pvlib.location.Location(
        SITE_LATITUDE, SITE_LONGITUDE, altitude=SITE_ALTITUDE_M
    )
    position = pvlib.solarposition.get_solarposition(
        mid, site.latitude, site.longitude, altitude=site.altitude
    )
    clearsky = site.get_clearsky(mid, model="ineichen", solar_position=position)

    # Below the horizon the cosine turns negative, but no irradiance arrives.
    cos_zenith = np.cos(np.radians(position["apparent_zenith"].to_numpy())).clip(0.0)

    return pd.DataFrame(
        {
            "sun_elevation": position["apparent_elevation"].to_numpy(),
            "sun_azimuth": position["azimuth"].to_numpy(),
            "cos_zenith": cos_zenith,
            "cs_ghi": clearsky["ghi"].to_numpy(),
        },
        index=index,
    )


def clear_sky_index(ghi: pd.Series, cs_ghi: pd.Series) -> pd.Series:
    """Clear-sky index kt = GHI / clear-sky GHI, capped and zero without sun."""
    night = cs_ghi < MIN_CS_GHI_WM2
    kt = ghi.div(cs_ghi.where(~night)).clip(0.0, KT_MAX)
    return kt.mask(night, 0.0).rename("kt")


def diffuse_fraction(ghi: pd.Series, dhi: pd.Series) -> pd.Series:
    """Diffuse share of the GHI; undefined without irradiance and set to 0 there."""
    dark = ghi <= 0.0
    # Open-Meteo's direct component is horizontal (direct + diffuse = GHI), so the
    # clip only guards the numerics.
    fraction = dhi.div(ghi.where(~dark)).clip(0.0, 1.0)
    return fraction.mask(dark, 0.0).rename("diffuse_fraction")


def cyclic_day_of_year(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Season as a point on the unit circle; 365.25 keeps leap years in phase."""
    angle = 2 * np.pi * (index.dayofyear - 1) / 365.25
    return pd.DataFrame(
        {"doy_sin": np.sin(angle), "doy_cos": np.cos(angle)}, index=index
    )


def lag_features(series: pd.Series, lags: tuple[int, ...] = LAG_HOURS) -> pd.DataFrame:
    """Time-shifted copies of a series, named after their offset in hours."""
    out = pd.DataFrame(index=series.index)
    for lag in lags:
        # Shifting the index (not the rows) keeps the offset correct across gaps.
        shifted = series.shift(freq=pd.Timedelta(hours=lag))
        out[f"{series.name}_lag{lag}h"] = shifted.reindex(series.index)
    return out


def baseline_inputs(X: pd.DataFrame) -> pd.DataFrame:
    """Lagged clear-sky index the naive references need."""
    return lag_features(X["kt"], lags=(min(LAG_HOURS),))


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build feature matrix, target and meta columns from the joined PV+weather table.

    X holds only what is known at the issue time: deterministic solar geometry, the
    weather of the target hour and feed-in lagged past the gate. Rows are kept, so the
    lag warm-up at the start of the series stays visible as NaN instead of vanishing.
    """
    missing = [col for col in ("pv_mwh", *HOURLY_VARS) if col not in df.columns]
    if missing:
        raise ValueError(f"Spalten fehlen im Modell-Input: {missing}")

    pv = df["pv_mwh"]
    capacity = rolling_capacity(pv)
    cf = capacity_factor(pv, capacity)

    first_cf = cf.first_valid_index()
    if first_cf is None:
        raise ValueError("Zeitreihe zu kurz für die Kapazitätsschätzung")

    solar = solar_geometry(df.index)
    X = pd.concat(
        [
            solar,
            clear_sky_index(df["shortwave_radiation"], solar["cs_ghi"]),
            diffuse_fraction(df["shortwave_radiation"], df["diffuse_radiation"]),
            cyclic_day_of_year(df.index),
            df[HOURLY_VARS],
            lag_features(cf),
        ],
        axis=1,
    )

    # Outside the warm-up the matrix has to be complete; a hole is a pipeline error.
    warmup_end = first_cf + pd.Timedelta(hours=max(LAG_HOURS))
    per_col = X.loc[warmup_end:].isna().sum()
    if per_col.any():
        holes = int(per_col.sum())
        raise ValueError(
            f"{holes} NaN-Werte nach der Warm-up-Phase:\n{per_col[per_col > 0]}"
        )

    meta = pd.concat([pv, capacity], axis=1)

    logger.info(
        f"Features: {X.shape[1]} Spalten für {len(X)} Stunden, vollständig ab "
        f"{warmup_end:%Y-%m-%d %H:%M} (Lags {', '.join(f'{h} h' for h in LAG_HOURS)}, "
        f"Gate {ISSUE_HOUR_UTC}:00 UTC am Vortag)"
    )
    return X, cf, meta
