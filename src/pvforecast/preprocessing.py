"""Data preprocessing and feature engineering pipeline."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from pvforecast import config
from pvforecast.config import (
    CAPACITY_RAW,
    FORECAST_RAW,
    HOURLY_VARS,
    INTERIM_DIR,
    MODEL_INPUT,
    MODEL_VARS,
    PROCESSED_DIR,
    PV_HOURLY,
    PV_RAW,
    PV_RAW_HOURLY,
    RADIATION_VARS,
    SITES,
    WEATHER_RAW,
)

logger = logging.getLogger(__name__)

PV_COLUMN = "pv_mwh"
FORECAST_COLUMN = "pv_fcst_mwh"
HOURS_PER_DAY = 24

# Everything ingest.py writes; the derived tables are rebuilt when one is newer.
RAW_FILES = (PV_RAW, PV_RAW_HOURLY, FORECAST_RAW, WEATHER_RAW, CAPACITY_RAW)

# 2015 serves as warm-up for the 365-day capacity window; model data starts after it.
DATA_START = pd.Timestamp("2016-01-01", tz="UTC")

# Hourly data uses the sun position at the interval midpoint.
HOUR_MIDPOINT = pd.Timedelta(minutes=30)

# Day-ahead gate: 12:00 local time on D-1.
ISSUE_HOUR_UTC = 10

# t-24h would leak past the gate.
LAG_HOURS = (48, 168)

# Below this clear-sky irradiance the ratio is twilight noise; kt is set to 0 there.
MIN_CS_GHI_WM2 = 20.0

# kt runs above 1 at low sun and under the Linke climatology; the cap bounds that.
KT_MAX = 1.5

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

# S2 adds the raw NWP weather of the target hour, S3 its transformations.
WEATHER = list(MODEL_VARS)
SOLAR_PHYSICS = ["kt", "diffuse_fraction"]

STAGES = {
    "S1": HISTORY,
    "S2": HISTORY + WEATHER,
    "S3": HISTORY + WEATHER + SOLAR_PHYSICS,
}


def load_raw_quarterhour(path, column: str = PV_COLUMN) -> pd.DataFrame:
    """Read a raw quarter-hour CSV and build a UTC DatetimeIndex."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("time").drop(columns="timestamp_ms")
    logger.info(f"{len(df)} Viertelstundenwerte geladen: {path.name}")
    return df[[column]]


def build_hourly_series(path, column: str = PV_COLUMN) -> pd.DataFrame:
    """Aggregate quarter-hour data to hourly data."""
    df = load_raw_quarterhour(path, column)
    hourly = df[column].resample("h").sum(min_count=4).to_frame()

    start = pd.Timestamp(config.PERIOD_START, tz="UTC")
    end = pd.Timestamp(config.PERIOD_END, tz="UTC")
    hourly = hourly.loc[start:end]

    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    missing = full.difference(hourly.index)
    if not missing.empty:
        raise ValueError(f"{len(missing)} Lücken im Stundenraster, z. B. {missing[0]}")

    hourly = hourly.reindex(full)
    hourly.index.name = "time"

    holes = int(hourly[column].isna().sum())
    if holes:
        raise ValueError(f"{holes} NaN-Werte in der Stundenreihe {column}")

    logger.info(f"{len(df)} Viertelstunden zu {len(hourly)} Stunden aggregiert")
    return hourly


def load_weather(path) -> pd.DataFrame:
    """Read the raw long-format weather CSV and build a UTC 'time' column."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if "site" not in df.columns:
        raise ValueError("Wetterdatei ohne 'site'-Spalte")

    logger.info(f"{len(df)} Wetterzeilen, {df['site'].nunique()} Standorte geladen")
    return df


def align_radiation_labels(weather: pd.DataFrame) -> pd.DataFrame:
    """Shift radiation data to SMARD's hourly labels."""
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
        f"Strahlung je Standort um 1 h nach vorn ausgerichtet -> {len(out)} Zeilen"
    )
    return out


def spatial_mean(weather: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-site weather to one national hourly mean, equally weighted."""
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
    """Inner-join PV and weather on the hourly UTC index; a missing hour is an error."""
    missing = pv.index.difference(weather.index)
    if not missing.empty:
        raise ValueError(f"Wetter fehlt für {len(missing)} PV-Stunden")

    joined = pv.join(weather, how="inner").asfreq("h")
    joined.index.name = "time"
    if len(joined) != len(pv):
        raise ValueError(f"Join ergab {len(joined)} statt {len(pv)} Stunden")

    holes = int(joined.isna().sum().sum())
    if holes:
        raise ValueError(f"{holes} NaN-Werte im Join")

    logger.info(f"Join: {len(joined)} Stunden, {joined.shape[1]} Spalten")
    return joined


def load_capacity(path=CAPACITY_RAW) -> pd.DataFrame:
    """The monthly installed capacity table in GW, indexed by UTC month start."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    logger.info(f"{len(df)} Kapazitäts-Monatswerte geladen: {path.name}")
    return df


def rolling_capacity(
    pv: pd.Series, window: str = "365D", q: float = 0.995, shift_h: int = 48
) -> pd.Series:
    """Empirical capacity: trailing quantile of the feed-in, shifted forward."""
    if not pv.index.is_monotonic_increasing:
        raise ValueError("Index muss aufsteigend sortiert sein")
    if pv.index.has_duplicates:
        raise ValueError("Index enthält doppelte Zeitstempel")
    if pv.index.tz is None:
        raise ValueError("Index muss zeitzonenbehaftet sein (UTC)")

    # Shifting the index (not the rows) is what makes the leakage guarantee hold.
    trailing = pv.rolling(window).quantile(q)
    capacity = trailing.shift(freq=pd.Timedelta(hours=shift_h)).reindex(pv.index)
    capacity.name = "cap_roll_mwh"

    logger.info(f"Rollierende Kapazität verfügbar ab {capacity.first_valid_index()}")
    return capacity


def capacity_factor(pv: pd.Series, capacity: pd.Series) -> pd.Series:
    """Normalise the feed-in by the rolling capacity estimate."""
    if (capacity <= 0).any():
        raise ValueError("Kapazitätsschätzung enthält Werte <= 0")
    return (pv / capacity).rename("cf")


def to_energy(cf: pd.Series, capacity: pd.Series) -> pd.Series:
    """Back-transform a capacity factor into MWh."""
    return (cf * capacity).rename(PV_COLUMN)


def issue_time(target) -> pd.DatetimeIndex:
    """Gate of the day-ahead forecast for the given target hours: 10:00 UTC on D-1."""
    return target.normalize() - (
        pd.Timedelta(days=1) - pd.Timedelta(hours=ISSUE_HOUR_UTC)
    )


def solar_geometry(index: pd.DatetimeIndex, sites: tuple = SITES) -> pd.DataFrame:
    """Solar position and Ineichen clear-sky GHI, averaged over the weather sites."""
    mid = index + HOUR_MIDPOINT
    elevation, cos_zenith, cs_ghi, azimuth_sin, azimuth_cos = [], [], [], [], []

    for _, latitude, longitude, altitude in sites:
        site = pvlib.location.Location(latitude, longitude, altitude=altitude)
        position = pvlib.solarposition.get_solarposition(
            mid, site.latitude, site.longitude, altitude=site.altitude
        )
        clearsky = site.get_clearsky(mid, model="ineichen", solar_position=position)

        elevation.append(position["apparent_elevation"].to_numpy())
        # Below the horizon the cosine turns negative, but no irradiance arrives.
        cos_zenith.append(
            np.cos(np.radians(position["apparent_zenith"].to_numpy())).clip(0.0)
        )
        cs_ghi.append(clearsky["ghi"].to_numpy())

        radians = np.radians(position["azimuth"].to_numpy())
        azimuth_sin.append(np.sin(radians))
        azimuth_cos.append(np.cos(radians))

    # Circular mean: a plain average would break across the 0/360 wrap at night.
    azimuth = (
        np.degrees(
            np.arctan2(np.mean(azimuth_sin, axis=0), np.mean(azimuth_cos, axis=0))
        )
        % 360
    )

    return pd.DataFrame(
        {
            "sun_elevation": np.mean(elevation, axis=0),
            "sun_azimuth": azimuth,
            "cos_zenith": np.mean(cos_zenith, axis=0),
            "cs_ghi": np.mean(cs_ghi, axis=0),
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


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build the feature matrix, target, and metadata."""
    missing = [col for col in (PV_COLUMN, *HOURLY_VARS) if col not in df.columns]
    if missing:
        raise ValueError(f"Spalten fehlen im Modell-Input: {missing}")

    pv = df[PV_COLUMN]
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
            df[MODEL_VARS],
            lag_features(cf),
        ],
        axis=1,
    )

    # Outside the warm-up the matrix has to be complete; a hole is a pipeline error.
    warmup_end = first_cf + pd.Timedelta(hours=max(LAG_HOURS))
    per_col = X.loc[warmup_end:].isna().sum()
    if per_col.any():
        raise ValueError(f"NaN-Werte nach der Warm-up-Phase:\n{per_col[per_col > 0]}")

    meta = pd.concat([pv, capacity], axis=1)
    logger.info(
        f"Features: {X.shape[1]} Spalten für {len(X)} Stunden, vollständig ab "
        f"{warmup_end:%Y-%m-%d %H:%M} (Gate {ISSUE_HOUR_UTC}:00 UTC am Vortag)"
    )
    return X, cf, meta


def select(X: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Reduce the feature matrix to one stage; a missing column is a pipeline error."""
    if stage not in STAGES:
        raise ValueError(
            f"Unbekannte Feature-Stufe: {stage!r} (bekannt: {list(STAGES)})"
        )

    cols = STAGES[stage]
    missing = [col for col in cols if col not in X.columns]
    if missing:
        raise ValueError(f"Stufe {stage} braucht fehlende Spalten: {missing}")
    return X[cols]


def build_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load model inputs for the fold loop."""
    df = pd.read_parquet(MODEL_INPUT)
    X, y, meta = build_features(df)

    # The naive references read the lagged clear-sky index as a plain column.
    X = pd.concat([X, lag_features(X["kt"], lags=(min(LAG_HOURS),))], axis=1)
    X, y, meta = X.loc[DATA_START:], y.loc[DATA_START:], meta.loc[DATA_START:]

    # Monthly nameplate power, held constant within the month it was reported.
    capacity = load_capacity()["solar_ac_gw"] * 1000
    meta["cap_ac_mw"] = capacity.reindex(X.index.union(capacity.index)).ffill()[X.index]

    tso = pd.Series(dtype=float)
    if cfg["include_tso"]:
        forecast = build_hourly_series(FORECAST_RAW, FORECAST_COLUMN)
        missing = X.index.difference(forecast.index)
        if not missing.empty:
            raise ValueError(f"ÜNB-Prognose fehlt für {len(missing)} Zielstunden")
        tso = forecast.loc[X.index, FORECAST_COLUMN]
        logger.info(f"ÜNB-Prognose auf {len(tso)} Zielstunden ausgerichtet")

    return X, y, meta, tso


def build_model_input() -> None:
    """Rebuild data/interim and data/processed from the raw files."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pv = build_hourly_series(PV_RAW, PV_COLUMN)
    pv.to_parquet(PV_HOURLY)
    logger.info(f"{len(pv)} Stunden gespeichert: {PV_HOURLY}")

    weather = spatial_mean(align_radiation_labels(load_weather(WEATHER_RAW)))
    joined = join_pv_weather(pv, weather)
    joined.to_parquet(MODEL_INPUT)
    logger.info(f"{len(joined)} Stunden gespeichert: {MODEL_INPUT}")


def ensure_model_input(force: bool = False) -> Path:
    """Bring data/ up to date"""
    missing = [path for path in RAW_FILES if not path.is_file()]
    if missing:
        logger.info(f"Rohdaten fehlen: {', '.join(p.name for p in missing)}")
        from pvforecast import ingest

        ingest.main()

    if not force and MODEL_INPUT.is_file():
        newest_raw = max(path.stat().st_mtime for path in RAW_FILES)
        if MODEL_INPUT.stat().st_mtime >= newest_raw:
            logger.info(f"Modell-Input ist aktuell: {MODEL_INPUT.name}")
            return MODEL_INPUT

    build_model_input()
    return MODEL_INPUT


if __name__ == "__main__":
    config.setup_logging()
    ensure_model_input(force=True)
