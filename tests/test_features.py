"""Tests for the target variable and the feature matrix in pvforecast.features."""

import re

import numpy as np
import pandas as pd
import pvlib
import pytest

from pvforecast import features
from pvforecast.config import PROCESSED_DIR
from pvforecast.data.openmeteo import HOURLY_VARS

WINDOW = "10D"
SHIFT_H = 48

MODEL_INPUT = PROCESSED_DIR / "pv_weather_hourly.parquet"

LAG_COLS = ["cf_lag48h", "cf_lag168h"]

# Features that need no lag
ISSUE_TIME_SAFE = {
    "sun_elevation",
    "sun_azimuth",
    "cos_zenith",
    "cs_ghi",
    "kt",
    "diffuse_fraction",
    "doy_sin",
    "doy_cos",
    *HOURLY_VARS,
}


def _flat_series(days: int = 60) -> pd.Series:
    idx = pd.date_range("2020-01-01 00:00", periods=days * 24, freq="h", tz="UTC")
    return pd.Series(1.0, index=idx, name="pv_mwh")


def _model_input(days: int = 60) -> pd.DataFrame:
    """Synthetic joined PV + weather frame with a plausible daily radiation cycle."""
    idx = pd.date_range("2020-01-01 00:00", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(0)

    daylight = np.clip(np.sin((idx.hour.to_numpy() - 6) * np.pi / 12), 0, None)
    ghi = daylight * rng.uniform(0.2, 1.0, len(idx)) * 700
    dhi = ghi * rng.uniform(0.2, 0.9, len(idx))

    return pd.DataFrame(
        {
            # Offset keeps the feed-in positive, so the capacity stays > 0.
            "pv_mwh": ghi * 50 + 1.0,
            "shortwave_radiation": ghi,
            "direct_radiation": ghi - dhi,
            "diffuse_radiation": dhi,
            "temperature_2m": 10 + 10 * daylight,
            "cloud_cover": rng.integers(0, 101, len(idx)),
            "relative_humidity_2m": rng.integers(30, 101, len(idx)),
            "wind_speed_10m": rng.uniform(0, 15, len(idx)),
        },
        index=idx,
    )


def test_rolling_capacity_ignores_future_values():
    """Nothing from t - shift_h onwards may reach cap_roll(t)."""
    pv = _flat_series()
    t = pv.index[24 * 40]
    base = features.rolling_capacity(pv, window=WINDOW, shift_h=SHIFT_H)

    leaked = pv.copy()
    leaked.loc[t - pd.Timedelta(hours=SHIFT_H - 1) :] = 1e6
    out = features.rolling_capacity(leaked, window=WINDOW, shift_h=SHIFT_H)

    assert out.loc[t] == base.loc[t]


def test_rolling_capacity_reacts_to_values_inside_the_window():
    """Counterpart to the leakage test: the window is not simply blind."""
    pv = _flat_series()
    t = pv.index[24 * 40]
    base = features.rolling_capacity(pv, window=WINDOW, shift_h=SHIFT_H)

    spiked = pv.copy()
    # The last five hours the window ending at t - shift_h is still allowed to see.
    spiked.loc[
        t - pd.Timedelta(hours=SHIFT_H + 4) : t - pd.Timedelta(hours=SHIFT_H)
    ] = 1e6
    out = features.rolling_capacity(spiked, window=WINDOW, shift_h=SHIFT_H)

    assert out.loc[t] > base.loc[t]


def test_rolling_capacity_shift_is_time_based():
    pv = _flat_series()
    t = pv.index[24 * 40]

    capacity = features.rolling_capacity(pv, window=WINDOW, shift_h=SHIFT_H)
    trailing = pv.rolling(WINDOW).quantile(0.995)

    assert capacity.loc[t] == trailing.loc[t - pd.Timedelta(hours=SHIFT_H)]


def test_rolling_capacity_warmup_is_nan_not_filled():
    pv = _flat_series()

    capacity = features.rolling_capacity(pv, window=WINDOW, shift_h=SHIFT_H)

    # The first shift_h hours cannot have a predecessor -> NaN, never back-filled.
    assert capacity.iloc[:SHIFT_H].isna().all()
    assert capacity.index.equals(pv.index)


def test_rolling_capacity_raises_on_unsorted_index():
    pv = _flat_series(days=2)

    with pytest.raises(ValueError, match="aufsteigend"):
        features.rolling_capacity(pv.iloc[::-1])


def test_rolling_capacity_raises_on_duplicate_index():
    pv = _flat_series(days=2)
    doubled = pd.concat([pv, pv.iloc[[-1]]])

    with pytest.raises(ValueError, match="doppelte"):
        features.rolling_capacity(doubled)


def test_rolling_capacity_raises_on_naive_index():
    pv = _flat_series(days=2)

    with pytest.raises(ValueError, match="zeitzonenbehaftet"):
        features.rolling_capacity(pv.tz_localize(None))


def test_capacity_factor_round_trip():
    pv = _flat_series()
    capacity = features.rolling_capacity(pv, window=WINDOW, shift_h=SHIFT_H)

    cf = features.capacity_factor(pv, capacity)
    restored = features.to_energy(cf, capacity)

    assert cf.name == "cf"
    valid = restored.notna()
    assert valid.any()
    pd.testing.assert_series_equal(
        restored[valid], pv[valid], check_names=False, check_freq=False
    )


def test_capacity_factor_raises_on_non_positive_capacity():
    pv = _flat_series(days=2)
    capacity = pd.Series(0.0, index=pv.index)

    with pytest.raises(ValueError, match="<= 0"):
        features.capacity_factor(pv, capacity)


def test_data_start_leaves_the_warmup_behind():
    # 2015 is warm-up; the first usable model timestamp is the start of 2016.
    assert features.DATA_START == pd.Timestamp("2016-01-01", tz="UTC")


def test_issue_time_is_the_conservative_noon_gate():
    day = pd.date_range("2020-06-15", periods=24, freq="h", tz="UTC")

    gate = features.issue_time(day)

    # One gate per target day: 12:00 CEST on D-1, the earlier of the two local noons.
    assert (gate == pd.Timestamp("2020-06-14 10:00", tz="UTC")).all()


def test_every_feature_name_clears_the_issue_time():
    """Name-level rule: unlagged features must be known at gate closure."""
    X, _, _ = features.build_features(_model_input())

    # Worst case of a target day: the 23:00 hour sits furthest from its gate.
    latest = pd.Timestamp("2020-06-15 23:00", tz="UTC")
    min_lag = latest - features.issue_time(latest)

    for col in X.columns:
        if col in ISSUE_TIME_SAFE:
            continue
        match = re.fullmatch(r"(.+)_lag(\d+)h", col)
        assert match, f"Feature ohne Lag-Kennzeichnung: {col}"
        assert pd.Timedelta(hours=int(match.group(2))) >= min_lag, col


def test_lag_hours_exclude_the_leaking_day_lag():
    latest = pd.Timestamp("2020-06-15 23:00", tz="UTC")
    min_lag = latest - features.issue_time(latest)

    # t-24h is what the rule rejects; both configured lags clear it.
    assert pd.Timedelta(hours=24) < min_lag
    assert all(pd.Timedelta(hours=lag) >= min_lag for lag in features.LAG_HOURS)


def test_features_ignore_feed_in_after_the_issue_time():
    """Nothing observed after the gate may reach a feature of the target day."""
    df = _model_input()
    day = df.index[24 * 40]
    gate = features.issue_time(day)

    leaked = df.copy()
    leaked.loc[gate + pd.Timedelta(hours=1) :, "pv_mwh"] *= 1e3

    base, _, _ = features.build_features(df)
    out, _, _ = features.build_features(leaked)

    target = out.index.normalize() == day
    pd.testing.assert_frame_equal(out[target], base[target])


def test_features_react_to_feed_in_before_the_issue_time():
    """Counterpart to the leakage test: the 48 h lag is not simply blind."""
    df = _model_input()
    day = df.index[24 * 40]

    leaked = df.copy()
    # The last full day the 48 h lag still reads for the target day.
    visible = df.index.normalize() == day - pd.Timedelta(days=2)
    leaked.loc[visible, "pv_mwh"] *= 1e3

    base, _, _ = features.build_features(df)
    out, _, _ = features.build_features(leaked)

    target = out.index.normalize() == day
    assert (out.loc[target, "cf_lag48h"] != base.loc[target, "cf_lag48h"]).all()


def test_lag_features_are_the_target_shifted_by_the_named_offset():
    df = _model_input()

    X, y, _ = features.build_features(df)

    t = df.index[24 * 40]
    for lag in features.LAG_HOURS:
        assert X.loc[t, f"cf_lag{lag}h"] == y.loc[t - pd.Timedelta(hours=lag)]


def test_features_are_nan_free_after_the_warmup():
    df = _model_input()

    X, y, _ = features.build_features(df)

    warmup_end = y.first_valid_index() + pd.Timedelta(hours=max(features.LAG_HOURS))
    assert X.loc[warmup_end:].notna().all().all()
    # Only the lags carry the warm-up; everything else is complete from hour one.
    assert X.drop(columns=LAG_COLS).notna().all().all()
    assert X.loc[: warmup_end - pd.Timedelta(hours=1), "cf_lag168h"].isna().any()


def test_build_features_keeps_index_target_and_meta_aligned():
    df = _model_input()

    X, y, meta = features.build_features(df)

    assert X.index.equals(df.index)
    assert y.index.equals(df.index)
    assert y.name == "cf"
    assert list(meta.columns) == ["pv_mwh", "cap_roll_mwh"]
    # Meta is for the back-transform only and must not leak into the matrix.
    assert not set(meta.columns) & set(X.columns)


def test_build_features_raises_on_missing_columns():
    df = _model_input(days=30).drop(columns=["cloud_cover"])

    with pytest.raises(ValueError, match="Spalten fehlen"):
        features.build_features(df)


def test_build_features_raises_when_the_series_is_too_short():
    df = _model_input(days=1)

    with pytest.raises(ValueError, match="zu kurz"):
        features.build_features(df)


def test_solar_geometry_is_evaluated_at_the_hour_middle():
    idx = pd.date_range("2020-06-21 00:00", periods=24, freq="h", tz="UTC")
    one_site = (features.SITES[0],)
    _, latitude, longitude, altitude = features.SITES[0]

    out = features.solar_geometry(idx, sites=one_site)

    at_label = pvlib.solarposition.get_solarposition(
        idx, latitude, longitude, altitude=altitude
    )
    at_middle = pvlib.solarposition.get_solarposition(
        idx + features.HOUR_MIDPOINT, latitude, longitude, altitude=altitude
    )
    assert np.allclose(out["sun_elevation"], at_middle["apparent_elevation"])
    assert not np.allclose(out["sun_elevation"], at_label["apparent_elevation"])


def test_solar_geometry_averages_over_the_sites():
    """The national geometry must sit between its northernmost and southernmost site."""
    idx = pd.date_range("2020-06-21 10:00", periods=4, freq="h", tz="UTC")
    north = features.solar_geometry(idx, sites=(features.SITES[0],))
    south = features.solar_geometry(idx, sites=(features.SITES[3],))

    out = features.solar_geometry(idx)

    assert (north["sun_elevation"] < out["sun_elevation"]).all()
    assert (out["sun_elevation"] < south["sun_elevation"]).all()


def test_solar_geometry_azimuth_survives_the_wrap():
    """A plain mean of azimuths would land near 180 degrees around midnight."""
    idx = pd.date_range("2020-06-21 00:00", periods=24, freq="h", tz="UTC")

    out = features.solar_geometry(idx)

    assert out["sun_azimuth"].between(0, 360).all()
    # Around solar midnight the sun sits due north, not due south.
    assert out.loc[idx[0], "sun_azimuth"] < 45 or out.loc[idx[0], "sun_azimuth"] > 315


def test_solar_geometry_covers_the_night():
    idx = pd.date_range("2020-12-21 00:00", periods=24, freq="h", tz="UTC")

    out = features.solar_geometry(idx)

    night = out["sun_elevation"] < 0
    assert night.any() and not night.all()
    assert (out.loc[night, "cos_zenith"] == 0).all()
    assert (out.loc[night, "cs_ghi"] == 0).all()
    assert out["cs_ghi"].max() > 0


def test_clear_sky_index_is_capped_and_zero_without_sun():
    idx = pd.date_range("2020-06-21 00:00", periods=4, freq="h", tz="UTC")
    ghi = pd.Series([0.0, 5.0, 400.0, 900.0], index=idx)
    cs_ghi = pd.Series([0.0, 10.0, 500.0, 300.0], index=idx)

    kt = features.clear_sky_index(ghi, cs_ghi)

    assert kt.name == "kt"
    # Night and twilight below the floor -> 0, then the plain ratio, then the cap.
    assert kt.tolist() == [0.0, 0.0, 0.8, features.KT_MAX]


def test_diffuse_fraction_is_bounded_and_zero_in_the_dark():
    idx = pd.date_range("2020-06-21 00:00", periods=3, freq="h", tz="UTC")
    ghi = pd.Series([0.0, 200.0, 400.0], index=idx)
    dhi = pd.Series([0.0, 50.0, 400.0], index=idx)

    fraction = features.diffuse_fraction(ghi, dhi)

    assert fraction.name == "diffuse_fraction"
    assert fraction.tolist() == [0.0, 0.25, 1.0]


def test_cyclic_day_of_year_closes_the_circle():
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="D", tz="UTC")

    out = features.cyclic_day_of_year(idx)

    assert np.allclose(out["doy_sin"] ** 2 + out["doy_cos"] ** 2, 1.0)
    # Year start and year end are neighbours on the circle, not opposites.
    step = np.hypot(
        out["doy_sin"].iloc[-1] - out["doy_sin"].iloc[0],
        out["doy_cos"].iloc[-1] - out["doy_cos"].iloc[0],
    )
    assert step < 0.05


@pytest.mark.skipif(
    not MODEL_INPUT.exists(), reason="data/processed ist nicht eingecheckt"
)
def test_build_features_on_model_input():
    """The stored model input must yield a complete matrix from DATA_START on."""
    df = pd.read_parquet(MODEL_INPUT)

    X, y, meta = features.build_features(df)

    usable = X.loc[features.DATA_START :]
    assert usable.notna().all().all()
    assert y.loc[features.DATA_START :].notna().all()
    assert meta.loc[features.DATA_START :].notna().all().all()
    assert usable["kt"].between(0.0, features.KT_MAX).all()
    assert usable["diffuse_fraction"].between(0.0, 1.0).all()
    assert usable["cs_ghi"].min() == 0.0
    # Radiation and clear-sky must peak in the same hours, not half a day apart.
    assert usable["shortwave_radiation"].corr(usable["cs_ghi"]) > 0.8
