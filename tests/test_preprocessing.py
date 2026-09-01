"""Tests that guard against data leakage."""

import numpy as np
import pandas as pd
import pytest

from pvforecast import preprocessing
from pvforecast.config import MODEL_VARS, RADIATION_VARS, REDUNDANT_VAR
from pvforecast.preprocessing import (
    STAGES,
    align_radiation_labels,
    build_features,
    clear_sky_index,
    issue_time,
    rolling_capacity,
    select,
    spatial_mean,
    to_energy,
)


def test_rolling_capacity_ignores_future_values():
    index = pd.date_range("2020-01-01", periods=24 * 400, freq="h", tz="UTC")
    pv = pd.Series(100.0, index=index, name="pv_mwh")

    spike = pv.copy()
    spike.iloc[-1] = 1e6

    # The last value may not reach any earlier capacity estimate.
    assert rolling_capacity(pv).equals(rolling_capacity(spike))


def test_rolling_capacity_shift_is_time_based():
    index = pd.date_range("2020-01-01", periods=24 * 400, freq="h", tz="UTC")
    pv = pd.Series(np.linspace(1, 500, len(index)), index=index, name="pv_mwh")
    capacity = rolling_capacity(pv, shift_h=48)

    plain = pv.rolling("365D").quantile(0.995)
    target = index[-1]
    assert capacity.loc[target] == pytest.approx(
        plain.loc[target - pd.Timedelta(hours=48)]
    )


def test_rolling_capacity_rejects_a_naive_index():
    pv = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=10, freq="h"))
    with pytest.raises(ValueError, match="zeitzonenbehaftet"):
        rolling_capacity(pv)


def test_capacity_factor_round_trip(dataset):
    _, y, meta = dataset
    energy = to_energy(y, meta["cap_roll_mwh"])
    assert np.allclose(energy, meta["pv_mwh"])


def test_issue_time_is_the_day_ahead_gate():
    target = pd.DatetimeIndex(["2024-06-15 13:00"], tz="UTC")
    assert issue_time(target)[0] == pd.Timestamp("2024-06-14 10:00", tz="UTC")


def test_no_feature_reaches_past_the_issue_time(joined_frame):
    """Feed-in after the gate must not move a single feature of the target hour."""
    target = pd.Timestamp("2022-06-15 13:00", tz="UTC")
    gate = issue_time(pd.DatetimeIndex([target], tz="UTC"))[0]

    tampered = joined_frame.copy()
    window = (tampered.index > gate) & (tampered.index <= target)
    assert window.sum() > 0
    tampered.loc[window, "pv_mwh"] *= 7.0

    before, _, _ = build_features(joined_frame)
    after, _, _ = build_features(tampered)
    pd.testing.assert_series_equal(before.loc[target], after.loc[target])


def test_features_are_nan_free_after_the_warmup(dataset):
    X, y, meta = dataset
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert (meta["cap_roll_mwh"] > 0).all()


def test_clear_sky_index_is_capped_and_zero_without_sun():
    cs_ghi = pd.Series([0.0, 10.0, 800.0, 800.0])
    ghi = pd.Series([50.0, 50.0, 2000.0, 400.0])
    kt = clear_sky_index(ghi, cs_ghi)

    assert kt.iloc[0] == 0.0 and kt.iloc[1] == 0.0
    assert kt.iloc[2] == preprocessing.KT_MAX
    assert kt.iloc[3] == pytest.approx(0.5)


def test_align_radiation_labels_shifts_only_radiation():
    index = pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            "time": list(index) * 2,
            "site": ["nord"] * 4 + ["ost"] * 4,
            **{name: list(range(4)) * 2 for name in RADIATION_VARS},
            "temperature_2m": list(range(100, 104)) * 2,
        }
    )
    out = align_radiation_labels(weather)

    nord = out[out["site"] == "nord"].reset_index(drop=True)
    # Each site loses its last hour and the radiation moves one step forward.
    assert len(nord) == 3
    assert nord["shortwave_radiation"].tolist() == [1, 2, 3]
    assert nord["temperature_2m"].tolist() == [100, 101, 102]


def test_spatial_mean_raises_when_a_site_is_missing_an_hour():
    index = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            "time": list(index) + list(index[:2]),
            "site": ["nord"] * 3 + ["ost"] * 2,
            **{name: 1.0 for name in preprocessing.HOURLY_VARS},
        }
    )
    with pytest.raises(ValueError, match="ohne alle"):
        spatial_mean(weather)


def test_feature_stages_are_nested_and_never_rank_deficient(dataset):
    X, _, _ = dataset
    assert set(STAGES["S1"]) < set(STAGES["S2"]) < set(STAGES["S3"])

    for stage in STAGES:
        columns = select(X, stage).columns
        # All three radiation columns would make the design matrix singular.
        assert REDUNDANT_VAR not in columns
        assert set(columns) <= set(X.columns)

    assert np.linalg.matrix_rank(select(X, "S3").to_numpy()) == len(STAGES["S3"])
    assert REDUNDANT_VAR not in MODEL_VARS
