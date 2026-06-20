"""Tests for the cleaning/resampling invariants in pvforecast.data.clean."""

import numpy as np
import pandas as pd
import pytest

from pvforecast.data import clean


def _to_ms(index: pd.DatetimeIndex) -> np.ndarray:
    """Epoch milliseconds, matching the raw SMARD CSV format."""
    return (index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")


def test_aggregate_full_hour_sums_quarters():
    # Four quarter-hours form one complete hour -> their sum.
    idx = pd.date_range("2020-01-01 00:00", periods=4, freq="15min", tz="UTC")
    df = pd.DataFrame({"pv_mwh": [1.0, 2.0, 3.0, 4.0]}, index=idx)

    hourly = clean.aggregate_to_hourly(df)

    assert len(hourly) == 1
    assert hourly["pv_mwh"].iloc[0] == 10.0


def test_aggregate_incomplete_hour_is_nan():
    # Only three of four quarter-hours -> min_count=4 marks the hour NaN.
    idx = pd.date_range("2020-01-01 00:00", periods=3, freq="15min", tz="UTC")
    df = pd.DataFrame({"pv_mwh": [1.0, 2.0, 3.0]}, index=idx)

    hourly = clean.aggregate_to_hourly(df)

    assert hourly["pv_mwh"].isna().all()


def test_ensure_regular_grid_raises_on_gap():
    # 01:00 is missing between 00:00 and 02:00.
    idx = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 02:00"]).tz_localize("UTC")
    df = pd.DataFrame({"pv_mwh": [1.0, 2.0]}, index=idx)

    with pytest.raises(ValueError, match="Lücken"):
        clean.ensure_regular_grid(df)


def test_ensure_regular_grid_raises_on_nan():
    idx = pd.date_range("2020-01-01 00:00", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"pv_mwh": [1.0, np.nan, 3.0]}, index=idx)

    with pytest.raises(ValueError, match="NaN"):
        clean.ensure_regular_grid(df)


def test_ensure_regular_grid_passes_clean_series():
    idx = pd.date_range("2020-01-01 00:00", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"pv_mwh": [1.0, 2.0, 3.0]}, index=idx)

    out = clean.ensure_regular_grid(df)

    assert out.index.freq is not None
    assert out.index.name == "time"
    assert out["pv_mwh"].notna().all()


def test_trim_to_period_is_inclusive():
    idx = pd.date_range("2020-01-01 00:00", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"pv_mwh": range(5)}, index=idx)
    start = pd.Timestamp("2020-01-01 01:00", tz="UTC")
    end = pd.Timestamp("2020-01-01 03:00", tz="UTC")

    out = clean.trim_to_period(df, start, end)

    assert out.index.min() == start
    assert out.index.max() == end
    assert len(out) == 3


def test_build_hourly_series_end_to_end(tmp_path):
    # Eight quarter-hours = two full hours, each summing to 4.0.
    idx = pd.date_range("2020-01-01 00:00", periods=8, freq="15min", tz="UTC")
    raw = pd.DataFrame({"timestamp_ms": _to_ms(idx), "pv_mwh": [1.0] * 8})
    csv = tmp_path / "raw.csv"
    raw.to_csv(csv, index=False)

    start = pd.Timestamp("2020-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2020-01-01 01:00", tz="UTC")
    out = clean.build_hourly_series(csv, start, end)

    assert len(out) == 2
    assert (out["pv_mwh"] == 4.0).all()
    assert out.index.name == "time"
