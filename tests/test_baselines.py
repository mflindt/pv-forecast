"""Tests for the reference forecasts in pvforecast.baselines."""

import numpy as np
import pandas as pd
import pytest

from pvforecast import baselines


def _frame(days: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix and capacity factor with a plausible daily cycle."""
    idx = pd.date_range("2020-03-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(0)

    cs_ghi = np.clip(np.sin((idx.hour.to_numpy() - 6) * np.pi / 12), 0, None) * 800
    kt = np.where(cs_ghi > 20, rng.uniform(0.3, 1.1, len(idx)), 0.0)
    cf = pd.Series(kt * cs_ghi / 1000, index=idx, name="cf")

    X = pd.DataFrame({"cs_ghi": cs_ghi, "kt": kt}, index=idx)
    for col, source in (("cf_lag48h", cf), ("kt_lag48h", X["kt"])):
        X[col] = source.shift(freq=pd.Timedelta(hours=48)).reindex(idx)

    usable = X.dropna().index
    return X.loc[usable], cf.loc[usable]


def test_climatology_learns_the_daily_cycle():
    X, y = _frame()

    model = baselines.Climatology().fit(X, y)
    pred = model.predict(X)

    # Midday has to come out above midnight, otherwise nothing was learned.
    assert pred[pred.index.hour == 12].mean() > pred[pred.index.hour == 0].mean()
    assert (pred >= 0).all()


def test_climatology_ignores_the_test_block():
    """Leakage test: fitting on train only, whatever happens later."""
    X, y = _frame()
    split = X.index[24 * 20]
    train = X.index < split

    base = baselines.Climatology().fit(X[train], y[train]).predict(X[~train])

    leaked = y.copy()
    leaked[~train] *= 100
    out = baselines.Climatology().fit(X[train], leaked[train]).predict(X[~train])

    pd.testing.assert_series_equal(out, base)


def test_climatology_falls_back_on_an_unseen_cell():
    """A month-hour cell the training block never saw must not produce NaN."""
    X, y = _frame(days=10)

    model = baselines.Climatology().fit(X, y)
    july = X.set_index(X.index + pd.DateOffset(months=4))

    pred = model.predict(july)

    assert pred.notna().all()
    assert np.allclose(pred, model.fallback_)


def test_climatology_raises_before_fit():
    X, _ = _frame()

    with pytest.raises(ValueError, match="nicht gefittet"):
        baselines.Climatology().predict(X)


def test_persistence_returns_the_lagged_column():
    X, y = _frame()

    pred = baselines.Persistence().fit(X, y).predict(X)

    pd.testing.assert_series_equal(pred, X["cf_lag48h"].rename(pred.name))


def test_persistence_raises_without_its_column():
    X, y = _frame()

    with pytest.raises(ValueError, match="cf_lag48h"):
        baselines.Persistence().fit(X.drop(columns="cf_lag48h"), y)


def test_clearsky_persistence_beats_plain_persistence():
    """The point of R2: removing the daily cycle makes it the fair naive competitor."""
    X, y = _frame()

    r1 = baselines.Persistence().fit(X, y).predict(X)
    r2 = baselines.ClearSkyPersistence().fit(X, y).predict(X)

    assert (r2 - y).abs().mean() < (r1 - y).abs().mean()


def test_clearsky_persistence_calibration_is_fitted():
    X, y = _frame()

    model = baselines.ClearSkyPersistence().fit(X, y)

    # beta converts W/m2 into a capacity factor, so it is small but positive.
    assert 0 < model.beta_ < 0.1


def test_clearsky_persistence_is_zero_at_night():
    X, y = _frame()

    pred = baselines.ClearSkyPersistence().fit(X, y).predict(X)

    assert (pred[X["cs_ghi"] == 0] == 0).all()


def test_clearsky_persistence_raises_before_fit():
    X, _ = _frame()

    with pytest.raises(ValueError, match="nicht gefittet"):
        baselines.ClearSkyPersistence().predict(X)


def test_build_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unbekannte Referenz"):
        baselines.build("R9_magic")


def test_combined_reference_is_at_least_as_good_as_its_parts():
    """The point of R3: it cannot be beaten by either component it is built from."""
    X, y = _frame()

    r0 = baselines.Climatology().fit(X, y).predict(X)
    r1 = baselines.Persistence().fit(X, y).predict(X)
    r3 = baselines.CombinedReference().fit(X, y).predict(X)

    def mse(pred):
        return float(((pred - y) ** 2).mean())

    assert mse(r3) <= min(mse(r0), mse(r1)) + 1e-12


def test_combined_reference_weight_stays_a_combination():
    X, y = _frame()

    model = baselines.CombinedReference().fit(X, y)

    assert 0.0 <= model.weight_ <= 1.0


def test_combined_reference_raises_before_fit():
    X, _ = _frame()

    with pytest.raises(ValueError, match="nicht gefittet"):
        baselines.CombinedReference().predict(X)


def test_skill_reference_is_the_combined_reference():
    # Skill against a weak baseline would flatter every model; R3 is the strictest
    # naive reference at this lag.
    assert baselines.SKILL_REFERENCE == baselines.CombinedReference.name
