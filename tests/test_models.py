"""All forecasters follow the same interface."""

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from pvforecast import models
from pvforecast.models import MODELS, REFERENCES, ClearSkyPersistence, CombinedReference

ALL_NAMES = list(REFERENCES) + list(MODELS)


# Enough rows to carry the signal, few enough to keep the suite quick.
SAMPLE_HOURS = 24 * 200


def fit_window(dataset):
    """A training and a test block from the shared dataset."""
    X, y, _ = dataset
    X, y = X.iloc[-SAMPLE_HOURS:], y.iloc[-SAMPLE_HOURS:]
    split = len(X) // 2
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


@lru_cache(maxsize=1)
def tabpfn_ready() -> bool:
    """TabPFN-3 needs the gpu extra and accepted licence terms for its weights."""
    try:
        from tabpfn import TabPFNRegressor

        X = pd.DataFrame({"a": np.linspace(0, 1, 32), "b": np.linspace(1, 0, 32)})
        TabPFNRegressor(random_state=0).fit(X, X["a"])
    except Exception:
        return False
    return True


def build(name: str, params: dict | None = None, seed: int = models.DEFAULT_SEED):
    """Instantiate a forecaster; LightGBM gets a small round count for the tests."""
    if name == models.TabPFN3.name and not tabpfn_ready():
        pytest.skip("TabPFN-3-Gewichte nicht verfügbar (gpu-Extra, HF_TOKEN)")
    params = dict(params or {})
    if name == "lightgbm":
        params.setdefault("n_estimators", 60)
    return models.build(name, params, seed)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_forecaster_fits_and_predicts_a_series(name, dataset):
    X_train, y_train, X_test, _ = fit_window(dataset)
    prediction = build(name).fit(X_train, y_train).predict(X_test)

    assert isinstance(prediction, pd.Series)
    assert prediction.index.equals(X_test.index)
    assert prediction.notna().all()
    # Capacity factors are non-negative by construction.
    assert (prediction >= 0).all()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_forecaster_learns_the_signal(name, dataset):
    """Beating a flat mean forecast is the lowest bar a forecaster has to clear."""
    X_train, y_train, X_test, y_test = fit_window(dataset)
    prediction = build(name).fit(X_train, y_train).predict(X_test)

    flat = float(y_train.mean())
    assert np.abs(prediction - y_test).mean() < np.abs(flat - y_test).mean()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_predicting_before_fitting_raises(name, dataset):
    X, _, _ = dataset
    estimator = build(name)
    if name == "R1_persistence":
        pytest.skip("R1 ist zustandslos: der Lag ist bereits eine Spalte")

    with pytest.raises(ValueError, match="gefittet"):
        estimator.predict(X)


def test_combined_reference_is_at_least_as_good_as_its_parts(dataset):
    """R3 is the skill baseline and cannot be outperformed on its training block."""
    X_train, y_train, _, _ = fit_window(dataset)

    combined = build("R3_combined").fit(X_train, y_train).predict(X_train)
    climatology = build("R0_climatology").fit(X_train, y_train).predict(X_train)
    persistence = build("R1_persistence").fit(X_train, y_train).predict(X_train)

    loss = ((combined - y_train) ** 2).mean()
    assert loss <= ((climatology - y_train) ** 2).mean() + 1e-12
    assert loss <= ((persistence - y_train) ** 2).mean() + 1e-12


def test_combined_reference_weight_stays_a_combination(dataset):
    X_train, y_train, _, _ = fit_window(dataset)
    reference = CombinedReference().fit(X_train, y_train)
    assert 0.0 <= reference.weight_ <= 1.0


def test_clearsky_persistence_calibration_is_fitted(dataset):
    X_train, y_train, _, _ = fit_window(dataset)
    reference = ClearSkyPersistence().fit(X_train, y_train)
    assert reference.beta_ is not None and reference.beta_ > 0


def test_ridge_alpha_shrinks_the_fit(dataset):
    X_train, y_train, X_test, _ = fit_window(dataset)
    weak = build("ridge", {"alpha": 1e-3}).fit(X_train, y_train).predict(X_test)
    strong = build("ridge", {"alpha": 1e6}).fit(X_train, y_train).predict(X_test)

    assert strong.std() < weak.std()


def test_lightgbm_early_stopping_freezes_the_round_count(dataset):
    X_train, y_train, X_test, y_test = fit_window(dataset)
    estimator = models.build("lightgbm", {"learning_rate": 0.1})
    estimator.fit(X_train, y_train, validation=(X_test, y_test))

    frozen = estimator.freeze()["n_estimators"]
    assert 0 < frozen < models.MAX_ROUNDS


def test_lightgbm_is_reproducible_under_one_seed(dataset):
    X_train, y_train, X_test, _ = fit_window(dataset)
    params = {"subsample": 0.6, "n_estimators": 30}

    first = (
        models.build("lightgbm", params, seed=7).fit(X_train, y_train).predict(X_test)
    )
    second = (
        models.build("lightgbm", params, seed=7).fit(X_train, y_train).predict(X_test)
    )
    other = (
        models.build("lightgbm", params, seed=8).fit(X_train, y_train).predict(X_test)
    )

    pd.testing.assert_series_equal(first, second)
    assert not first.equals(other)
