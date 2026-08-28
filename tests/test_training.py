"""Tests for splits, tuning, and the fold loop."""

import numpy as np
import pandas as pd
import pytest

import main
from pvforecast import preprocessing, training
from pvforecast.evaluation import PREDICTION_COLUMNS, check_predictions
from pvforecast.training import (
    HOLDOUT_START,
    information_set,
    inner_split,
    rolling_origin_days,
    sample_configs,
)


@pytest.fixture(scope="session")
def folds(dataset):
    X, _, _ = dataset
    return rolling_origin_days(X.index, n_folds=3, test_days=40, gap_hours=48)


def test_gap_between_train_and_test(folds):
    """Without the gap the 48 h lags of the first test hours would sit in training."""
    for train, test in folds:
        assert test.min() - train.max() >= pd.Timedelta(hours=48)
        assert train.intersection(test).empty


def test_test_blocks_are_disjoint_and_ordered(folds):
    blocks = [test for _, test in folds]
    for earlier, later in zip(blocks, blocks[1:], strict=False):
        assert earlier.max() < later.min()
        assert earlier.intersection(later).empty


def test_holdout_year_is_never_touched(dataset):
    X, _, _ = dataset
    index = X.index.append(
        pd.date_range("2025-01-01", periods=24 * 200, freq="h", tz="UTC")
    )

    for train, test in rolling_origin_days(index, n_folds=2, test_days=40):
        assert train.max() < HOLDOUT_START
        assert test.max() < HOLDOUT_START


def test_expanding_grows_while_sliding_stays_constant(dataset):
    X, _, _ = dataset
    expanding = rolling_origin_days(X.index, n_folds=3, test_days=40, mode="expanding")
    sliding = rolling_origin_days(X.index, n_folds=3, test_days=40, mode="sliding")

    assert len(expanding[0][0]) < len(expanding[-1][0])
    assert len({len(train) for train, _ in sliding}) == 1


def test_rolling_origin_raises_when_history_is_too_short(dataset):
    X, _, _ = dataset
    with pytest.raises(ValueError, match="reichen nicht"):
        rolling_origin_days(X.index, n_folds=100, test_days=90)


def test_inner_split_never_reaches_into_the_test_fold(folds):
    """The most critical guard: tuning must not see a single test hour."""
    for train, test in folds:
        core, validation = inner_split(train, validation_days=40, gap_hours=48)

        assert core.intersection(validation).empty
        assert core.intersection(test).empty
        assert validation.intersection(test).empty
        # The validation block is the most recent part of the training window.
        assert validation.max() == train.max()
        assert validation.min() - core.max() >= pd.Timedelta(hours=48)


def test_inner_split_raises_when_the_window_is_too_short(folds):
    train, _ = folds[0]
    with pytest.raises(ValueError, match="reichen nicht"):
        inner_split(train, validation_days=10_000)


def test_sample_configs_is_distinct_and_reproducible():
    space = {"a": [1, 2, 3], "b": [10, 20, 30, 40]}
    drawn = sample_configs(space, 5, np.random.default_rng(42))
    again = sample_configs(space, 5, np.random.default_rng(42))

    assert len(drawn) == 5
    assert len({tuple(sorted(c.items())) for c in drawn}) == 5
    assert drawn == again
    # A budget beyond the grid enumerates it instead of drawing forever.
    assert len(sample_configs(space, 99, np.random.default_rng(0))) == 12


def test_information_set_separates_perfect_prog_from_history():
    assert information_set("S1") == training.HISTORY_ONLY
    assert information_set("S3") == training.PERFECT_PROG
    assert information_set(training.REFERENCE_FEATURESET) == training.HISTORY_ONLY


def test_predictions_satisfy_the_long_format_contract(dataset, small_config):
    X, y, meta = dataset
    predictions, hyperparams, spans = training.run_folds(
        small_config, X, y, meta, pd.Series(dtype=float)
    )

    check_predictions(predictions)
    assert list(predictions.columns) == list(PREDICTION_COLUMNS)
    assert set(predictions["model"]) == set(
        small_config["references"] + small_config["models"]
    )
    # Every forecaster is scored on exactly the same rows.
    counts = predictions.groupby("model", observed=True)["time"].count()
    assert counts.nunique() == 1

    assert (
        len(hyperparams)
        == len(small_config["models"]) * small_config["splits"]["n_folds"]
    )
    assert len(spans) == small_config["splits"]["n_folds"]


def test_predictions_never_come_from_a_model_that_saw_the_test_fold(
    dataset, small_config
):
    """Corrupting the target inside the test fold must not move a single prediction."""
    X, y, meta = dataset
    clean, _, spans = training.run_folds(
        small_config, X, y, meta, pd.Series(dtype=float)
    )

    test_hours = (y.index >= spans["test_start"].min()) & (
        y.index <= spans["test_end"].max()
    )
    tampered_y = y.copy()
    tampered_y[test_hours] = 0.0
    tampered_meta = meta.copy()
    tampered_meta.loc[test_hours, "pv_mwh"] = 0.0

    tampered, _, _ = training.run_folds(
        small_config, X, tampered_y, tampered_meta, pd.Series(dtype=float)
    )
    pd.testing.assert_series_equal(clean["y_pred_mwh"], tampered["y_pred_mwh"])


def test_pipeline_writes_every_artefact_including_figures(
    dataset, small_config, tmp_path, monkeypatch
):
    """The end-to-end smoke test: one call, all four steps, all outputs."""
    X, y, meta = dataset
    monkeypatch.setattr(
        preprocessing, "build_dataset", lambda cfg: (X, y, meta, pd.Series(dtype=float))
    )
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    out = main.run_pipeline(small_config)

    expected = {
        "predictions.parquet",
        "metrics_agg.csv",
        "metrics_fold.csv",
        "strata.csv",
        "tests.csv",
        "folds.csv",
        "hyperparams.json",
        "summary.md",
        "config_resolved.yaml",
    }
    assert expected <= {p.name for p in out.iterdir()}

    figures = sorted(p.name for p in (out / "figures").glob("*.png"))
    assert len(figures) >= 7
    assert figures[0] == "00_splits.png"
    assert (out.parent / "latest").resolve() == out


def test_limit_context_keeps_the_most_recent_rows(dataset):
    """Taking the oldest rows instead would invert the whole context experiment."""
    X, _, _ = dataset
    window = X.index[:1000]

    limited = training.limit_context(window, X["sun_elevation"], context_rows=100)
    assert len(limited) == 100
    assert limited[-1] == window[-1]
    assert limited[0] == window[-100]


def test_limit_context_drops_only_night_hours(dataset):
    """Daylight training must not move the window, only thin it."""
    X, _, _ = dataset
    window = X.index[:1000]

    limited = training.limit_context(window, X["sun_elevation"], daylight_only=True)
    assert 0 < len(limited) < len(window)
    assert (X.loc[limited, "sun_elevation"] > 5.0).all()
