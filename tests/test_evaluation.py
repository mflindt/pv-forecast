"""Metrics have to mean what the thesis says they mean."""

import numpy as np
import pandas as pd
import pytest

from pvforecast import evaluation
from pvforecast.evaluation import (
    DAYLIGHT_ELEVATION_DEG,
    STRATA,
    add_skill,
    aggregate_folds,
    check_predictions,
    daylight_mask,
    evaluate,
    point_metrics,
    runtime,
    score,
    significance_test,
    stratify_all,
)


def test_point_metrics_on_a_constant_error():
    truth = pd.Series([100.0, 200.0, 300.0])
    metrics = point_metrics(truth, truth + 10.0, pd.Series([1000.0] * 3))

    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["rmse"] == pytest.approx(10.0)
    assert metrics["mbe"] == pytest.approx(10.0)
    assert metrics["nmae"] == pytest.approx(0.01)
    assert metrics["n"] == 3


def test_point_metrics_normalise_per_timestamp():
    """Each error is divided by the capacity of its own hour, not by a global mean."""
    truth = pd.Series([0.0, 0.0])
    capacity = pd.Series([100.0, 1000.0])
    metrics = point_metrics(truth, pd.Series([10.0, 10.0]), capacity)

    assert metrics["nmae"] == pytest.approx((0.1 + 0.01) / 2)


def test_point_metrics_rejects_an_empty_set():
    empty = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="Leere Auswertungsmenge"):
        point_metrics(empty, empty, empty)


def test_daylight_mask_uses_the_astronomical_threshold():
    elevation = pd.Series([-10.0, 0.0, DAYLIGHT_ELEVATION_DEG, 30.0])
    assert daylight_mask(elevation).tolist() == [False, False, False, True]


def test_runtime_counts_fits_not_forecast_hours(prediction_frame):
    """Averaging over rows would weight a fit by the length of its test block."""
    table = runtime(prediction_frame)
    lgbm = table[table["model"] == "lightgbm"].iloc[0]

    assert set(table["model"]) == {"R3_combined", "lightgbm"}
    # Two folds, one seed: two fits, not 2 x 480 forecast hours.
    assert lgbm["n_fits"] == 2
    assert lgbm["fit_seconds"] == pytest.approx(3.2)
    assert lgbm["fit_seconds_total"] == pytest.approx(6.4)
    assert lgbm["predict_seconds"] == pytest.approx(0.05)


def test_daylight_filter_changes_the_metric(prediction_frame):
    day = evaluate(prediction_frame, groupby=("model",), daylight_only=True)
    full = evaluate(prediction_frame, groupby=("model",), daylight_only=False)

    assert (day["n"] < full["n"]).all()
    assert not np.allclose(day["nmae"], full["nmae"])


def test_metrics_record_how_they_were_normalised(prediction_frame):
    metrics = evaluate(prediction_frame, groupby=("model",), normaliser="cap_ac")
    assert (metrics["normaliser"] == "cap_ac").all()
    assert metrics["daylight_only"].all()


def test_evaluate_rejects_an_unknown_normaliser(prediction_frame):
    with pytest.raises(ValueError, match="Unbekannte Normierung"):
        evaluate(prediction_frame, normaliser="nameplate")


def test_check_predictions_rejects_a_broken_frame(prediction_frame):
    with pytest.raises(ValueError, match="Spalten fehlen"):
        check_predictions(prediction_frame.drop(columns="kt"))

    with_nan = prediction_frame.copy()
    with_nan.loc[0, "y_pred_mwh"] = np.nan
    with pytest.raises(ValueError, match="NaN-Werte"):
        check_predictions(with_nan)

    zero_capacity = prediction_frame.copy()
    zero_capacity.loc[0, "cap_roll_mwh"] = 0.0
    with pytest.raises(ValueError, match="<= 0"):
        check_predictions(zero_capacity)


def test_skill_score_is_zero_against_itself(prediction_frame):
    metrics = evaluate(prediction_frame, groupby=("model", "featureset", "fold"))
    scored = add_skill(metrics, "R3_combined")

    reference = scored[scored["model"] == "R3_combined"]
    assert np.allclose(reference["skill"], 0.0)
    # The better forecast has to come out with a positive skill score.
    assert (scored[scored["model"] == "lightgbm"]["skill"] > 0).all()


def test_add_skill_raises_on_a_missing_reference(prediction_frame):
    metrics = evaluate(prediction_frame, groupby=("model", "fold"))
    with pytest.raises(ValueError, match="fehlt in den Metriken"):
        add_skill(metrics, "R9_does_not_exist")


def test_aggregate_folds_separates_seed_spread_from_fold_spread():
    """Repeats are collapsed per fold first; otherwise the spread mixes two sources."""
    rows = []
    for fold, level in ((1, 0.10), (2, 0.20)):
        for seed, offset in ((42, -0.01), (43, 0.01)):
            rows.append(
                {
                    "model": "lightgbm",
                    "featureset": "S3",
                    "context_rows": 0,
                    "fold": fold,
                    "seed": seed,
                    **{s: level + offset for s in evaluation.SCORES},
                }
            )
    agg = aggregate_folds(pd.DataFrame(rows)).iloc[0]

    assert agg["nmae_mean"] == pytest.approx(0.15)
    # Spread over the two fold means (0.10, 0.20), not over the four raw rows.
    assert agg["nmae_std"] == pytest.approx(np.std([0.10, 0.20], ddof=1))
    assert agg["nmae_sd_seed"] == pytest.approx(np.std([-0.01, 0.01], ddof=1))
    assert agg["n_folds"] == 2


def test_aggregate_folds_needs_a_fold_column():
    with pytest.raises(ValueError, match="'fold'-Spalte"):
        aggregate_folds(pd.DataFrame({"model": ["a"], "nmae": [0.1]}))


def test_stratify_all_covers_every_stratum(prediction_frame):
    strata = stratify_all(prediction_frame)

    assert set(strata["stratum_type"]) == set(STRATA)
    # One string column, so the mixed keys of the four stratifications share a CSV.
    assert strata["stratum"].map(type).eq(str).all()
    hours = strata[strata["stratum_type"] == "hour"]["stratum"].astype(int)
    # Only daylight hours survive the mask.
    assert hours.max() < 24 and len(set(hours)) < 24


def test_significance_test_finds_a_real_difference(prediction_frame):
    tests = significance_test(prediction_frame, "R3_combined")
    row = tests.iloc[0]

    assert row["model"] == "lightgbm"
    # The better forecast carries a negative loss differential.
    assert row["mean_loss_diff"] < 0
    assert row["p_holm"] < 0.05
    assert row["n_days"] >= evaluation.MIN_TEST_DAYS


def test_significance_test_stays_silent_without_a_difference(prediction_frame):
    twin = prediction_frame[prediction_frame["model"] == "R3_combined"].copy()
    twin["model"] = "R3_copy"
    tests = significance_test(pd.concat([prediction_frame, twin]), "R3_combined")

    assert tests.loc[tests["model"] == "R3_copy", "p_holm"].iloc[0] == pytest.approx(
        1.0
    )


def test_score_produces_every_table(prediction_frame, small_config):
    results = score(small_config, prediction_frame)

    assert set(results) == {"metrics_fold", "metrics_agg", "strata", "runtime", "tests"}
    assert "skill" in results["metrics_agg"].columns
    assert "nmae_std" in results["metrics_agg"].columns
    # Best model first, so the head of the table is the headline result.
    assert results["metrics_agg"]["nmae"].is_monotonic_increasing


def test_merge_rejects_runs_with_unequal_coverage(prediction_frame):
    """A model scored on fewer hours than another is a silent unfair comparison."""
    full = prediction_frame[prediction_frame["model"] == "R3_combined"]
    short = prediction_frame[prediction_frame["model"] == "lightgbm"].iloc[:100]

    with pytest.raises(ValueError, match="unterschiedlich viele Stunden"):
        evaluation.merge_predictions([full, short])
