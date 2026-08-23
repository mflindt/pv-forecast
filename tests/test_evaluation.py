"""Tests for the forecast metrics in pvforecast.evaluation."""

import numpy as np
import pandas as pd
import pytest

from pvforecast import evaluation


def _predictions(n_hours: int = 96, error: float = 100.0) -> pd.DataFrame:
    """Long-format frame with a constant over-forecast and a daily sun cycle."""
    time = pd.date_range("2024-06-01", periods=n_hours, freq="h", tz="UTC")
    elevation = 40 * np.sin((time.hour.to_numpy() - 6) * np.pi / 12)

    return pd.DataFrame(
        {
            "time": time,
            "split": "cv",
            "fold": 1,
            "model": "M",
            "featureset": "S3",
            "seed": 0,
            "y_true_mwh": 1000.0,
            "y_pred_mwh": 1000.0 + error,
            "cap_roll_mwh": 10_000.0,
            "cap_ac_mw": 20_000.0,
            "sun_elevation": elevation,
            "kt": np.clip(elevation / 40, 0, None),
        }
    )


def test_point_metrics_on_a_constant_error():
    metrics = evaluation.point_metrics(
        pd.Series([100.0, 200.0]),
        pd.Series([110.0, 210.0]),
        pd.Series([1000.0, 1000.0]),
    )

    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["rmse"] == pytest.approx(10.0)
    # A pure over-forecast has to show up as a positive bias.
    assert metrics["mbe"] == pytest.approx(10.0)
    assert metrics["nmae"] == pytest.approx(0.01)


def test_point_metrics_normalise_per_timestamp():
    """A growing capacity must not be replaced by a single divisor."""
    metrics = evaluation.point_metrics(
        pd.Series([0.0, 0.0]), pd.Series([100.0, 100.0]), pd.Series([1000.0, 100.0])
    )

    # Mean of 0.1 and 1.0, not 200 / 1100.
    assert metrics["nmae"] == pytest.approx(0.55)


def test_point_metrics_raises_on_empty_input():
    with pytest.raises(ValueError, match="Leere Auswertungsmenge"):
        evaluation.point_metrics(
            pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
        )


def test_daylight_mask_uses_the_astronomical_threshold():
    elevation = pd.Series([-10.0, 0.0, 5.0, 5.1, 30.0])

    mask = evaluation.daylight_mask(elevation)

    assert mask.tolist() == [False, False, False, True, True]


def test_daylight_filter_changes_the_metric():
    """Night-time zeros dilute every metric -- that is why they are excluded."""
    predictions = _predictions()

    day = evaluation.evaluate(predictions, groupby=("model",), daylight_only=True)
    full = evaluation.evaluate(predictions, groupby=("model",), daylight_only=False)

    assert day["n"].iloc[0] < full["n"].iloc[0]


def test_evaluate_reports_both_normalisers():
    predictions = _predictions(error=100.0)

    roll = evaluation.evaluate(predictions, groupby=("model",), normaliser="cap_roll")
    ac = evaluation.evaluate(predictions, groupby=("model",), normaliser="cap_ac")

    assert roll["nmae"].iloc[0] == pytest.approx(0.01)
    # The AC nameplate is twice the rolling capacity here, so the ratio halves.
    assert ac["nmae"].iloc[0] == pytest.approx(0.005)


def test_evaluate_rejects_an_unknown_normaliser():
    with pytest.raises(ValueError, match="Unbekannte Normierung"):
        evaluation.evaluate(_predictions(), normaliser="peak")


def test_check_predictions_rejects_missing_columns():
    predictions = _predictions().drop(columns="kt")

    with pytest.raises(ValueError, match="Spalten fehlen"):
        evaluation.check_predictions(predictions)


def test_check_predictions_rejects_nan():
    predictions = _predictions()
    predictions.loc[0, "y_pred_mwh"] = np.nan

    with pytest.raises(ValueError, match="NaN-Werte"):
        evaluation.check_predictions(predictions)


def test_check_predictions_rejects_non_positive_capacity():
    predictions = _predictions()
    predictions.loc[0, "cap_roll_mwh"] = 0.0

    with pytest.raises(ValueError, match="cap_roll_mwh"):
        evaluation.check_predictions(predictions)


def test_skill_score_is_zero_against_itself():
    good = _predictions(error=50.0).assign(model="good")
    reference = _predictions(error=100.0).assign(model="ref")
    metrics = evaluation.evaluate(pd.concat([good, reference]))

    scored = evaluation.add_skill(metrics, reference="ref")

    assert scored.set_index("model").loc["ref", "skill"] == pytest.approx(0.0)
    # Half the error of the reference means a skill score of 0.5.
    assert scored.set_index("model").loc["good", "skill"] == pytest.approx(0.5)


def test_skill_works_on_pooled_metrics_without_folds():
    """The headline skill score is computed on pooled metrics, not per fold."""
    good = _predictions(error=50.0).assign(model="good")
    reference = _predictions(error=100.0).assign(model="ref")
    pooled = evaluation.evaluate(pd.concat([good, reference]), groupby=("model",))

    scored = evaluation.add_skill(pooled, reference="ref")

    assert scored.set_index("model").loc["good", "skill"] == pytest.approx(0.5)


def test_add_skill_raises_on_an_ambiguous_reference():
    """Several reference rows per key would silently multiply the metric rows."""
    seeds = [_predictions(error=100.0).assign(model="ref", seed=s) for s in (1, 2)]
    metrics = evaluation.evaluate(pd.concat(seeds), groupby=("model", "seed"))

    with pytest.raises(ValueError, match="nicht eindeutig"):
        evaluation.add_skill(metrics, reference="ref")


def test_add_skill_raises_on_missing_reference():
    metrics = evaluation.evaluate(_predictions())

    with pytest.raises(ValueError, match="fehlt in den Metriken"):
        evaluation.add_skill(metrics, reference="R2_clearsky_persistence")


def test_aggregate_folds_reports_mean_and_spread():
    folds = [
        _predictions(error=e).assign(fold=i) for i, e in enumerate([50.0, 150.0], 1)
    ]
    metrics = evaluation.evaluate(pd.concat(folds))

    agg = evaluation.aggregate_folds(metrics)

    assert agg["n_folds"].iloc[0] == 2
    assert agg["nmae_mean"].iloc[0] == pytest.approx(0.01)
    assert agg["nmae_std"].iloc[0] > 0


def test_metrics_record_how_they_were_normalised():
    """A metrics CSV has to be readable without knowing how it was produced."""
    metrics = evaluation.evaluate(_predictions(), normaliser="cap_ac")

    assert (metrics["normaliser"] == "cap_ac").all()
    assert metrics["daylight_only"].all()


def test_aggregate_folds_counts_distinct_folds_not_rows():
    """Several seeds per fold must not inflate the fold count."""
    runs = [_predictions().assign(fold=f, seed=s) for f in (1, 2) for s in (42, 43)]
    metrics = evaluation.evaluate(pd.concat(runs), groupby=("model", "fold", "seed"))

    agg = evaluation.aggregate_folds(metrics, groupby=("model",))

    assert agg["n_folds"].iloc[0] == 2


def test_aggregate_folds_needs_a_fold_column():
    pooled = evaluation.evaluate(_predictions(), groupby=("model",))

    with pytest.raises(ValueError, match="'fold'-Spalte"):
        evaluation.aggregate_folds(pooled, groupby=("model",))


def test_kt_bins_cover_the_cloud_regimes():
    bins = evaluation.kt_bins(pd.Series([0.0, 0.5, 0.9, 1.4]))

    assert bins.tolist() == ["bedeckt", "teilbewölkt", "klar", "klar"]


def test_stratify_by_hour_keeps_only_daylight():
    strata = evaluation.stratify(_predictions(), by="hour")

    assert strata["stratum"].min() > 6
    assert set(strata["model"]) == {"M"}


def test_stratify_rejects_an_unknown_key():
    with pytest.raises(ValueError, match="Unbekannte Schichtung"):
        evaluation.stratify(_predictions(), by="weekday")
