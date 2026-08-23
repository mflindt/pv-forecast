"""Tests for the feature stages in pvforecast.featuresets."""

import pandas as pd
import pytest

from pvforecast import featuresets
from pvforecast.config import PROCESSED_DIR
from pvforecast.features import build_features

MODEL_INPUT = PROCESSED_DIR / "pv_weather_hourly.parquet"


def test_stages_are_nested():
    s1, s2, s3 = (set(featuresets.columns(s)) for s in ("S1", "S2", "S3"))

    assert s1 < s2 < s3


def test_solar_physics_only_adds_transformations():
    """S3 carries no new information, only a different representation of S2."""
    added = set(featuresets.columns("S3")) - set(featuresets.columns("S2"))

    assert added == {"kt", "diffuse_fraction"}


def test_history_stage_needs_no_external_source():
    s1 = featuresets.columns("S1")

    # Solar geometry is computed from the timestamp, so it belongs to S1.
    assert "cs_ghi" in s1
    assert "shortwave_radiation" not in s1


def test_columns_rejects_an_unknown_stage():
    with pytest.raises(ValueError, match="Unbekannte Feature-Stufe"):
        featuresets.columns("S4")


def test_select_reduces_the_matrix():
    X = pd.DataFrame(0.0, index=range(3), columns=featuresets.columns("S3"))

    out = featuresets.select(X, "S1")

    assert list(out.columns) == featuresets.columns("S1")


def test_select_raises_on_a_missing_column():
    X = pd.DataFrame(0.0, index=range(3), columns=["kt"])

    with pytest.raises(ValueError, match="fehlende Spalten"):
        featuresets.select(X, "S1")


@pytest.mark.skipif(not MODEL_INPUT.exists(), reason="Modell-Input nicht gebaut")
def test_full_stage_matches_the_built_feature_matrix():
    """S3 must be exactly what features.build_features produces -- no drift."""
    X, _, _ = build_features(pd.read_parquet(MODEL_INPUT).head(24 * 400))

    assert set(featuresets.columns(featuresets.FULL_STAGE)) == set(X.columns)
